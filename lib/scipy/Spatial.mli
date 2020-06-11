(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module ConvexHull : sig
type tag = [`ConvexHull]
type t = [`ConvexHull | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?incremental:bool -> ?qhull_options:string -> points:[>`Ndarray] Np.Obj.t -> unit -> t
(**
ConvexHull(points, incremental=False, qhull_options=None)

Convex hulls in N dimensions.

.. versionadded:: 0.12.0

Parameters
----------
points : ndarray of floats, shape (npoints, ndim)
    Coordinates of points to construct a convex hull from
incremental : bool, optional
    Allow adding new points incrementally. This takes up some additional
    resources.
qhull_options : str, optional
    Additional options to pass to Qhull. See Qhull manual
    for details. (Default: 'Qx' for ndim > 4 and '' otherwise)
    Option 'Qt' is always enabled.

Attributes
----------
points : ndarray of double, shape (npoints, ndim)
    Coordinates of input points.
vertices : ndarray of ints, shape (nvertices,)
    Indices of points forming the vertices of the convex hull.
    For 2-D convex hulls, the vertices are in counterclockwise order.
    For other dimensions, they are in input order.
simplices : ndarray of ints, shape (nfacet, ndim)
    Indices of points forming the simplical facets of the convex hull.
neighbors : ndarray of ints, shape (nfacet, ndim)
    Indices of neighbor facets for each facet.
    The kth neighbor is opposite to the kth vertex.
    -1 denotes no neighbor.
equations : ndarray of double, shape (nfacet, ndim+1)
    [normal, offset] forming the hyperplane equation of the facet
    (see `Qhull documentation <http://www.qhull.org/>`__  for more).
coplanar : ndarray of int, shape (ncoplanar, 3)
    Indices of coplanar points and the corresponding indices of
    the nearest facets and nearest vertex indices.  Coplanar
    points are input points which were *not* included in the
    triangulation due to numerical precision issues.

    If option 'Qc' is not specified, this list is not computed.
good : ndarray of bool or None
    A one-dimensional Boolean array indicating which facets are
    good. Used with options that compute good facets, e.g. QGn
    and QG-n. Good facets are defined as those that are
    visible (n) or invisible (-n) from point n, where
    n is the nth point in 'points'. The 'good' attribute may be
    used as an index into 'simplices' to return the good (visible)
    facets: simplices[good]. A facet is visible from the outside
    of the hull only, and neither coplanarity nor degeneracy count
    as cases of visibility.

    If a 'QGn' or 'QG-n' option is not specified, None is returned.

    .. versionadded:: 1.3.0
area : float
    Area of the convex hull.

    .. versionadded:: 0.17.0
volume : float
    Volume of the convex hull.

    .. versionadded:: 0.17.0

Raises
------
QhullError
    Raised when Qhull encounters an error condition, such as
    geometrical degeneracy when options to resolve are not enabled.
ValueError
    Raised if an incompatible array is given as input.

Notes
-----
The convex hull is computed using the
`Qhull library <http://www.qhull.org/>`__.

Examples
--------

Convex hull of a random set of points:

>>> from scipy.spatial import ConvexHull, convex_hull_plot_2d
>>> points = np.random.rand(30, 2)   # 30 random points in 2-D
>>> hull = ConvexHull(points)

Plot it:

>>> import matplotlib.pyplot as plt
>>> plt.plot(points[:,0], points[:,1], 'o')
>>> for simplex in hull.simplices:
...     plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

We could also have directly used the vertices of the hull, which
for 2-D are guaranteed to be in counterclockwise order:

>>> plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
>>> plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
>>> plt.show()

Facets visible from a point:

Create a square and add a point above the square.

>>> generators = np.array([[0.2, 0.2],
...                        [0.2, 0.4],
...                        [0.4, 0.4],
...                        [0.4, 0.2],
...                        [0.3, 0.6]])

Call ConvexHull with the QG option. QG4 means 
compute the portions of the hull not including
point 4, indicating the facets that are visible 
from point 4.

>>> hull = ConvexHull(points=generators,
...                   qhull_options='QG4')

The 'good' array indicates which facets are 
visible from point 4.

>>> print(hull.simplices)
    [[1 0]
     [1 2]
     [3 0]
     [3 2]]
>>> print(hull.good)
    [False  True False False]

Now plot it, highlighting the visible facets.

>>> fig = plt.figure()
>>> ax = fig.add_subplot(1,1,1)
>>> for visible_facet in hull.simplices[hull.good]:
...     ax.plot(hull.points[visible_facet, 0],
...             hull.points[visible_facet, 1],
...             color='violet',
...             lw=6)
>>> convex_hull_plot_2d(hull, ax=ax)
    <Figure size 640x480 with 1 Axes> # may vary
>>> plt.show()

References
----------
.. [Qhull] http://www.qhull.org/
*)

val add_points : ?restart:bool -> points:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
add_points(points, restart=False)

Process a set of additional new points.

Parameters
----------
points : ndarray
    New points to add. The dimensionality should match that of the
    initial points.
restart : bool, optional
    Whether to restart processing from scratch, rather than
    adding points incrementally.

Raises
------
QhullError
    Raised when Qhull encounters an error condition, such as
    geometrical degeneracy when options to resolve are not enabled.

See Also
--------
close

Notes
-----
You need to specify ``incremental=True`` when constructing the
object to be able to add points incrementally. Incremental addition
of points is also not possible after `close` has been called.
*)

val close : [> tag] Obj.t -> Py.Object.t
(**
close()

Finish incremental processing.

Call this to free resources taken up by Qhull, when using the
incremental mode. After calling this, adding more points is no
longer possible.
*)


(** Attribute points: get value or raise Not_found if None.*)
val points : t -> Py.Object.t

(** Attribute points: get value as an option. *)
val points_opt : t -> (Py.Object.t) option


(** Attribute vertices: get value or raise Not_found if None.*)
val vertices : t -> Py.Object.t

(** Attribute vertices: get value as an option. *)
val vertices_opt : t -> (Py.Object.t) option


(** Attribute simplices: get value or raise Not_found if None.*)
val simplices : t -> Py.Object.t

(** Attribute simplices: get value as an option. *)
val simplices_opt : t -> (Py.Object.t) option


(** Attribute neighbors: get value or raise Not_found if None.*)
val neighbors : t -> Py.Object.t

(** Attribute neighbors: get value as an option. *)
val neighbors_opt : t -> (Py.Object.t) option


(** Attribute equations: get value or raise Not_found if None.*)
val equations : t -> Py.Object.t

(** Attribute equations: get value as an option. *)
val equations_opt : t -> (Py.Object.t) option


(** Attribute coplanar: get value or raise Not_found if None.*)
val coplanar : t -> Py.Object.t

(** Attribute coplanar: get value as an option. *)
val coplanar_opt : t -> (Py.Object.t) option


(** Attribute good: get value or raise Not_found if None.*)
val good : t -> Py.Object.t

(** Attribute good: get value as an option. *)
val good_opt : t -> (Py.Object.t) option


(** Attribute area: get value or raise Not_found if None.*)
val area : t -> float

(** Attribute area: get value as an option. *)
val area_opt : t -> (float) option


(** Attribute volume: get value or raise Not_found if None.*)
val volume : t -> float

(** Attribute volume: get value as an option. *)
val volume_opt : t -> (float) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Delaunay : sig
type tag = [`Delaunay]
type t = [`Delaunay | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?furthest_site:bool -> ?incremental:bool -> ?qhull_options:string -> points:[>`Ndarray] Np.Obj.t -> unit -> t
(**
Delaunay(points, furthest_site=False, incremental=False, qhull_options=None)

Delaunay tessellation in N dimensions.

.. versionadded:: 0.9

Parameters
----------
points : ndarray of floats, shape (npoints, ndim)
    Coordinates of points to triangulate
furthest_site : bool, optional
    Whether to compute a furthest-site Delaunay triangulation.
    Default: False

    .. versionadded:: 0.12.0
incremental : bool, optional
    Allow adding new points incrementally. This takes up some additional
    resources.
qhull_options : str, optional
    Additional options to pass to Qhull. See Qhull manual for
    details. Option 'Qt' is always enabled.
    Default:'Qbb Qc Qz Qx Q12' for ndim > 4 and 'Qbb Qc Qz Q12' otherwise.
    Incremental mode omits 'Qz'.

    .. versionadded:: 0.12.0

Attributes
----------
points : ndarray of double, shape (npoints, ndim)
    Coordinates of input points.
simplices : ndarray of ints, shape (nsimplex, ndim+1)
    Indices of the points forming the simplices in the triangulation.
    For 2-D, the points are oriented counterclockwise.
neighbors : ndarray of ints, shape (nsimplex, ndim+1)
    Indices of neighbor simplices for each simplex.
    The kth neighbor is opposite to the kth vertex.
    For simplices at the boundary, -1 denotes no neighbor.
equations : ndarray of double, shape (nsimplex, ndim+2)
    [normal, offset] forming the hyperplane equation of the facet
    on the paraboloid
    (see `Qhull documentation <http://www.qhull.org/>`__ for more).
paraboloid_scale, paraboloid_shift : float
    Scale and shift for the extra paraboloid dimension
    (see `Qhull documentation <http://www.qhull.org/>`__ for more).
transform : ndarray of double, shape (nsimplex, ndim+1, ndim)
    Affine transform from ``x`` to the barycentric coordinates ``c``.
    This is defined by::

        T c = x - r

    At vertex ``j``, ``c_j = 1`` and the other coordinates zero.

    For simplex ``i``, ``transform[i,:ndim,:ndim]`` contains
    inverse of the matrix ``T``, and ``transform[i,ndim,:]``
    contains the vector ``r``.

    If the simplex is degenerate or nearly degenerate, its
    barycentric transform contains NaNs.
vertex_to_simplex : ndarray of int, shape (npoints,)
    Lookup array, from a vertex, to some simplex which it is a part of.
    If qhull option 'Qc' was not specified, the list will contain -1
    for points that are not vertices of the tessellation.
convex_hull : ndarray of int, shape (nfaces, ndim)
    Vertices of facets forming the convex hull of the point set.
    The array contains the indices of the points belonging to
    the (N-1)-dimensional facets that form the convex hull
    of the triangulation.

    .. note::

       Computing convex hulls via the Delaunay triangulation is
       inefficient and subject to increased numerical instability.
       Use `ConvexHull` instead.
coplanar : ndarray of int, shape (ncoplanar, 3)
    Indices of coplanar points and the corresponding indices of
    the nearest facet and the nearest vertex.  Coplanar
    points are input points which were *not* included in the
    triangulation due to numerical precision issues.

    If option 'Qc' is not specified, this list is not computed.

    .. versionadded:: 0.12.0
vertices
    Same as `simplices`, but deprecated.
vertex_neighbor_vertices : tuple of two ndarrays of int; (indptr, indices)
    Neighboring vertices of vertices. The indices of neighboring
    vertices of vertex `k` are ``indices[indptr[k]:indptr[k+1]]``.
furthest_site
    True if this was a furthest site triangulation and False if not.

    .. versionadded:: 1.4.0

Raises
------
QhullError
    Raised when Qhull encounters an error condition, such as
    geometrical degeneracy when options to resolve are not enabled.
ValueError
    Raised if an incompatible array is given as input.

Notes
-----
The tessellation is computed using the Qhull library
`Qhull library <http://www.qhull.org/>`__.

.. note::

   Unless you pass in the Qhull option 'QJ', Qhull does not
   guarantee that each input point appears as a vertex in the
   Delaunay triangulation. Omitted points are listed in the
   `coplanar` attribute.

Examples
--------
Triangulation of a set of points:

>>> points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])
>>> from scipy.spatial import Delaunay
>>> tri = Delaunay(points)

We can plot it:

>>> import matplotlib.pyplot as plt
>>> plt.triplot(points[:,0], points[:,1], tri.simplices)
>>> plt.plot(points[:,0], points[:,1], 'o')
>>> plt.show()

Point indices and coordinates for the two triangles forming the
triangulation:

>>> tri.simplices
array([[2, 3, 0],                 # may vary
       [3, 1, 0]], dtype=int32)

Note that depending on how rounding errors go, the simplices may
be in a different order than above.

>>> points[tri.simplices]
array([[[ 1. ,  0. ],            # may vary
        [ 1. ,  1. ],
        [ 0. ,  0. ]],
       [[ 1. ,  1. ],
        [ 0. ,  1.1],
        [ 0. ,  0. ]]])

Triangle 0 is the only neighbor of triangle 1, and it's opposite to
vertex 1 of triangle 1:

>>> tri.neighbors[1]
array([-1,  0, -1], dtype=int32)
>>> points[tri.simplices[1,1]]
array([ 0. ,  1.1])

We can find out which triangle points are in:

>>> p = np.array([(0.1, 0.2), (1.5, 0.5), (0.5, 1.05)])
>>> tri.find_simplex(p)
array([ 1, -1, 1], dtype=int32)

The returned integers in the array are the indices of the simplex the
corresponding point is in. If -1 is returned, the point is in no simplex.
Be aware that the shortcut in the following example only works corretcly
for valid points as invalid points result in -1 which is itself a valid
index for the last simplex in the list.

>>> p_valids = np.array([(0.1, 0.2), (0.5, 1.05)])
>>> tri.simplices[tri.find_simplex(p_valids)]
array([[3, 1, 0],                 # may vary
       [3, 1, 0]], dtype=int32)

We can also compute barycentric coordinates in triangle 1 for
these points:

>>> b = tri.transform[1,:2].dot(np.transpose(p - tri.transform[1,2]))
>>> np.c_[np.transpose(b), 1 - b.sum(axis=0)]
array([[ 0.1       ,  0.09090909,  0.80909091],
       [ 1.5       , -0.90909091,  0.40909091],
       [ 0.5       ,  0.5       ,  0.        ]])

The coordinates for the first point are all positive, meaning it
is indeed inside the triangle. The third point is on a vertex,
hence its null third coordinate.
*)

val add_points : ?restart:bool -> points:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
add_points(points, restart=False)

Process a set of additional new points.

Parameters
----------
points : ndarray
    New points to add. The dimensionality should match that of the
    initial points.
restart : bool, optional
    Whether to restart processing from scratch, rather than
    adding points incrementally.

Raises
------
QhullError
    Raised when Qhull encounters an error condition, such as
    geometrical degeneracy when options to resolve are not enabled.

See Also
--------
close

Notes
-----
You need to specify ``incremental=True`` when constructing the
object to be able to add points incrementally. Incremental addition
of points is also not possible after `close` has been called.
*)

val close : [> tag] Obj.t -> Py.Object.t
(**
close()

Finish incremental processing.

Call this to free resources taken up by Qhull, when using the
incremental mode. After calling this, adding more points is no
longer possible.
*)

val find_simplex : ?bruteforce:bool -> ?tol:float -> xi:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
find_simplex(self, xi, bruteforce=False, tol=None)

Find the simplices containing the given points.

Parameters
----------
tri : DelaunayInfo
    Delaunay triangulation
xi : ndarray of double, shape (..., ndim)
    Points to locate
bruteforce : bool, optional
    Whether to only perform a brute-force search
tol : float, optional
    Tolerance allowed in the inside-triangle check.
    Default is ``100*eps``.

Returns
-------
i : ndarray of int, same shape as `xi`
    Indices of simplices containing each point.
    Points outside the triangulation get the value -1.

Notes
-----
This uses an algorithm adapted from Qhull's ``qh_findbestfacet``,
which makes use of the connection between a convex hull and a
Delaunay triangulation. After finding the simplex closest to
the point in N+1 dimensions, the algorithm falls back to
directed search in N dimensions.
*)

val lift_points : x:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
lift_points(self, x)

Lift points to the Qhull paraboloid.
*)

val plane_distance : xi:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
plane_distance(self, xi)

Compute hyperplane distances to the point `xi` from all simplices.
*)


(** Attribute points: get value or raise Not_found if None.*)
val points : t -> Py.Object.t

(** Attribute points: get value as an option. *)
val points_opt : t -> (Py.Object.t) option


(** Attribute simplices: get value or raise Not_found if None.*)
val simplices : t -> Py.Object.t

(** Attribute simplices: get value as an option. *)
val simplices_opt : t -> (Py.Object.t) option


(** Attribute neighbors: get value or raise Not_found if None.*)
val neighbors : t -> Py.Object.t

(** Attribute neighbors: get value as an option. *)
val neighbors_opt : t -> (Py.Object.t) option


(** Attribute equations: get value or raise Not_found if None.*)
val equations : t -> Py.Object.t

(** Attribute equations: get value as an option. *)
val equations_opt : t -> (Py.Object.t) option


(** Attribute transform: get value or raise Not_found if None.*)
val transform : t -> Py.Object.t

(** Attribute transform: get value as an option. *)
val transform_opt : t -> (Py.Object.t) option


(** Attribute vertex_to_simplex: get value or raise Not_found if None.*)
val vertex_to_simplex : t -> Py.Object.t

(** Attribute vertex_to_simplex: get value as an option. *)
val vertex_to_simplex_opt : t -> (Py.Object.t) option


(** Attribute convex_hull: get value or raise Not_found if None.*)
val convex_hull : t -> Py.Object.t

(** Attribute convex_hull: get value as an option. *)
val convex_hull_opt : t -> (Py.Object.t) option


(** Attribute coplanar: get value or raise Not_found if None.*)
val coplanar : t -> Py.Object.t

(** Attribute coplanar: get value as an option. *)
val coplanar_opt : t -> (Py.Object.t) option


(** Attribute vertices: get value or raise Not_found if None.*)
val vertices : t -> Py.Object.t

(** Attribute vertices: get value as an option. *)
val vertices_opt : t -> (Py.Object.t) option


(** Attribute vertex_neighbor_vertices: get value or raise Not_found if None.*)
val vertex_neighbor_vertices : t -> Py.Object.t

(** Attribute vertex_neighbor_vertices: get value as an option. *)
val vertex_neighbor_vertices_opt : t -> (Py.Object.t) option


(** Attribute furthest_site: get value or raise Not_found if None.*)
val furthest_site : t -> Py.Object.t

(** Attribute furthest_site: get value as an option. *)
val furthest_site_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module HalfspaceIntersection : sig
type tag = [`HalfspaceIntersection]
type t = [`HalfspaceIntersection | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?incremental:bool -> ?qhull_options:string -> halfspaces:[>`Ndarray] Np.Obj.t -> interior_point:[>`Ndarray] Np.Obj.t -> unit -> t
(**
HalfspaceIntersection(halfspaces, interior_point, incremental=False, qhull_options=None)

Halfspace intersections in N dimensions.

.. versionadded:: 0.19.0

Parameters
----------
halfspaces : ndarray of floats, shape (nineq, ndim+1)
    Stacked Inequalities of the form Ax + b <= 0 in format [A; b]
interior_point : ndarray of floats, shape (ndim,)
    Point clearly inside the region defined by halfspaces. Also called a feasible
    point, it can be obtained by linear programming.
incremental : bool, optional
    Allow adding new halfspaces incrementally. This takes up some additional
    resources.
qhull_options : str, optional
    Additional options to pass to Qhull. See Qhull manual
    for details. (Default: 'Qx' for ndim > 4 and '' otherwise)
    Option 'H' is always enabled.

Attributes
----------
halfspaces : ndarray of double, shape (nineq, ndim+1)
    Input halfspaces.
interior_point :ndarray of floats, shape (ndim,)
    Input interior point.
intersections : ndarray of double, shape (ninter, ndim)
    Intersections of all halfspaces.
dual_points : ndarray of double, shape (nineq, ndim)
    Dual points of the input halfspaces.
dual_facets : list of lists of ints
    Indices of points forming the (non necessarily simplicial) facets of
    the dual convex hull.
dual_vertices : ndarray of ints, shape (nvertices,)
    Indices of halfspaces forming the vertices of the dual convex hull.
    For 2-D convex hulls, the vertices are in counterclockwise order.
    For other dimensions, they are in input order.
dual_equations : ndarray of double, shape (nfacet, ndim+1)
    [normal, offset] forming the hyperplane equation of the dual facet
    (see `Qhull documentation <http://www.qhull.org/>`__  for more).
dual_area : float
    Area of the dual convex hull
dual_volume : float
    Volume of the dual convex hull

Raises
------
QhullError
    Raised when Qhull encounters an error condition, such as
    geometrical degeneracy when options to resolve are not enabled.
ValueError
    Raised if an incompatible array is given as input.

Notes
-----
The intersections are computed using the
`Qhull library <http://www.qhull.org/>`__.
This reproduces the 'qhalf' functionality of Qhull.

Examples
--------

Halfspace intersection of planes forming some polygon

>>> from scipy.spatial import HalfspaceIntersection
>>> halfspaces = np.array([[-1, 0., 0.],
...                        [0., -1., 0.],
...                        [2., 1., -4.],
...                        [-0.5, 1., -2.]])
>>> feasible_point = np.array([0.5, 0.5])
>>> hs = HalfspaceIntersection(halfspaces, feasible_point)

Plot halfspaces as filled regions and intersection points:

>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> ax = fig.add_subplot('111', aspect='equal')
>>> xlim, ylim = (-1, 3), (-1, 3)
>>> ax.set_xlim(xlim)
>>> ax.set_ylim(ylim)
>>> x = np.linspace(-1, 3, 100)
>>> symbols = ['-', '+', 'x', '*']
>>> signs = [0, 0, -1, -1]
>>> fmt = {'color': None, 'edgecolor': 'b', 'alpha': 0.5}
>>> for h, sym, sign in zip(halfspaces, symbols, signs):
...     hlist = h.tolist()
...     fmt['hatch'] = sym
...     if h[1]== 0:
...         ax.axvline(-h[2]/h[0], label='{}x+{}y+{}=0'.format( *hlist))
...         xi = np.linspace(xlim[sign], -h[2]/h[0], 100)
...         ax.fill_between(xi, ylim[0], ylim[1], **fmt)
...     else:
...         ax.plot(x, (-h[2]-h[0]*x)/h[1], label='{}x+{}y+{}=0'.format( *hlist))
...         ax.fill_between(x, (-h[2]-h[0]*x)/h[1], ylim[sign], **fmt)
>>> x, y = zip( *hs.intersections)
>>> ax.plot(x, y, 'o', markersize=8)

By default, qhull does not provide with a way to compute an interior point.
This can easily be computed using linear programming. Considering halfspaces
of the form :math:`Ax + b \leq 0`, solving the linear program:

.. math::

    max \: y

    s.t. Ax + y ||A_i|| \leq -b

With :math:`A_i` being the rows of A, i.e. the normals to each plane.

Will yield a point x that is furthest inside the convex polyhedron. To
be precise, it is the center of the largest hypersphere of radius y
inscribed in the polyhedron. This point is called the Chebyshev center
of the polyhedron (see [1]_ 4.3.1, pp148-149). The
equations outputted by Qhull are always normalized.

>>> from scipy.optimize import linprog
>>> from matplotlib.patches import Circle
>>> norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1),
...     (halfspaces.shape[0], 1))
>>> c = np.zeros((halfspaces.shape[1],))
>>> c[-1] = -1
>>> A = np.hstack((halfspaces[:, :-1], norm_vector))
>>> b = - halfspaces[:, -1:]
>>> res = linprog(c, A_ub=A, b_ub=b)
>>> x = res.x[:-1]
>>> y = res.x[-1]
>>> circle = Circle(x, radius=y, alpha=0.3)
>>> ax.add_patch(circle)
>>> plt.legend(bbox_to_anchor=(1.6, 1.0))
>>> plt.show()

References
----------
.. [Qhull] http://www.qhull.org/
.. [1] S. Boyd, L. Vandenberghe, Convex Optimization, available
       at http://stanford.edu/~boyd/cvxbook/
*)

val add_halfspaces : ?restart:bool -> halfspaces:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
add_halfspaces(halfspaces, restart=False)

Process a set of additional new halfspaces.

Parameters
----------
halfspaces : ndarray
    New halfspaces to add. The dimensionality should match that of the
    initial halfspaces.
restart : bool, optional
    Whether to restart processing from scratch, rather than
    adding halfspaces incrementally.

Raises
------
QhullError
    Raised when Qhull encounters an error condition, such as
    geometrical degeneracy when options to resolve are not enabled.

See Also
--------
close

Notes
-----
You need to specify ``incremental=True`` when constructing the
object to be able to add halfspaces incrementally. Incremental addition
of halfspaces is also not possible after `close` has been called.
*)

val close : [> tag] Obj.t -> Py.Object.t
(**
close()

Finish incremental processing.

Call this to free resources taken up by Qhull, when using the
incremental mode. After calling this, adding more points is no
longer possible.
*)


(** Attribute halfspaces: get value or raise Not_found if None.*)
val halfspaces : t -> Py.Object.t

(** Attribute halfspaces: get value as an option. *)
val halfspaces_opt : t -> (Py.Object.t) option


(** Attribute interior_point: get value or raise Not_found if None.*)
val interior_point : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute interior_point: get value as an option. *)
val interior_point_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute intersections: get value or raise Not_found if None.*)
val intersections : t -> Py.Object.t

(** Attribute intersections: get value as an option. *)
val intersections_opt : t -> (Py.Object.t) option


(** Attribute dual_points: get value or raise Not_found if None.*)
val dual_points : t -> Py.Object.t

(** Attribute dual_points: get value as an option. *)
val dual_points_opt : t -> (Py.Object.t) option


(** Attribute dual_facets: get value or raise Not_found if None.*)
val dual_facets : t -> Py.Object.t

(** Attribute dual_facets: get value as an option. *)
val dual_facets_opt : t -> (Py.Object.t) option


(** Attribute dual_vertices: get value or raise Not_found if None.*)
val dual_vertices : t -> Py.Object.t

(** Attribute dual_vertices: get value as an option. *)
val dual_vertices_opt : t -> (Py.Object.t) option


(** Attribute dual_equations: get value or raise Not_found if None.*)
val dual_equations : t -> Py.Object.t

(** Attribute dual_equations: get value as an option. *)
val dual_equations_opt : t -> (Py.Object.t) option


(** Attribute dual_area: get value or raise Not_found if None.*)
val dual_area : t -> float

(** Attribute dual_area: get value as an option. *)
val dual_area_opt : t -> (float) option


(** Attribute dual_volume: get value or raise Not_found if None.*)
val dual_volume : t -> float

(** Attribute dual_volume: get value as an option. *)
val dual_volume_opt : t -> (float) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module KDTree : sig
type tag = [`KDTree]
type t = [`KDTree | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?leafsize:int -> data:Py.Object.t -> unit -> t
(**
kd-tree for quick nearest-neighbor lookup

This class provides an index into a set of k-dimensional points which
can be used to rapidly look up the nearest neighbors of any point.

Parameters
----------
data : (N,K) array_like
    The data points to be indexed. This array is not copied, and
    so modifying this data will result in bogus results.
leafsize : int, optional
    The number of points at which the algorithm switches over to
    brute-force.  Has to be positive.

Raises
------
RuntimeError
    The maximum recursion limit can be exceeded for large data
    sets.  If this happens, either increase the value for the `leafsize`
    parameter or increase the recursion limit by::

        >>> import sys
        >>> sys.setrecursionlimit(10000)

See Also
--------
cKDTree : Implementation of `KDTree` in Cython

Notes
-----
The algorithm used is described in Maneewongvatana and Mount 1999.
The general idea is that the kd-tree is a binary tree, each of whose
nodes represents an axis-aligned hyperrectangle. Each node specifies
an axis and splits the set of points based on whether their coordinate
along that axis is greater than or less than a particular value.

During construction, the axis and splitting point are chosen by the
'sliding midpoint' rule, which ensures that the cells do not all
become long and thin.

The tree can be queried for the r closest neighbors of any given point
(optionally returning only those within some maximum distance of the
point). It can also be queried, with a substantial gain in efficiency,
for the r approximate closest neighbors.

For large dimensions (20 is already large) do not expect this to run
significantly faster than brute force. High-dimensional nearest-neighbor
queries are a substantial open problem in computer science.

The tree also supports all-neighbors queries, both with arrays of points
and with other kd-trees. These do use a reasonably efficient algorithm,
but the kd-tree is not necessarily the best data structure for this
sort of calculation.
*)

val count_neighbors : ?p:[`T1_p_infinity of Py.Object.t | `F of float] -> other:Py.Object.t -> r:[`Ndarray of [>`Ndarray] Np.Obj.t | `F of float] -> [> tag] Obj.t -> Py.Object.t
(**
Count how many nearby pairs can be formed.

Count the number of pairs (x1,x2) can be formed, with x1 drawn
from self and x2 drawn from ``other``, and where
``distance(x1, x2, p) <= r``.
This is the 'two-point correlation' described in Gray and Moore 2000,
'N-body problems in statistical learning', and the code here is based
on their algorithm.

Parameters
----------
other : KDTree instance
    The other tree to draw points from.
r : float or one-dimensional array of floats
    The radius to produce a count for. Multiple radii are searched with
    a single tree traversal.
p : float, 1<=p<=infinity, optional
    Which Minkowski p-norm to use

Returns
-------
result : int or 1-D array of ints
    The number of pairs. Note that this is internally stored in a numpy
    int, and so may overflow if very large (2e9).
*)

val query : ?k:int -> ?eps:Py.Object.t -> ?p:[`T1_p_infinity of Py.Object.t | `F of float] -> ?distance_upper_bound:Py.Object.t -> x:[`Ndarray of [>`Ndarray] Np.Obj.t | `Last_dimension_self_m of Py.Object.t] -> [> tag] Obj.t -> (Py.Object.t * Py.Object.t)
(**
Query the kd-tree for nearest neighbors

Parameters
----------
x : array_like, last dimension self.m
    An array of points to query.
k : int, optional
    The number of nearest neighbors to return.
eps : nonnegative float, optional
    Return approximate nearest neighbors; the kth returned value
    is guaranteed to be no further than (1+eps) times the
    distance to the real kth nearest neighbor.
p : float, 1<=p<=infinity, optional
    Which Minkowski p-norm to use.
    1 is the sum-of-absolute-values 'Manhattan' distance
    2 is the usual Euclidean distance
    infinity is the maximum-coordinate-difference distance
distance_upper_bound : nonnegative float, optional
    Return only neighbors within this distance. This is used to prune
    tree searches, so if you are doing a series of nearest-neighbor
    queries, it may help to supply the distance to the nearest neighbor
    of the most recent point.

Returns
-------
d : float or array of floats
    The distances to the nearest neighbors.
    If x has shape tuple+(self.m,), then d has shape tuple if
    k is one, or tuple+(k,) if k is larger than one. Missing
    neighbors (e.g. when k > n or distance_upper_bound is
    given) are indicated with infinite distances.  If k is None,
    then d is an object array of shape tuple, containing lists
    of distances. In either case the hits are sorted by distance
    (nearest first).
i : integer or array of integers
    The locations of the neighbors in self.data. i is the same
    shape as d.

Examples
--------
>>> from scipy import spatial
>>> x, y = np.mgrid[0:5, 2:8]
>>> tree = spatial.KDTree(list(zip(x.ravel(), y.ravel())))
>>> tree.data
array([[0, 2],
       [0, 3],
       [0, 4],
       [0, 5],
       [0, 6],
       [0, 7],
       [1, 2],
       [1, 3],
       [1, 4],
       [1, 5],
       [1, 6],
       [1, 7],
       [2, 2],
       [2, 3],
       [2, 4],
       [2, 5],
       [2, 6],
       [2, 7],
       [3, 2],
       [3, 3],
       [3, 4],
       [3, 5],
       [3, 6],
       [3, 7],
       [4, 2],
       [4, 3],
       [4, 4],
       [4, 5],
       [4, 6],
       [4, 7]])
>>> pts = np.array([[0, 0], [2.1, 2.9]])
>>> tree.query(pts)
(array([ 2.        ,  0.14142136]), array([ 0, 13]))
>>> tree.query(pts[0])
(2.0, 0)
*)

val query_ball_point : ?p:float -> ?eps:Py.Object.t -> x:[`Ndarray of [>`Ndarray] Np.Obj.t | `Shape_tuple_self_m_ of Py.Object.t] -> r:float -> [> tag] Obj.t -> Py.Object.t
(**
Find all points within distance r of point(s) x.

Parameters
----------
x : array_like, shape tuple + (self.m,)
    The point or points to search for neighbors of.
r : positive float
    The radius of points to return.
p : float, optional
    Which Minkowski p-norm to use.  Should be in the range [1, inf].
eps : nonnegative float, optional
    Approximate search. Branches of the tree are not explored if their
    nearest points are further than ``r / (1 + eps)``, and branches are
    added in bulk if their furthest points are nearer than
    ``r * (1 + eps)``.

Returns
-------
results : list or array of lists
    If `x` is a single point, returns a list of the indices of the
    neighbors of `x`. If `x` is an array of points, returns an object
    array of shape tuple containing lists of neighbors.

Notes
-----
If you have many points whose neighbors you want to find, you may save
substantial amounts of time by putting them in a KDTree and using
query_ball_tree.

Examples
--------
>>> from scipy import spatial
>>> x, y = np.mgrid[0:5, 0:5]
>>> points = np.c_[x.ravel(), y.ravel()]
>>> tree = spatial.KDTree(points)
>>> tree.query_ball_point([2, 0], 1)
[5, 10, 11, 15]

Query multiple points and plot the results:

>>> import matplotlib.pyplot as plt
>>> points = np.asarray(points)
>>> plt.plot(points[:,0], points[:,1], '.')
>>> for results in tree.query_ball_point(([2, 0], [3, 3]), 1):
...     nearby_points = points[results]
...     plt.plot(nearby_points[:,0], nearby_points[:,1], 'o')
>>> plt.margins(0.1, 0.1)
>>> plt.show()
*)

val query_ball_tree : ?p:float -> ?eps:float -> other:Py.Object.t -> r:float -> [> tag] Obj.t -> Py.Object.t
(**
Find all pairs of points whose distance is at most r

Parameters
----------
other : KDTree instance
    The tree containing points to search against.
r : float
    The maximum distance, has to be positive.
p : float, optional
    Which Minkowski norm to use.  `p` has to meet the condition
    ``1 <= p <= infinity``.
eps : float, optional
    Approximate search.  Branches of the tree are not explored
    if their nearest points are further than ``r/(1+eps)``, and
    branches are added in bulk if their furthest points are nearer
    than ``r * (1+eps)``.  `eps` has to be non-negative.

Returns
-------
results : list of lists
    For each element ``self.data[i]`` of this tree, ``results[i]`` is a
    list of the indices of its neighbors in ``other.data``.
*)

val query_pairs : ?p:float -> ?eps:float -> r:float -> [> tag] Obj.t -> Py.Object.t
(**
Find all pairs of points within a distance.

Parameters
----------
r : positive float
    The maximum distance.
p : float, optional
    Which Minkowski norm to use.  `p` has to meet the condition
    ``1 <= p <= infinity``.
eps : float, optional
    Approximate search.  Branches of the tree are not explored
    if their nearest points are further than ``r/(1+eps)``, and
    branches are added in bulk if their furthest points are nearer
    than ``r * (1+eps)``.  `eps` has to be non-negative.

Returns
-------
results : set
    Set of pairs ``(i,j)``, with ``i < j``, for which the corresponding
    positions are close.
*)

val sparse_distance_matrix : ?p:float -> other:Py.Object.t -> max_distance:float -> [> tag] Obj.t -> Py.Object.t
(**
Compute a sparse distance matrix

Computes a distance matrix between two KDTrees, leaving as zero
any distance greater than max_distance.

Parameters
----------
other : KDTree

max_distance : positive float

p : float, optional

Returns
-------
result : dok_matrix
    Sparse matrix representing the results in 'dictionary of keys' format.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Rectangle : sig
type tag = [`Rectangle]
type t = [`Object | `Rectangle] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : maxes:Py.Object.t -> mins:Py.Object.t -> unit -> t
(**
Hyperrectangle class.

Represents a Cartesian product of intervals.
*)

val max_distance_point : ?p:float -> x:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the maximum distance between input and points in the hyperrectangle.

Parameters
----------
x : array_like
    Input array.
p : float, optional
    Input.
*)

val max_distance_rectangle : ?p:float -> other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Compute the maximum distance between points in the two hyperrectangles.

Parameters
----------
other : hyperrectangle
    Input.
p : float, optional
    Input.
*)

val min_distance_point : ?p:float -> x:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Return the minimum distance between input and points in the hyperrectangle.

Parameters
----------
x : array_like
    Input.
p : float, optional
    Input.
*)

val min_distance_rectangle : ?p:float -> other:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Compute the minimum distance between points in the two hyperrectangles.

Parameters
----------
other : hyperrectangle
    Input.
p : float
    Input.
*)

val split : d:int -> split:float -> [> tag] Obj.t -> Py.Object.t
(**
Produce two hyperrectangles by splitting.

In general, if you need to compute maximum and minimum
distances to the children, it can be done more efficiently
by updating the maximum and minimum distances to the parent.

Parameters
----------
d : int
    Axis to split hyperrectangle along.
split : float
    Position along axis `d` to split at.
*)

val volume : [> tag] Obj.t -> Py.Object.t
(**
Total volume.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module SphericalVoronoi : sig
type tag = [`SphericalVoronoi]
type t = [`Object | `SphericalVoronoi] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?radius:float -> ?center:[>`Ndarray] Np.Obj.t -> ?threshold:float -> points:[>`Ndarray] Np.Obj.t -> unit -> t
(**
Voronoi diagrams on the surface of a sphere.

.. versionadded:: 0.18.0

Parameters
----------
points : ndarray of floats, shape (npoints, ndim)
    Coordinates of points from which to construct a spherical
    Voronoi diagram.
radius : float, optional
    Radius of the sphere (Default: 1)
center : ndarray of floats, shape (ndim,)
    Center of sphere (Default: origin)
threshold : float
    Threshold for detecting duplicate points and
    mismatches between points and sphere parameters.
    (Default: 1e-06)

Attributes
----------
points : double array of shape (npoints, ndim)
    the points in `ndim` dimensions to generate the Voronoi diagram from
radius : double
    radius of the sphere
center : double array of shape (ndim,)
    center of the sphere
vertices : double array of shape (nvertices, ndim)
    Voronoi vertices corresponding to points
regions : list of list of integers of shape (npoints, _ )
    the n-th entry is a list consisting of the indices
    of the vertices belonging to the n-th point in points

Raises
------
ValueError
    If there are duplicates in `points`.
    If the provided `radius` is not consistent with `points`.

Notes
-----
The spherical Voronoi diagram algorithm proceeds as follows. The Convex
Hull of the input points (generators) is calculated, and is equivalent to
their Delaunay triangulation on the surface of the sphere [Caroli]_.
The Convex Hull neighbour information is then used to
order the Voronoi region vertices around each generator. The latter
approach is substantially less sensitive to floating point issues than
angle-based methods of Voronoi region vertex sorting.

Empirical assessment of spherical Voronoi algorithm performance suggests
quadratic time complexity (loglinear is optimal, but algorithms are more
challenging to implement).

References
----------
.. [Caroli] Caroli et al. Robust and Efficient Delaunay triangulations of
            points on or close to a sphere. Research Report RR-7004, 2009.

See Also
--------
Voronoi : Conventional Voronoi diagrams in N dimensions.

Examples
--------
Do some imports and take some points on a cube:

>>> from matplotlib import colors
>>> from mpl_toolkits.mplot3d.art3d import Poly3DCollection
>>> import matplotlib.pyplot as plt
>>> from scipy.spatial import SphericalVoronoi
>>> from mpl_toolkits.mplot3d import proj3d
>>> # set input data
>>> points = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0],
...                    [0, 1, 0], [0, -1, 0], [-1, 0, 0], ])

Calculate the spherical Voronoi diagram:

>>> radius = 1
>>> center = np.array([0, 0, 0])
>>> sv = SphericalVoronoi(points, radius, center)

Generate plot:

>>> # sort vertices (optional, helpful for plotting)
>>> sv.sort_vertices_of_regions()
>>> fig = plt.figure()
>>> ax = fig.add_subplot(111, projection='3d')
>>> # plot the unit sphere for reference (optional)
>>> u = np.linspace(0, 2 * np.pi, 100)
>>> v = np.linspace(0, np.pi, 100)
>>> x = np.outer(np.cos(u), np.sin(v))
>>> y = np.outer(np.sin(u), np.sin(v))
>>> z = np.outer(np.ones(np.size(u)), np.cos(v))
>>> ax.plot_surface(x, y, z, color='y', alpha=0.1)
>>> # plot generator points
>>> ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b')
>>> # plot Voronoi vertices
>>> ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2],
...                    c='g')
>>> # indicate Voronoi regions (as Euclidean polygons)
>>> for region in sv.regions:
...    random_color = colors.rgb2hex(np.random.rand(3))
...    polygon = Poly3DCollection([sv.vertices[region]], alpha=1.0)
...    polygon.set_color(random_color)
...    ax.add_collection3d(polygon)
>>> plt.show()
*)

val sort_vertices_of_regions : [> tag] Obj.t -> Py.Object.t
(**
Sort indices of the vertices to be (counter-)clockwise ordered.

Raises
------
TypeError
    If the points are not three-dimensional.

Notes
-----
For each region in regions, it sorts the indices of the Voronoi
vertices such that the resulting points are in a clockwise or
counterclockwise order around the generator point.

This is done as follows: Recall that the n-th region in regions
surrounds the n-th generator in points and that the k-th
Voronoi vertex in vertices is the circumcenter of the k-th triangle
in _tri.simplices.  For each region n, we choose the first triangle
(=Voronoi vertex) in _tri.simplices and a vertex of that triangle
not equal to the center n. These determine a unique neighbor of that
triangle, which is then chosen as the second triangle. The second
triangle will have a unique vertex not equal to the current vertex or
the center. This determines a unique neighbor of the second triangle,
which is then chosen as the third triangle and so forth. We proceed
through all the triangles (=Voronoi vertices) belonging to the
generator in points and obtain a sorted version of the vertices
of its surrounding region.
*)


(** Attribute points: get value or raise Not_found if None.*)
val points : t -> Py.Object.t

(** Attribute points: get value as an option. *)
val points_opt : t -> (Py.Object.t) option


(** Attribute radius: get value or raise Not_found if None.*)
val radius : t -> float

(** Attribute radius: get value as an option. *)
val radius_opt : t -> (float) option


(** Attribute center: get value or raise Not_found if None.*)
val center : t -> Py.Object.t

(** Attribute center: get value as an option. *)
val center_opt : t -> (Py.Object.t) option


(** Attribute vertices: get value or raise Not_found if None.*)
val vertices : t -> Py.Object.t

(** Attribute vertices: get value as an option. *)
val vertices_opt : t -> (Py.Object.t) option


(** Attribute regions: get value or raise Not_found if None.*)
val regions : t -> Py.Object.t

(** Attribute regions: get value as an option. *)
val regions_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Voronoi : sig
type tag = [`Voronoi]
type t = [`Object | `Voronoi] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?furthest_site:bool -> ?incremental:bool -> ?qhull_options:string -> points:[>`Ndarray] Np.Obj.t -> unit -> t
(**
Voronoi(points, furthest_site=False, incremental=False, qhull_options=None)

Voronoi diagrams in N dimensions.

.. versionadded:: 0.12.0

Parameters
----------
points : ndarray of floats, shape (npoints, ndim)
    Coordinates of points to construct a convex hull from
furthest_site : bool, optional
    Whether to compute a furthest-site Voronoi diagram. Default: False
incremental : bool, optional
    Allow adding new points incrementally. This takes up some additional
    resources.
qhull_options : str, optional
    Additional options to pass to Qhull. See Qhull manual
    for details. (Default: 'Qbb Qc Qz Qx' for ndim > 4 and
    'Qbb Qc Qz' otherwise. Incremental mode omits 'Qz'.)

Attributes
----------
points : ndarray of double, shape (npoints, ndim)
    Coordinates of input points.
vertices : ndarray of double, shape (nvertices, ndim)
    Coordinates of the Voronoi vertices.
ridge_points : ndarray of ints, shape ``(nridges, 2)``
    Indices of the points between which each Voronoi ridge lies.
ridge_vertices : list of list of ints, shape ``(nridges, * )``
    Indices of the Voronoi vertices forming each Voronoi ridge.
regions : list of list of ints, shape ``(nregions, * )``
    Indices of the Voronoi vertices forming each Voronoi region.
    -1 indicates vertex outside the Voronoi diagram.
point_region : list of ints, shape (npoints)
    Index of the Voronoi region for each input point.
    If qhull option 'Qc' was not specified, the list will contain -1
    for points that are not associated with a Voronoi region.
furthest_site
    True if this was a furthest site triangulation and False if not.

    .. versionadded:: 1.4.0

Raises
------
QhullError
    Raised when Qhull encounters an error condition, such as
    geometrical degeneracy when options to resolve are not enabled.
ValueError
    Raised if an incompatible array is given as input.

Notes
-----
The Voronoi diagram is computed using the
`Qhull library <http://www.qhull.org/>`__.

Examples
--------
Voronoi diagram for a set of point:

>>> points = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
...                    [2, 0], [2, 1], [2, 2]])
>>> from scipy.spatial import Voronoi, voronoi_plot_2d
>>> vor = Voronoi(points)

Plot it:

>>> import matplotlib.pyplot as plt
>>> fig = voronoi_plot_2d(vor)
>>> plt.show()

The Voronoi vertices:

>>> vor.vertices
array([[0.5, 0.5],
       [0.5, 1.5],
       [1.5, 0.5],
       [1.5, 1.5]])

There is a single finite Voronoi region, and four finite Voronoi
ridges:

>>> vor.regions
[[], [-1, 0], [-1, 1], [1, -1, 0], [3, -1, 2], [-1, 3], [-1, 2], [0, 1, 3, 2], [2, -1, 0], [3, -1, 1]]
>>> vor.ridge_vertices
[[-1, 0], [-1, 0], [-1, 1], [-1, 1], [0, 1], [-1, 3], [-1, 2], [2, 3], [-1, 3], [-1, 2], [1, 3], [0, 2]]

The ridges are perpendicular between lines drawn between the following
input points:

>>> vor.ridge_points
array([[0, 3],
       [0, 1],
       [2, 5],
       [2, 1],
       [1, 4],
       [7, 8],
       [7, 6],
       [7, 4],
       [8, 5],
       [6, 3],
       [4, 5],
       [4, 3]], dtype=int32)
*)

val add_points : ?restart:bool -> points:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
add_points(points, restart=False)

Process a set of additional new points.

Parameters
----------
points : ndarray
    New points to add. The dimensionality should match that of the
    initial points.
restart : bool, optional
    Whether to restart processing from scratch, rather than
    adding points incrementally.

Raises
------
QhullError
    Raised when Qhull encounters an error condition, such as
    geometrical degeneracy when options to resolve are not enabled.

See Also
--------
close

Notes
-----
You need to specify ``incremental=True`` when constructing the
object to be able to add points incrementally. Incremental addition
of points is also not possible after `close` has been called.
*)

val close : [> tag] Obj.t -> Py.Object.t
(**
close()

Finish incremental processing.

Call this to free resources taken up by Qhull, when using the
incremental mode. After calling this, adding more points is no
longer possible.
*)


(** Attribute points: get value or raise Not_found if None.*)
val points : t -> Py.Object.t

(** Attribute points: get value as an option. *)
val points_opt : t -> (Py.Object.t) option


(** Attribute vertices: get value or raise Not_found if None.*)
val vertices : t -> Py.Object.t

(** Attribute vertices: get value as an option. *)
val vertices_opt : t -> (Py.Object.t) option


(** Attribute ridge_points: get value or raise Not_found if None.*)
val ridge_points : t -> Py.Object.t

(** Attribute ridge_points: get value as an option. *)
val ridge_points_opt : t -> (Py.Object.t) option


(** Attribute ridge_vertices: get value or raise Not_found if None.*)
val ridge_vertices : t -> Py.Object.t

(** Attribute ridge_vertices: get value as an option. *)
val ridge_vertices_opt : t -> (Py.Object.t) option


(** Attribute regions: get value or raise Not_found if None.*)
val regions : t -> Py.Object.t

(** Attribute regions: get value as an option. *)
val regions_opt : t -> (Py.Object.t) option


(** Attribute point_region: get value or raise Not_found if None.*)
val point_region : t -> Py.Object.t

(** Attribute point_region: get value as an option. *)
val point_region_opt : t -> (Py.Object.t) option


(** Attribute furthest_site: get value or raise Not_found if None.*)
val furthest_site : t -> Py.Object.t

(** Attribute furthest_site: get value as an option. *)
val furthest_site_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module CKDTree : sig
type tag = [`CKDTree]
type t = [`CKDTree | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?leafsize:Py.Object.t -> ?compact_nodes:bool -> ?copy_data:bool -> ?balanced_tree:bool -> ?boxsize:[`S of string | `F of float | `Ndarray of [>`Ndarray] Np.Obj.t | `I of int | `Bool of bool] -> data:[>`Ndarray] Np.Obj.t -> unit -> t
(**
cKDTree(data, leafsize=16, compact_nodes=True, copy_data=False,
        balanced_tree=True, boxsize=None)

kd-tree for quick nearest-neighbor lookup

This class provides an index into a set of k-dimensional points
which can be used to rapidly look up the nearest neighbors of any
point.

The algorithm used is described in Maneewongvatana and Mount 1999.
The general idea is that the kd-tree is a binary trie, each of whose
nodes represents an axis-aligned hyperrectangle. Each node specifies
an axis and splits the set of points based on whether their coordinate
along that axis is greater than or less than a particular value.

During construction, the axis and splitting point are chosen by the
'sliding midpoint' rule, which ensures that the cells do not all
become long and thin.

The tree can be queried for the r closest neighbors of any given point
(optionally returning only those within some maximum distance of the
point). It can also be queried, with a substantial gain in efficiency,
for the r approximate closest neighbors.

For large dimensions (20 is already large) do not expect this to run
significantly faster than brute force. High-dimensional nearest-neighbor
queries are a substantial open problem in computer science.

Parameters
----------
data : array_like, shape (n,m)
    The n data points of dimension m to be indexed. This array is
    not copied unless this is necessary to produce a contiguous
    array of doubles, and so modifying this data will result in
    bogus results. The data are also copied if the kd-tree is built
    with copy_data=True.
leafsize : positive int, optional
    The number of points at which the algorithm switches over to
    brute-force. Default: 16.
compact_nodes : bool, optional
    If True, the kd-tree is built to shrink the hyperrectangles to
    the actual data range. This usually gives a more compact tree that
    is robust against degenerated input data and gives faster queries
    at the expense of longer build time. Default: True.
copy_data : bool, optional
    If True the data is always copied to protect the kd-tree against
    data corruption. Default: False.
balanced_tree : bool, optional
    If True, the median is used to split the hyperrectangles instead of
    the midpoint. This usually gives a more compact tree and
    faster queries at the expense of longer build time. Default: True.
boxsize : array_like or scalar, optional
    Apply a m-d toroidal topology to the KDTree.. The topology is generated
    by :math:`x_i + n_i L_i` where :math:`n_i` are integers and :math:`L_i`
    is the boxsize along i-th dimension. The input data shall be wrapped
    into :math:`[0, L_i)`. A ValueError is raised if any of the data is
    outside of this bound.

Attributes
----------
data : ndarray, shape (n,m)
    The n data points of dimension m to be indexed. This array is
    not copied unless this is necessary to produce a contiguous
    array of doubles. The data are also copied if the kd-tree is built
    with `copy_data=True`.
leafsize : positive int
    The number of points at which the algorithm switches over to
    brute-force.
m : int
    The dimension of a single data-point.
n : int
    The number of data points.
maxes : ndarray, shape (m,)
    The maximum value in each dimension of the n data points.
mins : ndarray, shape (m,)
    The minimum value in each dimension of the n data points.
tree : object, class cKDTreeNode
    This class exposes a Python view of the root node in the cKDTree object.
size : int
    The number of nodes in the tree.

See Also
--------
KDTree : Implementation of `cKDTree` in pure Python
*)

val count_neighbors : ?p:float -> ?weights:[`Ndarray of [>`Ndarray] Np.Obj.t | `Tuple of Py.Object.t] -> ?cumulative:bool -> other:Py.Object.t -> r:[`Ndarray of [>`Ndarray] Np.Obj.t | `F of float] -> [> tag] Obj.t -> Py.Object.t
(**
count_neighbors(self, other, r, p=2., weights=None, cumulative=True)

Count how many nearby pairs can be formed. (pair-counting)

Count the number of pairs (x1,x2) can be formed, with x1 drawn
from self and x2 drawn from ``other``, and where
``distance(x1, x2, p) <= r``.

Data points on self and other are optionally weighted by the ``weights``
argument. (See below)

The algorithm we implement here is based on [1]_. See notes for further discussion.

Parameters
----------
other : cKDTree instance
    The other tree to draw points from, can be the same tree as self.
r : float or one-dimensional array of floats
    The radius to produce a count for. Multiple radii are searched with
    a single tree traversal.
    If the count is non-cumulative(``cumulative=False``), ``r`` defines
    the edges of the bins, and must be non-decreasing.
p : float, optional
    1<=p<=infinity.
    Which Minkowski p-norm to use.
    Default 2.0.
    A finite large p may cause a ValueError if overflow can occur.
weights : tuple, array_like, or None, optional
    If None, the pair-counting is unweighted.
    If given as a tuple, weights[0] is the weights of points in ``self``, and
    weights[1] is the weights of points in ``other``; either can be None to
    indicate the points are unweighted.
    If given as an array_like, weights is the weights of points in ``self``
    and ``other``. For this to make sense, ``self`` and ``other`` must be the
    same tree. If ``self`` and ``other`` are two different trees, a ``ValueError``
    is raised.
    Default: None
cumulative : bool, optional
    Whether the returned counts are cumulative. When cumulative is set to ``False``
    the algorithm is optimized to work with a large number of bins (>10) specified
    by ``r``. When ``cumulative`` is set to True, the algorithm is optimized to work
    with a small number of ``r``. Default: True

Returns
-------
result : scalar or 1-D array
    The number of pairs. For unweighted counts, the result is integer.
    For weighted counts, the result is float.
    If cumulative is False, ``result[i]`` contains the counts with
    ``(-inf if i == 0 else r[i-1]) < R <= r[i]``

Notes
-----
Pair-counting is the basic operation used to calculate the two point
correlation functions from a data set composed of position of objects.

Two point correlation function measures the clustering of objects and
is widely used in cosmology to quantify the large scale structure
in our Universe, but it may be useful for data analysis in other fields
where self-similar assembly of objects also occur.

The Landy-Szalay estimator for the two point correlation function of
``D`` measures the clustering signal in ``D``. [2]_

For example, given the position of two sets of objects,

- objects ``D`` (data) contains the clustering signal, and

- objects ``R`` (random) that contains no signal,

.. math::

     \xi(r) = \frac{<D, D> - 2 f <D, R> + f^2<R, R>}{f^2<R, R>},

where the brackets represents counting pairs between two data sets
in a finite bin around ``r`` (distance), corresponding to setting
`cumulative=False`, and ``f = float(len(D)) / float(len(R))`` is the
ratio between number of objects from data and random.

The algorithm implemented here is loosely based on the dual-tree
algorithm described in [1]_. We switch between two different
pair-cumulation scheme depending on the setting of ``cumulative``.
The computing time of the method we use when for
``cumulative == False`` does not scale with the total number of bins.
The algorithm for ``cumulative == True`` scales linearly with the
number of bins, though it is slightly faster when only
1 or 2 bins are used. [5]_.

As an extension to the naive pair-counting,
weighted pair-counting counts the product of weights instead
of number of pairs.
Weighted pair-counting is used to estimate marked correlation functions
([3]_, section 2.2),
or to properly calculate the average of data per distance bin
(e.g. [4]_, section 2.1 on redshift).

.. [1] Gray and Moore,
       'N-body problems in statistical learning',
       Mining the sky, 2000,
       https://arxiv.org/abs/astro-ph/0012333

.. [2] Landy and Szalay,
       'Bias and variance of angular correlation functions',
       The Astrophysical Journal, 1993,
       http://adsabs.harvard.edu/abs/1993ApJ...412...64L

.. [3] Sheth, Connolly and Skibba,
       'Marked correlations in galaxy formation models',
       Arxiv e-print, 2005,
       https://arxiv.org/abs/astro-ph/0511773

.. [4] Hawkins, et al.,
       'The 2dF Galaxy Redshift Survey: correlation functions,
       peculiar velocities and the matter density of the Universe',
       Monthly Notices of the Royal Astronomical Society, 2002,
       http://adsabs.harvard.edu/abs/2003MNRAS.346...78H

.. [5] https://github.com/scipy/scipy/pull/5647#issuecomment-168474926
*)

val query_ball_point : ?p:float -> ?eps:Py.Object.t -> x:[`Ndarray of [>`Ndarray] Np.Obj.t | `Shape_tuple_self_m_ of Py.Object.t] -> r:[`Ndarray of [>`Ndarray] Np.Obj.t | `F of float] -> [> tag] Obj.t -> Py.Object.t
(**
query_ball_point(self, x, r, p=2., eps=0)

Find all points within distance r of point(s) x.

Parameters
----------
x : array_like, shape tuple + (self.m,)
    The point or points to search for neighbors of.
r : array_like, float
    The radius of points to return, shall broadcast to the length of x.
p : float, optional
    Which Minkowski p-norm to use.  Should be in the range [1, inf].
    A finite large p may cause a ValueError if overflow can occur.
eps : nonnegative float, optional
    Approximate search. Branches of the tree are not explored if their
    nearest points are further than ``r / (1 + eps)``, and branches are
    added in bulk if their furthest points are nearer than
    ``r * (1 + eps)``.
n_jobs : int, optional
    Number of jobs to schedule for parallel processing. If -1 is given
    all processors are used. Default: 1.
return_sorted : bool, optional
    Sorts returned indicies if True and does not sort them if False. If
    None, does not sort single point queries, but does sort
    multi-point queries which was the behavior before this option
    was added.

    .. versionadded:: 1.2.0
return_length: bool, optional
    Return the number of points inside the radius instead of a list
    of the indices.
    .. versionadded:: 1.3.0

Returns
-------
results : list or array of lists
    If `x` is a single point, returns a list of the indices of the
    neighbors of `x`. If `x` is an array of points, returns an object
    array of shape tuple containing lists of neighbors.

Notes
-----
If you have many points whose neighbors you want to find, you may save
substantial amounts of time by putting them in a cKDTree and using
query_ball_tree.

Examples
--------
>>> from scipy import spatial
>>> x, y = np.mgrid[0:4, 0:4]
>>> points = np.c_[x.ravel(), y.ravel()]
>>> tree = spatial.cKDTree(points)
>>> tree.query_ball_point([2, 0], 1)
[4, 8, 9, 12]
*)

val query_ball_tree : ?p:float -> ?eps:float -> other:Py.Object.t -> r:float -> [> tag] Obj.t -> Py.Object.t
(**
query_ball_tree(self, other, r, p=2., eps=0)

Find all pairs of points whose distance is at most r

Parameters
----------
other : cKDTree instance
    The tree containing points to search against.
r : float
    The maximum distance, has to be positive.
p : float, optional
    Which Minkowski norm to use.  `p` has to meet the condition
    ``1 <= p <= infinity``.
    A finite large p may cause a ValueError if overflow can occur.
eps : float, optional
    Approximate search.  Branches of the tree are not explored
    if their nearest points are further than ``r/(1+eps)``, and
    branches are added in bulk if their furthest points are nearer
    than ``r * (1+eps)``.  `eps` has to be non-negative.

Returns
-------
results : list of lists
    For each element ``self.data[i]`` of this tree, ``results[i]`` is a
    list of the indices of its neighbors in ``other.data``.
*)

val query_pairs : ?p:float -> ?eps:float -> r:float -> [> tag] Obj.t -> Py.Object.t
(**
query_pairs(self, r, p=2., eps=0)

Find all pairs of points whose distance is at most r.

Parameters
----------
r : positive float
    The maximum distance.
p : float, optional
    Which Minkowski norm to use.  ``p`` has to meet the condition
    ``1 <= p <= infinity``.
    A finite large p may cause a ValueError if overflow can occur.
eps : float, optional
    Approximate search.  Branches of the tree are not explored
    if their nearest points are further than ``r/(1+eps)``, and
    branches are added in bulk if their furthest points are nearer
    than ``r * (1+eps)``.  `eps` has to be non-negative.
output_type : string, optional
    Choose the output container, 'set' or 'ndarray'. Default: 'set'

Returns
-------
results : set or ndarray
    Set of pairs ``(i,j)``, with ``i < j``, for which the corresponding
    positions are close. If output_type is 'ndarray', an ndarry is
    returned instead of a set.
*)

val sparse_distance_matrix : ?p:[`T1_p_infinity of Py.Object.t | `F of float] -> other:Py.Object.t -> max_distance:float -> [> tag] Obj.t -> Py.Object.t
(**
sparse_distance_matrix(self, other, max_distance, p=2.)

Compute a sparse distance matrix

Computes a distance matrix between two cKDTrees, leaving as zero
any distance greater than max_distance.

Parameters
----------
other : cKDTree

max_distance : positive float

p : float, 1<=p<=infinity
    Which Minkowski p-norm to use.
    A finite large p may cause a ValueError if overflow can occur.

output_type : string, optional
    Which container to use for output data. Options: 'dok_matrix',
    'coo_matrix', 'dict', or 'ndarray'. Default: 'dok_matrix'.

Returns
-------
result : dok_matrix, coo_matrix, dict or ndarray
    Sparse matrix representing the results in 'dictionary of keys'
    format. If a dict is returned the keys are (i,j) tuples of indices.
    If output_type is 'ndarray' a record array with fields 'i', 'j',
    and 'v' is returned,
*)


(** Attribute data: get value or raise Not_found if None.*)
val data : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute data: get value as an option. *)
val data_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute leafsize: get value or raise Not_found if None.*)
val leafsize : t -> Py.Object.t

(** Attribute leafsize: get value as an option. *)
val leafsize_opt : t -> (Py.Object.t) option


(** Attribute m: get value or raise Not_found if None.*)
val m : t -> int

(** Attribute m: get value as an option. *)
val m_opt : t -> (int) option


(** Attribute n: get value or raise Not_found if None.*)
val n : t -> int

(** Attribute n: get value as an option. *)
val n_opt : t -> (int) option


(** Attribute maxes: get value or raise Not_found if None.*)
val maxes : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute maxes: get value as an option. *)
val maxes_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute mins: get value or raise Not_found if None.*)
val mins : t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t

(** Attribute mins: get value as an option. *)
val mins_opt : t -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t) option


(** Attribute tree: get value or raise Not_found if None.*)
val tree : t -> Py.Object.t

(** Attribute tree: get value as an option. *)
val tree_opt : t -> (Py.Object.t) option


(** Attribute size: get value or raise Not_found if None.*)
val size : t -> int

(** Attribute size: get value as an option. *)
val size_opt : t -> (int) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Ckdtree : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module CKDTreeNode : sig
type tag = [`CKDTreeNode]
type t = [`CKDTreeNode | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t


(** Attribute level: get value or raise Not_found if None.*)
val level : t -> int

(** Attribute level: get value as an option. *)
val level_opt : t -> (int) option


(** Attribute split_dim: get value or raise Not_found if None.*)
val split_dim : t -> int

(** Attribute split_dim: get value as an option. *)
val split_dim_opt : t -> (int) option


(** Attribute split: get value or raise Not_found if None.*)
val split : t -> float

(** Attribute split: get value as an option. *)
val split_opt : t -> (float) option


(** Attribute children: get value or raise Not_found if None.*)
val children : t -> int

(** Attribute children: get value as an option. *)
val children_opt : t -> (int) option


(** Attribute data_points: get value or raise Not_found if None.*)
val data_points : t -> Py.Object.t

(** Attribute data_points: get value as an option. *)
val data_points_opt : t -> (Py.Object.t) option


(** Attribute indices: get value or raise Not_found if None.*)
val indices : t -> Py.Object.t

(** Attribute indices: get value as an option. *)
val indices_opt : t -> (Py.Object.t) option


(** Attribute lesser: get value or raise Not_found if None.*)
val lesser : t -> Py.Object.t

(** Attribute lesser: get value as an option. *)
val lesser_opt : t -> (Py.Object.t) option


(** Attribute greater: get value or raise Not_found if None.*)
val greater : t -> Py.Object.t

(** Attribute greater: get value as an option. *)
val greater_opt : t -> (Py.Object.t) option


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Coo_entries : sig
type tag = [`Coo_entries]
type t = [`Coo_entries | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Ordered_pairs : sig
type tag = [`Ordered_pairs]
type t = [`Object | `Ordered_pairs] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val cpu_count : unit -> Py.Object.t
(**
Returns the number of CPUs in the system
*)


end

module Distance : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module MetricInfo : sig
type tag = [`MetricInfo]
type t = [`MetricInfo | `Object] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?types:Py.Object.t -> ?validator:Py.Object.t -> aka:Py.Object.t -> unit -> t
(**
MetricInfo(aka, types, validator)
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
Return number of occurrences of value.
*)

val index : ?start:Py.Object.t -> ?stop:Py.Object.t -> value:Py.Object.t -> [> tag] Obj.t -> Py.Object.t
(**
Return first index of value.

Raises ValueError if the value is not present.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Partial : sig
type tag = [`Partial]
type t = [`Object | `Partial] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?keywords:(string * Py.Object.t) list -> func:Py.Object.t -> Py.Object.t list -> t
(**
partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

val braycurtis : ?w:[>`Ndarray] Np.Obj.t -> u:[>`Ndarray] Np.Obj.t -> v:[>`Ndarray] Np.Obj.t -> unit -> float
(**
Compute the Bray-Curtis distance between two 1-D arrays.

Bray-Curtis distance is defined as

.. math::

   \sum{ |u_i-v_i| } / \sum{ |u_i+v_i| }

The Bray-Curtis distance is in the range [0, 1] if all coordinates are
positive, and is undefined if the inputs are of length zero.

Parameters
----------
u : (N,) array_like
    Input array.
v : (N,) array_like
    Input array.
w : (N,) array_like, optional
    The weights for each value in `u` and `v`. Default is None,
    which gives each value a weight of 1.0

Returns
-------
braycurtis : double
    The Bray-Curtis distance between 1-D arrays `u` and `v`.

Examples
--------
>>> from scipy.spatial import distance
>>> distance.braycurtis([1, 0, 0], [0, 1, 0])
1.0
>>> distance.braycurtis([1, 1, 0], [0, 1, 0])
0.33333333333333331
*)

val callable : Py.Object.t -> Py.Object.t
(**
None
*)

val canberra : ?w:[>`Ndarray] Np.Obj.t -> u:[>`Ndarray] Np.Obj.t -> v:[>`Ndarray] Np.Obj.t -> unit -> float
(**
Compute the Canberra distance between two 1-D arrays.

The Canberra distance is defined as

.. math::

     d(u,v) = \sum_i \frac{ |u_i-v_i| }
                          { |u_i|+|v_i| }.

Parameters
----------
u : (N,) array_like
    Input array.
v : (N,) array_like
    Input array.
w : (N,) array_like, optional
    The weights for each value in `u` and `v`. Default is None,
    which gives each value a weight of 1.0

Returns
-------
canberra : double
    The Canberra distance between vectors `u` and `v`.

Notes
-----
When `u[i]` and `v[i]` are 0 for given i, then the fraction 0/0 = 0 is
used in the calculation.

Examples
--------
>>> from scipy.spatial import distance
>>> distance.canberra([1, 0, 0], [0, 1, 0])
2.0
>>> distance.canberra([1, 1, 0], [0, 1, 0])
1.0
*)

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

val chebyshev : ?w:[>`Ndarray] Np.Obj.t -> u:[>`Ndarray] Np.Obj.t -> v:[>`Ndarray] Np.Obj.t -> unit -> float
(**
Compute the Chebyshev distance.

Computes the Chebyshev distance between two 1-D arrays `u` and `v`,
which is defined as

.. math::

   \max_i { |u_i-v_i| }.

Parameters
----------
u : (N,) array_like
    Input vector.
v : (N,) array_like
    Input vector.
w : (N,) array_like, optional
    The weights for each value in `u` and `v`. Default is None,
    which gives each value a weight of 1.0

Returns
-------
chebyshev : double
    The Chebyshev distance between vectors `u` and `v`.

Examples
--------
>>> from scipy.spatial import distance
>>> distance.chebyshev([1, 0, 0], [0, 1, 0])
1
>>> distance.chebyshev([1, 1, 0], [0, 1, 0])
1
*)

val cityblock : ?w:[>`Ndarray] Np.Obj.t -> u:[>`Ndarray] Np.Obj.t -> v:[>`Ndarray] Np.Obj.t -> unit -> float
(**
Compute the City Block (Manhattan) distance.

Computes the Manhattan distance between two 1-D arrays `u` and `v`,
which is defined as

.. math::

   \sum_i {\left| u_i - v_i \right| }.

Parameters
----------
u : (N,) array_like
    Input array.
v : (N,) array_like
    Input array.
w : (N,) array_like, optional
    The weights for each value in `u` and `v`. Default is None,
    which gives each value a weight of 1.0

Returns
-------
cityblock : double
    The City Block (Manhattan) distance between vectors `u` and `v`.

Examples
--------
>>> from scipy.spatial import distance
>>> distance.cityblock([1, 0, 0], [0, 1, 0])
2
>>> distance.cityblock([1, 0, 0], [0, 2, 0])
3
>>> distance.cityblock([1, 0, 0], [1, 1, 0])
1
*)

val correlation : ?w:[>`Ndarray] Np.Obj.t -> ?centered:Py.Object.t -> u:[>`Ndarray] Np.Obj.t -> v:[>`Ndarray] Np.Obj.t -> unit -> float
(**
Compute the correlation distance between two 1-D arrays.

The correlation distance between `u` and `v`, is
defined as

.. math::

    1 - \frac{(u - \bar{u}) \cdot (v - \bar{v})}
              {{ ||(u - \bar{u})|| }_2 { ||(v - \bar{v})|| }_2}

where :math:`\bar{u}` is the mean of the elements of `u`
and :math:`x \cdot y` is the dot product of :math:`x` and :math:`y`.

Parameters
----------
u : (N,) array_like
    Input array.
v : (N,) array_like
    Input array.
w : (N,) array_like, optional
    The weights for each value in `u` and `v`. Default is None,
    which gives each value a weight of 1.0

Returns
-------
correlation : double
    The correlation distance between 1-D array `u` and `v`.
*)

val cosine : ?w:[>`Ndarray] Np.Obj.t -> u:[>`Ndarray] Np.Obj.t -> v:[>`Ndarray] Np.Obj.t -> unit -> float
(**
Compute the Cosine distance between 1-D arrays.

The Cosine distance between `u` and `v`, is defined as

.. math::

    1 - \frac{u \cdot v}
              { ||u||_2 ||v||_2}.

where :math:`u \cdot v` is the dot product of :math:`u` and
:math:`v`.

Parameters
----------
u : (N,) array_like
    Input array.
v : (N,) array_like
    Input array.
w : (N,) array_like, optional
    The weights for each value in `u` and `v`. Default is None,
    which gives each value a weight of 1.0

Returns
-------
cosine : double
    The Cosine distance between vectors `u` and `v`.

Examples
--------
>>> from scipy.spatial import distance
>>> distance.cosine([1, 0, 0], [0, 1, 0])
1.0
>>> distance.cosine([100, 0, 0], [0, 1, 0])
1.0
>>> distance.cosine([1, 1, 0], [0, 1, 0])
0.29289321881345254
*)

val dice : ?w:[>`Ndarray] Np.Obj.t -> u:[>`Ndarray] Np.Obj.t -> v:[>`Ndarray] Np.Obj.t -> unit -> float
(**
Compute the Dice dissimilarity between two boolean 1-D arrays.

The Dice dissimilarity between `u` and `v`, is

.. math::

     \frac{c_{TF} + c_{FT}}
          {2c_{TT} + c_{FT} + c_{TF}}

where :math:`c_{ij}` is the number of occurrences of
:math:`\mathtt{u[k]} = i` and :math:`\mathtt{v[k]} = j` for
:math:`k < n`.

Parameters
----------
u : (N,) ndarray, bool
    Input 1-D array.
v : (N,) ndarray, bool
    Input 1-D array.
w : (N,) array_like, optional
    The weights for each value in `u` and `v`. Default is None,
    which gives each value a weight of 1.0

Returns
-------
dice : double
    The Dice dissimilarity between 1-D arrays `u` and `v`.

Examples
--------
>>> from scipy.spatial import distance
>>> distance.dice([1, 0, 0], [0, 1, 0])
1.0
>>> distance.dice([1, 0, 0], [1, 1, 0])
0.3333333333333333
>>> distance.dice([1, 0, 0], [2, 0, 0])
-0.3333333333333333
*)

val directed_hausdorff : ?seed:[`I of int | `None] -> u:[>`Ndarray] Np.Obj.t -> v:Py.Object.t -> unit -> (float * int * int)
(**
Compute the directed Hausdorff distance between two N-D arrays.

Distances between pairs are calculated using a Euclidean metric.

Parameters
----------
u : (M,N) ndarray
    Input array.
v : (O,N) ndarray
    Input array.
seed : int or None
    Local `numpy.random.mtrand.RandomState` seed. Default is 0, a random
    shuffling of u and v that guarantees reproducibility.

Returns
-------
d : double
    The directed Hausdorff distance between arrays `u` and `v`,

index_1 : int
    index of point contributing to Hausdorff pair in `u`

index_2 : int
    index of point contributing to Hausdorff pair in `v`

Raises
------
ValueError
    An exception is thrown if `u` and `v` do not have
    the same number of columns.

Notes
-----
Uses the early break technique and the random sampling approach
described by [1]_. Although worst-case performance is ``O(m * o)``
(as with the brute force algorithm), this is unlikely in practice
as the input data would have to require the algorithm to explore
every single point interaction, and after the algorithm shuffles
the input points at that. The best case performance is O(m), which
is satisfied by selecting an inner loop distance that is less than
cmax and leads to an early break as often as possible. The authors
have formally shown that the average runtime is closer to O(m).

.. versionadded:: 0.19.0

References
----------
.. [1] A. A. Taha and A. Hanbury, 'An efficient algorithm for
       calculating the exact Hausdorff distance.' IEEE Transactions On
       Pattern Analysis And Machine Intelligence, vol. 37 pp. 2153-63,
       2015.

See Also
--------
scipy.spatial.procrustes : Another similarity test for two data sets

Examples
--------
Find the directed Hausdorff distance between two 2-D arrays of
coordinates:

>>> from scipy.spatial.distance import directed_hausdorff
>>> u = np.array([(1.0, 0.0),
...               (0.0, 1.0),
...               (-1.0, 0.0),
...               (0.0, -1.0)])
>>> v = np.array([(2.0, 0.0),
...               (0.0, 2.0),
...               (-2.0, 0.0),
...               (0.0, -4.0)])

>>> directed_hausdorff(u, v)[0]
2.23606797749979
>>> directed_hausdorff(v, u)[0]
3.0

Find the general (symmetric) Hausdorff distance between two 2-D
arrays of coordinates:

>>> max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
3.0

Find the indices of the points that generate the Hausdorff distance
(the Hausdorff pair):

>>> directed_hausdorff(v, u)[1:]
(3, 3)
*)

val euclidean : ?w:[>`Ndarray] Np.Obj.t -> u:[>`Ndarray] Np.Obj.t -> v:[>`Ndarray] Np.Obj.t -> unit -> float
(**
Computes the Euclidean distance between two 1-D arrays.

The Euclidean distance between 1-D arrays `u` and `v`, is defined as

.. math::

   { ||u-v|| }_2

   \left(\sum{(w_i |(u_i - v_i)|^2)}\right)^{1/2}

Parameters
----------
u : (N,) array_like
    Input array.
v : (N,) array_like
    Input array.
w : (N,) array_like, optional
    The weights for each value in `u` and `v`. Default is None,
    which gives each value a weight of 1.0

Returns
-------
euclidean : double
    The Euclidean distance between vectors `u` and `v`.

Examples
--------
>>> from scipy.spatial import distance
>>> distance.euclidean([1, 0, 0], [0, 1, 0])
1.4142135623730951
>>> distance.euclidean([1, 1, 0], [0, 1, 0])
1.0
*)

val hamming : ?w:[>`Ndarray] Np.Obj.t -> u:[>`Ndarray] Np.Obj.t -> v:[>`Ndarray] Np.Obj.t -> unit -> float
(**
Compute the Hamming distance between two 1-D arrays.

The Hamming distance between 1-D arrays `u` and `v`, is simply the
proportion of disagreeing components in `u` and `v`. If `u` and `v` are
boolean vectors, the Hamming distance is

.. math::

   \frac{c_{01} + c_{10}}{n}

where :math:`c_{ij}` is the number of occurrences of
:math:`\mathtt{u[k]} = i` and :math:`\mathtt{v[k]} = j` for
:math:`k < n`.

Parameters
----------
u : (N,) array_like
    Input array.
v : (N,) array_like
    Input array.
w : (N,) array_like, optional
    The weights for each value in `u` and `v`. Default is None,
    which gives each value a weight of 1.0

Returns
-------
hamming : double
    The Hamming distance between vectors `u` and `v`.

Examples
--------
>>> from scipy.spatial import distance
>>> distance.hamming([1, 0, 0], [0, 1, 0])
0.66666666666666663
>>> distance.hamming([1, 0, 0], [1, 1, 0])
0.33333333333333331
>>> distance.hamming([1, 0, 0], [2, 0, 0])
0.33333333333333331
>>> distance.hamming([1, 0, 0], [3, 0, 0])
0.33333333333333331
*)

val is_valid_dm : ?tol:float -> ?throw:bool -> ?name:string -> ?warning:bool -> d:[>`Ndarray] Np.Obj.t -> unit -> bool
(**
Return True if input array is a valid distance matrix.

Distance matrices must be 2-dimensional numpy arrays.
They must have a zero-diagonal, and they must be symmetric.

Parameters
----------
D : ndarray
    The candidate object to test for validity.
tol : float, optional
    The distance matrix should be symmetric. `tol` is the maximum
    difference between entries ``ij`` and ``ji`` for the distance
    metric to be considered symmetric.
throw : bool, optional
    An exception is thrown if the distance matrix passed is not valid.
name : str, optional
    The name of the variable to checked. This is useful if
    throw is set to True so the offending variable can be identified
    in the exception message when an exception is thrown.
warning : bool, optional
    Instead of throwing an exception, a warning message is
    raised.

Returns
-------
valid : bool
    True if the variable `D` passed is a valid distance matrix.

Notes
-----
Small numerical differences in `D` and `D.T` and non-zeroness of
the diagonal are ignored if they are within the tolerance specified
by `tol`.
*)

val is_valid_y : ?warning:bool -> ?throw:bool -> ?name:bool -> y:[>`Ndarray] Np.Obj.t -> unit -> Py.Object.t
(**
Return True if the input array is a valid condensed distance matrix.

Condensed distance matrices must be 1-dimensional numpy arrays.
Their length must be a binomial coefficient :math:`{n \choose 2}`
for some positive integer n.

Parameters
----------
y : ndarray
    The condensed distance matrix.
warning : bool, optional
    Invokes a warning if the variable passed is not a valid
    condensed distance matrix. The warning message explains why
    the distance matrix is not valid.  `name` is used when
    referencing the offending variable.
throw : bool, optional
    Throws an exception if the variable passed is not a valid
    condensed distance matrix.
name : bool, optional
    Used when referencing the offending variable in the
    warning or exception message.
*)

val jaccard : ?w:[>`Ndarray] Np.Obj.t -> u:[`Ndarray of [>`Ndarray] Np.Obj.t | `Bool of bool] -> v:[`Ndarray of [>`Ndarray] Np.Obj.t | `Bool of bool] -> unit -> float
(**
Compute the Jaccard-Needham dissimilarity between two boolean 1-D arrays.

The Jaccard-Needham dissimilarity between 1-D boolean arrays `u` and `v`,
is defined as

.. math::

   \frac{c_{TF} + c_{FT}}
        {c_{TT} + c_{FT} + c_{TF}}

where :math:`c_{ij}` is the number of occurrences of
:math:`\mathtt{u[k]} = i` and :math:`\mathtt{v[k]} = j` for
:math:`k < n`.

Parameters
----------
u : (N,) array_like, bool
    Input array.
v : (N,) array_like, bool
    Input array.
w : (N,) array_like, optional
    The weights for each value in `u` and `v`. Default is None,
    which gives each value a weight of 1.0

Returns
-------
jaccard : double
    The Jaccard distance between vectors `u` and `v`.

Notes
-----
When both `u` and `v` lead to a `0/0` division i.e. there is no overlap
between the items in the vectors the returned distance is 0. See the
Wikipedia page on the Jaccard index [1]_, and this paper [2]_.

.. versionchanged:: 1.2.0
    Previously, when `u` and `v` lead to a `0/0` division, the function
    would return NaN. This was changed to return 0 instead.

References
----------
.. [1] https://en.wikipedia.org/wiki/Jaccard_index
.. [2] S. Kosub, 'A note on the triangle inequality for the Jaccard
   distance', 2016, Available online: https://arxiv.org/pdf/1612.02696.pdf

Examples
--------
>>> from scipy.spatial import distance
>>> distance.jaccard([1, 0, 0], [0, 1, 0])
1.0
>>> distance.jaccard([1, 0, 0], [1, 1, 0])
0.5
>>> distance.jaccard([1, 0, 0], [1, 2, 0])
0.5
>>> distance.jaccard([1, 0, 0], [1, 1, 1])
0.66666666666666663
*)

val jensenshannon : ?base:float -> p:[>`Ndarray] Np.Obj.t -> q:[>`Ndarray] Np.Obj.t -> unit -> float
(**
Compute the Jensen-Shannon distance (metric) between
two 1-D probability arrays. This is the square root
of the Jensen-Shannon divergence.

The Jensen-Shannon distance between two probability
vectors `p` and `q` is defined as,

.. math::

   \sqrt{\frac{D(p \parallel m) + D(q \parallel m)}{2}}

where :math:`m` is the pointwise mean of :math:`p` and :math:`q`
and :math:`D` is the Kullback-Leibler divergence.

This routine will normalize `p` and `q` if they don't sum to 1.0.

Parameters
----------
p : (N,) array_like
    left probability vector
q : (N,) array_like
    right probability vector
base : double, optional
    the base of the logarithm used to compute the output
    if not given, then the routine uses the default base of
    scipy.stats.entropy.

Returns
-------
js : double
    The Jensen-Shannon distance between `p` and `q`

.. versionadded:: 1.2.0

Examples
--------
>>> from scipy.spatial import distance
>>> distance.jensenshannon([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 2.0)
1.0
>>> distance.jensenshannon([1.0, 0.0], [0.5, 0.5])
0.46450140402245893
>>> distance.jensenshannon([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
0.0
*)

val kulsinski : ?w:[>`Ndarray] Np.Obj.t -> u:[`Ndarray of [>`Ndarray] Np.Obj.t | `Bool of bool] -> v:[`Ndarray of [>`Ndarray] Np.Obj.t | `Bool of bool] -> unit -> float
(**
Compute the Kulsinski dissimilarity between two boolean 1-D arrays.

The Kulsinski dissimilarity between two boolean 1-D arrays `u` and `v`,
is defined as

.. math::

     \frac{c_{TF} + c_{FT} - c_{TT} + n}
          {c_{FT} + c_{TF} + n}

where :math:`c_{ij}` is the number of occurrences of
:math:`\mathtt{u[k]} = i` and :math:`\mathtt{v[k]} = j` for
:math:`k < n`.

Parameters
----------
u : (N,) array_like, bool
    Input array.
v : (N,) array_like, bool
    Input array.
w : (N,) array_like, optional
    The weights for each value in `u` and `v`. Default is None,
    which gives each value a weight of 1.0

Returns
-------
kulsinski : double
    The Kulsinski distance between vectors `u` and `v`.

Examples
--------
>>> from scipy.spatial import distance
>>> distance.kulsinski([1, 0, 0], [0, 1, 0])
1.0
>>> distance.kulsinski([1, 0, 0], [1, 1, 0])
0.75
>>> distance.kulsinski([1, 0, 0], [2, 1, 0])
0.33333333333333331
>>> distance.kulsinski([1, 0, 0], [3, 1, 0])
-0.5
*)

val mahalanobis : u:[>`Ndarray] Np.Obj.t -> v:[>`Ndarray] Np.Obj.t -> vi:[>`Ndarray] Np.Obj.t -> unit -> float
(**
Compute the Mahalanobis distance between two 1-D arrays.

The Mahalanobis distance between 1-D arrays `u` and `v`, is defined as

.. math::

   \sqrt{ (u-v) V^{-1} (u-v)^T }

where ``V`` is the covariance matrix.  Note that the argument `VI`
is the inverse of ``V``.

Parameters
----------
u : (N,) array_like
    Input array.
v : (N,) array_like
    Input array.
VI : ndarray
    The inverse of the covariance matrix.

Returns
-------
mahalanobis : double
    The Mahalanobis distance between vectors `u` and `v`.

Examples
--------
>>> from scipy.spatial import distance
>>> iv = [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]
>>> distance.mahalanobis([1, 0, 0], [0, 1, 0], iv)
1.0
>>> distance.mahalanobis([0, 2, 0], [0, 1, 0], iv)
1.0
>>> distance.mahalanobis([2, 0, 0], [0, 1, 0], iv)
1.7320508075688772
*)

val matching : ?kwds:(string * Py.Object.t) list -> Py.Object.t list -> Py.Object.t
(**
`matching` is deprecated!
spatial.distance.matching is deprecated in scipy 1.0.0; use spatial.distance.hamming instead.

    Compute the Hamming distance between two boolean 1-D arrays.

    This is a deprecated synonym for :func:`hamming`.
    
*)

val minkowski : ?p:int -> ?w:[>`Ndarray] Np.Obj.t -> u:[>`Ndarray] Np.Obj.t -> v:[>`Ndarray] Np.Obj.t -> unit -> float
(**
Compute the Minkowski distance between two 1-D arrays.

The Minkowski distance between 1-D arrays `u` and `v`,
is defined as

.. math::

   { ||u-v|| }_p = (\sum{ |u_i - v_i|^p})^{1/p}.


   \left(\sum{w_i(|(u_i - v_i)|^p)}\right)^{1/p}.

Parameters
----------
u : (N,) array_like
    Input array.
v : (N,) array_like
    Input array.
p : int
    The order of the norm of the difference :math:`{ ||u-v|| }_p`.
w : (N,) array_like, optional
    The weights for each value in `u` and `v`. Default is None,
    which gives each value a weight of 1.0

Returns
-------
minkowski : double
    The Minkowski distance between vectors `u` and `v`.

Examples
--------
>>> from scipy.spatial import distance
>>> distance.minkowski([1, 0, 0], [0, 1, 0], 1)
2.0
>>> distance.minkowski([1, 0, 0], [0, 1, 0], 2)
1.4142135623730951
>>> distance.minkowski([1, 0, 0], [0, 1, 0], 3)
1.2599210498948732
>>> distance.minkowski([1, 1, 0], [0, 1, 0], 1)
1.0
>>> distance.minkowski([1, 1, 0], [0, 1, 0], 2)
1.0
>>> distance.minkowski([1, 1, 0], [0, 1, 0], 3)
1.0
*)

val namedtuple : ?rename:Py.Object.t -> ?defaults:Py.Object.t -> ?module_:Py.Object.t -> typename:Py.Object.t -> field_names:Py.Object.t -> unit -> Py.Object.t
(**
Returns a new subclass of tuple with named fields.

>>> Point = namedtuple('Point', ['x', 'y'])
>>> Point.__doc__                   # docstring for the new class
'Point(x, y)'
>>> p = Point(11, y=22)             # instantiate with positional args or keywords
>>> p[0] + p[1]                     # indexable like a plain tuple
33
>>> x, y = p                        # unpack like a regular tuple
>>> x, y
(11, 22)
>>> p.x + p.y                       # fields also accessible by name
33
>>> d = p._asdict()                 # convert to a dictionary
>>> d['x']
11
>>> Point( **d)                      # convert from a dictionary
Point(x=11, y=22)
>>> p._replace(x=100)               # _replace() is like str.replace() but targets named fields
Point(x=100, y=22)
*)

val norm : ?ord:[`Fro | `PyObject of Py.Object.t] -> ?axis:[`T2_tuple_of_ints of Py.Object.t | `I of int] -> ?keepdims:bool -> ?check_finite:bool -> a:Py.Object.t -> unit -> Py.Object.t
(**
Matrix or vector norm.

This function is able to return one of seven different matrix norms,
or one of an infinite number of vector norms (described below), depending
on the value of the ``ord`` parameter.

Parameters
----------
a : (M,) or (M, N) array_like
    Input array.  If `axis` is None, `a` must be 1-D or 2-D.
ord : {non-zero int, inf, -inf, 'fro'}, optional
    Order of the norm (see table under ``Notes``). inf means numpy's
    `inf` object
axis : {int, 2-tuple of ints, None}, optional
    If `axis` is an integer, it specifies the axis of `a` along which to
    compute the vector norms.  If `axis` is a 2-tuple, it specifies the
    axes that hold 2-D matrices, and the matrix norms of these matrices
    are computed.  If `axis` is None then either a vector norm (when `a`
    is 1-D) or a matrix norm (when `a` is 2-D) is returned.
keepdims : bool, optional
    If this is set to True, the axes which are normed over are left in the
    result as dimensions with size one.  With this option the result will
    broadcast correctly against the original `a`.
check_finite : bool, optional
    Whether to check that the input matrix contains only finite numbers.
    Disabling may give a performance gain, but may result in problems
    (crashes, non-termination) if the inputs do contain infinities or NaNs.

Returns
-------
n : float or ndarray
    Norm of the matrix or vector(s).

Notes
-----
For values of ``ord <= 0``, the result is, strictly speaking, not a
mathematical 'norm', but it may still be useful for various numerical
purposes.

The following norms can be calculated:

=====  ============================  ==========================
ord    norm for matrices             norm for vectors
=====  ============================  ==========================
None   Frobenius norm                2-norm
'fro'  Frobenius norm                --
inf    max(sum(abs(x), axis=1))      max(abs(x))
-inf   min(sum(abs(x), axis=1))      min(abs(x))
0      --                            sum(x != 0)
1      max(sum(abs(x), axis=0))      as below
-1     min(sum(abs(x), axis=0))      as below
2      2-norm (largest sing. value)  as below
-2     smallest singular value       as below
other  --                            sum(abs(x)**ord)**(1./ord)
=====  ============================  ==========================

The Frobenius norm is given by [1]_:

    :math:`||A||_F = [\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

The ``axis`` and ``keepdims`` arguments are passed directly to
``numpy.linalg.norm`` and are only usable if they are supported
by the version of numpy in use.

References
----------
.. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
       Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

Examples
--------
>>> from scipy.linalg import norm
>>> a = np.arange(9) - 4.0
>>> a
array([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
>>> b = a.reshape((3, 3))
>>> b
array([[-4., -3., -2.],
       [-1.,  0.,  1.],
       [ 2.,  3.,  4.]])

>>> norm(a)
7.745966692414834
>>> norm(b)
7.745966692414834
>>> norm(b, 'fro')
7.745966692414834
>>> norm(a, np.inf)
4
>>> norm(b, np.inf)
9
>>> norm(a, -np.inf)
0
>>> norm(b, -np.inf)
2

>>> norm(a, 1)
20
>>> norm(b, 1)
7
>>> norm(a, -1)
-4.6566128774142013e-010
>>> norm(b, -1)
6
>>> norm(a, 2)
7.745966692414834
>>> norm(b, 2)
7.3484692283495345

>>> norm(a, -2)
0
>>> norm(b, -2)
1.8570331885190563e-016
>>> norm(a, 3)
5.8480354764257312
>>> norm(a, -3)
0
*)

val num_obs_dm : [>`Ndarray] Np.Obj.t -> int
(**
Return the number of original observations that correspond to a
square, redundant distance matrix.

Parameters
----------
d : ndarray
    The target distance matrix.

Returns
-------
num_obs_dm : int
    The number of observations in the redundant distance matrix.
*)

val num_obs_y : [>`Ndarray] Np.Obj.t -> int
(**
Return the number of original observations that correspond to a
condensed distance matrix.

Parameters
----------
Y : ndarray
    Condensed distance matrix.

Returns
-------
n : int
    The number of observations in the condensed distance matrix `Y`.
*)

val pdist : ?metric:[`Callable of Py.Object.t | `S of string] -> ?kwargs:(string * Py.Object.t) list -> x:[>`Ndarray] Np.Obj.t -> Py.Object.t list -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Pairwise distances between observations in n-dimensional space.

See Notes for common calling conventions.

Parameters
----------
X : ndarray
    An m by n array of m original observations in an
    n-dimensional space.
metric : str or function, optional
    The distance metric to use. The distance function can
    be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
    'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
    'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching',
    'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
    'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.
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
    Default: var(X, axis=0, ddof=1)

    VI : ndarray
    The inverse of the covariance matrix for Mahalanobis.
    Default: inv(cov(X.T)).T

    out : ndarray.
    The output array
    If not None, condensed distance matrix Y is stored in this array.
    Note: metric independent, it will become a regular keyword arg in a
    future scipy version

Returns
-------
Y : ndarray
    Returns a condensed distance matrix Y.  For
    each :math:`i` and :math:`j` (where :math:`i<j<m`),where m is the number
    of original observations. The metric ``dist(u=X[i], v=X[j])``
    is computed and stored in entry ``ij``.

See Also
--------
squareform : converts between condensed distance matrices and
             square distance matrices.

Notes
-----
See ``squareform`` for information on how to calculate the index of
this entry or to convert the condensed distance matrix to a
redundant square matrix.

The following are common calling conventions.

1. ``Y = pdist(X, 'euclidean')``

   Computes the distance between m points using Euclidean distance
   (2-norm) as the distance metric between the points. The points
   are arranged as m n-dimensional row vectors in the matrix X.

2. ``Y = pdist(X, 'minkowski', p=2.)``

   Computes the distances using the Minkowski distance
   :math:`||u-v||_p` (p-norm) where :math:`p \geq 1`.

3. ``Y = pdist(X, 'cityblock')``

   Computes the city block or Manhattan distance between the
   points.

4. ``Y = pdist(X, 'seuclidean', V=None)``

   Computes the standardized Euclidean distance. The standardized
   Euclidean distance between two n-vectors ``u`` and ``v`` is

   .. math::

      \sqrt{\sum {(u_i-v_i)^2 / V[x_i]}}


   V is the variance vector; V[i] is the variance computed over all
   the i'th components of the points.  If not passed, it is
   automatically computed.

5. ``Y = pdist(X, 'sqeuclidean')``

   Computes the squared Euclidean distance :math:`||u-v||_2^2` between
   the vectors.

6. ``Y = pdist(X, 'cosine')``

   Computes the cosine distance between vectors u and v,

   .. math::

      1 - \frac{u \cdot v}
               {{ ||u|| }_2 { ||v|| }_2}

   where :math:`||*||_2` is the 2-norm of its argument ``*``, and
   :math:`u \cdot v` is the dot product of ``u`` and ``v``.

7. ``Y = pdist(X, 'correlation')``

   Computes the correlation distance between vectors u and v. This is

   .. math::

      1 - \frac{(u - \bar{u}) \cdot (v - \bar{v})}
               {{ ||(u - \bar{u})|| }_2 { ||(v - \bar{v})|| }_2}

   where :math:`\bar{v}` is the mean of the elements of vector v,
   and :math:`x \cdot y` is the dot product of :math:`x` and :math:`y`.

8. ``Y = pdist(X, 'hamming')``

   Computes the normalized Hamming distance, or the proportion of
   those vector elements between two n-vectors ``u`` and ``v``
   which disagree. To save memory, the matrix ``X`` can be of type
   boolean.

9. ``Y = pdist(X, 'jaccard')``

   Computes the Jaccard distance between the points. Given two
   vectors, ``u`` and ``v``, the Jaccard distance is the
   proportion of those elements ``u[i]`` and ``v[i]`` that
   disagree.

10. ``Y = pdist(X, 'chebyshev')``

   Computes the Chebyshev distance between the points. The
   Chebyshev distance between two n-vectors ``u`` and ``v`` is the
   maximum norm-1 distance between their respective elements. More
   precisely, the distance is given by

   .. math::

      d(u,v) = \max_i { |u_i-v_i| }

11. ``Y = pdist(X, 'canberra')``

   Computes the Canberra distance between the points. The
   Canberra distance between two points ``u`` and ``v`` is

   .. math::

     d(u,v) = \sum_i \frac{ |u_i-v_i| }
                          { |u_i|+|v_i| }


12. ``Y = pdist(X, 'braycurtis')``

   Computes the Bray-Curtis distance between the points. The
   Bray-Curtis distance between two points ``u`` and ``v`` is


   .. math::

        d(u,v) = \frac{\sum_i { |u_i-v_i| }}
                       {\sum_i { |u_i+v_i| }}

13. ``Y = pdist(X, 'mahalanobis', VI=None)``

   Computes the Mahalanobis distance between the points. The
   Mahalanobis distance between two points ``u`` and ``v`` is
   :math:`\sqrt{(u-v)(1/V)(u-v)^T}` where :math:`(1/V)` (the ``VI``
   variable) is the inverse covariance. If ``VI`` is not None,
   ``VI`` will be used as the inverse covariance matrix.

14. ``Y = pdist(X, 'yule')``

   Computes the Yule distance between each pair of boolean
   vectors. (see yule function documentation)

15. ``Y = pdist(X, 'matching')``

   Synonym for 'hamming'.

16. ``Y = pdist(X, 'dice')``

   Computes the Dice distance between each pair of boolean
   vectors. (see dice function documentation)

17. ``Y = pdist(X, 'kulsinski')``

   Computes the Kulsinski distance between each pair of
   boolean vectors. (see kulsinski function documentation)

18. ``Y = pdist(X, 'rogerstanimoto')``

   Computes the Rogers-Tanimoto distance between each pair of
   boolean vectors. (see rogerstanimoto function documentation)

19. ``Y = pdist(X, 'russellrao')``

   Computes the Russell-Rao distance between each pair of
   boolean vectors. (see russellrao function documentation)

20. ``Y = pdist(X, 'sokalmichener')``

   Computes the Sokal-Michener distance between each pair of
   boolean vectors. (see sokalmichener function documentation)

21. ``Y = pdist(X, 'sokalsneath')``

   Computes the Sokal-Sneath distance between each pair of
   boolean vectors. (see sokalsneath function documentation)

22. ``Y = pdist(X, 'wminkowski', p=2, w=w)``

   Computes the weighted Minkowski distance between each pair of
   vectors. (see wminkowski function documentation)

23. ``Y = pdist(X, f)``

   Computes the distance between all pairs of vectors in X
   using the user supplied 2-arity function f. For example,
   Euclidean distance between the vectors could be computed
   as follows::

     dm = pdist(X, lambda u, v: np.sqrt(((u-v)**2).sum()))

   Note that you should avoid passing a reference to one of
   the distance functions defined in this library. For example,::

     dm = pdist(X, sokalsneath)

   would calculate the pair-wise distances between the vectors in
   X using the Python function sokalsneath. This would result in
   sokalsneath being called :math:`{n \choose 2}` times, which
   is inefficient. Instead, the optimized C version is more
   efficient, and we call it using the following syntax.::

     dm = pdist(X, 'sokalsneath')
*)

val rel_entr : ?out:[>`Ndarray] Np.Obj.t -> ?where:Py.Object.t -> x:Py.Object.t -> unit -> Py.Object.t
(**
rel_entr(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

rel_entr(x, y, out=None)

Elementwise function for computing relative entropy.

.. math::

    \mathrm{rel\_entr}(x, y) =
        \begin{cases}
            x \log(x / y) & x > 0, y > 0 \\
            0 & x = 0, y \ge 0 \\
            \infty & \text{otherwise}
        \end{cases}

Parameters
----------
x, y : array_like
    Input arrays
out : ndarray, optional
    Optional output array for the function results

Returns
-------
scalar or ndarray
    Relative entropy of the inputs

See Also
--------
entr, kl_div

Notes
-----
.. versionadded:: 0.15.0

This function is jointly convex in x and y.

The origin of this function is in convex programming; see
[1]_. Given two discrete probability distributions :math:`p_1,
\ldots, p_n` and :math:`q_1, \ldots, q_n`, to get the relative
entropy of statistics compute the sum

.. math::

    \sum_{i = 1}^n \mathrm{rel\_entr}(p_i, q_i).

See [2]_ for details.

References
----------
.. [1] Grant, Boyd, and Ye, 'CVX: Matlab Software for Disciplined Convex
    Programming', http://cvxr.com/cvx/
.. [2] Kullback-Leibler divergence,
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
*)

val rogerstanimoto : ?w:[>`Ndarray] Np.Obj.t -> u:[`Ndarray of [>`Ndarray] Np.Obj.t | `Bool of bool] -> v:[`Ndarray of [>`Ndarray] Np.Obj.t | `Bool of bool] -> unit -> float
(**
Compute the Rogers-Tanimoto dissimilarity between two boolean 1-D arrays.

The Rogers-Tanimoto dissimilarity between two boolean 1-D arrays
`u` and `v`, is defined as

.. math::
   \frac{R}
        {c_{TT} + c_{FF} + R}

where :math:`c_{ij}` is the number of occurrences of
:math:`\mathtt{u[k]} = i` and :math:`\mathtt{v[k]} = j` for
:math:`k < n` and :math:`R = 2(c_{TF} + c_{FT})`.

Parameters
----------
u : (N,) array_like, bool
    Input array.
v : (N,) array_like, bool
    Input array.
w : (N,) array_like, optional
    The weights for each value in `u` and `v`. Default is None,
    which gives each value a weight of 1.0

Returns
-------
rogerstanimoto : double
    The Rogers-Tanimoto dissimilarity between vectors
    `u` and `v`.

Examples
--------
>>> from scipy.spatial import distance
>>> distance.rogerstanimoto([1, 0, 0], [0, 1, 0])
0.8
>>> distance.rogerstanimoto([1, 0, 0], [1, 1, 0])
0.5
>>> distance.rogerstanimoto([1, 0, 0], [2, 0, 0])
-1.0
*)

val russellrao : ?w:[>`Ndarray] Np.Obj.t -> u:[`Ndarray of [>`Ndarray] Np.Obj.t | `Bool of bool] -> v:[`Ndarray of [>`Ndarray] Np.Obj.t | `Bool of bool] -> unit -> float
(**
Compute the Russell-Rao dissimilarity between two boolean 1-D arrays.

The Russell-Rao dissimilarity between two boolean 1-D arrays, `u` and
`v`, is defined as

.. math::

  \frac{n - c_{TT}}
       {n}

where :math:`c_{ij}` is the number of occurrences of
:math:`\mathtt{u[k]} = i` and :math:`\mathtt{v[k]} = j` for
:math:`k < n`.

Parameters
----------
u : (N,) array_like, bool
    Input array.
v : (N,) array_like, bool
    Input array.
w : (N,) array_like, optional
    The weights for each value in `u` and `v`. Default is None,
    which gives each value a weight of 1.0

Returns
-------
russellrao : double
    The Russell-Rao dissimilarity between vectors `u` and `v`.

Examples
--------
>>> from scipy.spatial import distance
>>> distance.russellrao([1, 0, 0], [0, 1, 0])
1.0
>>> distance.russellrao([1, 0, 0], [1, 1, 0])
0.6666666666666666
>>> distance.russellrao([1, 0, 0], [2, 0, 0])
0.3333333333333333
*)

val seuclidean : u:[>`Ndarray] Np.Obj.t -> v:[>`Ndarray] Np.Obj.t -> v':[>`Ndarray] Np.Obj.t -> unit -> float
(**
Return the standardized Euclidean distance between two 1-D arrays.

The standardized Euclidean distance between `u` and `v`.

Parameters
----------
u : (N,) array_like
    Input array.
v : (N,) array_like
    Input array.
V : (N,) array_like
    `V` is an 1-D array of component variances. It is usually computed
    among a larger collection vectors.

Returns
-------
seuclidean : double
    The standardized Euclidean distance between vectors `u` and `v`.

Examples
--------
>>> from scipy.spatial import distance
>>> distance.seuclidean([1, 0, 0], [0, 1, 0], [0.1, 0.1, 0.1])
4.4721359549995796
>>> distance.seuclidean([1, 0, 0], [0, 1, 0], [1, 0.1, 0.1])
3.3166247903553998
>>> distance.seuclidean([1, 0, 0], [0, 1, 0], [10, 0.1, 0.1])
3.1780497164141406
*)

val sokalmichener : ?w:[>`Ndarray] Np.Obj.t -> u:[`Ndarray of [>`Ndarray] Np.Obj.t | `Bool of bool] -> v:[`Ndarray of [>`Ndarray] Np.Obj.t | `Bool of bool] -> unit -> float
(**
Compute the Sokal-Michener dissimilarity between two boolean 1-D arrays.

The Sokal-Michener dissimilarity between boolean 1-D arrays `u` and `v`,
is defined as

.. math::

   \frac{R}
        {S + R}

where :math:`c_{ij}` is the number of occurrences of
:math:`\mathtt{u[k]} = i` and :math:`\mathtt{v[k]} = j` for
:math:`k < n`, :math:`R = 2 * (c_{TF} + c_{FT})` and
:math:`S = c_{FF} + c_{TT}`.

Parameters
----------
u : (N,) array_like, bool
    Input array.
v : (N,) array_like, bool
    Input array.
w : (N,) array_like, optional
    The weights for each value in `u` and `v`. Default is None,
    which gives each value a weight of 1.0

Returns
-------
sokalmichener : double
    The Sokal-Michener dissimilarity between vectors `u` and `v`.

Examples
--------
>>> from scipy.spatial import distance
>>> distance.sokalmichener([1, 0, 0], [0, 1, 0])
0.8
>>> distance.sokalmichener([1, 0, 0], [1, 1, 0])
0.5
>>> distance.sokalmichener([1, 0, 0], [2, 0, 0])
-1.0
*)

val sokalsneath : ?w:[>`Ndarray] Np.Obj.t -> u:[`Ndarray of [>`Ndarray] Np.Obj.t | `Bool of bool] -> v:[`Ndarray of [>`Ndarray] Np.Obj.t | `Bool of bool] -> unit -> float
(**
Compute the Sokal-Sneath dissimilarity between two boolean 1-D arrays.

The Sokal-Sneath dissimilarity between `u` and `v`,

.. math::

   \frac{R}
        {c_{TT} + R}

where :math:`c_{ij}` is the number of occurrences of
:math:`\mathtt{u[k]} = i` and :math:`\mathtt{v[k]} = j` for
:math:`k < n` and :math:`R = 2(c_{TF} + c_{FT})`.

Parameters
----------
u : (N,) array_like, bool
    Input array.
v : (N,) array_like, bool
    Input array.
w : (N,) array_like, optional
    The weights for each value in `u` and `v`. Default is None,
    which gives each value a weight of 1.0

Returns
-------
sokalsneath : double
    The Sokal-Sneath dissimilarity between vectors `u` and `v`.

Examples
--------
>>> from scipy.spatial import distance
>>> distance.sokalsneath([1, 0, 0], [0, 1, 0])
1.0
>>> distance.sokalsneath([1, 0, 0], [1, 1, 0])
0.66666666666666663
>>> distance.sokalsneath([1, 0, 0], [2, 1, 0])
0.0
>>> distance.sokalsneath([1, 0, 0], [3, 1, 0])
-2.0
*)

val sqeuclidean : ?w:[>`Ndarray] Np.Obj.t -> u:[>`Ndarray] Np.Obj.t -> v:[>`Ndarray] Np.Obj.t -> unit -> float
(**
Compute the squared Euclidean distance between two 1-D arrays.

The squared Euclidean distance between `u` and `v` is defined as

.. math::

   { ||u-v|| }_2^2

   \left(\sum{(w_i |(u_i - v_i)|^2)}\right)

Parameters
----------
u : (N,) array_like
    Input array.
v : (N,) array_like
    Input array.
w : (N,) array_like, optional
    The weights for each value in `u` and `v`. Default is None,
    which gives each value a weight of 1.0

Returns
-------
sqeuclidean : double
    The squared Euclidean distance between vectors `u` and `v`.

Examples
--------
>>> from scipy.spatial import distance
>>> distance.sqeuclidean([1, 0, 0], [0, 1, 0])
2.0
>>> distance.sqeuclidean([1, 1, 0], [0, 1, 0])
1.0
*)

val squareform : ?force:string -> ?checks:bool -> x:[>`Ndarray] Np.Obj.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Convert a vector-form distance vector to a square-form distance
matrix, and vice-versa.

Parameters
----------
X : ndarray
    Either a condensed or redundant distance matrix.
force : str, optional
    As with MATLAB(TM), if force is equal to ``'tovector'`` or
    ``'tomatrix'``, the input will be treated as a distance matrix or
    distance vector respectively.
checks : bool, optional
    If set to False, no checks will be made for matrix
    symmetry nor zero diagonals. This is useful if it is known that
    ``X - X.T1`` is small and ``diag(X)`` is close to zero.
    These values are ignored any way so they do not disrupt the
    squareform transformation.

Returns
-------
Y : ndarray
    If a condensed distance matrix is passed, a redundant one is
    returned, or if a redundant one is passed, a condensed distance
    matrix is returned.

Notes
-----
1. v = squareform(X)

   Given a square d-by-d symmetric distance matrix X,
   ``v = squareform(X)`` returns a ``d * (d-1) / 2`` (or
   :math:`{n \choose 2}`) sized vector v.

  :math:`v[{n \choose 2}-{n-i \choose 2} + (j-i-1)]` is the distance
  between points i and j. If X is non-square or asymmetric, an error
  is returned.

2. X = squareform(v)

  Given a ``d*(d-1)/2`` sized v for some integer ``d >= 2`` encoding
  distances as described, ``X = squareform(v)`` returns a d by d distance
  matrix X.  The ``X[i, j]`` and ``X[j, i]`` values are set to
  :math:`v[{n \choose 2}-{n-i \choose 2} + (j-i-1)]` and all
  diagonal elements are zero.

In SciPy 0.19.0, ``squareform`` stopped casting all input types to
float64, and started returning arrays of the same dtype as the input.
*)

val wminkowski : u:[>`Ndarray] Np.Obj.t -> v:[>`Ndarray] Np.Obj.t -> p:int -> w:[>`Ndarray] Np.Obj.t -> unit -> float
(**
Compute the weighted Minkowski distance between two 1-D arrays.

The weighted Minkowski distance between `u` and `v`, defined as

.. math::

   \left(\sum{(|w_i (u_i - v_i)|^p)}\right)^{1/p}.

Parameters
----------
u : (N,) array_like
    Input array.
v : (N,) array_like
    Input array.
p : int
    The order of the norm of the difference :math:`{ ||u-v|| }_p`.
w : (N,) array_like
    The weight vector.

Returns
-------
wminkowski : double
    The weighted Minkowski distance between vectors `u` and `v`.

Notes
-----
`wminkowski` is DEPRECATED. It implements a definition where weights
are powered. It is recommended to use the weighted version of `minkowski`
instead. This function will be removed in a future version of scipy.

Examples
--------
>>> from scipy.spatial import distance
>>> distance.wminkowski([1, 0, 0], [0, 1, 0], 1, np.ones(3))
2.0
>>> distance.wminkowski([1, 0, 0], [0, 1, 0], 2, np.ones(3))
1.4142135623730951
>>> distance.wminkowski([1, 0, 0], [0, 1, 0], 3, np.ones(3))
1.2599210498948732
>>> distance.wminkowski([1, 1, 0], [0, 1, 0], 1, np.ones(3))
1.0
>>> distance.wminkowski([1, 1, 0], [0, 1, 0], 2, np.ones(3))
1.0
>>> distance.wminkowski([1, 1, 0], [0, 1, 0], 3, np.ones(3))
1.0
*)

val yule : ?w:[>`Ndarray] Np.Obj.t -> u:[`Ndarray of [>`Ndarray] Np.Obj.t | `Bool of bool] -> v:[`Ndarray of [>`Ndarray] Np.Obj.t | `Bool of bool] -> unit -> float
(**
Compute the Yule dissimilarity between two boolean 1-D arrays.

The Yule dissimilarity is defined as

.. math::

     \frac{R}{c_{TT} * c_{FF} + \frac{R}{2}}

where :math:`c_{ij}` is the number of occurrences of
:math:`\mathtt{u[k]} = i` and :math:`\mathtt{v[k]} = j` for
:math:`k < n` and :math:`R = 2.0 * c_{TF} * c_{FT}`.

Parameters
----------
u : (N,) array_like, bool
    Input array.
v : (N,) array_like, bool
    Input array.
w : (N,) array_like, optional
    The weights for each value in `u` and `v`. Default is None,
    which gives each value a weight of 1.0

Returns
-------
yule : double
    The Yule dissimilarity between vectors `u` and `v`.

Examples
--------
>>> from scipy.spatial import distance
>>> distance.yule([1, 0, 0], [0, 1, 0])
2.0
>>> distance.yule([1, 1, 0], [0, 1, 0])
0.0
*)


end

module Kdtree : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val distance_matrix : ?p:[`T1_p_infinity of Py.Object.t | `F of float] -> ?threshold:Py.Object.t -> x:Py.Object.t -> y:Py.Object.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the distance matrix.

Returns the matrix of all pair-wise distances.

Parameters
----------
x : (M, K) array_like
    Matrix of M vectors in K dimensions.
y : (N, K) array_like
    Matrix of N vectors in K dimensions.
p : float, 1 <= p <= infinity
    Which Minkowski p-norm to use.
threshold : positive int
    If ``M * N * K`` > `threshold`, algorithm uses a Python loop instead
    of large temporary arrays.

Returns
-------
result : (M, N) ndarray
    Matrix containing the distance from every vector in `x` to every vector
    in `y`.

Examples
--------
>>> from scipy.spatial import distance_matrix
>>> distance_matrix([[0,0],[0,1]], [[1,0],[1,1]])
array([[ 1.        ,  1.41421356],
       [ 1.41421356,  1.        ]])
*)

val heappop : Py.Object.t -> Py.Object.t
(**
Pop the smallest item off the heap, maintaining the heap invariant.
*)

val heappush : heap:Py.Object.t -> item:Py.Object.t -> unit -> Py.Object.t
(**
Push item onto heap, maintaining the heap invariant.
*)

val minkowski_distance : ?p:[`T1_p_infinity of Py.Object.t | `F of float] -> x:Py.Object.t -> y:Py.Object.t -> unit -> Py.Object.t
(**
Compute the L**p distance between two arrays.

Parameters
----------
x : (M, K) array_like
    Input array.
y : (N, K) array_like
    Input array.
p : float, 1 <= p <= infinity
    Which Minkowski p-norm to use.

Examples
--------
>>> from scipy.spatial import minkowski_distance
>>> minkowski_distance([[0,0],[0,0]], [[1,1],[0,1]])
array([ 1.41421356,  1.        ])
*)

val minkowski_distance_p : ?p:[`T1_p_infinity of Py.Object.t | `F of float] -> x:Py.Object.t -> y:Py.Object.t -> unit -> Py.Object.t
(**
Compute the p-th power of the L**p distance between two arrays.

For efficiency, this function computes the L**p distance but does
not extract the pth root. If `p` is 1 or infinity, this is equal to
the actual L**p distance.

Parameters
----------
x : (M, K) array_like
    Input array.
y : (N, K) array_like
    Input array.
p : float, 1 <= p <= infinity
    Which Minkowski p-norm to use.

Examples
--------
>>> from scipy.spatial import minkowski_distance_p
>>> minkowski_distance_p([[0,0],[0,0]], [[1,1],[0,1]])
array([2, 1])
*)


end

module Qhull : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module QhullError : sig
type tag = [`QhullError]
type t = [`BaseException | `Object | `QhullError] Obj.t
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

val asbytes : Py.Object.t -> Py.Object.t
(**
None
*)

val tsearch : tri:Py.Object.t -> xi:Py.Object.t -> unit -> Py.Object.t
(**
tsearch(tri, xi)

Find simplices containing the given points. This function does the
same thing as `Delaunay.find_simplex`.

.. versionadded:: 0.9

See Also
--------
Delaunay.find_simplex


Examples
--------

>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from scipy.spatial import Delaunay, delaunay_plot_2d, tsearch

The Delaunay triangulation of a set of random points:

>>> pts = np.random.rand(20, 2)
>>> tri = Delaunay(pts)
>>> _ = delaunay_plot_2d(tri)

Find the simplices containing a given set of points:

>>> loc = np.random.uniform(0.2, 0.8, (5, 2))
>>> s = tsearch(tri, loc)
>>> plt.triplot(pts[:, 0], pts[:, 1], tri.simplices[s], 'b-', mask=s==-1)
>>> plt.scatter(loc[:, 0], loc[:, 1], c='r', marker='x')
>>> plt.show()
*)


end

module Transform : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

module Rotation : sig
type tag = [`Rotation]
type t = [`Object | `Rotation] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : ?normalize:Py.Object.t -> ?copy:Py.Object.t -> quat:Py.Object.t -> unit -> t
(**
Rotation in 3 dimensions.

This class provides an interface to initialize from and represent rotations
with:

- Quaternions
- Rotation Matrices
- Rotation Vectors
- Euler Angles

The following operations on rotations are supported:

- Application on vectors
- Rotation Composition
- Rotation Inversion
- Rotation Indexing

Indexing within a rotation is supported since multiple rotation transforms
can be stored within a single `Rotation` instance.

To create `Rotation` objects use ``from_...`` methods (see examples below).
``Rotation(...)`` is not supposed to be instantiated directly.

Methods
-------
__len__
from_quat
from_matrix
from_rotvec
from_euler
as_quat
as_matrix
as_rotvec
as_euler
apply
__mul__
inv
magnitude
mean
reduce
create_group
__getitem__
identity
random
align_vectors

See Also
--------
Slerp

Notes
-----
.. versionadded: 1.2.0

Examples
--------
>>> from scipy.spatial.transform import Rotation as R

A `Rotation` instance can be initialized in any of the above formats and
converted to any of the others. The underlying object is independent of the
representation used for initialization.

Consider a counter-clockwise rotation of 90 degrees about the z-axis. This
corresponds to the following quaternion (in scalar-last format):

>>> r = R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])

The rotation can be expressed in any of the other formats:

>>> r.as_matrix()
array([[ 2.22044605e-16, -1.00000000e+00,  0.00000000e+00],
[ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00],
[ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
>>> r.as_rotvec()
array([0.        , 0.        , 1.57079633])
>>> r.as_euler('zyx', degrees=True)
array([90.,  0.,  0.])

The same rotation can be initialized using a rotation matrix:

>>> r = R.from_matrix([[0, -1, 0],
...                    [1, 0, 0],
...                    [0, 0, 1]])

Representation in other formats:

>>> r.as_quat()
array([0.        , 0.        , 0.70710678, 0.70710678])
>>> r.as_rotvec()
array([0.        , 0.        , 1.57079633])
>>> r.as_euler('zyx', degrees=True)
array([90.,  0.,  0.])

The rotation vector corresponding to this rotation is given by:

>>> r = R.from_rotvec(np.pi/2 * np.array([0, 0, 1]))

Representation in other formats:

>>> r.as_quat()
array([0.        , 0.        , 0.70710678, 0.70710678])
>>> r.as_matrix()
array([[ 2.22044605e-16, -1.00000000e+00,  0.00000000e+00],
       [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
>>> r.as_euler('zyx', degrees=True)
array([90.,  0.,  0.])

The ``from_euler`` method is quite flexible in the range of input formats
it supports. Here we initialize a single rotation about a single axis:

>>> r = R.from_euler('z', 90, degrees=True)

Again, the object is representation independent and can be converted to any
other format:

>>> r.as_quat()
array([0.        , 0.        , 0.70710678, 0.70710678])
>>> r.as_matrix()
array([[ 2.22044605e-16, -1.00000000e+00,  0.00000000e+00],
       [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
>>> r.as_rotvec()
array([0.        , 0.        , 1.57079633])

It is also possible to initialize multiple rotations in a single instance
using any of the `from_...` functions. Here we initialize a stack of 3
rotations using the ``from_euler`` method:

>>> r = R.from_euler('zyx', [
... [90, 0, 0],
... [0, 45, 0],
... [45, 60, 30]], degrees=True)

The other representations also now return a stack of 3 rotations. For
example:

>>> r.as_quat()
array([[0.        , 0.        , 0.70710678, 0.70710678],
       [0.        , 0.38268343, 0.        , 0.92387953],
       [0.39190384, 0.36042341, 0.43967974, 0.72331741]])

Applying the above rotations onto a vector:

>>> v = [1, 2, 3]
>>> r.apply(v)
array([[-2.        ,  1.        ,  3.        ],
       [ 2.82842712,  2.        ,  1.41421356],
       [ 2.24452282,  0.78093109,  2.89002836]])

A `Rotation` instance can be indexed and sliced as if it were a single
1D array or list:

>>> r.as_quat()
array([[0.        , 0.        , 0.70710678, 0.70710678],
       [0.        , 0.38268343, 0.        , 0.92387953],
       [0.39190384, 0.36042341, 0.43967974, 0.72331741]])
>>> p = r[0]
>>> p.as_matrix()
array([[ 2.22044605e-16, -1.00000000e+00,  0.00000000e+00],
       [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
>>> q = r[1:3]
>>> q.as_quat()
array([[0.        , 0.38268343, 0.        , 0.92387953],
       [0.39190384, 0.36042341, 0.43967974, 0.72331741]])

Multiple rotations can be composed using the ``*`` operator:

>>> r1 = R.from_euler('z', 90, degrees=True)
>>> r2 = R.from_rotvec([np.pi/4, 0, 0])
>>> v = [1, 2, 3]
>>> r2.apply(r1.apply(v))
array([-2.        , -1.41421356,  2.82842712])
>>> r3 = r2 * r1 # Note the order
>>> r3.apply(v)
array([-2.        , -1.41421356,  2.82842712])

Finally, it is also possible to invert rotations:

>>> r1 = R.from_euler('z', [90, 45], degrees=True)
>>> r2 = r1.inv()
>>> r2.as_euler('zyx', degrees=True)
array([[-90.,   0.,   0.],
       [-45.,   0.,   0.]])

These examples serve as an overview into the `Rotation` class and highlight
major functionalities. For more thorough examples of the range of input and
output formats supported, consult the individual method's examples.
*)

val __getitem__ : indexer:[`Slice of Np.Wrap_utils.Slice.t | `PyObject of Py.Object.t] -> [> tag] Obj.t -> Py.Object.t
(**
Extract rotation(s) at given index(es) from object.

Create a new `Rotation` instance containing a subset of rotations
stored in this object.

Parameters
----------
indexer : index, slice, or index array
    Specifies which rotation(s) to extract. A single indexer must be
    specified, i.e. as if indexing a 1 dimensional array or list.

Returns
-------
rotation : `Rotation` instance
    Contains
        - a single rotation, if `indexer` is a single index
        - a stack of rotation(s), if `indexer` is a slice, or and index
          array.

Examples
--------
>>> from scipy.spatial.transform import Rotation as R
>>> r = R.from_quat([
... [1, 1, 0, 0],
... [0, 1, 0, 1],
... [1, 1, -1, 0]])
>>> r.as_quat()
array([[ 0.70710678,  0.70710678,  0.        ,  0.        ],
       [ 0.        ,  0.70710678,  0.        ,  0.70710678],
       [ 0.57735027,  0.57735027, -0.57735027,  0.        ]])

Indexing using a single index:

>>> p = r[0]
>>> p.as_quat()
array([0.70710678, 0.70710678, 0.        , 0.        ])

Array slicing:

>>> q = r[1:3]
>>> q.as_quat()
array([[ 0.        ,  0.70710678,  0.        ,  0.70710678],
       [ 0.57735027,  0.57735027, -0.57735027,  0.        ]])
*)

val align_vectors : ?weights:[>`Ndarray] Np.Obj.t -> ?return_sensitivity:bool -> a:[>`Ndarray] Np.Obj.t -> b:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> (Py.Object.t * float * [`ArrayLike|`Ndarray|`Object] Np.Obj.t)
(**
Estimate a rotation to optimally align two sets of vectors.

Find a rotation between frames A and B which best aligns a set of
vectors `a` and `b` observed in these frames. The following loss
function is minimized to solve for the rotation matrix
:math:`C`:

.. math::

    L(C) = \frac{1}{2} \sum_{i = 1}^{n} w_i \lVert \mathbf{a}_i -
    C \mathbf{b}_i \rVert^2 ,

where :math:`w_i`'s are the `weights` corresponding to each vector.

The rotation is estimated with Kabsch algorithm [1]_.

Parameters
----------
a : array_like, shape (N, 3)
    Vector components observed in initial frame A. Each row of `a`
    denotes a vector.
b : array_like, shape (N, 3)
    Vector components observed in another frame B. Each row of `b`
    denotes a vector.
weights : array_like shape (N,), optional
    Weights describing the relative importance of the vector
    observations. If None (default), then all values in `weights` are
    assumed to be 1.
return_sensitivity : bool, optional
    Whether to return the sensitivity matrix. See Notes for details.
    Default is False.

Returns
-------
estimated_rotation : `Rotation` instance
    Best estimate of the rotation that transforms `b` to `a`.
rmsd : float
    Root mean square distance (weighted) between the given set of
    vectors after alignment. It is equal to ``sqrt(2 * minimum_loss)``,
    where ``minimum_loss`` is the loss function evaluated for the
    found optimal rotation.
sensitivity_matrix : ndarray, shape (3, 3)
    Sensitivity matrix of the estimated rotation estimate as explained
    in Notes. Returned only when `return_sensitivity` is True.

Notes
-----
This method can also compute the sensitivity of the estimated rotation
to small perturbations of the vector measurements. Specifically we
consider the rotation estimate error as a small rotation vector of
frame A. The sensitivity matrix is proportional to the covariance of
this rotation vector assuming that the vectors in `a` was measured with
errors significantly less than their lengths. To get the true
covariance matrix, the returned sensitivity matrix must be multiplied
by harmonic mean [3]_ of variance in each observation. Note that
`weights` are supposed to be inversely proportional to the observation
variances to get consistent results. For example, if all vectors are
measured with the same accuracy of 0.01 (`weights` must be all equal),
then you should multiple the sensitivity matrix by 0.01**2 to get the
covariance.

Refer to [2]_ for more rigorous discussion of the covariance
estimation.

References
----------
.. [1] https://en.wikipedia.org/wiki/Kabsch_algorithm
.. [2] F. Landis Markley,
        'Attitude determination using vector observations: a fast
        optimal matrix algorithm', Journal of Astronautical Sciences,
        Vol. 41, No.2, 1993, pp. 261-280.
.. [3] https://en.wikipedia.org/wiki/Harmonic_mean
*)

val apply : ?inverse:bool -> vectors:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Apply this rotation to a set of vectors.

If the original frame rotates to the final frame by this rotation, then
its application to a vector can be seen in two ways:

    - As a projection of vector components expressed in the final frame
      to the original frame.
    - As the physical rotation of a vector being glued to the original
      frame as it rotates. In this case the vector components are
      expressed in the original frame before and after the rotation.

In terms of rotation matricies, this application is the same as
``self.as_matrix().dot(vectors)``.

Parameters
----------
vectors : array_like, shape (3,) or (N, 3)
    Each `vectors[i]` represents a vector in 3D space. A single vector
    can either be specified with shape `(3, )` or `(1, 3)`. The number
    of rotations and number of vectors given must follow standard numpy
    broadcasting rules: either one of them equals unity or they both
    equal each other.
inverse : boolean, optional
    If True then the inverse of the rotation(s) is applied to the input
    vectors. Default is False.

Returns
-------
rotated_vectors : ndarray, shape (3,) or (N, 3)
    Result of applying rotation on input vectors.
    Shape depends on the following cases:

        - If object contains a single rotation (as opposed to a stack
          with a single rotation) and a single vector is specified with
          shape ``(3,)``, then `rotated_vectors` has shape ``(3,)``.
        - In all other cases, `rotated_vectors` has shape ``(N, 3)``,
          where ``N`` is either the number of rotations or vectors.

Examples
--------
>>> from scipy.spatial.transform import Rotation as R

Single rotation applied on a single vector:

>>> vector = np.array([1, 0, 0])
>>> r = R.from_rotvec([0, 0, np.pi/2])
>>> r.as_matrix()
array([[ 2.22044605e-16, -1.00000000e+00,  0.00000000e+00],
       [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
>>> r.apply(vector)
array([2.22044605e-16, 1.00000000e+00, 0.00000000e+00])
>>> r.apply(vector).shape
(3,)

Single rotation applied on multiple vectors:

>>> vectors = np.array([
... [1, 0, 0],
... [1, 2, 3]])
>>> r = R.from_rotvec([0, 0, np.pi/4])
>>> r.as_matrix()
array([[ 0.70710678, -0.70710678,  0.        ],
       [ 0.70710678,  0.70710678,  0.        ],
       [ 0.        ,  0.        ,  1.        ]])
>>> r.apply(vectors)
array([[ 0.70710678,  0.70710678,  0.        ],
       [-0.70710678,  2.12132034,  3.        ]])
>>> r.apply(vectors).shape
(2, 3)

Multiple rotations on a single vector:

>>> r = R.from_rotvec([[0, 0, np.pi/4], [np.pi/2, 0, 0]])
>>> vector = np.array([1,2,3])
>>> r.as_matrix()
array([[[ 7.07106781e-01, -7.07106781e-01,  0.00000000e+00],
        [ 7.07106781e-01,  7.07106781e-01,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]],
       [[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  2.22044605e-16, -1.00000000e+00],
        [ 0.00000000e+00,  1.00000000e+00,  2.22044605e-16]]])
>>> r.apply(vector)
array([[-0.70710678,  2.12132034,  3.        ],
       [ 1.        , -3.        ,  2.        ]])
>>> r.apply(vector).shape
(2, 3)

Multiple rotations on multiple vectors. Each rotation is applied on the
corresponding vector:

>>> r = R.from_euler('zxy', [
... [0, 0, 90],
... [45, 30, 60]], degrees=True)
>>> vectors = [
... [1, 2, 3],
... [1, 0, -1]]
>>> r.apply(vectors)
array([[ 3.        ,  2.        , -1.        ],
       [-0.09026039,  1.11237244, -0.86860844]])
>>> r.apply(vectors).shape
(2, 3)

It is also possible to apply the inverse rotation:

>>> r = R.from_euler('zxy', [
... [0, 0, 90],
... [45, 30, 60]], degrees=True)
>>> vectors = [
... [1, 2, 3],
... [1, 0, -1]]
>>> r.apply(vectors, inverse=True)
array([[-3.        ,  2.        ,  1.        ],
       [ 1.09533535, -0.8365163 ,  0.3169873 ]])
*)

val as_euler : ?degrees:bool -> seq:[`S of string | `Length_3 of Py.Object.t] -> [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Represent as Euler angles.

Any orientation can be expressed as a composition of 3 elementary
rotations. Once the axis sequence has been chosen, Euler angles define
the angle of rotation around each respective axis [1]_.

The algorithm from [2]_ has been used to calculate Euler angles for the
rotation about a given sequence of axes.

Euler angles suffer from the problem of gimbal lock [3]_, where the
representation loses a degree of freedom and it is not possible to
determine the first and third angles uniquely. In this case,
a warning is raised, and the third angle is set to zero. Note however
that the returned angles still represent the correct rotation.

Parameters
----------
seq : string, length 3
    3 characters belonging to the set {'X', 'Y', 'Z'} for intrinsic
    rotations, or {'x', 'y', 'z'} for extrinsic rotations [1]_.
    Adjacent axes cannot be the same.
    Extrinsic and intrinsic rotations cannot be mixed in one function
    call.
degrees : boolean, optional
    Returned angles are in degrees if this flag is True, else they are
    in radians. Default is False.

Returns
-------
angles : ndarray, shape (3,) or (N, 3)
    Shape depends on shape of inputs used to initialize object.
    The returned angles are in the range:

    - First angle belongs to [-180, 180] degrees (both inclusive)
    - Third angle belongs to [-180, 180] degrees (both inclusive)
    - Second angle belongs to:

        - [-90, 90] degrees if all axes are different (like xyz)
        - [0, 180] degrees if first and third axes are the same
          (like zxz)

References
----------
.. [1] https://en.wikipedia.org/wiki/Euler_angles#Definition_by_intrinsic_rotations
.. [2] Malcolm D. Shuster, F. Landis Markley, 'General formula for
       extraction the Euler angles', Journal of guidance, control, and
       dynamics, vol. 29.1, pp. 215-221. 2006
.. [3] https://en.wikipedia.org/wiki/Gimbal_lock#In_applied_mathematics

Examples
--------
>>> from scipy.spatial.transform import Rotation as R

Represent a single rotation:

>>> r = R.from_rotvec([0, 0, np.pi/2])
>>> r.as_euler('zxy', degrees=True)
array([90.,  0.,  0.])
>>> r.as_euler('zxy', degrees=True).shape
(3,)

Represent a stack of single rotation:

>>> r = R.from_rotvec([[0, 0, np.pi/2]])
>>> r.as_euler('zxy', degrees=True)
array([[90.,  0.,  0.]])
>>> r.as_euler('zxy', degrees=True).shape
(1, 3)

Represent multiple rotations in a single object:

>>> r = R.from_rotvec([
... [0, 0, np.pi/2],
... [0, -np.pi/3, 0],
... [np.pi/4, 0, 0]])
>>> r.as_euler('zxy', degrees=True)
array([[ 90.,   0.,   0.],
       [  0.,   0., -60.],
       [  0.,  45.,   0.]])
>>> r.as_euler('zxy', degrees=True).shape
(3, 3)
*)

val as_matrix : [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Represent as rotation matrix.

3D rotations can be represented using rotation matrices, which
are 3 x 3 real orthogonal matrices with determinant equal to +1 [1]_.

Returns
-------
matrix : ndarray, shape (3, 3) or (N, 3, 3)
    Shape depends on shape of inputs used for initialization.

References
----------
.. [1] https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions

Examples
--------
>>> from scipy.spatial.transform import Rotation as R

Represent a single rotation:

>>> r = R.from_rotvec([0, 0, np.pi/2])
>>> r.as_matrix()
array([[ 2.22044605e-16, -1.00000000e+00,  0.00000000e+00],
       [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
>>> r.as_matrix().shape
(3, 3)

Represent a stack with a single rotation:

>>> r = R.from_quat([[1, 1, 0, 0]])
>>> r.as_matrix()
array([[[ 0.,  1.,  0.],
        [ 1.,  0.,  0.],
        [ 0.,  0., -1.]]])
>>> r.as_matrix().shape
(1, 3, 3)

Represent multiple rotations:

>>> r = R.from_rotvec([[np.pi/2, 0, 0], [0, 0, np.pi/2]])
>>> r.as_matrix()
array([[[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  2.22044605e-16, -1.00000000e+00],
        [ 0.00000000e+00,  1.00000000e+00,  2.22044605e-16]],
       [[ 2.22044605e-16, -1.00000000e+00,  0.00000000e+00],
        [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]])
>>> r.as_matrix().shape
(2, 3, 3)
*)

val as_quat : [> tag] Obj.t -> Py.Object.t
(**
Represent as quaternions.

Rotations in 3 dimensions can be represented using unit norm
quaternions [1]_. The mapping from quaternions to rotations is
two-to-one, i.e. quaternions ``q`` and ``-q``, where ``-q`` simply
reverses the sign of each component, represent the same spatial
rotation.

Returns
-------
quat : `numpy.ndarray`, shape (4,) or (N, 4)
    Shape depends on shape of inputs used for initialization.

References
----------
.. [1] https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

Examples
--------
>>> from scipy.spatial.transform import Rotation as R

Represent a single rotation:

>>> r = R.from_matrix([[0, -1, 0],
...                    [1, 0, 0],
...                    [0, 0, 1]])
>>> r.as_quat()
array([0.        , 0.        , 0.70710678, 0.70710678])
>>> r.as_quat().shape
(4,)

Represent a stack with a single rotation:

>>> r = R.from_quat([[0, 0, 0, 1]])
>>> r.as_quat().shape
(1, 4)

Represent multiple rotations in a single object:

>>> r = R.from_rotvec([[np.pi, 0, 0], [0, 0, np.pi/2]])
>>> r.as_quat().shape
(2, 4)
*)

val as_rotvec : [> tag] Obj.t -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Represent as rotation vectors.

A rotation vector is a 3 dimensional vector which is co-directional to
the axis of rotation and whose norm gives the angle of rotation (in
radians) [1]_.

Returns
-------
rotvec : ndarray, shape (3,) or (N, 3)
    Shape depends on shape of inputs used for initialization.

References
----------
.. [1] https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Rotation_vector

Examples
--------
>>> from scipy.spatial.transform import Rotation as R

Represent a single rotation:

>>> r = R.from_euler('z', 90, degrees=True)
>>> r.as_rotvec()
array([0.        , 0.        , 1.57079633])
>>> r.as_rotvec().shape
(3,)

Represent a stack with a single rotation:

>>> r = R.from_quat([[0, 0, 1, 1]])
>>> r.as_rotvec()
array([[0.        , 0.        , 1.57079633]])
>>> r.as_rotvec().shape
(1, 3)

Represent multiple rotations in a single object:

>>> r = R.from_quat([[0, 0, 1, 1], [1, 1, 0, 1]])
>>> r.as_rotvec()
array([[0.        , 0.        , 1.57079633],
       [1.35102172, 1.35102172, 0.        ]])
>>> r.as_rotvec().shape
(2, 3)
*)

val create_group : ?axis:int -> group:string -> [> tag] Obj.t -> Py.Object.t
(**
Create a 3D rotation group.

Parameters
----------
group : string
    The name of the group. Must be one of 'I', 'O', 'T', 'Dn', 'Cn',
    where `n` is a positive integer. The groups are:

        * I: Icosahedral group
        * O: Octahedral group
        * T: Tetrahedral group
        * D: Dicyclic group
        * C: Cyclic group

axis : integer
    The cyclic rotation axis. Must be one of ['X', 'Y', 'Z'] (or
    lowercase). Default is 'Z'. Ignored for groups 'I', 'O', and 'T'.

Returns
-------
rotation : `Rotation` instance
    Object containing the elements of the rotation group.

Notes
-----
This method generates rotation groups only. The full 3-dimensional
point groups [PointGroups]_ also contain reflections.

References
----------
.. [PointGroups] `Point groups
   <https://en.wikipedia.org/wiki/Point_groups_in_three_dimensions>`_
   on Wikipedia.
*)

val from_dcm : ?kwds:(string * Py.Object.t) list -> Py.Object.t list -> [> tag] Obj.t -> Py.Object.t
(**
`from_dcm` is deprecated!
from_dcm is renamed to from_matrix in scipy 1.4.0 and will be removed in scipy 1.6.0
*)

val from_euler : ?degrees:bool -> seq:string -> angles:[`Ndarray of [>`Ndarray] Np.Obj.t | `F of float] -> [> tag] Obj.t -> Py.Object.t
(**
Initialize from Euler angles.

Rotations in 3-D can be represented by a sequence of 3
rotations around a sequence of axes. In theory, any three axes spanning
the 3-D Euclidean space are enough. In practice, the axes of rotation are
chosen to be the basis vectors.

The three rotations can either be in a global frame of reference
(extrinsic) or in a body centred frame of reference (intrinsic), which
is attached to, and moves with, the object under rotation [1]_.

Parameters
----------
seq : string
    Specifies sequence of axes for rotations. Up to 3 characters
    belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations, or
    {'x', 'y', 'z'} for extrinsic rotations. Extrinsic and intrinsic
    rotations cannot be mixed in one function call.
angles : float or array_like, shape (N,) or (N, [1 or 2 or 3])
    Euler angles specified in radians (`degrees` is False) or degrees
    (`degrees` is True).
    For a single character `seq`, `angles` can be:

    - a single value
    - array_like with shape (N,), where each `angle[i]`
      corresponds to a single rotation
    - array_like with shape (N, 1), where each `angle[i, 0]`
      corresponds to a single rotation

    For 2- and 3-character wide `seq`, `angles` can be:

    - array_like with shape (W,) where `W` is the width of
      `seq`, which corresponds to a single rotation with `W` axes
    - array_like with shape (N, W) where each `angle[i]`
      corresponds to a sequence of Euler angles describing a single
      rotation

degrees : bool, optional
    If True, then the given angles are assumed to be in degrees.
    Default is False.

Returns
-------
rotation : `Rotation` instance
    Object containing the rotation represented by the sequence of
    rotations around given axes with given angles.

References
----------
.. [1] https://en.wikipedia.org/wiki/Euler_angles#Definition_by_intrinsic_rotations

Examples
--------
>>> from scipy.spatial.transform import Rotation as R

Initialize a single rotation along a single axis:

>>> r = R.from_euler('x', 90, degrees=True)
>>> r.as_quat().shape
(4,)

Initialize a single rotation with a given axis sequence:

>>> r = R.from_euler('zyx', [90, 45, 30], degrees=True)
>>> r.as_quat().shape
(4,)

Initialize a stack with a single rotation around a single axis:

>>> r = R.from_euler('x', [90], degrees=True)
>>> r.as_quat().shape
(1, 4)

Initialize a stack with a single rotation with an axis sequence:

>>> r = R.from_euler('zyx', [[90, 45, 30]], degrees=True)
>>> r.as_quat().shape
(1, 4)

Initialize multiple elementary rotations in one object:

>>> r = R.from_euler('x', [90, 45, 30], degrees=True)
>>> r.as_quat().shape
(3, 4)

Initialize multiple rotations in one object:

>>> r = R.from_euler('zyx', [[90, 45, 30], [35, 45, 90]], degrees=True)
>>> r.as_quat().shape
(2, 4)
*)

val from_matrix : matrix:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Initialize from rotation matrix.

Rotations in 3 dimensions can be represented with 3 x 3 proper
orthogonal matrices [1]_. If the input is not proper orthogonal,
an approximation is created using the method described in [2]_.

Parameters
----------
matrix : array_like, shape (N, 3, 3) or (3, 3)
    A single matrix or a stack of matrices, where ``matrix[i]`` is
    the i-th matrix.

Returns
-------
rotation : `Rotation` instance
    Object containing the rotations represented by the rotation
    matrices.

References
----------
.. [1] https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
.. [2] F. Landis Markley, 'Unit Quaternion from Rotation Matrix',
       Journal of guidance, control, and dynamics vol. 31.2, pp.
       440-442, 2008.

Examples
--------
>>> from scipy.spatial.transform import Rotation as R

Initialize a single rotation:

>>> r = R.from_matrix([
... [0, -1, 0],
... [1, 0, 0],
... [0, 0, 1]])
>>> r.as_matrix().shape
(3, 3)

Initialize multiple rotations in a single object:

>>> r = R.from_matrix([
... [
...     [0, -1, 0],
...     [1, 0, 0],
...     [0, 0, 1],
... ],
... [
...     [1, 0, 0],
...     [0, 0, -1],
...     [0, 1, 0],
... ]])
>>> r.as_matrix().shape
(2, 3, 3)

If input matrices are not special orthogonal (orthogonal with
determinant equal to +1), then a special orthogonal estimate is stored:

>>> a = np.array([
... [0, -0.5, 0],
... [0.5, 0, 0],
... [0, 0, 0.5]])
>>> np.linalg.det(a)
0.12500000000000003
>>> r = R.from_matrix(a)
>>> matrix = r.as_matrix()
>>> matrix
array([[-0.38461538, -0.92307692,  0.        ],
       [ 0.92307692, -0.38461538,  0.        ],
       [ 0.        ,  0.        ,  1.        ]])
>>> np.linalg.det(matrix)
1.0000000000000002

It is also possible to have a stack containing a single rotation:

>>> r = R.from_matrix([[
... [0, -1, 0],
... [1, 0, 0],
... [0, 0, 1]]])
>>> r.as_matrix()
array([[[ 0., -1.,  0.],
        [ 1.,  0.,  0.],
        [ 0.,  0.,  1.]]])
>>> r.as_matrix().shape
(1, 3, 3)
*)

val from_quat : ?normalized:Py.Object.t -> quat:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Initialize from quaternions.

3D rotations can be represented using unit-norm quaternions [1]_.

Parameters
----------
quat : array_like, shape (N, 4) or (4,)
    Each row is a (possibly non-unit norm) quaternion in scalar-last
    (x, y, z, w) format. Each quaternion will be normalized to unit
    norm.
normalized
    Deprecated argument. Has no effect, input `quat` is always
    normalized.

    .. deprecated:: 1.4.0

Returns
-------
rotation : `Rotation` instance
    Object containing the rotations represented by input quaternions.

References
----------
.. [1] https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

Examples
--------
>>> from scipy.spatial.transform import Rotation as R

Initialize a single rotation:

>>> r = R.from_quat([1, 0, 0, 0])
>>> r.as_quat()
array([1., 0., 0., 0.])
>>> r.as_quat().shape
(4,)

Initialize multiple rotations in a single object:

>>> r = R.from_quat([
... [1, 0, 0, 0],
... [0, 0, 0, 1]
... ])
>>> r.as_quat()
array([[1., 0., 0., 0.],
       [0., 0., 0., 1.]])
>>> r.as_quat().shape
(2, 4)

It is also possible to have a stack of a single rotation:

>>> r = R.from_quat([[0, 0, 0, 1]])
>>> r.as_quat()
array([[0., 0., 0., 1.]])
>>> r.as_quat().shape
(1, 4)

Quaternions are normalized before initialization.

>>> r = R.from_quat([0, 0, 1, 1])
>>> r.as_quat()
array([0.        , 0.        , 0.70710678, 0.70710678])
*)

val from_rotvec : rotvec:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Initialize from rotation vectors.

A rotation vector is a 3 dimensional vector which is co-directional to
the axis of rotation and whose norm gives the angle of rotation (in
radians) [1]_.

Parameters
----------
rotvec : array_like, shape (N, 3) or (3,)
    A single vector or a stack of vectors, where `rot_vec[i]` gives
    the ith rotation vector.

Returns
-------
rotation : `Rotation` instance
    Object containing the rotations represented by input rotation
    vectors.

References
----------
.. [1] https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Rotation_vector

Examples
--------
>>> from scipy.spatial.transform import Rotation as R

Initialize a single rotation:

>>> r = R.from_rotvec(np.pi/2 * np.array([0, 0, 1]))
>>> r.as_rotvec()
array([0.        , 0.        , 1.57079633])
>>> r.as_rotvec().shape
(3,)

Initialize multiple rotations in one object:

>>> r = R.from_rotvec([
... [0, 0, np.pi/2],
... [np.pi/2, 0, 0]])
>>> r.as_rotvec()
array([[0.        , 0.        , 1.57079633],
       [1.57079633, 0.        , 0.        ]])
>>> r.as_rotvec().shape
(2, 3)

It is also possible to have a stack of a single rotaton:

>>> r = R.from_rotvec([[0, 0, np.pi/2]])
>>> r.as_rotvec().shape
(1, 3)
*)

val identity : ?num:int -> [> tag] Obj.t -> Py.Object.t
(**
Get identity rotation(s).

Composition with the identity rotation has no effect.

Parameters
----------
num : int or None, optional
    Number of identity rotations to generate. If None (default), then a
    single rotation is generated.

Returns
-------
identity : Rotation object
    The identity rotation.
*)

val inv : [> tag] Obj.t -> Py.Object.t
(**
Invert this rotation.

Composition of a rotation with its inverse results in an identity
transformation.

Returns
-------
inverse : `Rotation` instance
    Object containing inverse of the rotations in the current instance.

Examples
--------
>>> from scipy.spatial.transform import Rotation as R

Inverting a single rotation:

>>> p = R.from_euler('z', 45, degrees=True)
>>> q = p.inv()
>>> q.as_euler('zyx', degrees=True)
array([-45.,   0.,   0.])

Inverting multiple rotations:

>>> p = R.from_rotvec([[0, 0, np.pi/3], [-np.pi/4, 0, 0]])
>>> q = p.inv()
>>> q.as_rotvec()
array([[-0.        , -0.        , -1.04719755],
       [ 0.78539816, -0.        , -0.        ]])
*)

val magnitude : [> tag] Obj.t -> Py.Object.t
(**
Get the magnitude(s) of the rotation(s).

Returns
-------
magnitude : ndarray or float
    Angle(s) in radians, float if object contains a single rotation
    and ndarray if object contains multiple rotations.

Examples
--------
>>> from scipy.spatial.transform import Rotation as R
>>> r = R.from_quat(np.eye(4))
>>> r.magnitude()
array([3.14159265, 3.14159265, 3.14159265, 0.        ])

Magnitude of a single rotation:

>>> r[0].magnitude()
3.141592653589793
*)

val match_vectors : ?kwds:(string * Py.Object.t) list -> Py.Object.t list -> [> tag] Obj.t -> Py.Object.t
(**
`match_vectors` is deprecated!
match_vectors is deprecated in favor of align_vectors in scipy 1.4.0 and will be removed in scipy 1.6.0

Deprecated in favor of `align_vectors`.
*)

val mean : ?weights:[>`Ndarray] Np.Obj.t -> [> tag] Obj.t -> Py.Object.t
(**
Get the mean of the rotations.

Parameters
----------
weights : array_like shape (N,), optional
    Weights describing the relative importance of the rotations. If
    None (default), then all values in `weights` are assumed to be
    equal.

Returns
-------
mean : `Rotation` instance
    Object containing the mean of the rotations in the current
    instance.

Notes
-----
The mean used is the chordal L2 mean (also called the projected or
induced arithmetic mean). If ``p`` is a set of rotations with mean
``m``, then ``m`` is the rotation which minimizes
``(weights[:, None, None] * (p.as_matrix() - m.as_matrix())**2).sum()``.

Examples
--------
>>> from scipy.spatial.transform import Rotation as R
>>> r = R.from_euler('zyx', [[0, 0, 0],
...                          [1, 0, 0],
...                          [0, 1, 0],
...                          [0, 0, 1]], degrees=True)
>>> r.mean().as_euler('zyx', degrees=True)
array([0.24945696, 0.25054542, 0.24945696])
*)

val random : ?num:int -> ?random_state:[`RandomState of Py.Object.t | `I of int] -> [> tag] Obj.t -> Py.Object.t
(**
Generate uniformly distributed rotations.

Parameters
----------
num : int or None, optional
    Number of random rotations to generate. If None (default), then a
    single rotation is generated.
random_state : int, RandomState instance or None, optional
    Accepts an integer as a seed for the random generator or a
    RandomState object. If None (default), uses global `numpy.random`
    random state.

Returns
-------
random_rotation : `Rotation` instance
    Contains a single rotation if `num` is None. Otherwise contains a
    stack of `num` rotations.

Examples
--------
>>> from scipy.spatial.transform import Rotation as R

Sample a single rotation:

>>> R.random(random_state=1234).as_euler('zxy', degrees=True)
array([-110.5976185 ,   55.32758512,   76.3289269 ])

Sample a stack of rotations:

>>> R.random(5, random_state=1234).as_euler('zxy', degrees=True)
array([[-110.5976185 ,   55.32758512,   76.3289269 ],
       [ -91.59132005,  -14.3629884 ,  -93.91933182],
       [  25.23835501,   45.02035145, -121.67867086],
       [ -51.51414184,  -15.29022692, -172.46870023],
       [ -81.63376847,  -27.39521579,    2.60408416]])
*)

val reduce : ?left:Py.Object.t -> ?right:Py.Object.t -> ?return_indices:bool -> [> tag] Obj.t -> Py.Object.t
(**
Reduce this rotation with the provided rotation groups.

Reduction of a rotation ``p`` is a transformation of the form
``q = l * p * r``, where ``l`` and ``r`` are chosen from `left` and
`right` respectively, such that rotation ``q`` has the smallest
magnitude.

If `left` and `right` are rotation groups representing symmetries of
two objects rotated by ``p``, then ``q`` is the rotation of the
smallest magnitude to align these objects considering their symmetries.

Parameters
----------
left : `Rotation` instance, optional
    Object containing the left rotation(s). Default value (None)
    corresponds to the identity rotation.
right : `Rotation` instance, optional
    Object containing the right rotation(s). Default value (None)
    corresponds to the identity rotation.
return_indices : bool, optional
    Whether to return the indices of the rotations from `left` and
    `right` used for reduction.

Returns
-------
reduced : `Rotation` instance
    Object containing reduced rotations.
left_best, right_best: integer ndarray
    Indices of elements from `left` and `right` used for reduction.
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module RotationSpline : sig
type tag = [`RotationSpline]
type t = [`Object | `RotationSpline] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : times:[>`Ndarray] Np.Obj.t -> rotations:Py.Object.t -> unit -> t
(**
Interpolate rotations with continuous angular rate and acceleration.

The rotation vectors between each consecutive orientation are cubic
functions of time and it is guaranteed that angular rate and acceleration
are continuous. Such interpolation are analogous to cubic spline
interpolation.

Refer to [1]_ for math and implementation details.

Parameters
----------
times : array_like, shape (N,)
    Times of the known rotations. At least 2 times must be specified.
rotations : `Rotation` instance
    Rotations to perform the interpolation between. Must contain N
    rotations.

Methods
-------
__call__

References
----------
.. [1] `Smooth Attitude Interpolation
        <https://github.com/scipy/scipy/files/2932755/attitude_interpolation.pdf>`_

Examples
--------
>>> from scipy.spatial.transform import Rotation, RotationSpline

Define the sequence of times and rotations from the Euler angles:

>>> times = [0, 10, 20, 40]
>>> angles = [[-10, 20, 30], [0, 15, 40], [-30, 45, 30], [20, 45, 90]]
>>> rotations = Rotation.from_euler('XYZ', angles, degrees=True)

Create the interpolator object:

>>> spline = RotationSpline(times, rotations)

Interpolate the Euler angles, angular rate and acceleration:

>>> angular_rate = np.rad2deg(spline(times, 1))
>>> angular_acceleration = np.rad2deg(spline(times, 2))
>>> times_plot = np.linspace(times[0], times[-1], 100)
>>> angles_plot = spline(times_plot).as_euler('XYZ', degrees=True)
>>> angular_rate_plot = np.rad2deg(spline(times_plot, 1))
>>> angular_acceleration_plot = np.rad2deg(spline(times_plot, 2))

On this plot you see that Euler angles are continuous and smooth:

>>> import matplotlib.pyplot as plt
>>> plt.plot(times_plot, angles_plot)
>>> plt.plot(times, angles, 'x')
>>> plt.title('Euler angles')
>>> plt.show()

The angular rate is also smooth:

>>> plt.plot(times_plot, angular_rate_plot)
>>> plt.plot(times, angular_rate, 'x')
>>> plt.title('Angular rate')
>>> plt.show()

The angular acceleration is continuous, but not smooth. Also note that
the angular acceleration is not a piecewise-linear function, because
it is different from the second derivative of the rotation vector (which
is a piecewise-linear function as in the cubic spline).

>>> plt.plot(times_plot, angular_acceleration_plot)
>>> plt.plot(times, angular_acceleration, 'x')
>>> plt.title('Angular acceleration')
>>> plt.show()
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Slerp : sig
type tag = [`Slerp]
type t = [`Object | `Slerp] Obj.t
val of_pyobject : Py.Object.t -> t
val to_pyobject : [> tag] Obj.t -> Py.Object.t

val create : times:[>`Ndarray] Np.Obj.t -> rotations:Py.Object.t -> unit -> t
(**
Spherical Linear Interpolation of Rotations.

The interpolation between consecutive rotations is performed as a rotation
around a fixed axis with a constant angular velocity [1]_. This ensures
that the interpolated rotations follow the shortest path between initial
and final orientations.

Parameters
----------
times : array_like, shape (N,)
    Times of the known rotations. At least 2 times must be specified.
rotations : `Rotation` instance
    Rotations to perform the interpolation between. Must contain N
    rotations.

Methods
-------
__call__

See Also
--------
Rotation

Notes
-----
.. versionadded:: 1.2.0

References
----------
.. [1] https://en.wikipedia.org/wiki/Slerp#Quaternion_Slerp

Examples
--------
>>> from scipy.spatial.transform import Rotation as R
>>> from scipy.spatial.transform import Slerp

Setup the fixed keyframe rotations and times:

>>> key_rots = R.random(5, random_state=2342345)
>>> key_times = [0, 1, 2, 3, 4]

Create the interpolator object:

>>> slerp = Slerp(key_times, key_rots)

Interpolate the rotations at the given times:

>>> times = [0, 0.5, 0.25, 1, 1.5, 2, 2.75, 3, 3.25, 3.60, 4]
>>> interp_rots = slerp(times)

The keyframe rotations expressed as Euler angles:

>>> key_rots.as_euler('xyz', degrees=True)
array([[ 14.31443779, -27.50095894,  -3.7275787 ],
       [ -1.79924227, -24.69421529, 164.57701743],
       [146.15020772,  43.22849451, -31.34891088],
       [ 46.39959442,  11.62126073, -45.99719267],
       [-88.94647804, -49.64400082, -65.80546984]])

The interpolated rotations expressed as Euler angles. These agree with the
keyframe rotations at both endpoints of the range of keyframe times.

>>> interp_rots.as_euler('xyz', degrees=True)
array([[  14.31443779,  -27.50095894,   -3.7275787 ],
       [   4.74588574,  -32.44683966,   81.25139984],
       [  10.71094749,  -31.56690154,   38.06896408],
       [  -1.79924227,  -24.69421529,  164.57701743],
       [  11.72796022,   51.64207311, -171.7374683 ],
       [ 146.15020772,   43.22849451,  -31.34891088],
       [  68.10921869,   20.67625074,  -48.74886034],
       [  46.39959442,   11.62126073,  -45.99719267],
       [  12.35552615,    4.21525086,  -64.89288124],
       [ -30.08117143,  -19.90769513,  -78.98121326],
       [ -88.94647804,  -49.64400082,  -65.80546984]])
*)


(** Print the object to a human-readable representation. *)
val to_string : t -> string


(** Print the object to a human-readable representation. *)
val show : t -> string

(** Pretty-print the object to a formatter. *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]


end

module Rotation' : sig
(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val check_random_state : Py.Object.t -> Py.Object.t
(**
Turn seed into a np.random.RandomState instance

If seed is None (or np.random), return the RandomState singleton used
by np.random.
If seed is an int, return a new RandomState instance seeded with seed.
If seed is already a RandomState instance, return it.
If seed is a new-style np.random.Generator, return it.
Otherwise raise ValueError.
*)

val create_group : ?axis:Py.Object.t -> cls:Py.Object.t -> group:Py.Object.t -> unit -> Py.Object.t
(**
None
*)


end


end

val convex_hull_plot_2d : ?ax:Py.Object.t -> hull:Py.Object.t -> unit -> Py.Object.t
(**
Plot the given convex hull diagram in 2-D

Parameters
----------
hull : scipy.spatial.ConvexHull instance
    Convex hull to plot
ax : matplotlib.axes.Axes instance, optional
    Axes to plot on

Returns
-------
fig : matplotlib.figure.Figure instance
    Figure for the plot

See Also
--------
ConvexHull

Notes
-----
Requires Matplotlib.


Examples
--------

>>> import matplotlib.pyplot as plt
>>> from scipy.spatial import ConvexHull, convex_hull_plot_2d

The convex hull of a random set of points:

>>> points = np.random.rand(30, 2)
>>> hull = ConvexHull(points)

Plot it:

>>> _ = convex_hull_plot_2d(hull)
>>> plt.show()
*)

val delaunay_plot_2d : ?ax:Py.Object.t -> tri:Py.Object.t -> unit -> Py.Object.t
(**
Plot the given Delaunay triangulation in 2-D

Parameters
----------
tri : scipy.spatial.Delaunay instance
    Triangulation to plot
ax : matplotlib.axes.Axes instance, optional
    Axes to plot on

Returns
-------
fig : matplotlib.figure.Figure instance
    Figure for the plot

See Also
--------
Delaunay
matplotlib.pyplot.triplot

Notes
-----
Requires Matplotlib.

Examples
--------

>>> import matplotlib.pyplot as plt
>>> from scipy.spatial import Delaunay, delaunay_plot_2d

The Delaunay triangulation of a set of random points:

>>> points = np.random.rand(30, 2)
>>> tri = Delaunay(points)

Plot it:

>>> _ = delaunay_plot_2d(tri)
>>> plt.show()
*)

val distance_matrix : ?p:[`T1_p_infinity of Py.Object.t | `F of float] -> ?threshold:Py.Object.t -> x:Py.Object.t -> y:Py.Object.t -> unit -> [`ArrayLike|`Ndarray|`Object] Np.Obj.t
(**
Compute the distance matrix.

Returns the matrix of all pair-wise distances.

Parameters
----------
x : (M, K) array_like
    Matrix of M vectors in K dimensions.
y : (N, K) array_like
    Matrix of N vectors in K dimensions.
p : float, 1 <= p <= infinity
    Which Minkowski p-norm to use.
threshold : positive int
    If ``M * N * K`` > `threshold`, algorithm uses a Python loop instead
    of large temporary arrays.

Returns
-------
result : (M, N) ndarray
    Matrix containing the distance from every vector in `x` to every vector
    in `y`.

Examples
--------
>>> from scipy.spatial import distance_matrix
>>> distance_matrix([[0,0],[0,1]], [[1,0],[1,1]])
array([[ 1.        ,  1.41421356],
       [ 1.41421356,  1.        ]])
*)

val minkowski_distance : ?p:[`T1_p_infinity of Py.Object.t | `F of float] -> x:Py.Object.t -> y:Py.Object.t -> unit -> Py.Object.t
(**
Compute the L**p distance between two arrays.

Parameters
----------
x : (M, K) array_like
    Input array.
y : (N, K) array_like
    Input array.
p : float, 1 <= p <= infinity
    Which Minkowski p-norm to use.

Examples
--------
>>> from scipy.spatial import minkowski_distance
>>> minkowski_distance([[0,0],[0,0]], [[1,1],[0,1]])
array([ 1.41421356,  1.        ])
*)

val minkowski_distance_p : ?p:[`T1_p_infinity of Py.Object.t | `F of float] -> x:Py.Object.t -> y:Py.Object.t -> unit -> Py.Object.t
(**
Compute the p-th power of the L**p distance between two arrays.

For efficiency, this function computes the L**p distance but does
not extract the pth root. If `p` is 1 or infinity, this is equal to
the actual L**p distance.

Parameters
----------
x : (M, K) array_like
    Input array.
y : (N, K) array_like
    Input array.
p : float, 1 <= p <= infinity
    Which Minkowski p-norm to use.

Examples
--------
>>> from scipy.spatial import minkowski_distance_p
>>> minkowski_distance_p([[0,0],[0,0]], [[1,1],[0,1]])
array([2, 1])
*)

val procrustes : data1:[>`Ndarray] Np.Obj.t -> data2:[>`Ndarray] Np.Obj.t -> unit -> ([`ArrayLike|`Ndarray|`Object] Np.Obj.t * [`ArrayLike|`Ndarray|`Object] Np.Obj.t * float)
(**
Procrustes analysis, a similarity test for two data sets.

Each input matrix is a set of points or vectors (the rows of the matrix).
The dimension of the space is the number of columns of each matrix. Given
two identically sized matrices, procrustes standardizes both such that:

- :math:`tr(AA^{T}) = 1`.

- Both sets of points are centered around the origin.

Procrustes ([1]_, [2]_) then applies the optimal transform to the second
matrix (including scaling/dilation, rotations, and reflections) to minimize
:math:`M^{2}=\sum(data1-data2)^{2}`, or the sum of the squares of the
pointwise differences between the two input datasets.

This function was not designed to handle datasets with different numbers of
datapoints (rows).  If two data sets have different dimensionality
(different number of columns), simply add columns of zeros to the smaller
of the two.

Parameters
----------
data1 : array_like
    Matrix, n rows represent points in k (columns) space `data1` is the
    reference data, after it is standardised, the data from `data2` will be
    transformed to fit the pattern in `data1` (must have >1 unique points).
data2 : array_like
    n rows of data in k space to be fit to `data1`.  Must be the  same
    shape ``(numrows, numcols)`` as data1 (must have >1 unique points).

Returns
-------
mtx1 : array_like
    A standardized version of `data1`.
mtx2 : array_like
    The orientation of `data2` that best fits `data1`. Centered, but not
    necessarily :math:`tr(AA^{T}) = 1`.
disparity : float
    :math:`M^{2}` as defined above.

Raises
------
ValueError
    If the input arrays are not two-dimensional.
    If the shape of the input arrays is different.
    If the input arrays have zero columns or zero rows.

See Also
--------
scipy.linalg.orthogonal_procrustes
scipy.spatial.distance.directed_hausdorff : Another similarity test
  for two data sets

Notes
-----
- The disparity should not depend on the order of the input matrices, but
  the output matrices will, as only the first output matrix is guaranteed
  to be scaled such that :math:`tr(AA^{T}) = 1`.

- Duplicate data points are generally ok, duplicating a data point will
  increase its effect on the procrustes fit.

- The disparity scales as the number of points per input matrix.

References
----------
.. [1] Krzanowski, W. J. (2000). 'Principles of Multivariate analysis'.
.. [2] Gower, J. C. (1975). 'Generalized procrustes analysis'.

Examples
--------
>>> from scipy.spatial import procrustes

The matrix ``b`` is a rotated, shifted, scaled and mirrored version of
``a`` here:

>>> a = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], 'd')
>>> b = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], 'd')
>>> mtx1, mtx2, disparity = procrustes(a, b)
>>> round(disparity)
0.0
*)

val tsearch : tri:Py.Object.t -> xi:Py.Object.t -> unit -> Py.Object.t
(**
tsearch(tri, xi)

Find simplices containing the given points. This function does the
same thing as `Delaunay.find_simplex`.

.. versionadded:: 0.9

See Also
--------
Delaunay.find_simplex


Examples
--------

>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from scipy.spatial import Delaunay, delaunay_plot_2d, tsearch

The Delaunay triangulation of a set of random points:

>>> pts = np.random.rand(20, 2)
>>> tri = Delaunay(pts)
>>> _ = delaunay_plot_2d(tri)

Find the simplices containing a given set of points:

>>> loc = np.random.uniform(0.2, 0.8, (5, 2))
>>> s = tsearch(tri, loc)
>>> plt.triplot(pts[:, 0], pts[:, 1], tri.simplices[s], 'b-', mask=s==-1)
>>> plt.scatter(loc[:, 0], loc[:, 1], c='r', marker='x')
>>> plt.show()
*)

val voronoi_plot_2d : ?ax:Py.Object.t -> ?kw:(string * Py.Object.t) list -> vor:Py.Object.t -> unit -> Py.Object.t
(**
Plot the given Voronoi diagram in 2-D

Parameters
----------
vor : scipy.spatial.Voronoi instance
    Diagram to plot
ax : matplotlib.axes.Axes instance, optional
    Axes to plot on
show_points: bool, optional
    Add the Voronoi points to the plot.
show_vertices : bool, optional
    Add the Voronoi vertices to the plot.
line_colors : string, optional
    Specifies the line color for polygon boundaries
line_width : float, optional
    Specifies the line width for polygon boundaries
line_alpha: float, optional
    Specifies the line alpha for polygon boundaries
point_size: float, optional
    Specifies the size of points


Returns
-------
fig : matplotlib.figure.Figure instance
    Figure for the plot

See Also
--------
Voronoi

Notes
-----
Requires Matplotlib.

Examples
--------
Set of point:

>>> import matplotlib.pyplot as plt
>>> points = np.random.rand(10,2) #random

Voronoi diagram of the points:

>>> from scipy.spatial import Voronoi, voronoi_plot_2d
>>> vor = Voronoi(points)

using `voronoi_plot_2d` for visualisation:

>>> fig = voronoi_plot_2d(vor)

using `voronoi_plot_2d` for visualisation with enhancements:

>>> fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',
...                 line_width=2, line_alpha=0.6, point_size=2)
>>> plt.show()
*)

