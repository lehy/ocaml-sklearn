let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.spatial"

let get_py name = Py.Module.get __wrap_namespace name
module ConvexHull = struct
type tag = [`ConvexHull]
type t = [`ConvexHull | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?incremental ?qhull_options ~points () =
   Py.Module.get_function_with_keywords __wrap_namespace "ConvexHull"
     [||]
     (Wrap_utils.keyword_args [("incremental", Wrap_utils.Option.map incremental Py.Bool.of_bool); ("qhull_options", Wrap_utils.Option.map qhull_options Py.String.of_string); ("points", Some(points |> Np.Obj.to_pyobject))])
     |> of_pyobject
let add_points ?restart ~points self =
   Py.Module.get_function_with_keywords (to_pyobject self) "add_points"
     [||]
     (Wrap_utils.keyword_args [("restart", Wrap_utils.Option.map restart Py.Bool.of_bool); ("points", Some(points |> Np.Obj.to_pyobject))])

let close self =
   Py.Module.get_function_with_keywords (to_pyobject self) "close"
     [||]
     []


let points_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "points" with
  | None -> failwith "attribute points not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let points self = match points_opt self with
  | None -> raise Not_found
  | Some x -> x

let vertices_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "vertices" with
  | None -> failwith "attribute vertices not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let vertices self = match vertices_opt self with
  | None -> raise Not_found
  | Some x -> x

let simplices_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "simplices" with
  | None -> failwith "attribute simplices not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let simplices self = match simplices_opt self with
  | None -> raise Not_found
  | Some x -> x

let neighbors_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "neighbors" with
  | None -> failwith "attribute neighbors not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let neighbors self = match neighbors_opt self with
  | None -> raise Not_found
  | Some x -> x

let equations_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "equations" with
  | None -> failwith "attribute equations not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let equations self = match equations_opt self with
  | None -> raise Not_found
  | Some x -> x

let coplanar_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "coplanar" with
  | None -> failwith "attribute coplanar not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let coplanar self = match coplanar_opt self with
  | None -> raise Not_found
  | Some x -> x

let good_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "good" with
  | None -> failwith "attribute good not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let good self = match good_opt self with
  | None -> raise Not_found
  | Some x -> x

let area_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "area" with
  | None -> failwith "attribute area not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let area self = match area_opt self with
  | None -> raise Not_found
  | Some x -> x

let volume_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "volume" with
  | None -> failwith "attribute volume not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let volume self = match volume_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Delaunay = struct
type tag = [`Delaunay]
type t = [`Delaunay | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?furthest_site ?incremental ?qhull_options ~points () =
   Py.Module.get_function_with_keywords __wrap_namespace "Delaunay"
     [||]
     (Wrap_utils.keyword_args [("furthest_site", Wrap_utils.Option.map furthest_site Py.Bool.of_bool); ("incremental", Wrap_utils.Option.map incremental Py.Bool.of_bool); ("qhull_options", Wrap_utils.Option.map qhull_options Py.String.of_string); ("points", Some(points |> Np.Obj.to_pyobject))])
     |> of_pyobject
let add_points ?restart ~points self =
   Py.Module.get_function_with_keywords (to_pyobject self) "add_points"
     [||]
     (Wrap_utils.keyword_args [("restart", Wrap_utils.Option.map restart Py.Bool.of_bool); ("points", Some(points |> Np.Obj.to_pyobject))])

let close self =
   Py.Module.get_function_with_keywords (to_pyobject self) "close"
     [||]
     []

let find_simplex ?bruteforce ?tol ~xi self =
   Py.Module.get_function_with_keywords (to_pyobject self) "find_simplex"
     [||]
     (Wrap_utils.keyword_args [("bruteforce", Wrap_utils.Option.map bruteforce Py.Bool.of_bool); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("xi", Some(xi ))])

let lift_points ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "lift_points"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let plane_distance ~xi self =
   Py.Module.get_function_with_keywords (to_pyobject self) "plane_distance"
     [||]
     (Wrap_utils.keyword_args [("xi", Some(xi ))])


let points_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "points" with
  | None -> failwith "attribute points not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let points self = match points_opt self with
  | None -> raise Not_found
  | Some x -> x

let simplices_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "simplices" with
  | None -> failwith "attribute simplices not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let simplices self = match simplices_opt self with
  | None -> raise Not_found
  | Some x -> x

let neighbors_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "neighbors" with
  | None -> failwith "attribute neighbors not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let neighbors self = match neighbors_opt self with
  | None -> raise Not_found
  | Some x -> x

let equations_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "equations" with
  | None -> failwith "attribute equations not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let equations self = match equations_opt self with
  | None -> raise Not_found
  | Some x -> x

let transform_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "transform" with
  | None -> failwith "attribute transform not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let transform self = match transform_opt self with
  | None -> raise Not_found
  | Some x -> x

let vertex_to_simplex_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "vertex_to_simplex" with
  | None -> failwith "attribute vertex_to_simplex not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let vertex_to_simplex self = match vertex_to_simplex_opt self with
  | None -> raise Not_found
  | Some x -> x

let convex_hull_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "convex_hull" with
  | None -> failwith "attribute convex_hull not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let convex_hull self = match convex_hull_opt self with
  | None -> raise Not_found
  | Some x -> x

let coplanar_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "coplanar" with
  | None -> failwith "attribute coplanar not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let coplanar self = match coplanar_opt self with
  | None -> raise Not_found
  | Some x -> x

let vertices_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "vertices" with
  | None -> failwith "attribute vertices not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let vertices self = match vertices_opt self with
  | None -> raise Not_found
  | Some x -> x

let vertex_neighbor_vertices_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "vertex_neighbor_vertices" with
  | None -> failwith "attribute vertex_neighbor_vertices not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let vertex_neighbor_vertices self = match vertex_neighbor_vertices_opt self with
  | None -> raise Not_found
  | Some x -> x

let furthest_site_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "furthest_site" with
  | None -> failwith "attribute furthest_site not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let furthest_site self = match furthest_site_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module HalfspaceIntersection = struct
type tag = [`HalfspaceIntersection]
type t = [`HalfspaceIntersection | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?incremental ?qhull_options ~halfspaces ~interior_point () =
   Py.Module.get_function_with_keywords __wrap_namespace "HalfspaceIntersection"
     [||]
     (Wrap_utils.keyword_args [("incremental", Wrap_utils.Option.map incremental Py.Bool.of_bool); ("qhull_options", Wrap_utils.Option.map qhull_options Py.String.of_string); ("halfspaces", Some(halfspaces |> Np.Obj.to_pyobject)); ("interior_point", Some(interior_point |> Np.Obj.to_pyobject))])
     |> of_pyobject
let add_halfspaces ?restart ~halfspaces self =
   Py.Module.get_function_with_keywords (to_pyobject self) "add_halfspaces"
     [||]
     (Wrap_utils.keyword_args [("restart", Wrap_utils.Option.map restart Py.Bool.of_bool); ("halfspaces", Some(halfspaces |> Np.Obj.to_pyobject))])

let close self =
   Py.Module.get_function_with_keywords (to_pyobject self) "close"
     [||]
     []


let halfspaces_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "halfspaces" with
  | None -> failwith "attribute halfspaces not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let halfspaces self = match halfspaces_opt self with
  | None -> raise Not_found
  | Some x -> x

let interior_point_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "interior_point" with
  | None -> failwith "attribute interior_point not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let interior_point self = match interior_point_opt self with
  | None -> raise Not_found
  | Some x -> x

let intersections_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "intersections" with
  | None -> failwith "attribute intersections not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let intersections self = match intersections_opt self with
  | None -> raise Not_found
  | Some x -> x

let dual_points_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dual_points" with
  | None -> failwith "attribute dual_points not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let dual_points self = match dual_points_opt self with
  | None -> raise Not_found
  | Some x -> x

let dual_facets_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dual_facets" with
  | None -> failwith "attribute dual_facets not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let dual_facets self = match dual_facets_opt self with
  | None -> raise Not_found
  | Some x -> x

let dual_vertices_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dual_vertices" with
  | None -> failwith "attribute dual_vertices not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let dual_vertices self = match dual_vertices_opt self with
  | None -> raise Not_found
  | Some x -> x

let dual_equations_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dual_equations" with
  | None -> failwith "attribute dual_equations not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let dual_equations self = match dual_equations_opt self with
  | None -> raise Not_found
  | Some x -> x

let dual_area_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dual_area" with
  | None -> failwith "attribute dual_area not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let dual_area self = match dual_area_opt self with
  | None -> raise Not_found
  | Some x -> x

let dual_volume_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dual_volume" with
  | None -> failwith "attribute dual_volume not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let dual_volume self = match dual_volume_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module KDTree = struct
type tag = [`KDTree]
type t = [`KDTree | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?leafsize ~data () =
   Py.Module.get_function_with_keywords __wrap_namespace "KDTree"
     [||]
     (Wrap_utils.keyword_args [("leafsize", Wrap_utils.Option.map leafsize Py.Int.of_int); ("data", Some(data ))])
     |> of_pyobject
                  let count_neighbors ?p ~other ~r self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "count_neighbors"
                       [||]
                       (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p (function
| `T1_p_infinity x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)); ("other", Some(other )); ("r", Some(r |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])

                  let query ?k ?eps ?p ?distance_upper_bound ~x self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "query"
                       [||]
                       (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("eps", eps); ("p", Wrap_utils.Option.map p (function
| `T1_p_infinity x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)); ("distance_upper_bound", distance_upper_bound); ("x", Some(x |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Last_dimension_self_m x -> Wrap_utils.id x
)))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let query_ball_point ?p ?eps ~x ~r self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "query_ball_point"
                       [||]
                       (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p Py.Float.of_float); ("eps", eps); ("x", Some(x |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Shape_tuple_self_m_ x -> Wrap_utils.id x
))); ("r", Some(r |> Py.Float.of_float))])

let query_ball_tree ?p ?eps ~other ~r self =
   Py.Module.get_function_with_keywords (to_pyobject self) "query_ball_tree"
     [||]
     (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p Py.Float.of_float); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("other", Some(other )); ("r", Some(r |> Py.Float.of_float))])

let query_pairs ?p ?eps ~r self =
   Py.Module.get_function_with_keywords (to_pyobject self) "query_pairs"
     [||]
     (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p Py.Float.of_float); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("r", Some(r |> Py.Float.of_float))])

let sparse_distance_matrix ?p ~other ~max_distance self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sparse_distance_matrix"
     [||]
     (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p Py.Float.of_float); ("other", Some(other )); ("max_distance", Some(max_distance |> Py.Float.of_float))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Rectangle = struct
type tag = [`Rectangle]
type t = [`Object | `Rectangle] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ~maxes ~mins () =
   Py.Module.get_function_with_keywords __wrap_namespace "Rectangle"
     [||]
     (Wrap_utils.keyword_args [("maxes", Some(maxes )); ("mins", Some(mins ))])
     |> of_pyobject
let max_distance_point ?p ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "max_distance_point"
     [||]
     (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p Py.Float.of_float); ("x", Some(x |> Np.Obj.to_pyobject))])

let max_distance_rectangle ?p ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "max_distance_rectangle"
     [||]
     (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p Py.Float.of_float); ("other", Some(other ))])

let min_distance_point ?p ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "min_distance_point"
     [||]
     (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p Py.Float.of_float); ("x", Some(x |> Np.Obj.to_pyobject))])

let min_distance_rectangle ?p ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "min_distance_rectangle"
     [||]
     (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p Py.Float.of_float); ("other", Some(other ))])

let split ~d ~split self =
   Py.Module.get_function_with_keywords (to_pyobject self) "split"
     [||]
     (Wrap_utils.keyword_args [("d", Some(d |> Py.Int.of_int)); ("split", Some(split |> Py.Float.of_float))])

let volume self =
   Py.Module.get_function_with_keywords (to_pyobject self) "volume"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SphericalVoronoi = struct
type tag = [`SphericalVoronoi]
type t = [`Object | `SphericalVoronoi] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?radius ?center ?threshold ~points () =
   Py.Module.get_function_with_keywords __wrap_namespace "SphericalVoronoi"
     [||]
     (Wrap_utils.keyword_args [("radius", Wrap_utils.Option.map radius Py.Float.of_float); ("center", Wrap_utils.Option.map center Np.Obj.to_pyobject); ("threshold", Wrap_utils.Option.map threshold Py.Float.of_float); ("points", Some(points |> Np.Obj.to_pyobject))])
     |> of_pyobject
let sort_vertices_of_regions self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sort_vertices_of_regions"
     [||]
     []


let points_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "points" with
  | None -> failwith "attribute points not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let points self = match points_opt self with
  | None -> raise Not_found
  | Some x -> x

let radius_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "radius" with
  | None -> failwith "attribute radius not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let radius self = match radius_opt self with
  | None -> raise Not_found
  | Some x -> x

let center_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "center" with
  | None -> failwith "attribute center not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let center self = match center_opt self with
  | None -> raise Not_found
  | Some x -> x

let vertices_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "vertices" with
  | None -> failwith "attribute vertices not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let vertices self = match vertices_opt self with
  | None -> raise Not_found
  | Some x -> x

let regions_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "regions" with
  | None -> failwith "attribute regions not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let regions self = match regions_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Voronoi = struct
type tag = [`Voronoi]
type t = [`Object | `Voronoi] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?furthest_site ?incremental ?qhull_options ~points () =
   Py.Module.get_function_with_keywords __wrap_namespace "Voronoi"
     [||]
     (Wrap_utils.keyword_args [("furthest_site", Wrap_utils.Option.map furthest_site Py.Bool.of_bool); ("incremental", Wrap_utils.Option.map incremental Py.Bool.of_bool); ("qhull_options", Wrap_utils.Option.map qhull_options Py.String.of_string); ("points", Some(points |> Np.Obj.to_pyobject))])
     |> of_pyobject
let add_points ?restart ~points self =
   Py.Module.get_function_with_keywords (to_pyobject self) "add_points"
     [||]
     (Wrap_utils.keyword_args [("restart", Wrap_utils.Option.map restart Py.Bool.of_bool); ("points", Some(points |> Np.Obj.to_pyobject))])

let close self =
   Py.Module.get_function_with_keywords (to_pyobject self) "close"
     [||]
     []


let points_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "points" with
  | None -> failwith "attribute points not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let points self = match points_opt self with
  | None -> raise Not_found
  | Some x -> x

let vertices_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "vertices" with
  | None -> failwith "attribute vertices not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let vertices self = match vertices_opt self with
  | None -> raise Not_found
  | Some x -> x

let ridge_points_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "ridge_points" with
  | None -> failwith "attribute ridge_points not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let ridge_points self = match ridge_points_opt self with
  | None -> raise Not_found
  | Some x -> x

let ridge_vertices_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "ridge_vertices" with
  | None -> failwith "attribute ridge_vertices not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let ridge_vertices self = match ridge_vertices_opt self with
  | None -> raise Not_found
  | Some x -> x

let regions_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "regions" with
  | None -> failwith "attribute regions not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let regions self = match regions_opt self with
  | None -> raise Not_found
  | Some x -> x

let point_region_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "point_region" with
  | None -> failwith "attribute point_region not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let point_region self = match point_region_opt self with
  | None -> raise Not_found
  | Some x -> x

let furthest_site_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "furthest_site" with
  | None -> failwith "attribute furthest_site not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let furthest_site self = match furthest_site_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module CKDTree = struct
type tag = [`CKDTree]
type t = [`CKDTree | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?leafsize ?compact_nodes ?copy_data ?balanced_tree ?boxsize ~data () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cKDTree"
                       [||]
                       (Wrap_utils.keyword_args [("leafsize", leafsize); ("compact_nodes", Wrap_utils.Option.map compact_nodes Py.Bool.of_bool); ("copy_data", Wrap_utils.Option.map copy_data Py.Bool.of_bool); ("balanced_tree", Wrap_utils.Option.map balanced_tree Py.Bool.of_bool); ("boxsize", Wrap_utils.Option.map boxsize (function
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
)); ("data", Some(data |> Np.Obj.to_pyobject))])
                       |> of_pyobject
                  let count_neighbors ?p ?weights ?cumulative ~other ~r self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "count_neighbors"
                       [||]
                       (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p Py.Float.of_float); ("weights", Wrap_utils.Option.map weights (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple x -> Wrap_utils.id x
)); ("cumulative", Wrap_utils.Option.map cumulative Py.Bool.of_bool); ("other", Some(other )); ("r", Some(r |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])

                  let query_ball_point ?p ?eps ~x ~r self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "query_ball_point"
                       [||]
                       (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p Py.Float.of_float); ("eps", eps); ("x", Some(x |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Shape_tuple_self_m_ x -> Wrap_utils.id x
))); ("r", Some(r |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])

let query_ball_tree ?p ?eps ~other ~r self =
   Py.Module.get_function_with_keywords (to_pyobject self) "query_ball_tree"
     [||]
     (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p Py.Float.of_float); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("other", Some(other )); ("r", Some(r |> Py.Float.of_float))])

let query_pairs ?p ?eps ~r self =
   Py.Module.get_function_with_keywords (to_pyobject self) "query_pairs"
     [||]
     (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p Py.Float.of_float); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("r", Some(r |> Py.Float.of_float))])

                  let sparse_distance_matrix ?p ~other ~max_distance self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "sparse_distance_matrix"
                       [||]
                       (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p (function
| `T1_p_infinity x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)); ("other", Some(other )); ("max_distance", Some(max_distance |> Py.Float.of_float))])


let data_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "data" with
  | None -> failwith "attribute data not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let data self = match data_opt self with
  | None -> raise Not_found
  | Some x -> x

let leafsize_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "leafsize" with
  | None -> failwith "attribute leafsize not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let leafsize self = match leafsize_opt self with
  | None -> raise Not_found
  | Some x -> x

let m_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "m" with
  | None -> failwith "attribute m not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let m self = match m_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n" with
  | None -> failwith "attribute n not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n self = match n_opt self with
  | None -> raise Not_found
  | Some x -> x

let maxes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "maxes" with
  | None -> failwith "attribute maxes not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let maxes self = match maxes_opt self with
  | None -> raise Not_found
  | Some x -> x

let mins_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "mins" with
  | None -> failwith "attribute mins not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let mins self = match mins_opt self with
  | None -> raise Not_found
  | Some x -> x

let tree_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "tree" with
  | None -> failwith "attribute tree not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let tree self = match tree_opt self with
  | None -> raise Not_found
  | Some x -> x

let size_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "size" with
  | None -> failwith "attribute size not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let size self = match size_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Ckdtree = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.spatial.ckdtree"

let get_py name = Py.Module.get __wrap_namespace name
module CKDTreeNode = struct
type tag = [`CKDTreeNode]
type t = [`CKDTreeNode | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x

let level_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "level" with
  | None -> failwith "attribute level not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let level self = match level_opt self with
  | None -> raise Not_found
  | Some x -> x

let split_dim_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "split_dim" with
  | None -> failwith "attribute split_dim not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let split_dim self = match split_dim_opt self with
  | None -> raise Not_found
  | Some x -> x

let split_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "split" with
  | None -> failwith "attribute split not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let split self = match split_opt self with
  | None -> raise Not_found
  | Some x -> x

let children_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "children" with
  | None -> failwith "attribute children not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let children self = match children_opt self with
  | None -> raise Not_found
  | Some x -> x

let data_points_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "data_points" with
  | None -> failwith "attribute data_points not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let data_points self = match data_points_opt self with
  | None -> raise Not_found
  | Some x -> x

let indices_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "indices" with
  | None -> failwith "attribute indices not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let indices self = match indices_opt self with
  | None -> raise Not_found
  | Some x -> x

let lesser_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "lesser" with
  | None -> failwith "attribute lesser not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let lesser self = match lesser_opt self with
  | None -> raise Not_found
  | Some x -> x

let greater_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "greater" with
  | None -> failwith "attribute greater not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let greater self = match greater_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Coo_entries = struct
type tag = [`Coo_entries]
type t = [`Coo_entries | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Ordered_pairs = struct
type tag = [`Ordered_pairs]
type t = [`Object | `Ordered_pairs] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let cpu_count () =
   Py.Module.get_function_with_keywords __wrap_namespace "cpu_count"
     [||]
     []


end
module Distance = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.spatial.distance"

let get_py name = Py.Module.get __wrap_namespace name
module MetricInfo = struct
type tag = [`MetricInfo]
type t = [`MetricInfo | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?types ?validator ~aka () =
   Py.Module.get_function_with_keywords __wrap_namespace "MetricInfo"
     [||]
     (Wrap_utils.keyword_args [("types", types); ("validator", validator); ("aka", Some(aka ))])
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let count ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "count"
     (Array.of_list @@ List.concat [[value ]])
     []

let index ?start ?stop ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "index"
     (Array.of_list @@ List.concat [(match start with None -> [] | Some x -> [x ]);(match stop with None -> [] | Some x -> [x ]);[value ]])
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Partial = struct
type tag = [`Partial]
type t = [`Object | `Partial] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?keywords ~func args =
   Py.Module.get_function_with_keywords __wrap_namespace "partial"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("func", Some(func ))]) (match keywords with None -> [] | Some x -> x))
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let braycurtis ?w ~u ~v () =
   Py.Module.get_function_with_keywords __wrap_namespace "braycurtis"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("u", Some(u |> Np.Obj.to_pyobject)); ("v", Some(v |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let callable obj =
   Py.Module.get_function_with_keywords __wrap_namespace "callable"
     [||]
     (Wrap_utils.keyword_args [("obj", Some(obj ))])

let canberra ?w ~u ~v () =
   Py.Module.get_function_with_keywords __wrap_namespace "canberra"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("u", Some(u |> Np.Obj.to_pyobject)); ("v", Some(v |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
                  let cdist ?metric ?kwargs ~xa ~xb args =
                     Py.Module.get_function_with_keywords __wrap_namespace "cdist"
                       (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
                       (List.rev_append (Wrap_utils.keyword_args [("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("XA", Some(xa |> Np.Obj.to_pyobject)); ("XB", Some(xb |> Np.Obj.to_pyobject))]) (match kwargs with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let chebyshev ?w ~u ~v () =
   Py.Module.get_function_with_keywords __wrap_namespace "chebyshev"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("u", Some(u |> Np.Obj.to_pyobject)); ("v", Some(v |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let cityblock ?w ~u ~v () =
   Py.Module.get_function_with_keywords __wrap_namespace "cityblock"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("u", Some(u |> Np.Obj.to_pyobject)); ("v", Some(v |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let correlation ?w ?centered ~u ~v () =
   Py.Module.get_function_with_keywords __wrap_namespace "correlation"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("centered", centered); ("u", Some(u |> Np.Obj.to_pyobject)); ("v", Some(v |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let cosine ?w ~u ~v () =
   Py.Module.get_function_with_keywords __wrap_namespace "cosine"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("u", Some(u |> Np.Obj.to_pyobject)); ("v", Some(v |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let dice ?w ~u ~v () =
   Py.Module.get_function_with_keywords __wrap_namespace "dice"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("u", Some(u |> Np.Obj.to_pyobject)); ("v", Some(v |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
                  let directed_hausdorff ?seed ~u ~v () =
                     Py.Module.get_function_with_keywords __wrap_namespace "directed_hausdorff"
                       [||]
                       (Wrap_utils.keyword_args [("seed", Wrap_utils.Option.map seed (function
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("u", Some(u |> Np.Obj.to_pyobject)); ("v", Some(v ))])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2))))
let euclidean ?w ~u ~v () =
   Py.Module.get_function_with_keywords __wrap_namespace "euclidean"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("u", Some(u |> Np.Obj.to_pyobject)); ("v", Some(v |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let hamming ?w ~u ~v () =
   Py.Module.get_function_with_keywords __wrap_namespace "hamming"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("u", Some(u |> Np.Obj.to_pyobject)); ("v", Some(v |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let is_valid_dm ?tol ?throw ?name ?warning ~d () =
   Py.Module.get_function_with_keywords __wrap_namespace "is_valid_dm"
     [||]
     (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("throw", Wrap_utils.Option.map throw Py.Bool.of_bool); ("name", Wrap_utils.Option.map name Py.String.of_string); ("warning", Wrap_utils.Option.map warning Py.Bool.of_bool); ("D", Some(d |> Np.Obj.to_pyobject))])
     |> Py.Bool.to_bool
let is_valid_y ?warning ?throw ?name ~y () =
   Py.Module.get_function_with_keywords __wrap_namespace "is_valid_y"
     [||]
     (Wrap_utils.keyword_args [("warning", Wrap_utils.Option.map warning Py.Bool.of_bool); ("throw", Wrap_utils.Option.map throw Py.Bool.of_bool); ("name", Wrap_utils.Option.map name Py.Bool.of_bool); ("y", Some(y |> Np.Obj.to_pyobject))])

                  let jaccard ?w ~u ~v () =
                     Py.Module.get_function_with_keywords __wrap_namespace "jaccard"
                       [||]
                       (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("u", Some(u |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Bool x -> Py.Bool.of_bool x
))); ("v", Some(v |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Bool x -> Py.Bool.of_bool x
)))])
                       |> Py.Float.to_float
let jensenshannon ?base ~p ~q () =
   Py.Module.get_function_with_keywords __wrap_namespace "jensenshannon"
     [||]
     (Wrap_utils.keyword_args [("base", Wrap_utils.Option.map base Py.Float.of_float); ("p", Some(p |> Np.Obj.to_pyobject)); ("q", Some(q |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
                  let kulsinski ?w ~u ~v () =
                     Py.Module.get_function_with_keywords __wrap_namespace "kulsinski"
                       [||]
                       (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("u", Some(u |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Bool x -> Py.Bool.of_bool x
))); ("v", Some(v |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Bool x -> Py.Bool.of_bool x
)))])
                       |> Py.Float.to_float
let mahalanobis ~u ~v ~vi () =
   Py.Module.get_function_with_keywords __wrap_namespace "mahalanobis"
     [||]
     (Wrap_utils.keyword_args [("u", Some(u |> Np.Obj.to_pyobject)); ("v", Some(v |> Np.Obj.to_pyobject)); ("VI", Some(vi |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let matching ?kwds args =
   Py.Module.get_function_with_keywords __wrap_namespace "matching"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwds with None -> [] | Some x -> x)

let minkowski ?p ?w ~u ~v () =
   Py.Module.get_function_with_keywords __wrap_namespace "minkowski"
     [||]
     (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p Py.Int.of_int); ("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("u", Some(u |> Np.Obj.to_pyobject)); ("v", Some(v |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let namedtuple ?rename ?defaults ?module_ ~typename ~field_names () =
   Py.Module.get_function_with_keywords __wrap_namespace "namedtuple"
     [||]
     (Wrap_utils.keyword_args [("rename", rename); ("defaults", defaults); ("module", module_); ("typename", Some(typename )); ("field_names", Some(field_names ))])

                  let norm ?ord ?axis ?keepdims ?check_finite ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "norm"
                       [||]
                       (Wrap_utils.keyword_args [("ord", Wrap_utils.Option.map ord (function
| `Fro -> Py.String.of_string "fro"
| `PyObject x -> Wrap_utils.id x
)); ("axis", Wrap_utils.Option.map axis (function
| `T2_tuple_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a ))])

let num_obs_dm d =
   Py.Module.get_function_with_keywords __wrap_namespace "num_obs_dm"
     [||]
     (Wrap_utils.keyword_args [("d", Some(d |> Np.Obj.to_pyobject))])
     |> Py.Int.to_int
let num_obs_y y =
   Py.Module.get_function_with_keywords __wrap_namespace "num_obs_y"
     [||]
     (Wrap_utils.keyword_args [("Y", Some(y |> Np.Obj.to_pyobject))])
     |> Py.Int.to_int
                  let pdist ?metric ?kwargs ~x args =
                     Py.Module.get_function_with_keywords __wrap_namespace "pdist"
                       (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
                       (List.rev_append (Wrap_utils.keyword_args [("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("X", Some(x |> Np.Obj.to_pyobject))]) (match kwargs with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let rel_entr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "rel_entr"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

                  let rogerstanimoto ?w ~u ~v () =
                     Py.Module.get_function_with_keywords __wrap_namespace "rogerstanimoto"
                       [||]
                       (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("u", Some(u |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Bool x -> Py.Bool.of_bool x
))); ("v", Some(v |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Bool x -> Py.Bool.of_bool x
)))])
                       |> Py.Float.to_float
                  let russellrao ?w ~u ~v () =
                     Py.Module.get_function_with_keywords __wrap_namespace "russellrao"
                       [||]
                       (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("u", Some(u |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Bool x -> Py.Bool.of_bool x
))); ("v", Some(v |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Bool x -> Py.Bool.of_bool x
)))])
                       |> Py.Float.to_float
let seuclidean ~u ~v ~v' () =
   Py.Module.get_function_with_keywords __wrap_namespace "seuclidean"
     [||]
     (Wrap_utils.keyword_args [("u", Some(u |> Np.Obj.to_pyobject)); ("v", Some(v |> Np.Obj.to_pyobject)); ("V", Some(v' |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
                  let sokalmichener ?w ~u ~v () =
                     Py.Module.get_function_with_keywords __wrap_namespace "sokalmichener"
                       [||]
                       (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("u", Some(u |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Bool x -> Py.Bool.of_bool x
))); ("v", Some(v |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Bool x -> Py.Bool.of_bool x
)))])
                       |> Py.Float.to_float
                  let sokalsneath ?w ~u ~v () =
                     Py.Module.get_function_with_keywords __wrap_namespace "sokalsneath"
                       [||]
                       (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("u", Some(u |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Bool x -> Py.Bool.of_bool x
))); ("v", Some(v |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Bool x -> Py.Bool.of_bool x
)))])
                       |> Py.Float.to_float
let sqeuclidean ?w ~u ~v () =
   Py.Module.get_function_with_keywords __wrap_namespace "sqeuclidean"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("u", Some(u |> Np.Obj.to_pyobject)); ("v", Some(v |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let squareform ?force ?checks ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "squareform"
     [||]
     (Wrap_utils.keyword_args [("force", Wrap_utils.Option.map force Py.String.of_string); ("checks", Wrap_utils.Option.map checks Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let wminkowski ~u ~v ~p ~w () =
   Py.Module.get_function_with_keywords __wrap_namespace "wminkowski"
     [||]
     (Wrap_utils.keyword_args [("u", Some(u |> Np.Obj.to_pyobject)); ("v", Some(v |> Np.Obj.to_pyobject)); ("p", Some(p |> Py.Int.of_int)); ("w", Some(w |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
                  let yule ?w ~u ~v () =
                     Py.Module.get_function_with_keywords __wrap_namespace "yule"
                       [||]
                       (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("u", Some(u |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Bool x -> Py.Bool.of_bool x
))); ("v", Some(v |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Bool x -> Py.Bool.of_bool x
)))])
                       |> Py.Float.to_float

end
module Kdtree = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.spatial.kdtree"

let get_py name = Py.Module.get __wrap_namespace name
                  let distance_matrix ?p ?threshold ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "distance_matrix"
                       [||]
                       (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p (function
| `T1_p_infinity x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)); ("threshold", threshold); ("x", Some(x )); ("y", Some(y ))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let heappop heap =
   Py.Module.get_function_with_keywords __wrap_namespace "heappop"
     (Array.of_list @@ List.concat [[heap ]])
     []

let heappush ~heap ~item () =
   Py.Module.get_function_with_keywords __wrap_namespace "heappush"
     (Array.of_list @@ List.concat [[heap ];[item ]])
     []

                  let minkowski_distance ?p ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "minkowski_distance"
                       [||]
                       (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p (function
| `T1_p_infinity x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)); ("x", Some(x )); ("y", Some(y ))])

                  let minkowski_distance_p ?p ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "minkowski_distance_p"
                       [||]
                       (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p (function
| `T1_p_infinity x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)); ("x", Some(x )); ("y", Some(y ))])


end
module Qhull = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.spatial.qhull"

let get_py name = Py.Module.get __wrap_namespace name
module QhullError = struct
type tag = [`QhullError]
type t = [`BaseException | `Object | `QhullError] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_exception x = (x :> [`BaseException] Obj.t)
let with_traceback ~tb self =
   Py.Module.get_function_with_keywords (to_pyobject self) "with_traceback"
     [||]
     (Wrap_utils.keyword_args [("tb", Some(tb ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let asbytes s =
   Py.Module.get_function_with_keywords __wrap_namespace "asbytes"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s ))])

let tsearch ~tri ~xi () =
   Py.Module.get_function_with_keywords __wrap_namespace "tsearch"
     [||]
     (Wrap_utils.keyword_args [("tri", Some(tri )); ("xi", Some(xi ))])


end
module Transform = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.spatial.transform"

let get_py name = Py.Module.get __wrap_namespace name
module Rotation = struct
type tag = [`Rotation]
type t = [`Object | `Rotation] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?normalize ?copy ~quat () =
   Py.Module.get_function_with_keywords __wrap_namespace "Rotation"
     [||]
     (Wrap_utils.keyword_args [("normalize", normalize); ("copy", copy); ("quat", Some(quat ))])
     |> of_pyobject
                  let __getitem__ ~indexer self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
                       [||]
                       (Wrap_utils.keyword_args [("indexer", Some(indexer |> (function
| `Slice x -> Np.Wrap_utils.Slice.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])

let align_vectors ?weights ?return_sensitivity ~a ~b self =
   Py.Module.get_function_with_keywords (to_pyobject self) "align_vectors"
     [||]
     (Wrap_utils.keyword_args [("weights", Wrap_utils.Option.map weights Np.Obj.to_pyobject); ("return_sensitivity", Wrap_utils.Option.map return_sensitivity Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
let apply ?inverse ~vectors self =
   Py.Module.get_function_with_keywords (to_pyobject self) "apply"
     [||]
     (Wrap_utils.keyword_args [("inverse", Wrap_utils.Option.map inverse Py.Bool.of_bool); ("vectors", Some(vectors |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let as_euler ?degrees ~seq self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "as_euler"
                       [||]
                       (Wrap_utils.keyword_args [("degrees", Wrap_utils.Option.map degrees Py.Bool.of_bool); ("seq", Some(seq |> (function
| `S x -> Py.String.of_string x
| `Length_3 x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let as_matrix self =
   Py.Module.get_function_with_keywords (to_pyobject self) "as_matrix"
     [||]
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let as_quat self =
   Py.Module.get_function_with_keywords (to_pyobject self) "as_quat"
     [||]
     []

let as_rotvec self =
   Py.Module.get_function_with_keywords (to_pyobject self) "as_rotvec"
     [||]
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let create_group ?axis ~group self =
   Py.Module.get_function_with_keywords (to_pyobject self) "create_group"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("group", Some(group |> Py.String.of_string))])

let from_dcm ?kwds args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "from_dcm"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwds with None -> [] | Some x -> x)

                  let from_euler ?degrees ~seq ~angles self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "from_euler"
                       [||]
                       (Wrap_utils.keyword_args [("degrees", Wrap_utils.Option.map degrees Py.Bool.of_bool); ("seq", Some(seq |> Py.String.of_string)); ("angles", Some(angles |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])

let from_matrix ~matrix self =
   Py.Module.get_function_with_keywords (to_pyobject self) "from_matrix"
     [||]
     (Wrap_utils.keyword_args [("matrix", Some(matrix |> Np.Obj.to_pyobject))])

let from_quat ?normalized ~quat self =
   Py.Module.get_function_with_keywords (to_pyobject self) "from_quat"
     [||]
     (Wrap_utils.keyword_args [("normalized", normalized); ("quat", Some(quat |> Np.Obj.to_pyobject))])

let from_rotvec ~rotvec self =
   Py.Module.get_function_with_keywords (to_pyobject self) "from_rotvec"
     [||]
     (Wrap_utils.keyword_args [("rotvec", Some(rotvec |> Np.Obj.to_pyobject))])

let identity ?num self =
   Py.Module.get_function_with_keywords (to_pyobject self) "identity"
     [||]
     (Wrap_utils.keyword_args [("num", Wrap_utils.Option.map num Py.Int.of_int)])

let inv self =
   Py.Module.get_function_with_keywords (to_pyobject self) "inv"
     [||]
     []

let magnitude self =
   Py.Module.get_function_with_keywords (to_pyobject self) "magnitude"
     [||]
     []

let match_vectors ?kwds args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "match_vectors"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwds with None -> [] | Some x -> x)

let mean ?weights self =
   Py.Module.get_function_with_keywords (to_pyobject self) "mean"
     [||]
     (Wrap_utils.keyword_args [("weights", Wrap_utils.Option.map weights Np.Obj.to_pyobject)])

                  let random ?num ?random_state self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "random"
                       [||]
                       (Wrap_utils.keyword_args [("num", Wrap_utils.Option.map num Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state (function
| `RandomState x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))])

let reduce ?left ?right ?return_indices self =
   Py.Module.get_function_with_keywords (to_pyobject self) "reduce"
     [||]
     (Wrap_utils.keyword_args [("left", left); ("right", right); ("return_indices", Wrap_utils.Option.map return_indices Py.Bool.of_bool)])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RotationSpline = struct
type tag = [`RotationSpline]
type t = [`Object | `RotationSpline] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ~times ~rotations () =
   Py.Module.get_function_with_keywords __wrap_namespace "RotationSpline"
     [||]
     (Wrap_utils.keyword_args [("times", Some(times |> Np.Obj.to_pyobject)); ("rotations", Some(rotations ))])
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Slerp = struct
type tag = [`Slerp]
type t = [`Object | `Slerp] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ~times ~rotations () =
   Py.Module.get_function_with_keywords __wrap_namespace "Slerp"
     [||]
     (Wrap_utils.keyword_args [("times", Some(times |> Np.Obj.to_pyobject)); ("rotations", Some(rotations ))])
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Rotation' = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.spatial.transform.rotation"

let get_py name = Py.Module.get __wrap_namespace name
let check_random_state seed =
   Py.Module.get_function_with_keywords __wrap_namespace "check_random_state"
     [||]
     (Wrap_utils.keyword_args [("seed", Some(seed ))])

let create_group ?axis ~cls ~group () =
   Py.Module.get_function_with_keywords __wrap_namespace "create_group"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("cls", Some(cls )); ("group", Some(group ))])


end

end
let convex_hull_plot_2d ?ax ~hull () =
   Py.Module.get_function_with_keywords __wrap_namespace "convex_hull_plot_2d"
     [||]
     (Wrap_utils.keyword_args [("ax", ax); ("hull", Some(hull ))])

let delaunay_plot_2d ?ax ~tri () =
   Py.Module.get_function_with_keywords __wrap_namespace "delaunay_plot_2d"
     [||]
     (Wrap_utils.keyword_args [("ax", ax); ("tri", Some(tri ))])

                  let distance_matrix ?p ?threshold ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "distance_matrix"
                       [||]
                       (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p (function
| `T1_p_infinity x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)); ("threshold", threshold); ("x", Some(x )); ("y", Some(y ))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let minkowski_distance ?p ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "minkowski_distance"
                       [||]
                       (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p (function
| `T1_p_infinity x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)); ("x", Some(x )); ("y", Some(y ))])

                  let minkowski_distance_p ?p ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "minkowski_distance_p"
                       [||]
                       (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p (function
| `T1_p_infinity x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)); ("x", Some(x )); ("y", Some(y ))])

let procrustes ~data1 ~data2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "procrustes"
     [||]
     (Wrap_utils.keyword_args [("data1", Some(data1 |> Np.Obj.to_pyobject)); ("data2", Some(data2 |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let tsearch ~tri ~xi () =
   Py.Module.get_function_with_keywords __wrap_namespace "tsearch"
     [||]
     (Wrap_utils.keyword_args [("tri", Some(tri )); ("xi", Some(xi ))])

let voronoi_plot_2d ?ax ?kw ~vor () =
   Py.Module.get_function_with_keywords __wrap_namespace "voronoi_plot_2d"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("ax", ax); ("vor", Some(vor ))]) (match kw with None -> [] | Some x -> x))

