let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.cluster"

module AffinityPropagation = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?damping ?max_iter ?convergence_iter ?copy ?preference ?affinity ?verbose () =
                     Py.Module.get_function_with_keywords ns "AffinityPropagation"
                       [||]
                       (Wrap_utils.keyword_args [("damping", Wrap_utils.Option.map damping Py.Float.of_float); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("convergence_iter", Wrap_utils.Option.map convergence_iter Py.Int.of_int); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("preference", Wrap_utils.Option.map preference (function
| `Ndarray x -> Ndarray.to_pyobject x
| `Float x -> Py.Float.of_float x
)); ("affinity", Wrap_utils.Option.map affinity (function
| `Euclidean -> Py.String.of_string "euclidean"
| `Precomputed -> Py.String.of_string "precomputed"
)); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool)])

                  let fit ?y ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

                  let fit_predict ?y ~x self =
                     Py.Module.get_function_with_keywords self "fit_predict"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

                  let predict ~x self =
                     Py.Module.get_function_with_keywords self "predict"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let cluster_centers_indices_ self =
  match Py.Object.get_attr_string self "cluster_centers_indices_" with
| None -> raise (Wrap_utils.Attribute_not_found "cluster_centers_indices_")
| Some x -> Ndarray.of_pyobject x
let cluster_centers_ self =
  match Py.Object.get_attr_string self "cluster_centers_" with
| None -> raise (Wrap_utils.Attribute_not_found "cluster_centers_")
| Some x -> Ndarray.of_pyobject x
let labels_ self =
  match Py.Object.get_attr_string self "labels_" with
| None -> raise (Wrap_utils.Attribute_not_found "labels_")
| Some x -> Ndarray.of_pyobject x
let affinity_matrix_ self =
  match Py.Object.get_attr_string self "affinity_matrix_" with
| None -> raise (Wrap_utils.Attribute_not_found "affinity_matrix_")
| Some x -> Ndarray.of_pyobject x
let n_iter_ self =
  match Py.Object.get_attr_string self "n_iter_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_iter_")
| Some x -> Py.Int.to_int x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module AgglomerativeClustering = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_clusters ?affinity ?memory ?connectivity ?compute_full_tree ?linkage ?distance_threshold () =
                     Py.Module.get_function_with_keywords ns "AgglomerativeClustering"
                       [||]
                       (Wrap_utils.keyword_args [("n_clusters", Wrap_utils.Option.map n_clusters (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("affinity", Wrap_utils.Option.map affinity (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("memory", Wrap_utils.Option.map memory (function
| `String x -> Py.String.of_string x
| `JoblibMemory x -> Wrap_utils.id x
)); ("connectivity", Wrap_utils.Option.map connectivity (function
| `Ndarray x -> Ndarray.to_pyobject x
| `Callable x -> Wrap_utils.id x
)); ("compute_full_tree", Wrap_utils.Option.map compute_full_tree (function
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("linkage", Wrap_utils.Option.map linkage (function
| `Ward -> Py.String.of_string "ward"
| `Complete -> Py.String.of_string "complete"
| `Average -> Py.String.of_string "average"
| `Single -> Py.String.of_string "single"
)); ("distance_threshold", Wrap_utils.Option.map distance_threshold Py.Float.of_float)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let fit_predict ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_predict"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let n_clusters_ self =
  match Py.Object.get_attr_string self "n_clusters_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_clusters_")
| Some x -> Py.Int.to_int x
let labels_ self =
  match Py.Object.get_attr_string self "labels_" with
| None -> raise (Wrap_utils.Attribute_not_found "labels_")
| Some x -> Ndarray.of_pyobject x
let n_leaves_ self =
  match Py.Object.get_attr_string self "n_leaves_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_leaves_")
| Some x -> Py.Int.to_int x
let n_connected_components_ self =
  match Py.Object.get_attr_string self "n_connected_components_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_connected_components_")
| Some x -> Py.Int.to_int x
let children_ self =
  match Py.Object.get_attr_string self "children_" with
| None -> raise (Wrap_utils.Attribute_not_found "children_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Birch = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?threshold ?branching_factor ?n_clusters ?compute_labels ?copy () =
                     Py.Module.get_function_with_keywords ns "Birch"
                       [||]
                       (Wrap_utils.keyword_args [("threshold", Wrap_utils.Option.map threshold Py.Float.of_float); ("branching_factor", Wrap_utils.Option.map branching_factor Py.Int.of_int); ("n_clusters", Wrap_utils.Option.map n_clusters (function
| `Int x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)); ("compute_labels", Wrap_utils.Option.map compute_labels Py.Bool.of_bool); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

                  let fit ?y ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

let fit_predict ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_predict"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

                  let partial_fit ?x ?y self =
                     Py.Module.get_function_with_keywords self "partial_fit"
                       [||]
                       (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
| `None -> Py.String.of_string "None"
)); ("y", y)])

                  let predict ~x self =
                     Py.Module.get_function_with_keywords self "predict"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

                  let transform ~x self =
                     Py.Module.get_function_with_keywords self "transform"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let root_ self =
  match Py.Object.get_attr_string self "root_" with
| None -> raise (Wrap_utils.Attribute_not_found "root_")
| Some x -> Wrap_utils.id x
let dummy_leaf_ self =
  match Py.Object.get_attr_string self "dummy_leaf_" with
| None -> raise (Wrap_utils.Attribute_not_found "dummy_leaf_")
| Some x -> Wrap_utils.id x
let subcluster_centers_ self =
  match Py.Object.get_attr_string self "subcluster_centers_" with
| None -> raise (Wrap_utils.Attribute_not_found "subcluster_centers_")
| Some x -> Ndarray.of_pyobject x
let subcluster_labels_ self =
  match Py.Object.get_attr_string self "subcluster_labels_" with
| None -> raise (Wrap_utils.Attribute_not_found "subcluster_labels_")
| Some x -> Ndarray.of_pyobject x
let labels_ self =
  match Py.Object.get_attr_string self "labels_" with
| None -> raise (Wrap_utils.Attribute_not_found "labels_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module DBSCAN = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?eps ?min_samples ?metric ?metric_params ?algorithm ?leaf_size ?p ?n_jobs () =
                     Py.Module.get_function_with_keywords ns "DBSCAN"
                       [||]
                       (Wrap_utils.keyword_args [("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("min_samples", Wrap_utils.Option.map min_samples Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("metric_params", metric_params); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Auto -> Py.String.of_string "auto"
| `Ball_tree -> Py.String.of_string "ball_tree"
| `Kd_tree -> Py.String.of_string "kd_tree"
| `Brute -> Py.String.of_string "brute"
)); ("leaf_size", Wrap_utils.Option.map leaf_size Py.Int.of_int); ("p", Wrap_utils.Option.map p Py.Float.of_float); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
))])

                  let fit ?y ?sample_weight ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

                  let fit_predict ?y ?sample_weight ~x self =
                     Py.Module.get_function_with_keywords self "fit_predict"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let core_sample_indices_ self =
  match Py.Object.get_attr_string self "core_sample_indices_" with
| None -> raise (Wrap_utils.Attribute_not_found "core_sample_indices_")
| Some x -> Ndarray.of_pyobject x
let components_ self =
  match Py.Object.get_attr_string self "components_" with
| None -> raise (Wrap_utils.Attribute_not_found "components_")
| Some x -> Ndarray.of_pyobject x
let labels_ self =
  match Py.Object.get_attr_string self "labels_" with
| None -> raise (Wrap_utils.Attribute_not_found "labels_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module FeatureAgglomeration = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_clusters ?affinity ?memory ?connectivity ?compute_full_tree ?linkage ?pooling_func ?distance_threshold () =
                     Py.Module.get_function_with_keywords ns "FeatureAgglomeration"
                       [||]
                       (Wrap_utils.keyword_args [("n_clusters", Wrap_utils.Option.map n_clusters Py.Int.of_int); ("affinity", Wrap_utils.Option.map affinity (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("memory", Wrap_utils.Option.map memory (function
| `String x -> Py.String.of_string x
| `JoblibMemory x -> Wrap_utils.id x
)); ("connectivity", Wrap_utils.Option.map connectivity (function
| `Ndarray x -> Ndarray.to_pyobject x
| `Callable x -> Wrap_utils.id x
)); ("compute_full_tree", Wrap_utils.Option.map compute_full_tree (function
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("linkage", Wrap_utils.Option.map linkage (function
| `Ward -> Py.String.of_string "ward"
| `Complete -> Py.String.of_string "complete"
| `Average -> Py.String.of_string "average"
| `Single -> Py.String.of_string "single"
)); ("pooling_func", pooling_func); ("distance_threshold", Wrap_utils.Option.map distance_threshold Py.Float.of_float)])

let fit ?y ?params ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))]) (match params with None -> [] | Some x -> x))

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let inverse_transform ~xred self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("Xred", Some(xred |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let n_clusters_ self =
  match Py.Object.get_attr_string self "n_clusters_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_clusters_")
| Some x -> Py.Int.to_int x
let labels_ self =
  match Py.Object.get_attr_string self "labels_" with
| None -> raise (Wrap_utils.Attribute_not_found "labels_")
| Some x -> Wrap_utils.id x
let n_leaves_ self =
  match Py.Object.get_attr_string self "n_leaves_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_leaves_")
| Some x -> Py.Int.to_int x
let n_connected_components_ self =
  match Py.Object.get_attr_string self "n_connected_components_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_connected_components_")
| Some x -> Py.Int.to_int x
let children_ self =
  match Py.Object.get_attr_string self "children_" with
| None -> raise (Wrap_utils.Attribute_not_found "children_")
| Some x -> Ndarray.of_pyobject x
let distances_ self =
  match Py.Object.get_attr_string self "distances_" with
| None -> raise (Wrap_utils.Attribute_not_found "distances_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module KMeans = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_clusters ?init ?n_init ?max_iter ?tol ?precompute_distances ?verbose ?random_state ?copy_x ?n_jobs ?algorithm () =
                     Py.Module.get_function_with_keywords ns "KMeans"
                       [||]
                       (Wrap_utils.keyword_args [("n_clusters", Wrap_utils.Option.map n_clusters Py.Int.of_int); ("init", Wrap_utils.Option.map init (function
| `K_means_ -> Py.String.of_string "k-means++"
| `Random -> Py.String.of_string "random"
| `Ndarray x -> Ndarray.to_pyobject x
)); ("n_init", Wrap_utils.Option.map n_init Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("precompute_distances", Wrap_utils.Option.map precompute_distances (function
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
)); ("copy_x", Wrap_utils.Option.map copy_x Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Auto -> Py.String.of_string "auto"
| `Full -> Py.String.of_string "full"
| `Elkan -> Py.String.of_string "elkan"
))])

                  let fit ?y ?sample_weight ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

                  let fit_predict ?y ?sample_weight ~x self =
                     Py.Module.get_function_with_keywords self "fit_predict"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
                  let fit_transform ?y ?sample_weight ~x self =
                     Py.Module.get_function_with_keywords self "fit_transform"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

                  let predict ?sample_weight ~x self =
                     Py.Module.get_function_with_keywords self "predict"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
                  let score ?y ?sample_weight ~x self =
                     Py.Module.get_function_with_keywords self "score"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

                  let transform ~x self =
                     Py.Module.get_function_with_keywords self "transform"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let cluster_centers_ self =
  match Py.Object.get_attr_string self "cluster_centers_" with
| None -> raise (Wrap_utils.Attribute_not_found "cluster_centers_")
| Some x -> Ndarray.of_pyobject x
let labels_ self =
  match Py.Object.get_attr_string self "labels_" with
| None -> raise (Wrap_utils.Attribute_not_found "labels_")
| Some x -> Ndarray.of_pyobject x
let inertia_ self =
  match Py.Object.get_attr_string self "inertia_" with
| None -> raise (Wrap_utils.Attribute_not_found "inertia_")
| Some x -> Py.Float.to_float x
let n_iter_ self =
  match Py.Object.get_attr_string self "n_iter_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_iter_")
| Some x -> Py.Int.to_int x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MeanShift = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?bandwidth ?seeds ?bin_seeding ?min_bin_freq ?cluster_all ?n_jobs ?max_iter () =
                     Py.Module.get_function_with_keywords ns "MeanShift"
                       [||]
                       (Wrap_utils.keyword_args [("bandwidth", Wrap_utils.Option.map bandwidth Py.Float.of_float); ("seeds", Wrap_utils.Option.map seeds Ndarray.to_pyobject); ("bin_seeding", Wrap_utils.Option.map bin_seeding Py.Bool.of_bool); ("min_bin_freq", Wrap_utils.Option.map min_bin_freq Py.Int.of_int); ("cluster_all", Wrap_utils.Option.map cluster_all Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let fit_predict ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_predict"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

                  let predict ~x self =
                     Py.Module.get_function_with_keywords self "predict"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let cluster_centers_ self =
  match Py.Object.get_attr_string self "cluster_centers_" with
| None -> raise (Wrap_utils.Attribute_not_found "cluster_centers_")
| Some x -> Ndarray.of_pyobject x
let labels_ self =
  match Py.Object.get_attr_string self "labels_" with
| None -> raise (Wrap_utils.Attribute_not_found "labels_")
| Some x -> Wrap_utils.id x
let n_iter_ self =
  match Py.Object.get_attr_string self "n_iter_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_iter_")
| Some x -> Py.Int.to_int x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MiniBatchKMeans = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_clusters ?init ?max_iter ?batch_size ?verbose ?compute_labels ?random_state ?tol ?max_no_improvement ?init_size ?n_init ?reassignment_ratio () =
                     Py.Module.get_function_with_keywords ns "MiniBatchKMeans"
                       [||]
                       (Wrap_utils.keyword_args [("n_clusters", Wrap_utils.Option.map n_clusters Py.Int.of_int); ("init", Wrap_utils.Option.map init (function
| `K_means_ -> Py.String.of_string "k-means++"
| `Random -> Py.String.of_string "random"
| `Ndarray x -> Ndarray.to_pyobject x
)); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("batch_size", Wrap_utils.Option.map batch_size Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("compute_labels", Wrap_utils.Option.map compute_labels Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
)); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("max_no_improvement", Wrap_utils.Option.map max_no_improvement Py.Int.of_int); ("init_size", Wrap_utils.Option.map init_size Py.Int.of_int); ("n_init", Wrap_utils.Option.map n_init Py.Int.of_int); ("reassignment_ratio", Wrap_utils.Option.map reassignment_ratio Py.Float.of_float)])

                  let fit ?y ?sample_weight ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

                  let fit_predict ?y ?sample_weight ~x self =
                     Py.Module.get_function_with_keywords self "fit_predict"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
                  let fit_transform ?y ?sample_weight ~x self =
                     Py.Module.get_function_with_keywords self "fit_transform"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let partial_fit ?y ?sample_weight ~x self =
   Py.Module.get_function_with_keywords self "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))])

                  let predict ?sample_weight ~x self =
                     Py.Module.get_function_with_keywords self "predict"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
                  let score ?y ?sample_weight ~x self =
                     Py.Module.get_function_with_keywords self "score"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

                  let transform ~x self =
                     Py.Module.get_function_with_keywords self "transform"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let cluster_centers_ self =
  match Py.Object.get_attr_string self "cluster_centers_" with
| None -> raise (Wrap_utils.Attribute_not_found "cluster_centers_")
| Some x -> Ndarray.of_pyobject x
let labels_ self =
  match Py.Object.get_attr_string self "labels_" with
| None -> raise (Wrap_utils.Attribute_not_found "labels_")
| Some x -> Py.Int.to_int x
let inertia_ self =
  match Py.Object.get_attr_string self "inertia_" with
| None -> raise (Wrap_utils.Attribute_not_found "inertia_")
| Some x -> Py.Float.to_float x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module OPTICS = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?min_samples ?max_eps ?metric ?p ?metric_params ?cluster_method ?eps ?xi ?predecessor_correction ?min_cluster_size ?algorithm ?leaf_size ?n_jobs () =
                     Py.Module.get_function_with_keywords ns "OPTICS"
                       [||]
                       (Wrap_utils.keyword_args [("min_samples", Wrap_utils.Option.map min_samples (function
| `Int x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)); ("max_eps", Wrap_utils.Option.map max_eps Py.Float.of_float); ("metric", Wrap_utils.Option.map metric (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("p", Wrap_utils.Option.map p Py.Int.of_int); ("metric_params", metric_params); ("cluster_method", Wrap_utils.Option.map cluster_method Py.String.of_string); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("xi", Wrap_utils.Option.map xi (function
| `Float x -> Py.Float.of_float x
| `PyObject x -> Wrap_utils.id x
)); ("predecessor_correction", Wrap_utils.Option.map predecessor_correction Py.Bool.of_bool); ("min_cluster_size", Wrap_utils.Option.map min_cluster_size (function
| `Int x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Auto -> Py.String.of_string "auto"
| `Ball_tree -> Py.String.of_string "ball_tree"
| `Kd_tree -> Py.String.of_string "kd_tree"
| `Brute -> Py.String.of_string "brute"
)); ("leaf_size", Wrap_utils.Option.map leaf_size Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
))])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let fit_predict ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_predict"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let labels_ self =
  match Py.Object.get_attr_string self "labels_" with
| None -> raise (Wrap_utils.Attribute_not_found "labels_")
| Some x -> Ndarray.of_pyobject x
let reachability_ self =
  match Py.Object.get_attr_string self "reachability_" with
| None -> raise (Wrap_utils.Attribute_not_found "reachability_")
| Some x -> Ndarray.of_pyobject x
let ordering_ self =
  match Py.Object.get_attr_string self "ordering_" with
| None -> raise (Wrap_utils.Attribute_not_found "ordering_")
| Some x -> Ndarray.of_pyobject x
let core_distances_ self =
  match Py.Object.get_attr_string self "core_distances_" with
| None -> raise (Wrap_utils.Attribute_not_found "core_distances_")
| Some x -> Ndarray.of_pyobject x
let predecessor_ self =
  match Py.Object.get_attr_string self "predecessor_" with
| None -> raise (Wrap_utils.Attribute_not_found "predecessor_")
| Some x -> Ndarray.of_pyobject x
let cluster_hierarchy_ self =
  match Py.Object.get_attr_string self "cluster_hierarchy_" with
| None -> raise (Wrap_utils.Attribute_not_found "cluster_hierarchy_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SpectralBiclustering = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_clusters ?method_ ?n_components ?n_best ?svd_method ?n_svd_vecs ?mini_batch ?init ?n_init ?n_jobs ?random_state () =
                     Py.Module.get_function_with_keywords ns "SpectralBiclustering"
                       [||]
                       (Wrap_utils.keyword_args [("n_clusters", Wrap_utils.Option.map n_clusters (function
| `Int x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)); ("method", Wrap_utils.Option.map method_ (function
| `Bistochastic -> Py.String.of_string "bistochastic"
| `Scale -> Py.String.of_string "scale"
| `Log -> Py.String.of_string "log"
)); ("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("n_best", Wrap_utils.Option.map n_best Py.Int.of_int); ("svd_method", Wrap_utils.Option.map svd_method (function
| `Randomized -> Py.String.of_string "randomized"
| `Arpack -> Py.String.of_string "arpack"
)); ("n_svd_vecs", Wrap_utils.Option.map n_svd_vecs Py.Int.of_int); ("mini_batch", Wrap_utils.Option.map mini_batch Py.Bool.of_bool); ("init", Wrap_utils.Option.map init (function
| `K_means_ -> Py.String.of_string "k-means++"
| `Random -> Py.String.of_string "random"
| `PyObject x -> Wrap_utils.id x
)); ("n_init", Wrap_utils.Option.map n_init Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
))])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let get_indices ~i self =
   Py.Module.get_function_with_keywords self "get_indices"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i |> Py.Int.of_int))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let get_shape ~i self =
   Py.Module.get_function_with_keywords self "get_shape"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i |> Py.Int.of_int))])
     |> Py.Int.to_int
let get_submatrix ~i ~data self =
   Py.Module.get_function_with_keywords self "get_submatrix"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i |> Py.Int.of_int)); ("data", Some(data ))])
     |> Ndarray.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let rows_ self =
  match Py.Object.get_attr_string self "rows_" with
| None -> raise (Wrap_utils.Attribute_not_found "rows_")
| Some x -> Ndarray.of_pyobject x
let columns_ self =
  match Py.Object.get_attr_string self "columns_" with
| None -> raise (Wrap_utils.Attribute_not_found "columns_")
| Some x -> Ndarray.of_pyobject x
let row_labels_ self =
  match Py.Object.get_attr_string self "row_labels_" with
| None -> raise (Wrap_utils.Attribute_not_found "row_labels_")
| Some x -> Ndarray.of_pyobject x
let column_labels_ self =
  match Py.Object.get_attr_string self "column_labels_" with
| None -> raise (Wrap_utils.Attribute_not_found "column_labels_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SpectralClustering = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_clusters ?eigen_solver ?n_components ?random_state ?n_init ?gamma ?affinity ?n_neighbors ?eigen_tol ?assign_labels ?degree ?coef0 ?kernel_params ?n_jobs () =
                     Py.Module.get_function_with_keywords ns "SpectralClustering"
                       [||]
                       (Wrap_utils.keyword_args [("n_clusters", Wrap_utils.Option.map n_clusters Py.Int.of_int); ("eigen_solver", Wrap_utils.Option.map eigen_solver (function
| `Arpack -> Py.String.of_string "arpack"
| `Lobpcg -> Py.String.of_string "lobpcg"
| `Amg -> Py.String.of_string "amg"
| `None -> Py.String.of_string "None"
)); ("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("n_init", Wrap_utils.Option.map n_init Py.Int.of_int); ("gamma", Wrap_utils.Option.map gamma Py.Float.of_float); ("affinity", Wrap_utils.Option.map affinity (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("eigen_tol", Wrap_utils.Option.map eigen_tol Py.Float.of_float); ("assign_labels", Wrap_utils.Option.map assign_labels (function
| `Kmeans -> Py.String.of_string "kmeans"
| `Discretize -> Py.String.of_string "discretize"
)); ("degree", Wrap_utils.Option.map degree Py.Float.of_float); ("coef0", Wrap_utils.Option.map coef0 Py.Float.of_float); ("kernel_params", kernel_params); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
))])

                  let fit ?y ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

                  let fit_predict ?y ~x self =
                     Py.Module.get_function_with_keywords self "fit_predict"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let affinity_matrix_ self =
  match Py.Object.get_attr_string self "affinity_matrix_" with
| None -> raise (Wrap_utils.Attribute_not_found "affinity_matrix_")
| Some x -> Ndarray.of_pyobject x
let labels_ self =
  match Py.Object.get_attr_string self "labels_" with
| None -> raise (Wrap_utils.Attribute_not_found "labels_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SpectralCoclustering = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_clusters ?svd_method ?n_svd_vecs ?mini_batch ?init ?n_init ?n_jobs ?random_state () =
                     Py.Module.get_function_with_keywords ns "SpectralCoclustering"
                       [||]
                       (Wrap_utils.keyword_args [("n_clusters", Wrap_utils.Option.map n_clusters Py.Int.of_int); ("svd_method", Wrap_utils.Option.map svd_method (function
| `Randomized -> Py.String.of_string "randomized"
| `Arpack -> Py.String.of_string "arpack"
)); ("n_svd_vecs", Wrap_utils.Option.map n_svd_vecs Py.Int.of_int); ("mini_batch", Wrap_utils.Option.map mini_batch Py.Bool.of_bool); ("init", Wrap_utils.Option.map init (function
| `Random -> Py.String.of_string "random"
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("n_init", Wrap_utils.Option.map n_init Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
))])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let get_indices ~i self =
   Py.Module.get_function_with_keywords self "get_indices"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i |> Py.Int.of_int))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let get_shape ~i self =
   Py.Module.get_function_with_keywords self "get_shape"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i |> Py.Int.of_int))])
     |> Py.Int.to_int
let get_submatrix ~i ~data self =
   Py.Module.get_function_with_keywords self "get_submatrix"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i |> Py.Int.of_int)); ("data", Some(data ))])
     |> Ndarray.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let rows_ self =
  match Py.Object.get_attr_string self "rows_" with
| None -> raise (Wrap_utils.Attribute_not_found "rows_")
| Some x -> Ndarray.of_pyobject x
let columns_ self =
  match Py.Object.get_attr_string self "columns_" with
| None -> raise (Wrap_utils.Attribute_not_found "columns_")
| Some x -> Ndarray.of_pyobject x
let row_labels_ self =
  match Py.Object.get_attr_string self "row_labels_" with
| None -> raise (Wrap_utils.Attribute_not_found "row_labels_")
| Some x -> Ndarray.of_pyobject x
let column_labels_ self =
  match Py.Object.get_attr_string self "column_labels_" with
| None -> raise (Wrap_utils.Attribute_not_found "column_labels_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let affinity_propagation ?preference ?convergence_iter ?max_iter ?damping ?copy ?verbose ?return_n_iter ~s () =
                     Py.Module.get_function_with_keywords ns "affinity_propagation"
                       [||]
                       (Wrap_utils.keyword_args [("preference", Wrap_utils.Option.map preference (function
| `Ndarray x -> Ndarray.to_pyobject x
| `Float x -> Py.Float.of_float x
)); ("convergence_iter", Wrap_utils.Option.map convergence_iter Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("damping", Wrap_utils.Option.map damping Py.Float.of_float); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("return_n_iter", Wrap_utils.Option.map return_n_iter Py.Bool.of_bool); ("S", Some(s |> Ndarray.to_pyobject))])
                       |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2))))
let cluster_optics_dbscan ~reachability ~core_distances ~ordering ~eps () =
   Py.Module.get_function_with_keywords ns "cluster_optics_dbscan"
     [||]
     (Wrap_utils.keyword_args [("reachability", Some(reachability |> Ndarray.to_pyobject)); ("core_distances", Some(core_distances |> Ndarray.to_pyobject)); ("ordering", Some(ordering |> Ndarray.to_pyobject)); ("eps", Some(eps |> Py.Float.of_float))])
     |> Ndarray.of_pyobject
                  let cluster_optics_xi ?min_cluster_size ?xi ?predecessor_correction ~reachability ~predecessor ~ordering ~min_samples () =
                     Py.Module.get_function_with_keywords ns "cluster_optics_xi"
                       [||]
                       (Wrap_utils.keyword_args [("min_cluster_size", Wrap_utils.Option.map min_cluster_size (function
| `Int x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)); ("xi", Wrap_utils.Option.map xi (function
| `Float x -> Py.Float.of_float x
| `PyObject x -> Wrap_utils.id x
)); ("predecessor_correction", Wrap_utils.Option.map predecessor_correction Py.Bool.of_bool); ("reachability", Some(reachability |> Ndarray.to_pyobject)); ("predecessor", Some(predecessor |> Ndarray.to_pyobject)); ("ordering", Some(ordering |> Ndarray.to_pyobject)); ("min_samples", Some(min_samples |> (function
| `Int x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)))])
                       |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
                  let compute_optics_graph ~x ~min_samples ~max_eps ~metric ~p ~metric_params ~algorithm ~leaf_size ~n_jobs () =
                     Py.Module.get_function_with_keywords ns "compute_optics_graph"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject)); ("min_samples", Some(min_samples |> (function
| `Int x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
))); ("max_eps", Some(max_eps |> Py.Float.of_float)); ("metric", Some(metric |> (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
))); ("p", Some(p |> Py.Int.of_int)); ("metric_params", Some(metric_params )); ("algorithm", Some(algorithm |> (function
| `Auto -> Py.String.of_string "auto"
| `Ball_tree -> Py.String.of_string "ball_tree"
| `Kd_tree -> Py.String.of_string "kd_tree"
| `Brute -> Py.String.of_string "brute"
))); ("leaf_size", Some(leaf_size |> Py.Int.of_int)); ("n_jobs", Some(n_jobs |> (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)))])
                       |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1)), (Ndarray.of_pyobject (Py.Tuple.get x 2)), (Ndarray.of_pyobject (Py.Tuple.get x 3))))
                  let dbscan ?eps ?min_samples ?metric ?metric_params ?algorithm ?leaf_size ?p ?sample_weight ?n_jobs ~x () =
                     Py.Module.get_function_with_keywords ns "dbscan"
                       [||]
                       (Wrap_utils.keyword_args [("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("min_samples", Wrap_utils.Option.map min_samples Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("metric_params", metric_params); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Auto -> Py.String.of_string "auto"
| `Ball_tree -> Py.String.of_string "ball_tree"
| `Kd_tree -> Py.String.of_string "kd_tree"
| `Brute -> Py.String.of_string "brute"
)); ("leaf_size", Wrap_utils.Option.map leaf_size Py.Int.of_int); ("p", Wrap_utils.Option.map p Py.Float.of_float); ("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let estimate_bandwidth ?quantile ?n_samples ?random_state ?n_jobs ~x () =
                     Py.Module.get_function_with_keywords ns "estimate_bandwidth"
                       [||]
                       (Wrap_utils.keyword_args [("quantile", Wrap_utils.Option.map quantile Py.Float.of_float); ("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("X", Some(x |> Ndarray.to_pyobject))])
                       |> Py.Float.to_float
let get_bin_seeds ?min_bin_freq ~x ~bin_size () =
   Py.Module.get_function_with_keywords ns "get_bin_seeds"
     [||]
     (Wrap_utils.keyword_args [("min_bin_freq", Wrap_utils.Option.map min_bin_freq Py.Int.of_int); ("X", Some(x |> Ndarray.to_pyobject)); ("bin_size", Some(bin_size |> Py.Float.of_float))])
     |> Ndarray.of_pyobject
                  let k_means ?sample_weight ?init ?precompute_distances ?n_init ?max_iter ?verbose ?tol ?random_state ?copy_x ?n_jobs ?algorithm ?return_n_iter ~x ~n_clusters () =
                     Py.Module.get_function_with_keywords ns "k_means"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("init", Wrap_utils.Option.map init (function
| `K_means_ -> Py.String.of_string "k-means++"
| `Random -> Py.String.of_string "random"
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("precompute_distances", Wrap_utils.Option.map precompute_distances (function
| `Bool x -> Py.Bool.of_bool x
| `Auto -> Py.String.of_string "auto"
)); ("n_init", Wrap_utils.Option.map n_init Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("copy_x", Wrap_utils.Option.map copy_x Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Auto -> Py.String.of_string "auto"
| `Full -> Py.String.of_string "full"
| `Elkan -> Py.String.of_string "elkan"
)); ("return_n_iter", Wrap_utils.Option.map return_n_iter Py.Bool.of_bool); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
))); ("n_clusters", Some(n_clusters |> Py.Int.of_int))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2)), (Py.Int.to_int (Py.Tuple.get x 3))))
                  let linkage_tree ?connectivity ?n_clusters ?linkage ?affinity ~x () =
                     Py.Module.get_function_with_keywords ns "linkage_tree"
                       [||]
                       (Wrap_utils.keyword_args [("connectivity", Wrap_utils.Option.map connectivity Csr_matrix.to_pyobject); ("n_clusters", Wrap_utils.Option.map n_clusters Py.Int.of_int); ("linkage", Wrap_utils.Option.map linkage (function
| `Average -> Py.String.of_string "average"
| `Complete -> Py.String.of_string "complete"
| `Single -> Py.String.of_string "single"
)); ("affinity", Wrap_utils.Option.map affinity (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("return_distance", Some(true |> Py.Bool.of_bool)); ("X", Some(x |> Ndarray.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3)), (Ndarray.of_pyobject (Py.Tuple.get x 4))))
                  let mean_shift ?bandwidth ?seeds ?bin_seeding ?min_bin_freq ?cluster_all ?max_iter ?n_jobs ~x () =
                     Py.Module.get_function_with_keywords ns "mean_shift"
                       [||]
                       (Wrap_utils.keyword_args [("bandwidth", Wrap_utils.Option.map bandwidth Py.Float.of_float); ("seeds", Wrap_utils.Option.map seeds (function
| `Ndarray x -> Ndarray.to_pyobject x
| `None -> Py.String.of_string "None"
)); ("bin_seeding", Wrap_utils.Option.map bin_seeding Py.Bool.of_bool); ("min_bin_freq", Wrap_utils.Option.map min_bin_freq Py.Int.of_int); ("cluster_all", Wrap_utils.Option.map cluster_all Py.Bool.of_bool); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("X", Some(x |> Ndarray.to_pyobject))])
                       |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
                  let spectral_clustering ?n_clusters ?n_components ?eigen_solver ?random_state ?n_init ?eigen_tol ?assign_labels ~affinity () =
                     Py.Module.get_function_with_keywords ns "spectral_clustering"
                       [||]
                       (Wrap_utils.keyword_args [("n_clusters", Wrap_utils.Option.map n_clusters Py.Int.of_int); ("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("eigen_solver", Wrap_utils.Option.map eigen_solver (function
| `Arpack -> Py.String.of_string "arpack"
| `Lobpcg -> Py.String.of_string "lobpcg"
| `Amg -> Py.String.of_string "amg"
| `None -> Py.String.of_string "None"
)); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("n_init", Wrap_utils.Option.map n_init Py.Int.of_int); ("eigen_tol", Wrap_utils.Option.map eigen_tol Py.Float.of_float); ("assign_labels", Wrap_utils.Option.map assign_labels (function
| `Kmeans -> Py.String.of_string "kmeans"
| `Discretize -> Py.String.of_string "discretize"
)); ("affinity", Some(affinity |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

let ward_tree ?connectivity ?n_clusters ~x () =
   Py.Module.get_function_with_keywords ns "ward_tree"
     [||]
     (Wrap_utils.keyword_args [("connectivity", Wrap_utils.Option.map connectivity Csr_matrix.to_pyobject); ("n_clusters", Wrap_utils.Option.map n_clusters Py.Int.of_int); ("return_distance", Some(true |> Py.Bool.of_bool)); ("X", Some(x |> Ndarray.to_pyobject))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3)), (Wrap_utils.id (Py.Tuple.get x 4)), (Wrap_utils.id (Py.Tuple.get x 5))))
