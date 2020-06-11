let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.cluster"

let get_py name = Py.Module.get __wrap_namespace name
module AffinityPropagation = struct
type tag = [`AffinityPropagation]
type t = [`AffinityPropagation | `BaseEstimator | `ClusterMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_cluster x = (x :> [`ClusterMixin] Obj.t)
                  let create ?damping ?max_iter ?convergence_iter ?copy ?preference ?affinity ?verbose () =
                     Py.Module.get_function_with_keywords __wrap_namespace "AffinityPropagation"
                       [||]
                       (Wrap_utils.keyword_args [("damping", Wrap_utils.Option.map damping Py.Float.of_float); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("convergence_iter", Wrap_utils.Option.map convergence_iter Py.Int.of_int); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("preference", Wrap_utils.Option.map preference Np.Obj.to_pyobject); ("affinity", Wrap_utils.Option.map affinity (function
| `Euclidean -> Py.String.of_string "euclidean"
| `Precomputed -> Py.String.of_string "precomputed"
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int)])
                       |> of_pyobject
let fit ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> of_pyobject
let fit_predict ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_predict"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let predict ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject

let cluster_centers_indices_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "cluster_centers_indices_" with
  | None -> failwith "attribute cluster_centers_indices_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let cluster_centers_indices_ self = match cluster_centers_indices_opt self with
  | None -> raise Not_found
  | Some x -> x

let cluster_centers_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "cluster_centers_" with
  | None -> failwith "attribute cluster_centers_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let cluster_centers_ self = match cluster_centers_opt self with
  | None -> raise Not_found
  | Some x -> x

let labels_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "labels_" with
  | None -> failwith "attribute labels_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let labels_ self = match labels_opt self with
  | None -> raise Not_found
  | Some x -> x

let affinity_matrix_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "affinity_matrix_" with
  | None -> failwith "attribute affinity_matrix_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let affinity_matrix_ self = match affinity_matrix_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module AgglomerativeClustering = struct
type tag = [`AgglomerativeClustering]
type t = [`AgglomerativeClustering | `BaseEstimator | `ClusterMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_cluster x = (x :> [`ClusterMixin] Obj.t)
                  let create ?n_clusters ?affinity ?memory ?connectivity ?compute_full_tree ?linkage ?distance_threshold () =
                     Py.Module.get_function_with_keywords __wrap_namespace "AgglomerativeClustering"
                       [||]
                       (Wrap_utils.keyword_args [("n_clusters", Wrap_utils.Option.map n_clusters (function
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("affinity", Wrap_utils.Option.map affinity (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("memory", Wrap_utils.Option.map memory (function
| `S x -> Py.String.of_string x
| `Object_with_the_joblib_Memory_interface x -> Wrap_utils.id x
)); ("connectivity", Wrap_utils.Option.map connectivity (function
| `Callable x -> Wrap_utils.id x
| `Arr x -> Np.Obj.to_pyobject x
)); ("compute_full_tree", Wrap_utils.Option.map compute_full_tree (function
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("linkage", Wrap_utils.Option.map linkage (function
| `Ward -> Py.String.of_string "ward"
| `Complete -> Py.String.of_string "complete"
| `Average -> Py.String.of_string "average"
| `Single -> Py.String.of_string "single"
)); ("distance_threshold", Wrap_utils.Option.map distance_threshold Py.Float.of_float)])
                       |> of_pyobject
let fit ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> of_pyobject
let fit_predict ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_predict"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject

let n_clusters_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_clusters_" with
  | None -> failwith "attribute n_clusters_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_clusters_ self = match n_clusters_opt self with
  | None -> raise Not_found
  | Some x -> x

let labels_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "labels_" with
  | None -> failwith "attribute labels_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let labels_ self = match labels_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_leaves_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_leaves_" with
  | None -> failwith "attribute n_leaves_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_leaves_ self = match n_leaves_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_connected_components_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_connected_components_" with
  | None -> failwith "attribute n_connected_components_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_connected_components_ self = match n_connected_components_opt self with
  | None -> raise Not_found
  | Some x -> x

let children_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "children_" with
  | None -> failwith "attribute children_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let children_ self = match children_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Birch = struct
type tag = [`Birch]
type t = [`BaseEstimator | `Birch | `ClusterMixin | `Object | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_cluster x = (x :> [`ClusterMixin] Obj.t)
                  let create ?threshold ?branching_factor ?n_clusters ?compute_labels ?copy () =
                     Py.Module.get_function_with_keywords __wrap_namespace "Birch"
                       [||]
                       (Wrap_utils.keyword_args [("threshold", Wrap_utils.Option.map threshold Py.Float.of_float); ("branching_factor", Wrap_utils.Option.map branching_factor Py.Int.of_int); ("n_clusters", Wrap_utils.Option.map n_clusters (function
| `None -> Py.none
| `I x -> Py.Int.of_int x
| `ClusterMixin x -> Np.Obj.to_pyobject x
)); ("compute_labels", Wrap_utils.Option.map compute_labels Py.Bool.of_bool); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])
                       |> of_pyobject
let fit ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> of_pyobject
let fit_predict ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_predict"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let partial_fit ?x ?y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("y", y)])
     |> of_pyobject
let predict ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

let root_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "root_" with
  | None -> failwith "attribute root_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let root_ self = match root_opt self with
  | None -> raise Not_found
  | Some x -> x

let dummy_leaf_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dummy_leaf_" with
  | None -> failwith "attribute dummy_leaf_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let dummy_leaf_ self = match dummy_leaf_opt self with
  | None -> raise Not_found
  | Some x -> x

let subcluster_centers_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "subcluster_centers_" with
  | None -> failwith "attribute subcluster_centers_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let subcluster_centers_ self = match subcluster_centers_opt self with
  | None -> raise Not_found
  | Some x -> x

let subcluster_labels_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "subcluster_labels_" with
  | None -> failwith "attribute subcluster_labels_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let subcluster_labels_ self = match subcluster_labels_opt self with
  | None -> raise Not_found
  | Some x -> x

let labels_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "labels_" with
  | None -> failwith "attribute labels_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let labels_ self = match labels_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module DBSCAN = struct
type tag = [`DBSCAN]
type t = [`BaseEstimator | `ClusterMixin | `DBSCAN | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_cluster x = (x :> [`ClusterMixin] Obj.t)
                  let create ?eps ?min_samples ?metric ?metric_params ?algorithm ?leaf_size ?p ?n_jobs () =
                     Py.Module.get_function_with_keywords __wrap_namespace "DBSCAN"
                       [||]
                       (Wrap_utils.keyword_args [("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("min_samples", Wrap_utils.Option.map min_samples Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("metric_params", Wrap_utils.Option.map metric_params Dict.to_pyobject); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Auto -> Py.String.of_string "auto"
| `Ball_tree -> Py.String.of_string "ball_tree"
| `Kd_tree -> Py.String.of_string "kd_tree"
| `Brute -> Py.String.of_string "brute"
)); ("leaf_size", Wrap_utils.Option.map leaf_size Py.Int.of_int); ("p", Wrap_utils.Option.map p Py.Float.of_float); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int)])
                       |> of_pyobject
let fit ?y ?sample_weight ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> of_pyobject
let fit_predict ?y ?sample_weight ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_predict"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject

let core_sample_indices_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "core_sample_indices_" with
  | None -> failwith "attribute core_sample_indices_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let core_sample_indices_ self = match core_sample_indices_opt self with
  | None -> raise Not_found
  | Some x -> x

let components_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "components_" with
  | None -> failwith "attribute components_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let components_ self = match components_opt self with
  | None -> raise Not_found
  | Some x -> x

let labels_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "labels_" with
  | None -> failwith "attribute labels_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let labels_ self = match labels_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module FeatureAgglomeration = struct
type tag = [`FeatureAgglomeration]
type t = [`BaseEstimator | `ClusterMixin | `FeatureAgglomeration | `Object | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_cluster x = (x :> [`ClusterMixin] Obj.t)
                  let create ?n_clusters ?affinity ?memory ?connectivity ?compute_full_tree ?linkage ?pooling_func ?distance_threshold () =
                     Py.Module.get_function_with_keywords __wrap_namespace "FeatureAgglomeration"
                       [||]
                       (Wrap_utils.keyword_args [("n_clusters", Wrap_utils.Option.map n_clusters Py.Int.of_int); ("affinity", Wrap_utils.Option.map affinity (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("memory", Wrap_utils.Option.map memory (function
| `S x -> Py.String.of_string x
| `Object_with_the_joblib_Memory_interface x -> Wrap_utils.id x
)); ("connectivity", Wrap_utils.Option.map connectivity (function
| `Callable x -> Wrap_utils.id x
| `Arr x -> Np.Obj.to_pyobject x
)); ("compute_full_tree", Wrap_utils.Option.map compute_full_tree (function
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("linkage", Wrap_utils.Option.map linkage (function
| `Ward -> Py.String.of_string "ward"
| `Complete -> Py.String.of_string "complete"
| `Average -> Py.String.of_string "average"
| `Single -> Py.String.of_string "single"
)); ("pooling_func", pooling_func); ("distance_threshold", Wrap_utils.Option.map distance_threshold Py.Float.of_float)])
                       |> of_pyobject
let fit ?y ?params ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))]) (match params with None -> [] | Some x -> x))
     |> of_pyobject
let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let inverse_transform ~xred self =
   Py.Module.get_function_with_keywords (to_pyobject self) "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("Xred", Some(xred |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

let n_clusters_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_clusters_" with
  | None -> failwith "attribute n_clusters_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_clusters_ self = match n_clusters_opt self with
  | None -> raise Not_found
  | Some x -> x

let labels_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "labels_" with
  | None -> failwith "attribute labels_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let labels_ self = match labels_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_leaves_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_leaves_" with
  | None -> failwith "attribute n_leaves_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_leaves_ self = match n_leaves_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_connected_components_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_connected_components_" with
  | None -> failwith "attribute n_connected_components_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_connected_components_ self = match n_connected_components_opt self with
  | None -> raise Not_found
  | Some x -> x

let children_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "children_" with
  | None -> failwith "attribute children_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let children_ self = match children_opt self with
  | None -> raise Not_found
  | Some x -> x

let distances_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "distances_" with
  | None -> failwith "attribute distances_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let distances_ self = match distances_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module KMeans = struct
type tag = [`KMeans]
type t = [`BaseEstimator | `ClusterMixin | `KMeans | `Object | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_cluster x = (x :> [`ClusterMixin] Obj.t)
                  let create ?n_clusters ?init ?n_init ?max_iter ?tol ?precompute_distances ?verbose ?random_state ?copy_x ?n_jobs ?algorithm () =
                     Py.Module.get_function_with_keywords __wrap_namespace "KMeans"
                       [||]
                       (Wrap_utils.keyword_args [("n_clusters", Wrap_utils.Option.map n_clusters Py.Int.of_int); ("init", Wrap_utils.Option.map init (function
| `Arr x -> Np.Obj.to_pyobject x
| `Random -> Py.String.of_string "random"
| `K_means_ -> Py.String.of_string "k-means++"
)); ("n_init", Wrap_utils.Option.map n_init Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("precompute_distances", Wrap_utils.Option.map precompute_distances (function
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("copy_x", Wrap_utils.Option.map copy_x Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Auto -> Py.String.of_string "auto"
| `Full -> Py.String.of_string "full"
| `Elkan -> Py.String.of_string "elkan"
))])
                       |> of_pyobject
let fit ?y ?sample_weight ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> of_pyobject
let fit_predict ?y ?sample_weight ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_predict"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let fit_transform ?y ?sample_weight ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let predict ?sample_weight ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let score ?y ?sample_weight ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

let cluster_centers_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "cluster_centers_" with
  | None -> failwith "attribute cluster_centers_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let cluster_centers_ self = match cluster_centers_opt self with
  | None -> raise Not_found
  | Some x -> x

let labels_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "labels_" with
  | None -> failwith "attribute labels_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let labels_ self = match labels_opt self with
  | None -> raise Not_found
  | Some x -> x

let inertia_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "inertia_" with
  | None -> failwith "attribute inertia_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let inertia_ self = match inertia_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MeanShift = struct
type tag = [`MeanShift]
type t = [`BaseEstimator | `ClusterMixin | `MeanShift | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_cluster x = (x :> [`ClusterMixin] Obj.t)
let create ?bandwidth ?seeds ?bin_seeding ?min_bin_freq ?cluster_all ?n_jobs ?max_iter () =
   Py.Module.get_function_with_keywords __wrap_namespace "MeanShift"
     [||]
     (Wrap_utils.keyword_args [("bandwidth", Wrap_utils.Option.map bandwidth Py.Float.of_float); ("seeds", Wrap_utils.Option.map seeds Np.Obj.to_pyobject); ("bin_seeding", Wrap_utils.Option.map bin_seeding Py.Bool.of_bool); ("min_bin_freq", Wrap_utils.Option.map min_bin_freq Py.Int.of_int); ("cluster_all", Wrap_utils.Option.map cluster_all Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int)])
     |> of_pyobject
let fit ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> of_pyobject
let fit_predict ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_predict"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let predict ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject

let cluster_centers_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "cluster_centers_" with
  | None -> failwith "attribute cluster_centers_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let cluster_centers_ self = match cluster_centers_opt self with
  | None -> raise Not_found
  | Some x -> x

let labels_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "labels_" with
  | None -> failwith "attribute labels_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let labels_ self = match labels_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MiniBatchKMeans = struct
type tag = [`MiniBatchKMeans]
type t = [`BaseEstimator | `ClusterMixin | `MiniBatchKMeans | `Object | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_cluster x = (x :> [`ClusterMixin] Obj.t)
                  let create ?n_clusters ?init ?max_iter ?batch_size ?verbose ?compute_labels ?random_state ?tol ?max_no_improvement ?init_size ?n_init ?reassignment_ratio () =
                     Py.Module.get_function_with_keywords __wrap_namespace "MiniBatchKMeans"
                       [||]
                       (Wrap_utils.keyword_args [("n_clusters", Wrap_utils.Option.map n_clusters Py.Int.of_int); ("init", Wrap_utils.Option.map init (function
| `Arr x -> Np.Obj.to_pyobject x
| `Random -> Py.String.of_string "random"
| `K_means_ -> Py.String.of_string "k-means++"
)); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("batch_size", Wrap_utils.Option.map batch_size Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("compute_labels", Wrap_utils.Option.map compute_labels Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("max_no_improvement", Wrap_utils.Option.map max_no_improvement Py.Int.of_int); ("init_size", Wrap_utils.Option.map init_size Py.Int.of_int); ("n_init", Wrap_utils.Option.map n_init Py.Int.of_int); ("reassignment_ratio", Wrap_utils.Option.map reassignment_ratio Py.Float.of_float)])
                       |> of_pyobject
let fit ?y ?sample_weight ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> of_pyobject
let fit_predict ?y ?sample_weight ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_predict"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let fit_transform ?y ?sample_weight ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let partial_fit ?y ?sample_weight ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> of_pyobject
let predict ?sample_weight ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let score ?y ?sample_weight ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

let cluster_centers_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "cluster_centers_" with
  | None -> failwith "attribute cluster_centers_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let cluster_centers_ self = match cluster_centers_opt self with
  | None -> raise Not_found
  | Some x -> x

let labels_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "labels_" with
  | None -> failwith "attribute labels_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let labels_ self = match labels_opt self with
  | None -> raise Not_found
  | Some x -> x

let inertia_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "inertia_" with
  | None -> failwith "attribute inertia_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let inertia_ self = match inertia_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module OPTICS = struct
type tag = [`OPTICS]
type t = [`BaseEstimator | `ClusterMixin | `OPTICS | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_cluster x = (x :> [`ClusterMixin] Obj.t)
                  let create ?min_samples ?max_eps ?metric ?p ?metric_params ?cluster_method ?eps ?xi ?predecessor_correction ?min_cluster_size ?algorithm ?leaf_size ?n_jobs () =
                     Py.Module.get_function_with_keywords __wrap_namespace "OPTICS"
                       [||]
                       (Wrap_utils.keyword_args [("min_samples", Wrap_utils.Option.map min_samples (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("max_eps", Wrap_utils.Option.map max_eps Py.Float.of_float); ("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("p", Wrap_utils.Option.map p Py.Int.of_int); ("metric_params", Wrap_utils.Option.map metric_params Dict.to_pyobject); ("cluster_method", Wrap_utils.Option.map cluster_method Py.String.of_string); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("xi", Wrap_utils.Option.map xi (function
| `F x -> Py.Float.of_float x
| `Between_0_and_1 x -> Wrap_utils.id x
)); ("predecessor_correction", Wrap_utils.Option.map predecessor_correction Py.Bool.of_bool); ("min_cluster_size", Wrap_utils.Option.map min_cluster_size (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Auto -> Py.String.of_string "auto"
| `Ball_tree -> Py.String.of_string "ball_tree"
| `Kd_tree -> Py.String.of_string "kd_tree"
| `Brute -> Py.String.of_string "brute"
)); ("leaf_size", Wrap_utils.Option.map leaf_size Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int)])
                       |> of_pyobject
let fit ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> of_pyobject
let fit_predict ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_predict"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject

let labels_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "labels_" with
  | None -> failwith "attribute labels_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let labels_ self = match labels_opt self with
  | None -> raise Not_found
  | Some x -> x

let reachability_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "reachability_" with
  | None -> failwith "attribute reachability_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let reachability_ self = match reachability_opt self with
  | None -> raise Not_found
  | Some x -> x

let ordering_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "ordering_" with
  | None -> failwith "attribute ordering_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let ordering_ self = match ordering_opt self with
  | None -> raise Not_found
  | Some x -> x

let core_distances_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "core_distances_" with
  | None -> failwith "attribute core_distances_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let core_distances_ self = match core_distances_opt self with
  | None -> raise Not_found
  | Some x -> x

let predecessor_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "predecessor_" with
  | None -> failwith "attribute predecessor_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let predecessor_ self = match predecessor_opt self with
  | None -> raise Not_found
  | Some x -> x

let cluster_hierarchy_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "cluster_hierarchy_" with
  | None -> failwith "attribute cluster_hierarchy_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let cluster_hierarchy_ self = match cluster_hierarchy_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SpectralBiclustering = struct
type tag = [`SpectralBiclustering]
type t = [`BaseEstimator | `BaseSpectral | `BiclusterMixin | `Object | `SpectralBiclustering] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_spectral x = (x :> [`BaseSpectral] Obj.t)
let as_bicluster x = (x :> [`BiclusterMixin] Obj.t)
                  let create ?n_clusters ?method_ ?n_components ?n_best ?svd_method ?n_svd_vecs ?mini_batch ?init ?n_init ?n_jobs ?random_state () =
                     Py.Module.get_function_with_keywords __wrap_namespace "SpectralBiclustering"
                       [||]
                       (Wrap_utils.keyword_args [("n_clusters", Wrap_utils.Option.map n_clusters (function
| `Tuple x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("method", Wrap_utils.Option.map method_ (function
| `Bistochastic -> Py.String.of_string "bistochastic"
| `Scale -> Py.String.of_string "scale"
| `Log -> Py.String.of_string "log"
)); ("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("n_best", Wrap_utils.Option.map n_best Py.Int.of_int); ("svd_method", Wrap_utils.Option.map svd_method (function
| `Randomized -> Py.String.of_string "randomized"
| `Arpack -> Py.String.of_string "arpack"
)); ("n_svd_vecs", Wrap_utils.Option.map n_svd_vecs Py.Int.of_int); ("mini_batch", Wrap_utils.Option.map mini_batch Py.Bool.of_bool); ("init", Wrap_utils.Option.map init (function
| `Arr x -> Np.Obj.to_pyobject x
| `Random -> Py.String.of_string "random"
| `K_means_ -> Py.String.of_string "k-means++"
)); ("n_init", Wrap_utils.Option.map n_init Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
                       |> of_pyobject
let fit ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> of_pyobject
let get_indices ~i self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_indices"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i |> Py.Int.of_int))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let get_shape ~i self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_shape"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i |> Py.Int.of_int))])
     |> (fun x -> ((Py.Int.to_int (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
let get_submatrix ~i ~data self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_submatrix"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i |> Py.Int.of_int)); ("data", Some(data |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject

let rows_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "rows_" with
  | None -> failwith "attribute rows_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let rows_ self = match rows_opt self with
  | None -> raise Not_found
  | Some x -> x

let columns_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "columns_" with
  | None -> failwith "attribute columns_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let columns_ self = match columns_opt self with
  | None -> raise Not_found
  | Some x -> x

let row_labels_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "row_labels_" with
  | None -> failwith "attribute row_labels_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let row_labels_ self = match row_labels_opt self with
  | None -> raise Not_found
  | Some x -> x

let column_labels_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "column_labels_" with
  | None -> failwith "attribute column_labels_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let column_labels_ self = match column_labels_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SpectralClustering = struct
type tag = [`SpectralClustering]
type t = [`BaseEstimator | `ClusterMixin | `Object | `SpectralClustering] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_cluster x = (x :> [`ClusterMixin] Obj.t)
                  let create ?n_clusters ?eigen_solver ?n_components ?random_state ?n_init ?gamma ?affinity ?n_neighbors ?eigen_tol ?assign_labels ?degree ?coef0 ?kernel_params ?n_jobs () =
                     Py.Module.get_function_with_keywords __wrap_namespace "SpectralClustering"
                       [||]
                       (Wrap_utils.keyword_args [("n_clusters", Wrap_utils.Option.map n_clusters Py.Int.of_int); ("eigen_solver", Wrap_utils.Option.map eigen_solver (function
| `Arpack -> Py.String.of_string "arpack"
| `PyObject x -> Wrap_utils.id x
| `Lobpcg -> Py.String.of_string "lobpcg"
)); ("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("n_init", Wrap_utils.Option.map n_init Py.Int.of_int); ("gamma", Wrap_utils.Option.map gamma Py.Float.of_float); ("affinity", Wrap_utils.Option.map affinity (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("eigen_tol", Wrap_utils.Option.map eigen_tol Py.Float.of_float); ("assign_labels", Wrap_utils.Option.map assign_labels (function
| `Kmeans -> Py.String.of_string "kmeans"
| `Discretize -> Py.String.of_string "discretize"
)); ("degree", Wrap_utils.Option.map degree Py.Float.of_float); ("coef0", Wrap_utils.Option.map coef0 Py.Float.of_float); ("kernel_params", Wrap_utils.Option.map kernel_params Dict.to_pyobject); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int)])
                       |> of_pyobject
let fit ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> of_pyobject
let fit_predict ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_predict"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject

let affinity_matrix_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "affinity_matrix_" with
  | None -> failwith "attribute affinity_matrix_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let affinity_matrix_ self = match affinity_matrix_opt self with
  | None -> raise Not_found
  | Some x -> x

let labels_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "labels_" with
  | None -> failwith "attribute labels_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let labels_ self = match labels_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SpectralCoclustering = struct
type tag = [`SpectralCoclustering]
type t = [`BaseEstimator | `BaseSpectral | `BiclusterMixin | `Object | `SpectralCoclustering] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_spectral x = (x :> [`BaseSpectral] Obj.t)
let as_bicluster x = (x :> [`BiclusterMixin] Obj.t)
                  let create ?n_clusters ?svd_method ?n_svd_vecs ?mini_batch ?init ?n_init ?n_jobs ?random_state () =
                     Py.Module.get_function_with_keywords __wrap_namespace "SpectralCoclustering"
                       [||]
                       (Wrap_utils.keyword_args [("n_clusters", Wrap_utils.Option.map n_clusters Py.Int.of_int); ("svd_method", Wrap_utils.Option.map svd_method (function
| `Randomized -> Py.String.of_string "randomized"
| `Arpack -> Py.String.of_string "arpack"
)); ("n_svd_vecs", Wrap_utils.Option.map n_svd_vecs Py.Int.of_int); ("mini_batch", Wrap_utils.Option.map mini_batch Py.Bool.of_bool); ("init", Wrap_utils.Option.map init (function
| `Arr x -> Np.Obj.to_pyobject x
| `Random -> Py.String.of_string "random"
| `T_k_means_ x -> Wrap_utils.id x
)); ("n_init", Wrap_utils.Option.map n_init Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
                       |> of_pyobject
let fit ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> of_pyobject
let get_indices ~i self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_indices"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i |> Py.Int.of_int))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let get_shape ~i self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_shape"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i |> Py.Int.of_int))])
     |> (fun x -> ((Py.Int.to_int (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
let get_submatrix ~i ~data self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_submatrix"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i |> Py.Int.of_int)); ("data", Some(data |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject

let rows_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "rows_" with
  | None -> failwith "attribute rows_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let rows_ self = match rows_opt self with
  | None -> raise Not_found
  | Some x -> x

let columns_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "columns_" with
  | None -> failwith "attribute columns_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let columns_ self = match columns_opt self with
  | None -> raise Not_found
  | Some x -> x

let row_labels_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "row_labels_" with
  | None -> failwith "attribute row_labels_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let row_labels_ self = match row_labels_opt self with
  | None -> raise Not_found
  | Some x -> x

let column_labels_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "column_labels_" with
  | None -> failwith "attribute column_labels_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let column_labels_ self = match column_labels_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let affinity_propagation ?preference ?convergence_iter ?max_iter ?damping ?copy ?verbose ?return_n_iter ~s () =
   Py.Module.get_function_with_keywords __wrap_namespace "affinity_propagation"
     [||]
     (Wrap_utils.keyword_args [("preference", Wrap_utils.Option.map preference Np.Obj.to_pyobject); ("convergence_iter", Wrap_utils.Option.map convergence_iter Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("damping", Wrap_utils.Option.map damping Py.Float.of_float); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("return_n_iter", Wrap_utils.Option.map return_n_iter Py.Bool.of_bool); ("S", Some(s |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2))))
let cluster_optics_dbscan ~reachability ~core_distances ~ordering ~eps () =
   Py.Module.get_function_with_keywords __wrap_namespace "cluster_optics_dbscan"
     [||]
     (Wrap_utils.keyword_args [("reachability", Some(reachability |> Np.Obj.to_pyobject)); ("core_distances", Some(core_distances |> Np.Obj.to_pyobject)); ("ordering", Some(ordering |> Np.Obj.to_pyobject)); ("eps", Some(eps |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let cluster_optics_xi ?min_cluster_size ?xi ?predecessor_correction ~reachability ~predecessor ~ordering ~min_samples () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cluster_optics_xi"
                       [||]
                       (Wrap_utils.keyword_args [("min_cluster_size", Wrap_utils.Option.map min_cluster_size (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("xi", Wrap_utils.Option.map xi (function
| `F x -> Py.Float.of_float x
| `Between_0_and_1 x -> Wrap_utils.id x
)); ("predecessor_correction", Wrap_utils.Option.map predecessor_correction Py.Bool.of_bool); ("reachability", Some(reachability |> Np.Obj.to_pyobject)); ("predecessor", Some(predecessor |> Np.Obj.to_pyobject)); ("ordering", Some(ordering |> Np.Obj.to_pyobject)); ("min_samples", Some(min_samples |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let compute_optics_graph ~x ~min_samples ~max_eps ~metric ~p ~metric_params ~algorithm ~leaf_size ~n_jobs () =
                     Py.Module.get_function_with_keywords __wrap_namespace "compute_optics_graph"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("min_samples", Some(min_samples |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
))); ("max_eps", Some(max_eps |> Py.Float.of_float)); ("metric", Some(metric |> (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
))); ("p", Some(p |> Py.Int.of_int)); ("metric_params", Some(metric_params |> Dict.to_pyobject)); ("algorithm", Some(algorithm |> (function
| `Auto -> Py.String.of_string "auto"
| `Ball_tree -> Py.String.of_string "ball_tree"
| `Kd_tree -> Py.String.of_string "kd_tree"
| `Brute -> Py.String.of_string "brute"
))); ("leaf_size", Some(leaf_size |> Py.Int.of_int)); ("n_jobs", Some(n_jobs |> (function
| `I x -> Py.Int.of_int x
| `None -> Py.none
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 2)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 3))))
                  let dbscan ?eps ?min_samples ?metric ?metric_params ?algorithm ?leaf_size ?p ?sample_weight ?n_jobs ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "dbscan"
                       [||]
                       (Wrap_utils.keyword_args [("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("min_samples", Wrap_utils.Option.map min_samples Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("metric_params", Wrap_utils.Option.map metric_params Dict.to_pyobject); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Auto -> Py.String.of_string "auto"
| `Ball_tree -> Py.String.of_string "ball_tree"
| `Kd_tree -> Py.String.of_string "kd_tree"
| `Brute -> Py.String.of_string "brute"
)); ("leaf_size", Wrap_utils.Option.map leaf_size Py.Int.of_int); ("p", Wrap_utils.Option.map p Py.Float.of_float); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("X", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let estimate_bandwidth ?quantile ?n_samples ?random_state ?n_jobs ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "estimate_bandwidth"
     [||]
     (Wrap_utils.keyword_args [("quantile", Wrap_utils.Option.map quantile Py.Float.of_float); ("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let get_bin_seeds ?min_bin_freq ~x ~bin_size () =
   Py.Module.get_function_with_keywords __wrap_namespace "get_bin_seeds"
     [||]
     (Wrap_utils.keyword_args [("min_bin_freq", Wrap_utils.Option.map min_bin_freq Py.Int.of_int); ("X", Some(x |> Np.Obj.to_pyobject)); ("bin_size", Some(bin_size |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let k_means ?sample_weight ?init ?precompute_distances ?n_init ?max_iter ?verbose ?tol ?random_state ?copy_x ?n_jobs ?algorithm ?return_n_iter ~x ~n_clusters () =
                     Py.Module.get_function_with_keywords __wrap_namespace "k_means"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("init", init); ("precompute_distances", Wrap_utils.Option.map precompute_distances (function
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("n_init", Wrap_utils.Option.map n_init Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("copy_x", Wrap_utils.Option.map copy_x Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Auto -> Py.String.of_string "auto"
| `Full -> Py.String.of_string "full"
| `Elkan -> Py.String.of_string "elkan"
)); ("return_n_iter", Wrap_utils.Option.map return_n_iter Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject)); ("n_clusters", Some(n_clusters |> Py.Int.of_int))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2)), (Py.Int.to_int (Py.Tuple.get x 3))))
                  let linkage_tree ?connectivity ?n_clusters ?linkage ?affinity ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "linkage_tree"
                       [||]
                       (Wrap_utils.keyword_args [("connectivity", Wrap_utils.Option.map connectivity Np.Obj.to_pyobject); ("n_clusters", Wrap_utils.Option.map n_clusters Py.Int.of_int); ("linkage", Wrap_utils.Option.map linkage (function
| `Average -> Py.String.of_string "average"
| `Complete -> Py.String.of_string "complete"
| `Single -> Py.String.of_string "single"
)); ("affinity", Wrap_utils.Option.map affinity (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("return_distance", Some(true |> Py.Bool.of_bool)); ("X", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), ((fun py -> if Py.is_none py then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) py)) (Py.Tuple.get x 3)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 4))))
let mean_shift ?bandwidth ?seeds ?bin_seeding ?min_bin_freq ?cluster_all ?max_iter ?n_jobs ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "mean_shift"
     [||]
     (Wrap_utils.keyword_args [("bandwidth", Wrap_utils.Option.map bandwidth Py.Float.of_float); ("seeds", Wrap_utils.Option.map seeds Np.Obj.to_pyobject); ("bin_seeding", Wrap_utils.Option.map bin_seeding Py.Bool.of_bool); ("min_bin_freq", Wrap_utils.Option.map min_bin_freq Py.Int.of_int); ("cluster_all", Wrap_utils.Option.map cluster_all Py.Bool.of_bool); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let spectral_clustering ?n_clusters ?n_components ?eigen_solver ?random_state ?n_init ?eigen_tol ?assign_labels ~affinity () =
                     Py.Module.get_function_with_keywords __wrap_namespace "spectral_clustering"
                       [||]
                       (Wrap_utils.keyword_args [("n_clusters", Wrap_utils.Option.map n_clusters Py.Int.of_int); ("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("eigen_solver", Wrap_utils.Option.map eigen_solver (function
| `Arpack -> Py.String.of_string "arpack"
| `PyObject x -> Wrap_utils.id x
| `Lobpcg -> Py.String.of_string "lobpcg"
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("n_init", Wrap_utils.Option.map n_init Py.Int.of_int); ("eigen_tol", Wrap_utils.Option.map eigen_tol Py.Float.of_float); ("assign_labels", Wrap_utils.Option.map assign_labels (function
| `Kmeans -> Py.String.of_string "kmeans"
| `Discretize -> Py.String.of_string "discretize"
)); ("affinity", Some(affinity |> Np.Obj.to_pyobject))])

let ward_tree ?connectivity ?n_clusters ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ward_tree"
     [||]
     (Wrap_utils.keyword_args [("connectivity", Wrap_utils.Option.map connectivity Np.Obj.to_pyobject); ("n_clusters", Wrap_utils.Option.map n_clusters Py.Int.of_int); ("return_distance", Some(true |> Py.Bool.of_bool)); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), ((fun py -> if Py.is_none py then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) py)) (Py.Tuple.get x 3)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 4)), (Wrap_utils.id (Py.Tuple.get x 5))))
