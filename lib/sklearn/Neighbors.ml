let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.neighbors"

let get_py name = Py.Module.get __wrap_namespace name
module BallTree = struct
type tag = [`BallTree]
type t = [`BallTree | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let kernel_density ?kernel ?atol ?rtol ?breadth_first ?return_log ~x ~h self =
   Py.Module.get_function_with_keywords (to_pyobject self) "kernel_density"
     [||]
     (Wrap_utils.keyword_args [("kernel", Wrap_utils.Option.map kernel Py.String.of_string); ("atol", atol); ("rtol", rtol); ("breadth_first", Wrap_utils.Option.map breadth_first Py.Bool.of_bool); ("return_log", Wrap_utils.Option.map return_log Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject)); ("h", Some(h |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let query_radius ?count_only ~x ~r self =
   Py.Module.get_function_with_keywords (to_pyobject self) "query_radius"
     [||]
     (Wrap_utils.keyword_args [("count_only", Wrap_utils.Option.map count_only Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject)); ("r", Some(r ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))

let data_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "data" with
  | None -> failwith "attribute data not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let data self = match data_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module DistanceMetric = struct
type tag = [`DistanceMetric]
type t = [`DistanceMetric | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module KDTree = struct
type tag = [`KDTree]
type t = [`KDTree | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let kernel_density ?kernel ?atol ?rtol ?breadth_first ?return_log ~x ~h self =
   Py.Module.get_function_with_keywords (to_pyobject self) "kernel_density"
     [||]
     (Wrap_utils.keyword_args [("kernel", Wrap_utils.Option.map kernel Py.String.of_string); ("atol", atol); ("rtol", rtol); ("breadth_first", Wrap_utils.Option.map breadth_first Py.Bool.of_bool); ("return_log", Wrap_utils.Option.map return_log Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject)); ("h", Some(h |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let query_radius ?count_only ~x ~r self =
   Py.Module.get_function_with_keywords (to_pyobject self) "query_radius"
     [||]
     (Wrap_utils.keyword_args [("count_only", Wrap_utils.Option.map count_only Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject)); ("r", Some(r ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))

let data_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "data" with
  | None -> failwith "attribute data not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let data self = match data_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module KNeighborsClassifier = struct
type tag = [`KNeighborsClassifier]
type t = [`BaseEstimator | `ClassifierMixin | `KNeighborsClassifier | `KNeighborsMixin | `MultiOutputMixin | `NeighborsBase | `Object | `SupervisedIntegerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_neighbors x = (x :> [`NeighborsBase] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
let as_k_neighbors x = (x :> [`KNeighborsMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_supervised_integer x = (x :> [`SupervisedIntegerMixin] Obj.t)
                  let create ?n_neighbors ?weights ?algorithm ?leaf_size ?p ?metric ?metric_params ?n_jobs ?kwargs () =
                     Py.Module.get_function_with_keywords __wrap_namespace "KNeighborsClassifier"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("weights", Wrap_utils.Option.map weights (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Auto -> Py.String.of_string "auto"
| `Ball_tree -> Py.String.of_string "ball_tree"
| `Kd_tree -> Py.String.of_string "kd_tree"
| `Brute -> Py.String.of_string "brute"
)); ("leaf_size", Wrap_utils.Option.map leaf_size Py.Int.of_int); ("p", Wrap_utils.Option.map p Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("metric_params", Wrap_utils.Option.map metric_params Dict.to_pyobject); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int)]) (match kwargs with None -> [] | Some x -> x))
                       |> of_pyobject
                  let fit ~x ~y self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fit"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let kneighbors ?x ?n_neighbors self =
   Py.Module.get_function_with_keywords (to_pyobject self) "kneighbors"
     [||]
     (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("return_distance", Some(true |> Py.Bool.of_bool))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let kneighbors_graph ?x ?n_neighbors ?mode self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "kneighbors_graph"
                       [||]
                       (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Connectivity -> Py.String.of_string "connectivity"
| `Distance -> Py.String.of_string "distance"
))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Csr_matrix|`IndexMixin|`Object] Np.Obj.t))
let predict ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let predict_proba ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject

let classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let effective_metric_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "effective_metric_" with
  | None -> failwith "attribute effective_metric_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let effective_metric_ self = match effective_metric_opt self with
  | None -> raise Not_found
  | Some x -> x

let effective_metric_params_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "effective_metric_params_" with
  | None -> failwith "attribute effective_metric_params_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let effective_metric_params_ self = match effective_metric_params_opt self with
  | None -> raise Not_found
  | Some x -> x

let outputs_2d_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "outputs_2d_" with
  | None -> failwith "attribute outputs_2d_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Bool.to_bool x)

let outputs_2d_ self = match outputs_2d_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module KNeighborsRegressor = struct
type tag = [`KNeighborsRegressor]
type t = [`BaseEstimator | `KNeighborsMixin | `KNeighborsRegressor | `MultiOutputMixin | `NeighborsBase | `Object | `RegressorMixin | `SupervisedFloatMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_neighbors x = (x :> [`NeighborsBase] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
let as_k_neighbors x = (x :> [`KNeighborsMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_supervised_float x = (x :> [`SupervisedFloatMixin] Obj.t)
                  let create ?n_neighbors ?weights ?algorithm ?leaf_size ?p ?metric ?metric_params ?n_jobs ?kwargs () =
                     Py.Module.get_function_with_keywords __wrap_namespace "KNeighborsRegressor"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("weights", Wrap_utils.Option.map weights (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Auto -> Py.String.of_string "auto"
| `Ball_tree -> Py.String.of_string "ball_tree"
| `Kd_tree -> Py.String.of_string "kd_tree"
| `Brute -> Py.String.of_string "brute"
)); ("leaf_size", Wrap_utils.Option.map leaf_size Py.Int.of_int); ("p", Wrap_utils.Option.map p Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("metric_params", Wrap_utils.Option.map metric_params Dict.to_pyobject); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int)]) (match kwargs with None -> [] | Some x -> x))
                       |> of_pyobject
                  let fit ~x ~y self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fit"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let kneighbors ?x ?n_neighbors self =
   Py.Module.get_function_with_keywords (to_pyobject self) "kneighbors"
     [||]
     (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("return_distance", Some(true |> Py.Bool.of_bool))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let kneighbors_graph ?x ?n_neighbors ?mode self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "kneighbors_graph"
                       [||]
                       (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Connectivity -> Py.String.of_string "connectivity"
| `Distance -> Py.String.of_string "distance"
))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Csr_matrix|`IndexMixin|`Object] Np.Obj.t))
let predict ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject

let effective_metric_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "effective_metric_" with
  | None -> failwith "attribute effective_metric_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let effective_metric_ self = match effective_metric_opt self with
  | None -> raise Not_found
  | Some x -> x

let effective_metric_params_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "effective_metric_params_" with
  | None -> failwith "attribute effective_metric_params_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let effective_metric_params_ self = match effective_metric_params_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module KNeighborsTransformer = struct
type tag = [`KNeighborsTransformer]
type t = [`BaseEstimator | `KNeighborsMixin | `KNeighborsTransformer | `MultiOutputMixin | `NeighborsBase | `Object | `TransformerMixin | `UnsupervisedMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_neighbors x = (x :> [`NeighborsBase] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
let as_k_neighbors x = (x :> [`KNeighborsMixin] Obj.t)
let as_unsupervised x = (x :> [`UnsupervisedMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
                  let create ?mode ?n_neighbors ?algorithm ?leaf_size ?metric ?p ?metric_params ?n_jobs () =
                     Py.Module.get_function_with_keywords __wrap_namespace "KNeighborsTransformer"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Distance -> Py.String.of_string "distance"
| `Connectivity -> Py.String.of_string "connectivity"
)); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Auto -> Py.String.of_string "auto"
| `Ball_tree -> Py.String.of_string "ball_tree"
| `Kd_tree -> Py.String.of_string "kd_tree"
| `Brute -> Py.String.of_string "brute"
)); ("leaf_size", Wrap_utils.Option.map leaf_size Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("p", Wrap_utils.Option.map p Py.Int.of_int); ("metric_params", Wrap_utils.Option.map metric_params Dict.to_pyobject); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int)])
                       |> of_pyobject
                  let fit ?y ~x self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])
                       |> of_pyobject
let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let kneighbors ?x ?n_neighbors self =
   Py.Module.get_function_with_keywords (to_pyobject self) "kneighbors"
     [||]
     (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("return_distance", Some(true |> Py.Bool.of_bool))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let kneighbors_graph ?x ?n_neighbors ?mode self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "kneighbors_graph"
                       [||]
                       (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Connectivity -> Py.String.of_string "connectivity"
| `Distance -> Py.String.of_string "distance"
))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Csr_matrix|`IndexMixin|`Object] Np.Obj.t))
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
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module KernelDensity = struct
type tag = [`KernelDensity]
type t = [`BaseEstimator | `KernelDensity | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let create ?bandwidth ?algorithm ?kernel ?metric ?atol ?rtol ?breadth_first ?leaf_size ?metric_params () =
   Py.Module.get_function_with_keywords __wrap_namespace "KernelDensity"
     [||]
     (Wrap_utils.keyword_args [("bandwidth", Wrap_utils.Option.map bandwidth Py.Float.of_float); ("algorithm", Wrap_utils.Option.map algorithm Py.String.of_string); ("kernel", Wrap_utils.Option.map kernel Py.String.of_string); ("metric", Wrap_utils.Option.map metric Py.String.of_string); ("atol", Wrap_utils.Option.map atol Py.Float.of_float); ("rtol", Wrap_utils.Option.map rtol Py.Float.of_float); ("breadth_first", Wrap_utils.Option.map breadth_first Py.Bool.of_bool); ("leaf_size", Wrap_utils.Option.map leaf_size Py.Int.of_int); ("metric_params", Wrap_utils.Option.map metric_params Dict.to_pyobject)])
     |> of_pyobject
let fit ?y ?sample_weight ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let sample ?n_samples ?random_state self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sample"
     [||]
     (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let score ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let score_samples ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score_samples"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LocalOutlierFactor = struct
type tag = [`LocalOutlierFactor]
type t = [`BaseEstimator | `KNeighborsMixin | `LocalOutlierFactor | `MultiOutputMixin | `NeighborsBase | `Object | `OutlierMixin | `UnsupervisedMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_neighbors x = (x :> [`NeighborsBase] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
let as_k_neighbors x = (x :> [`KNeighborsMixin] Obj.t)
let as_unsupervised x = (x :> [`UnsupervisedMixin] Obj.t)
let as_outlier x = (x :> [`OutlierMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
                  let create ?n_neighbors ?algorithm ?leaf_size ?metric ?p ?metric_params ?contamination ?novelty ?n_jobs () =
                     Py.Module.get_function_with_keywords __wrap_namespace "LocalOutlierFactor"
                       [||]
                       (Wrap_utils.keyword_args [("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Auto -> Py.String.of_string "auto"
| `Ball_tree -> Py.String.of_string "ball_tree"
| `Kd_tree -> Py.String.of_string "kd_tree"
| `Brute -> Py.String.of_string "brute"
)); ("leaf_size", Wrap_utils.Option.map leaf_size Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("p", Wrap_utils.Option.map p Py.Int.of_int); ("metric_params", Wrap_utils.Option.map metric_params Dict.to_pyobject); ("contamination", Wrap_utils.Option.map contamination (function
| `F x -> Py.Float.of_float x
| `Auto -> Py.String.of_string "auto"
)); ("novelty", Wrap_utils.Option.map novelty Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int)])
                       |> of_pyobject
                  let fit ?y ~x self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])
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
let kneighbors ?x ?n_neighbors self =
   Py.Module.get_function_with_keywords (to_pyobject self) "kneighbors"
     [||]
     (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("return_distance", Some(true |> Py.Bool.of_bool))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let kneighbors_graph ?x ?n_neighbors ?mode self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "kneighbors_graph"
                       [||]
                       (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Connectivity -> Py.String.of_string "connectivity"
| `Distance -> Py.String.of_string "distance"
))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Csr_matrix|`IndexMixin|`Object] Np.Obj.t))
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject

let negative_outlier_factor_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "negative_outlier_factor_" with
  | None -> failwith "attribute negative_outlier_factor_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let negative_outlier_factor_ self = match negative_outlier_factor_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_neighbors_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_neighbors_" with
  | None -> failwith "attribute n_neighbors_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_neighbors_ self = match n_neighbors_opt self with
  | None -> raise Not_found
  | Some x -> x

let offset_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "offset_" with
  | None -> failwith "attribute offset_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let offset_ self = match offset_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module NearestCentroid = struct
type tag = [`NearestCentroid]
type t = [`BaseEstimator | `ClassifierMixin | `NearestCentroid | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
                  let create ?metric ?shrink_threshold () =
                     Py.Module.get_function_with_keywords __wrap_namespace "NearestCentroid"
                       [||]
                       (Wrap_utils.keyword_args [("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("shrink_threshold", Wrap_utils.Option.map shrink_threshold Py.Float.of_float)])
                       |> of_pyobject
let fit ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
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
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject

let centroids_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "centroids_" with
  | None -> failwith "attribute centroids_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let centroids_ self = match centroids_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module NearestNeighbors = struct
type tag = [`NearestNeighbors]
type t = [`BaseEstimator | `KNeighborsMixin | `MultiOutputMixin | `NearestNeighbors | `NeighborsBase | `Object | `RadiusNeighborsMixin | `UnsupervisedMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_radius_neighbors x = (x :> [`RadiusNeighborsMixin] Obj.t)
let as_neighbors x = (x :> [`NeighborsBase] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
let as_k_neighbors x = (x :> [`KNeighborsMixin] Obj.t)
let as_unsupervised x = (x :> [`UnsupervisedMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
                  let create ?n_neighbors ?radius ?algorithm ?leaf_size ?metric ?p ?metric_params ?n_jobs () =
                     Py.Module.get_function_with_keywords __wrap_namespace "NearestNeighbors"
                       [||]
                       (Wrap_utils.keyword_args [("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("radius", Wrap_utils.Option.map radius Py.Float.of_float); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Auto -> Py.String.of_string "auto"
| `Ball_tree -> Py.String.of_string "ball_tree"
| `Kd_tree -> Py.String.of_string "kd_tree"
| `Brute -> Py.String.of_string "brute"
)); ("leaf_size", Wrap_utils.Option.map leaf_size Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("p", Wrap_utils.Option.map p Py.Int.of_int); ("metric_params", Wrap_utils.Option.map metric_params Dict.to_pyobject); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int)])
                       |> of_pyobject
                  let fit ?y ~x self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])
                       |> of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let kneighbors ?x ?n_neighbors self =
   Py.Module.get_function_with_keywords (to_pyobject self) "kneighbors"
     [||]
     (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("return_distance", Some(true |> Py.Bool.of_bool))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let kneighbors_graph ?x ?n_neighbors ?mode self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "kneighbors_graph"
                       [||]
                       (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Connectivity -> Py.String.of_string "connectivity"
| `Distance -> Py.String.of_string "distance"
))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Csr_matrix|`IndexMixin|`Object] Np.Obj.t))
let radius_neighbors ?x ?radius ?sort_results self =
   Py.Module.get_function_with_keywords (to_pyobject self) "radius_neighbors"
     [||]
     (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("radius", Wrap_utils.Option.map radius Py.Float.of_float); ("return_distance", Some(true |> Py.Bool.of_bool)); ("sort_results", Wrap_utils.Option.map sort_results Py.Bool.of_bool)])
     |> (fun x -> ((Np.Numpy.Ndarray.List.of_pyobject (Py.Tuple.get x 0)), (Np.Numpy.Ndarray.List.of_pyobject (Py.Tuple.get x 1))))
                  let radius_neighbors_graph ?x ?radius ?mode ?sort_results self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "radius_neighbors_graph"
                       [||]
                       (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("radius", Wrap_utils.Option.map radius Py.Float.of_float); ("mode", Wrap_utils.Option.map mode (function
| `Connectivity -> Py.String.of_string "connectivity"
| `Distance -> Py.String.of_string "distance"
)); ("sort_results", Wrap_utils.Option.map sort_results Py.Bool.of_bool)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Csr_matrix|`IndexMixin|`Object] Np.Obj.t))
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject

let effective_metric_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "effective_metric_" with
  | None -> failwith "attribute effective_metric_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.String.to_string x)

let effective_metric_ self = match effective_metric_opt self with
  | None -> raise Not_found
  | Some x -> x

let effective_metric_params_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "effective_metric_params_" with
  | None -> failwith "attribute effective_metric_params_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let effective_metric_params_ self = match effective_metric_params_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module NeighborhoodComponentsAnalysis = struct
type tag = [`NeighborhoodComponentsAnalysis]
type t = [`BaseEstimator | `NeighborhoodComponentsAnalysis | `Object | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
                  let create ?n_components ?init ?warm_start ?max_iter ?tol ?callback ?verbose ?random_state () =
                     Py.Module.get_function_with_keywords __wrap_namespace "NeighborhoodComponentsAnalysis"
                       [||]
                       (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("init", Wrap_utils.Option.map init (function
| `S x -> Py.String.of_string x
| `Arr x -> Np.Obj.to_pyobject x
)); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("callback", callback); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
                       |> of_pyobject
let fit ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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

let components_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "components_" with
  | None -> failwith "attribute components_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let components_ self = match components_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x

let random_state_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "random_state_" with
  | None -> failwith "attribute random_state_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let random_state_ self = match random_state_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RadiusNeighborsClassifier = struct
type tag = [`RadiusNeighborsClassifier]
type t = [`BaseEstimator | `ClassifierMixin | `MultiOutputMixin | `NeighborsBase | `Object | `RadiusNeighborsClassifier | `RadiusNeighborsMixin | `SupervisedIntegerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_radius_neighbors x = (x :> [`RadiusNeighborsMixin] Obj.t)
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_neighbors x = (x :> [`NeighborsBase] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_supervised_integer x = (x :> [`SupervisedIntegerMixin] Obj.t)
                  let create ?radius ?weights ?algorithm ?leaf_size ?p ?metric ?outlier_label ?metric_params ?n_jobs ?kwargs () =
                     Py.Module.get_function_with_keywords __wrap_namespace "RadiusNeighborsClassifier"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("radius", Wrap_utils.Option.map radius Py.Float.of_float); ("weights", Wrap_utils.Option.map weights (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Auto -> Py.String.of_string "auto"
| `Ball_tree -> Py.String.of_string "ball_tree"
| `Kd_tree -> Py.String.of_string "kd_tree"
| `Brute -> Py.String.of_string "brute"
)); ("leaf_size", Wrap_utils.Option.map leaf_size Py.Int.of_int); ("p", Wrap_utils.Option.map p Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("outlier_label", Wrap_utils.Option.map outlier_label (function
| `Most_frequent -> Py.String.of_string "most_frequent"
| `Manual_label x -> Wrap_utils.id x
)); ("metric_params", Wrap_utils.Option.map metric_params Dict.to_pyobject); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int)]) (match kwargs with None -> [] | Some x -> x))
                       |> of_pyobject
                  let fit ~x ~y self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fit"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> of_pyobject
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
let predict_proba ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let radius_neighbors ?x ?radius ?sort_results self =
   Py.Module.get_function_with_keywords (to_pyobject self) "radius_neighbors"
     [||]
     (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("radius", Wrap_utils.Option.map radius Py.Float.of_float); ("return_distance", Some(true |> Py.Bool.of_bool)); ("sort_results", Wrap_utils.Option.map sort_results Py.Bool.of_bool)])
     |> (fun x -> ((Np.Numpy.Ndarray.List.of_pyobject (Py.Tuple.get x 0)), (Np.Numpy.Ndarray.List.of_pyobject (Py.Tuple.get x 1))))
                  let radius_neighbors_graph ?x ?radius ?mode ?sort_results self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "radius_neighbors_graph"
                       [||]
                       (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("radius", Wrap_utils.Option.map radius Py.Float.of_float); ("mode", Wrap_utils.Option.map mode (function
| `Connectivity -> Py.String.of_string "connectivity"
| `Distance -> Py.String.of_string "distance"
)); ("sort_results", Wrap_utils.Option.map sort_results Py.Bool.of_bool)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Csr_matrix|`IndexMixin|`Object] Np.Obj.t))
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject

let classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let effective_metric_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "effective_metric_" with
  | None -> failwith "attribute effective_metric_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let effective_metric_ self = match effective_metric_opt self with
  | None -> raise Not_found
  | Some x -> x

let effective_metric_params_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "effective_metric_params_" with
  | None -> failwith "attribute effective_metric_params_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let effective_metric_params_ self = match effective_metric_params_opt self with
  | None -> raise Not_found
  | Some x -> x

let outputs_2d_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "outputs_2d_" with
  | None -> failwith "attribute outputs_2d_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Bool.to_bool x)

let outputs_2d_ self = match outputs_2d_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RadiusNeighborsRegressor = struct
type tag = [`RadiusNeighborsRegressor]
type t = [`BaseEstimator | `MultiOutputMixin | `NeighborsBase | `Object | `RadiusNeighborsMixin | `RadiusNeighborsRegressor | `RegressorMixin | `SupervisedFloatMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_radius_neighbors x = (x :> [`RadiusNeighborsMixin] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_neighbors x = (x :> [`NeighborsBase] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_supervised_float x = (x :> [`SupervisedFloatMixin] Obj.t)
                  let create ?radius ?weights ?algorithm ?leaf_size ?p ?metric ?metric_params ?n_jobs ?kwargs () =
                     Py.Module.get_function_with_keywords __wrap_namespace "RadiusNeighborsRegressor"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("radius", Wrap_utils.Option.map radius Py.Float.of_float); ("weights", Wrap_utils.Option.map weights (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Auto -> Py.String.of_string "auto"
| `Ball_tree -> Py.String.of_string "ball_tree"
| `Kd_tree -> Py.String.of_string "kd_tree"
| `Brute -> Py.String.of_string "brute"
)); ("leaf_size", Wrap_utils.Option.map leaf_size Py.Int.of_int); ("p", Wrap_utils.Option.map p Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("metric_params", Wrap_utils.Option.map metric_params Dict.to_pyobject); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int)]) (match kwargs with None -> [] | Some x -> x))
                       |> of_pyobject
                  let fit ~x ~y self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fit"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> of_pyobject
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
let radius_neighbors ?x ?radius ?sort_results self =
   Py.Module.get_function_with_keywords (to_pyobject self) "radius_neighbors"
     [||]
     (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("radius", Wrap_utils.Option.map radius Py.Float.of_float); ("return_distance", Some(true |> Py.Bool.of_bool)); ("sort_results", Wrap_utils.Option.map sort_results Py.Bool.of_bool)])
     |> (fun x -> ((Np.Numpy.Ndarray.List.of_pyobject (Py.Tuple.get x 0)), (Np.Numpy.Ndarray.List.of_pyobject (Py.Tuple.get x 1))))
                  let radius_neighbors_graph ?x ?radius ?mode ?sort_results self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "radius_neighbors_graph"
                       [||]
                       (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("radius", Wrap_utils.Option.map radius Py.Float.of_float); ("mode", Wrap_utils.Option.map mode (function
| `Connectivity -> Py.String.of_string "connectivity"
| `Distance -> Py.String.of_string "distance"
)); ("sort_results", Wrap_utils.Option.map sort_results Py.Bool.of_bool)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Csr_matrix|`IndexMixin|`Object] Np.Obj.t))
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject

let effective_metric_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "effective_metric_" with
  | None -> failwith "attribute effective_metric_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let effective_metric_ self = match effective_metric_opt self with
  | None -> raise Not_found
  | Some x -> x

let effective_metric_params_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "effective_metric_params_" with
  | None -> failwith "attribute effective_metric_params_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let effective_metric_params_ self = match effective_metric_params_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RadiusNeighborsTransformer = struct
type tag = [`RadiusNeighborsTransformer]
type t = [`BaseEstimator | `MultiOutputMixin | `NeighborsBase | `Object | `RadiusNeighborsMixin | `RadiusNeighborsTransformer | `TransformerMixin | `UnsupervisedMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_radius_neighbors x = (x :> [`RadiusNeighborsMixin] Obj.t)
let as_neighbors x = (x :> [`NeighborsBase] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
let as_unsupervised x = (x :> [`UnsupervisedMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
                  let create ?mode ?radius ?algorithm ?leaf_size ?metric ?p ?metric_params ?n_jobs () =
                     Py.Module.get_function_with_keywords __wrap_namespace "RadiusNeighborsTransformer"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Distance -> Py.String.of_string "distance"
| `Connectivity -> Py.String.of_string "connectivity"
)); ("radius", Wrap_utils.Option.map radius Py.Float.of_float); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Auto -> Py.String.of_string "auto"
| `Ball_tree -> Py.String.of_string "ball_tree"
| `Kd_tree -> Py.String.of_string "kd_tree"
| `Brute -> Py.String.of_string "brute"
)); ("leaf_size", Wrap_utils.Option.map leaf_size Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("p", Wrap_utils.Option.map p Py.Int.of_int); ("metric_params", Wrap_utils.Option.map metric_params Dict.to_pyobject); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int)])
                       |> of_pyobject
                  let fit ?y ~x self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])
                       |> of_pyobject
let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let radius_neighbors ?x ?radius ?sort_results self =
   Py.Module.get_function_with_keywords (to_pyobject self) "radius_neighbors"
     [||]
     (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("radius", Wrap_utils.Option.map radius Py.Float.of_float); ("return_distance", Some(true |> Py.Bool.of_bool)); ("sort_results", Wrap_utils.Option.map sort_results Py.Bool.of_bool)])
     |> (fun x -> ((Np.Numpy.Ndarray.List.of_pyobject (Py.Tuple.get x 0)), (Np.Numpy.Ndarray.List.of_pyobject (Py.Tuple.get x 1))))
                  let radius_neighbors_graph ?x ?radius ?mode ?sort_results self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "radius_neighbors_graph"
                       [||]
                       (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("radius", Wrap_utils.Option.map radius Py.Float.of_float); ("mode", Wrap_utils.Option.map mode (function
| `Connectivity -> Py.String.of_string "connectivity"
| `Distance -> Py.String.of_string "distance"
)); ("sort_results", Wrap_utils.Option.map sort_results Py.Bool.of_bool)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Csr_matrix|`IndexMixin|`Object] Np.Obj.t))
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
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let kneighbors_graph ?mode ?metric ?p ?metric_params ?include_self ?n_jobs ~x ~n_neighbors () =
                     Py.Module.get_function_with_keywords __wrap_namespace "kneighbors_graph"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Connectivity -> Py.String.of_string "connectivity"
| `Distance -> Py.String.of_string "distance"
)); ("metric", Wrap_utils.Option.map metric Py.String.of_string); ("p", Wrap_utils.Option.map p Py.Int.of_int); ("metric_params", Wrap_utils.Option.map metric_params Dict.to_pyobject); ("include_self", Wrap_utils.Option.map include_self (function
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("X", Some(x |> (function
| `BallTree x -> Wrap_utils.id x
| `Arr x -> Np.Obj.to_pyobject x
))); ("n_neighbors", Some(n_neighbors |> Py.Int.of_int))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Csr_matrix|`IndexMixin|`Object] Np.Obj.t))
                  let radius_neighbors_graph ?mode ?metric ?p ?metric_params ?include_self ?n_jobs ~x ~radius () =
                     Py.Module.get_function_with_keywords __wrap_namespace "radius_neighbors_graph"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Connectivity -> Py.String.of_string "connectivity"
| `Distance -> Py.String.of_string "distance"
)); ("metric", Wrap_utils.Option.map metric Py.String.of_string); ("p", Wrap_utils.Option.map p Py.Int.of_int); ("metric_params", Wrap_utils.Option.map metric_params Dict.to_pyobject); ("include_self", Wrap_utils.Option.map include_self (function
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("X", Some(x |> (function
| `BallTree x -> Wrap_utils.id x
| `Arr x -> Np.Obj.to_pyobject x
))); ("radius", Some(radius |> Py.Float.of_float))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Csr_matrix|`IndexMixin|`Object] Np.Obj.t))
