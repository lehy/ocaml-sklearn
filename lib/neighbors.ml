let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.neighbors"

module KNeighborsClassifier = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_neighbors ?weights ?algorithm ?leaf_size ?p ?metric ?metric_params ?n_jobs ?kwargs () =
                     Py.Module.get_function_with_keywords ns "KNeighborsClassifier"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("weights", Wrap_utils.Option.map weights (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Auto -> Py.String.of_string "auto"
| `Ball_tree -> Py.String.of_string "ball_tree"
| `Kd_tree -> Py.String.of_string "kd_tree"
| `Brute -> Py.String.of_string "brute"
)); ("leaf_size", Wrap_utils.Option.map leaf_size Py.Int.of_int); ("p", Wrap_utils.Option.map p Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("metric_params", metric_params); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
))]) (match kwargs with None -> [] | Some x -> x))

                  let fit ~x ~y self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("y", Some(y |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let kneighbors ?x ?n_neighbors self =
   Py.Module.get_function_with_keywords self "kneighbors"
     [||]
     (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Ndarray.to_pyobject); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("return_distance", Some(true |> Py.Bool.of_bool))])
     |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
                  let kneighbors_graph ?x ?n_neighbors ?mode self =
                     Py.Module.get_function_with_keywords self "kneighbors_graph"
                       [||]
                       (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Ndarray.to_pyobject); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Connectivity -> Py.String.of_string "connectivity"
| `Distance -> Py.String.of_string "distance"
))])
                       |> Csr_matrix.of_pyobject
let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let predict_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let classes_ self =
  match Py.Object.get_attr_string self "classes_" with
| None -> raise (Wrap_utils.Attribute_not_found "classes_")
| Some x -> Ndarray.of_pyobject x
let effective_metric_ self =
  match Py.Object.get_attr_string self "effective_metric_" with
| None -> raise (Wrap_utils.Attribute_not_found "effective_metric_")
| Some x -> Wrap_utils.id x
let effective_metric_params_ self =
  match Py.Object.get_attr_string self "effective_metric_params_" with
| None -> raise (Wrap_utils.Attribute_not_found "effective_metric_params_")
| Some x -> Wrap_utils.id x
let outputs_2d_ self =
  match Py.Object.get_attr_string self "outputs_2d_" with
| None -> raise (Wrap_utils.Attribute_not_found "outputs_2d_")
| Some x -> Py.Bool.to_bool x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module KNeighborsRegressor = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_neighbors ?weights ?algorithm ?leaf_size ?p ?metric ?metric_params ?n_jobs ?kwargs () =
                     Py.Module.get_function_with_keywords ns "KNeighborsRegressor"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("weights", Wrap_utils.Option.map weights (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Auto -> Py.String.of_string "auto"
| `Ball_tree -> Py.String.of_string "ball_tree"
| `Kd_tree -> Py.String.of_string "kd_tree"
| `Brute -> Py.String.of_string "brute"
)); ("leaf_size", Wrap_utils.Option.map leaf_size Py.Int.of_int); ("p", Wrap_utils.Option.map p Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("metric_params", metric_params); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
))]) (match kwargs with None -> [] | Some x -> x))

                  let fit ~x ~y self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("y", Some(y |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let kneighbors ?x ?n_neighbors self =
   Py.Module.get_function_with_keywords self "kneighbors"
     [||]
     (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Ndarray.to_pyobject); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("return_distance", Some(true |> Py.Bool.of_bool))])
     |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
                  let kneighbors_graph ?x ?n_neighbors ?mode self =
                     Py.Module.get_function_with_keywords self "kneighbors_graph"
                       [||]
                       (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Ndarray.to_pyobject); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Connectivity -> Py.String.of_string "connectivity"
| `Distance -> Py.String.of_string "distance"
))])
                       |> Csr_matrix.of_pyobject
let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let effective_metric_ self =
  match Py.Object.get_attr_string self "effective_metric_" with
| None -> raise (Wrap_utils.Attribute_not_found "effective_metric_")
| Some x -> Wrap_utils.id x
let effective_metric_params_ self =
  match Py.Object.get_attr_string self "effective_metric_params_" with
| None -> raise (Wrap_utils.Attribute_not_found "effective_metric_params_")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module KNeighborsTransformer = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?mode ?n_neighbors ?algorithm ?leaf_size ?metric ?p ?metric_params ?n_jobs () =
                     Py.Module.get_function_with_keywords ns "KNeighborsTransformer"
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
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("p", Wrap_utils.Option.map p Py.Int.of_int); ("metric_params", metric_params); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int)])

                  let fit ?y ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])

let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let kneighbors ?x ?n_neighbors self =
   Py.Module.get_function_with_keywords self "kneighbors"
     [||]
     (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Ndarray.to_pyobject); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("return_distance", Some(true |> Py.Bool.of_bool))])
     |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
                  let kneighbors_graph ?x ?n_neighbors ?mode self =
                     Py.Module.get_function_with_keywords self "kneighbors_graph"
                       [||]
                       (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Ndarray.to_pyobject); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Connectivity -> Py.String.of_string "connectivity"
| `Distance -> Py.String.of_string "distance"
))])
                       |> Csr_matrix.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module KernelDensity = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?bandwidth ?algorithm ?kernel ?metric ?atol ?rtol ?breadth_first ?leaf_size ?metric_params () =
   Py.Module.get_function_with_keywords ns "KernelDensity"
     [||]
     (Wrap_utils.keyword_args [("bandwidth", Wrap_utils.Option.map bandwidth Py.Float.of_float); ("algorithm", Wrap_utils.Option.map algorithm Py.String.of_string); ("kernel", Wrap_utils.Option.map kernel Py.String.of_string); ("metric", Wrap_utils.Option.map metric Py.String.of_string); ("atol", Wrap_utils.Option.map atol Py.Float.of_float); ("rtol", Wrap_utils.Option.map rtol Py.Float.of_float); ("breadth_first", Wrap_utils.Option.map breadth_first Py.Bool.of_bool); ("leaf_size", Wrap_utils.Option.map leaf_size Py.Int.of_int); ("metric_params", metric_params)])

let fit ?y ?sample_weight ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("sample_weight", sample_weight); ("X", Some(x |> Ndarray.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

                  let sample ?n_samples ?random_state self =
                     Py.Module.get_function_with_keywords self "sample"
                       [||]
                       (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])
                       |> Ndarray.of_pyobject
let score ?y ~x self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let score_samples ~x self =
   Py.Module.get_function_with_keywords self "score_samples"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LocalOutlierFactor = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_neighbors ?algorithm ?leaf_size ?metric ?p ?metric_params ?contamination ?novelty ?n_jobs () =
                     Py.Module.get_function_with_keywords ns "LocalOutlierFactor"
                       [||]
                       (Wrap_utils.keyword_args [("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Auto -> Py.String.of_string "auto"
| `Ball_tree -> Py.String.of_string "ball_tree"
| `Kd_tree -> Py.String.of_string "kd_tree"
| `Brute -> Py.String.of_string "brute"
)); ("leaf_size", Wrap_utils.Option.map leaf_size Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("p", Wrap_utils.Option.map p Py.Int.of_int); ("metric_params", metric_params); ("contamination", Wrap_utils.Option.map contamination (function
| `Auto -> Py.String.of_string "auto"
| `Float x -> Py.Float.of_float x
)); ("novelty", Wrap_utils.Option.map novelty Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
))])

                  let fit ?y ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])

let fit_predict ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_predict"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let kneighbors ?x ?n_neighbors self =
   Py.Module.get_function_with_keywords self "kneighbors"
     [||]
     (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Ndarray.to_pyobject); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("return_distance", Some(true |> Py.Bool.of_bool))])
     |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
                  let kneighbors_graph ?x ?n_neighbors ?mode self =
                     Py.Module.get_function_with_keywords self "kneighbors_graph"
                       [||]
                       (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Ndarray.to_pyobject); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Connectivity -> Py.String.of_string "connectivity"
| `Distance -> Py.String.of_string "distance"
))])
                       |> Csr_matrix.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let negative_outlier_factor_ self =
  match Py.Object.get_attr_string self "negative_outlier_factor_" with
| None -> raise (Wrap_utils.Attribute_not_found "negative_outlier_factor_")
| Some x -> Ndarray.of_pyobject x
let n_neighbors_ self =
  match Py.Object.get_attr_string self "n_neighbors_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_neighbors_")
| Some x -> Py.Int.to_int x
let offset_ self =
  match Py.Object.get_attr_string self "offset_" with
| None -> raise (Wrap_utils.Attribute_not_found "offset_")
| Some x -> Py.Float.to_float x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module NearestCentroid = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?metric ?shrink_threshold () =
                     Py.Module.get_function_with_keywords ns "NearestCentroid"
                       [||]
                       (Wrap_utils.keyword_args [("metric", Wrap_utils.Option.map metric (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("shrink_threshold", Wrap_utils.Option.map shrink_threshold Py.Float.of_float)])

                  let fit ~x ~y self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
))); ("y", Some(y |> Ndarray.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let centroids_ self =
  match Py.Object.get_attr_string self "centroids_" with
| None -> raise (Wrap_utils.Attribute_not_found "centroids_")
| Some x -> Ndarray.of_pyobject x
let classes_ self =
  match Py.Object.get_attr_string self "classes_" with
| None -> raise (Wrap_utils.Attribute_not_found "classes_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module NearestNeighbors = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_neighbors ?radius ?algorithm ?leaf_size ?metric ?p ?metric_params ?n_jobs () =
                     Py.Module.get_function_with_keywords ns "NearestNeighbors"
                       [||]
                       (Wrap_utils.keyword_args [("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("radius", Wrap_utils.Option.map radius Py.Float.of_float); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Auto -> Py.String.of_string "auto"
| `Ball_tree -> Py.String.of_string "ball_tree"
| `Kd_tree -> Py.String.of_string "kd_tree"
| `Brute -> Py.String.of_string "brute"
)); ("leaf_size", Wrap_utils.Option.map leaf_size Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("p", Wrap_utils.Option.map p Py.Int.of_int); ("metric_params", metric_params); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
))])

                  let fit ?y ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let kneighbors ?x ?n_neighbors self =
   Py.Module.get_function_with_keywords self "kneighbors"
     [||]
     (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Ndarray.to_pyobject); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("return_distance", Some(true |> Py.Bool.of_bool))])
     |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
                  let kneighbors_graph ?x ?n_neighbors ?mode self =
                     Py.Module.get_function_with_keywords self "kneighbors_graph"
                       [||]
                       (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Ndarray.to_pyobject); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Connectivity -> Py.String.of_string "connectivity"
| `Distance -> Py.String.of_string "distance"
))])
                       |> Csr_matrix.of_pyobject
                  let radius_neighbors ?x ?radius ?sort_results self =
                     Py.Module.get_function_with_keywords self "radius_neighbors"
                       [||]
                       (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("radius", Wrap_utils.Option.map radius Py.Float.of_float); ("return_distance", Some(true |> Py.Bool.of_bool)); ("sort_results", Wrap_utils.Option.map sort_results Py.Bool.of_bool)])
                       |> (fun x -> (((fun py -> let len = Py.Sequence.length py in Array.init len (fun i -> Ndarray.of_pyobject (Py.Sequence.get_item py i))) (Py.Tuple.get x 0)), ((fun py -> let len = Py.Sequence.length py in Array.init len (fun i -> Ndarray.of_pyobject (Py.Sequence.get_item py i))) (Py.Tuple.get x 1))))
                  let radius_neighbors_graph ?x ?radius ?mode ?sort_results self =
                     Py.Module.get_function_with_keywords self "radius_neighbors_graph"
                       [||]
                       (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Ndarray.to_pyobject); ("radius", Wrap_utils.Option.map radius Py.Float.of_float); ("mode", Wrap_utils.Option.map mode (function
| `Connectivity -> Py.String.of_string "connectivity"
| `Distance -> Py.String.of_string "distance"
)); ("sort_results", Wrap_utils.Option.map sort_results Py.Bool.of_bool)])
                       |> Csr_matrix.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let effective_metric_ self =
  match Py.Object.get_attr_string self "effective_metric_" with
| None -> raise (Wrap_utils.Attribute_not_found "effective_metric_")
| Some x -> Py.String.to_string x
let effective_metric_params_ self =
  match Py.Object.get_attr_string self "effective_metric_params_" with
| None -> raise (Wrap_utils.Attribute_not_found "effective_metric_params_")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module NeighborhoodComponentsAnalysis = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_components ?init ?warm_start ?max_iter ?tol ?callback ?verbose ?random_state () =
                     Py.Module.get_function_with_keywords ns "NeighborhoodComponentsAnalysis"
                       [||]
                       (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("init", Wrap_utils.Option.map init (function
| `String x -> Py.String.of_string x
| `Ndarray x -> Ndarray.to_pyobject x
)); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("callback", callback); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
))])

let fit ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let components_ self =
  match Py.Object.get_attr_string self "components_" with
| None -> raise (Wrap_utils.Attribute_not_found "components_")
| Some x -> Ndarray.of_pyobject x
let n_iter_ self =
  match Py.Object.get_attr_string self "n_iter_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_iter_")
| Some x -> Py.Int.to_int x
let random_state_ self =
  match Py.Object.get_attr_string self "random_state_" with
| None -> raise (Wrap_utils.Attribute_not_found "random_state_")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RadiusNeighborsClassifier = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?radius ?weights ?algorithm ?leaf_size ?p ?metric ?outlier_label ?metric_params ?n_jobs ?kwargs () =
                     Py.Module.get_function_with_keywords ns "RadiusNeighborsClassifier"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("radius", Wrap_utils.Option.map radius Py.Float.of_float); ("weights", Wrap_utils.Option.map weights (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Auto -> Py.String.of_string "auto"
| `Ball_tree -> Py.String.of_string "ball_tree"
| `Kd_tree -> Py.String.of_string "kd_tree"
| `Brute -> Py.String.of_string "brute"
)); ("leaf_size", Wrap_utils.Option.map leaf_size Py.Int.of_int); ("p", Wrap_utils.Option.map p Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("outlier_label", Wrap_utils.Option.map outlier_label (function
| `Most_frequent -> Py.String.of_string "most_frequent"
| `PyObject x -> Wrap_utils.id x
)); ("metric_params", metric_params); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
))]) (match kwargs with None -> [] | Some x -> x))

                  let fit ~x ~y self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("y", Some(y |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let predict_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
                  let radius_neighbors ?x ?radius ?sort_results self =
                     Py.Module.get_function_with_keywords self "radius_neighbors"
                       [||]
                       (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("radius", Wrap_utils.Option.map radius Py.Float.of_float); ("return_distance", Some(true |> Py.Bool.of_bool)); ("sort_results", Wrap_utils.Option.map sort_results Py.Bool.of_bool)])
                       |> (fun x -> (((fun py -> let len = Py.Sequence.length py in Array.init len (fun i -> Ndarray.of_pyobject (Py.Sequence.get_item py i))) (Py.Tuple.get x 0)), ((fun py -> let len = Py.Sequence.length py in Array.init len (fun i -> Ndarray.of_pyobject (Py.Sequence.get_item py i))) (Py.Tuple.get x 1))))
                  let radius_neighbors_graph ?x ?radius ?mode ?sort_results self =
                     Py.Module.get_function_with_keywords self "radius_neighbors_graph"
                       [||]
                       (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Ndarray.to_pyobject); ("radius", Wrap_utils.Option.map radius Py.Float.of_float); ("mode", Wrap_utils.Option.map mode (function
| `Connectivity -> Py.String.of_string "connectivity"
| `Distance -> Py.String.of_string "distance"
)); ("sort_results", Wrap_utils.Option.map sort_results Py.Bool.of_bool)])
                       |> Csr_matrix.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let classes_ self =
  match Py.Object.get_attr_string self "classes_" with
| None -> raise (Wrap_utils.Attribute_not_found "classes_")
| Some x -> Ndarray.of_pyobject x
let effective_metric_ self =
  match Py.Object.get_attr_string self "effective_metric_" with
| None -> raise (Wrap_utils.Attribute_not_found "effective_metric_")
| Some x -> Wrap_utils.id x
let effective_metric_params_ self =
  match Py.Object.get_attr_string self "effective_metric_params_" with
| None -> raise (Wrap_utils.Attribute_not_found "effective_metric_params_")
| Some x -> Wrap_utils.id x
let outputs_2d_ self =
  match Py.Object.get_attr_string self "outputs_2d_" with
| None -> raise (Wrap_utils.Attribute_not_found "outputs_2d_")
| Some x -> Py.Bool.to_bool x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RadiusNeighborsRegressor = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?radius ?weights ?algorithm ?leaf_size ?p ?metric ?metric_params ?n_jobs ?kwargs () =
                     Py.Module.get_function_with_keywords ns "RadiusNeighborsRegressor"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("radius", Wrap_utils.Option.map radius Py.Float.of_float); ("weights", Wrap_utils.Option.map weights (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Auto -> Py.String.of_string "auto"
| `Ball_tree -> Py.String.of_string "ball_tree"
| `Kd_tree -> Py.String.of_string "kd_tree"
| `Brute -> Py.String.of_string "brute"
)); ("leaf_size", Wrap_utils.Option.map leaf_size Py.Int.of_int); ("p", Wrap_utils.Option.map p Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("metric_params", metric_params); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
))]) (match kwargs with None -> [] | Some x -> x))

                  let fit ~x ~y self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("y", Some(y |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
                  let radius_neighbors ?x ?radius ?sort_results self =
                     Py.Module.get_function_with_keywords self "radius_neighbors"
                       [||]
                       (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("radius", Wrap_utils.Option.map radius Py.Float.of_float); ("return_distance", Some(true |> Py.Bool.of_bool)); ("sort_results", Wrap_utils.Option.map sort_results Py.Bool.of_bool)])
                       |> (fun x -> (((fun py -> let len = Py.Sequence.length py in Array.init len (fun i -> Ndarray.of_pyobject (Py.Sequence.get_item py i))) (Py.Tuple.get x 0)), ((fun py -> let len = Py.Sequence.length py in Array.init len (fun i -> Ndarray.of_pyobject (Py.Sequence.get_item py i))) (Py.Tuple.get x 1))))
                  let radius_neighbors_graph ?x ?radius ?mode ?sort_results self =
                     Py.Module.get_function_with_keywords self "radius_neighbors_graph"
                       [||]
                       (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Ndarray.to_pyobject); ("radius", Wrap_utils.Option.map radius Py.Float.of_float); ("mode", Wrap_utils.Option.map mode (function
| `Connectivity -> Py.String.of_string "connectivity"
| `Distance -> Py.String.of_string "distance"
)); ("sort_results", Wrap_utils.Option.map sort_results Py.Bool.of_bool)])
                       |> Csr_matrix.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let effective_metric_ self =
  match Py.Object.get_attr_string self "effective_metric_" with
| None -> raise (Wrap_utils.Attribute_not_found "effective_metric_")
| Some x -> Wrap_utils.id x
let effective_metric_params_ self =
  match Py.Object.get_attr_string self "effective_metric_params_" with
| None -> raise (Wrap_utils.Attribute_not_found "effective_metric_params_")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RadiusNeighborsTransformer = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?mode ?radius ?algorithm ?leaf_size ?metric ?p ?metric_params ?n_jobs () =
                     Py.Module.get_function_with_keywords ns "RadiusNeighborsTransformer"
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
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("p", Wrap_utils.Option.map p Py.Int.of_int); ("metric_params", metric_params); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int)])

                  let fit ?y ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])

let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

                  let radius_neighbors ?x ?radius ?sort_results self =
                     Py.Module.get_function_with_keywords self "radius_neighbors"
                       [||]
                       (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("radius", Wrap_utils.Option.map radius Py.Float.of_float); ("return_distance", Some(true |> Py.Bool.of_bool)); ("sort_results", Wrap_utils.Option.map sort_results Py.Bool.of_bool)])
                       |> (fun x -> (((fun py -> let len = Py.Sequence.length py in Array.init len (fun i -> Ndarray.of_pyobject (Py.Sequence.get_item py i))) (Py.Tuple.get x 0)), ((fun py -> let len = Py.Sequence.length py in Array.init len (fun i -> Ndarray.of_pyobject (Py.Sequence.get_item py i))) (Py.Tuple.get x 1))))
                  let radius_neighbors_graph ?x ?radius ?mode ?sort_results self =
                     Py.Module.get_function_with_keywords self "radius_neighbors_graph"
                       [||]
                       (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Ndarray.to_pyobject); ("radius", Wrap_utils.Option.map radius Py.Float.of_float); ("mode", Wrap_utils.Option.map mode (function
| `Connectivity -> Py.String.of_string "connectivity"
| `Distance -> Py.String.of_string "distance"
)); ("sort_results", Wrap_utils.Option.map sort_results Py.Bool.of_bool)])
                       |> Csr_matrix.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let kneighbors_graph ?mode ?metric ?p ?metric_params ?include_self ?n_jobs ~x ~n_neighbors () =
                     Py.Module.get_function_with_keywords ns "kneighbors_graph"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Connectivity -> Py.String.of_string "connectivity"
| `Distance -> Py.String.of_string "distance"
)); ("metric", Wrap_utils.Option.map metric Py.String.of_string); ("p", Wrap_utils.Option.map p Py.Int.of_int); ("metric_params", metric_params); ("include_self", Wrap_utils.Option.map include_self (function
| `Bool x -> Py.Bool.of_bool x
| `Auto -> Py.String.of_string "auto"
)); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("n_neighbors", Some(n_neighbors |> Py.Int.of_int))])
                       |> Csr_matrix.of_pyobject
                  let radius_neighbors_graph ?mode ?metric ?p ?metric_params ?include_self ?n_jobs ~x ~radius () =
                     Py.Module.get_function_with_keywords ns "radius_neighbors_graph"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Connectivity -> Py.String.of_string "connectivity"
| `Distance -> Py.String.of_string "distance"
)); ("metric", Wrap_utils.Option.map metric Py.String.of_string); ("p", Wrap_utils.Option.map p Py.Int.of_int); ("metric_params", metric_params); ("include_self", Wrap_utils.Option.map include_self (function
| `Bool x -> Py.Bool.of_bool x
| `Auto -> Py.String.of_string "auto"
)); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("radius", Some(radius |> Py.Float.of_float))])
                       |> Csr_matrix.of_pyobject
