let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.multioutput"

module ABCMeta = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?kwargs ~name ~bases ~namespace () =
   Py.Module.get_function_with_keywords ns "ABCMeta"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("name", Some(name )); ("bases", Some(bases )); ("namespace", Some(namespace ))]) (match kwargs with None -> [] | Some x -> x))

let mro self =
   Py.Module.get_function_with_keywords self "mro"
     [||]
     []

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module BaseEstimator = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create () =
   Py.Module.get_function_with_keywords ns "BaseEstimator"
     [||]
     []

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ClassifierChain = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?order ?cv ?random_state ~base_estimator () =
                     Py.Module.get_function_with_keywords ns "ClassifierChain"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `Ndarray x -> Ndarray.to_pyobject x
| `Random -> Py.String.of_string "random"
)); ("cv", Wrap_utils.Option.map cv (function
| `Int x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Ndarray x -> Ndarray.to_pyobject x
)); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("base_estimator", Some(base_estimator ))])

let decision_function ~x self =
   Py.Module.get_function_with_keywords self "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
                  let fit ~x ~y self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
))); ("Y", Some(y ))])

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
                  let predict_proba ~x self =
                     Py.Module.get_function_with_keywords self "predict_proba"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
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
let estimators_ self =
  match Py.Object.get_attr_string self "estimators_" with
| None -> raise (Wrap_utils.Attribute_not_found "estimators_")
| Some x -> Wrap_utils.id x
let order_ self =
  match Py.Object.get_attr_string self "order_" with
| None -> raise (Wrap_utils.Attribute_not_found "order_")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ClassifierMixin = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create () =
   Py.Module.get_function_with_keywords ns "ClassifierMixin"
     [||]
     []

let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MetaEstimatorMixin = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create () =
   Py.Module.get_function_with_keywords ns "MetaEstimatorMixin"
     [||]
     []

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MultiOutputClassifier = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_jobs ~estimator () =
                     Py.Module.get_function_with_keywords ns "MultiOutputClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("estimator", Some(estimator ))])

                  let fit ?sample_weight ~x ~y self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", sample_weight); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
))); ("Y", Some(y ))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let partial_fit ?classes ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("classes", classes); ("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x )); ("y", Some(y ))])

let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> Ndarray.of_pyobject
let score ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let estimators_ self =
  match Py.Object.get_attr_string self "estimators_" with
| None -> raise (Wrap_utils.Attribute_not_found "estimators_")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MultiOutputRegressor = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_jobs ~estimator () =
                     Py.Module.get_function_with_keywords ns "MultiOutputRegressor"
                       [||]
                       (Wrap_utils.keyword_args [("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("estimator", Some(estimator ))])

let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x )); ("y", Some(y ))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let partial_fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x )); ("y", Some(y ))])

let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
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

let estimators_ self =
  match Py.Object.get_attr_string self "estimators_" with
| None -> raise (Wrap_utils.Attribute_not_found "estimators_")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Parallel = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_jobs ?backend ?verbose ?timeout ?pre_dispatch ?batch_size ?temp_folder ?max_nbytes ?mmap_mode ?prefer ?require () =
                     Py.Module.get_function_with_keywords ns "Parallel"
                       [||]
                       (Wrap_utils.keyword_args [("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("backend", backend); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("timeout", Wrap_utils.Option.map timeout Py.Float.of_float); ("pre_dispatch", Wrap_utils.Option.map pre_dispatch (function
| `All -> Py.String.of_string "all"
| `Int x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)); ("batch_size", Wrap_utils.Option.map batch_size (function
| `Int x -> Py.Int.of_int x
| `Auto -> Py.String.of_string "auto"
)); ("temp_folder", Wrap_utils.Option.map temp_folder Py.String.of_string); ("max_nbytes", max_nbytes); ("mmap_mode", Wrap_utils.Option.map mmap_mode (function
| `R_ -> Py.String.of_string "r+"
| `R -> Py.String.of_string "r"
| `W_ -> Py.String.of_string "w+"
| `C -> Py.String.of_string "c"
| `None -> Py.String.of_string "None"
)); ("prefer", Wrap_utils.Option.map prefer (function
| `Processes -> Py.String.of_string "processes"
| `Threads -> Py.String.of_string "threads"
| `None -> Py.String.of_string "None"
)); ("require", Wrap_utils.Option.map require (function
| `Sharedmem -> Py.String.of_string "sharedmem"
| `None -> Py.String.of_string "None"
))])

let debug ~msg self =
   Py.Module.get_function_with_keywords self "debug"
     [||]
     (Wrap_utils.keyword_args [("msg", Some(msg ))])

let dispatch_next self =
   Py.Module.get_function_with_keywords self "dispatch_next"
     [||]
     []

let dispatch_one_batch ~iterator self =
   Py.Module.get_function_with_keywords self "dispatch_one_batch"
     [||]
     (Wrap_utils.keyword_args [("iterator", Some(iterator ))])

let format ?indent ~obj self =
   Py.Module.get_function_with_keywords self "format"
     [||]
     (Wrap_utils.keyword_args [("indent", indent); ("obj", Some(obj ))])

let print_progress self =
   Py.Module.get_function_with_keywords self "print_progress"
     [||]
     []

let retrieve self =
   Py.Module.get_function_with_keywords self "retrieve"
     [||]
     []

let warn ~msg self =
   Py.Module.get_function_with_keywords self "warn"
     [||]
     (Wrap_utils.keyword_args [("msg", Some(msg ))])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RegressorChain = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?order ?cv ?random_state ~base_estimator () =
                     Py.Module.get_function_with_keywords ns "RegressorChain"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `Ndarray x -> Ndarray.to_pyobject x
| `Random -> Py.String.of_string "random"
)); ("cv", Wrap_utils.Option.map cv (function
| `Int x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Ndarray x -> Ndarray.to_pyobject x
)); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("base_estimator", Some(base_estimator ))])

                  let fit ~x ~y self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
))); ("Y", Some(y ))])

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
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let estimators_ self =
  match Py.Object.get_attr_string self "estimators_" with
| None -> raise (Wrap_utils.Attribute_not_found "estimators_")
| Some x -> Wrap_utils.id x
let order_ self =
  match Py.Object.get_attr_string self "order_" with
| None -> raise (Wrap_utils.Attribute_not_found "order_")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RegressorMixin = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create () =
   Py.Module.get_function_with_keywords ns "RegressorMixin"
     [||]
     []

let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let abstractmethod ~funcobj () =
   Py.Module.get_function_with_keywords ns "abstractmethod"
     [||]
     (Wrap_utils.keyword_args [("funcobj", Some(funcobj ))])

                  let check_X_y ?accept_sparse ?accept_large_sparse ?dtype ?order ?copy ?force_all_finite ?ensure_2d ?allow_nd ?multi_output ?ensure_min_samples ?ensure_min_features ?y_numeric ?warn_on_dtype ?estimator ~x ~y () =
                     Py.Module.get_function_with_keywords ns "check_X_y"
                       [||]
                       (Wrap_utils.keyword_args [("accept_sparse", Wrap_utils.Option.map accept_sparse (function
| `String x -> Py.String.of_string x
| `Bool x -> Py.Bool.of_bool x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
)); ("accept_large_sparse", Wrap_utils.Option.map accept_large_sparse Py.Bool.of_bool); ("dtype", Wrap_utils.Option.map dtype (function
| `String x -> Py.String.of_string x
| `Dtype x -> Wrap_utils.id x
| `TypeList x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
| `None -> Py.String.of_string "None"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("force_all_finite", Wrap_utils.Option.map force_all_finite (function
| `Bool x -> Py.Bool.of_bool x
| `Allow_nan -> Py.String.of_string "allow-nan"
)); ("ensure_2d", Wrap_utils.Option.map ensure_2d Py.Bool.of_bool); ("allow_nd", Wrap_utils.Option.map allow_nd Py.Bool.of_bool); ("multi_output", Wrap_utils.Option.map multi_output Py.Bool.of_bool); ("ensure_min_samples", Wrap_utils.Option.map ensure_min_samples Py.Int.of_int); ("ensure_min_features", Wrap_utils.Option.map ensure_min_features Py.Int.of_int); ("y_numeric", Wrap_utils.Option.map y_numeric Py.Bool.of_bool); ("warn_on_dtype", Wrap_utils.Option.map warn_on_dtype (function
| `Bool x -> Py.Bool.of_bool x
| `None -> Py.String.of_string "None"
)); ("estimator", Wrap_utils.Option.map estimator (function
| `String x -> Py.String.of_string x
| `Estimator x -> Wrap_utils.id x
)); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `ArrayLike x -> Wrap_utils.id x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
))); ("y", Some(y |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `ArrayLike x -> Wrap_utils.id x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let check_array ?accept_sparse ?accept_large_sparse ?dtype ?order ?copy ?force_all_finite ?ensure_2d ?allow_nd ?ensure_min_samples ?ensure_min_features ?warn_on_dtype ?estimator ~array () =
                     Py.Module.get_function_with_keywords ns "check_array"
                       [||]
                       (Wrap_utils.keyword_args [("accept_sparse", Wrap_utils.Option.map accept_sparse (function
| `String x -> Py.String.of_string x
| `Bool x -> Py.Bool.of_bool x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
)); ("accept_large_sparse", Wrap_utils.Option.map accept_large_sparse Py.Bool.of_bool); ("dtype", Wrap_utils.Option.map dtype (function
| `String x -> Py.String.of_string x
| `Dtype x -> Wrap_utils.id x
| `TypeList x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
| `None -> Py.String.of_string "None"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("force_all_finite", Wrap_utils.Option.map force_all_finite (function
| `Bool x -> Py.Bool.of_bool x
| `Allow_nan -> Py.String.of_string "allow-nan"
)); ("ensure_2d", Wrap_utils.Option.map ensure_2d Py.Bool.of_bool); ("allow_nd", Wrap_utils.Option.map allow_nd Py.Bool.of_bool); ("ensure_min_samples", Wrap_utils.Option.map ensure_min_samples Py.Int.of_int); ("ensure_min_features", Wrap_utils.Option.map ensure_min_features Py.Int.of_int); ("warn_on_dtype", Wrap_utils.Option.map warn_on_dtype (function
| `Bool x -> Py.Bool.of_bool x
| `None -> Py.String.of_string "None"
)); ("estimator", Wrap_utils.Option.map estimator (function
| `String x -> Py.String.of_string x
| `Estimator x -> Wrap_utils.id x
)); ("array", Some(array ))])

let check_classification_targets ~y () =
   Py.Module.get_function_with_keywords ns "check_classification_targets"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Ndarray.to_pyobject))])

                  let check_is_fitted ?attributes ?msg ?all_or_any ~estimator () =
                     Py.Module.get_function_with_keywords ns "check_is_fitted"
                       [||]
                       (Wrap_utils.keyword_args [("attributes", Wrap_utils.Option.map attributes (function
| `String x -> Py.String.of_string x
| `ArrayLike x -> Wrap_utils.id x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
)); ("msg", Wrap_utils.Option.map msg Py.String.of_string); ("all_or_any", Wrap_utils.Option.map all_or_any (function
| `Callable x -> Wrap_utils.id x
| `PyObject x -> Wrap_utils.id x
)); ("estimator", Some(estimator ))])

                  let check_random_state ~seed () =
                     Py.Module.get_function_with_keywords ns "check_random_state"
                       [||]
                       (Wrap_utils.keyword_args [("seed", Some(seed |> (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)))])

                  let clone ?safe ~estimator () =
                     Py.Module.get_function_with_keywords ns "clone"
                       [||]
                       (Wrap_utils.keyword_args [("safe", Wrap_utils.Option.map safe Py.Bool.of_bool); ("estimator", Some(estimator |> (function
| `Estimator x -> Wrap_utils.id x
| `ArrayLike x -> Wrap_utils.id x
| `PyObject x -> Wrap_utils.id x
)))])

                  let cross_val_predict ?y ?groups ?cv ?n_jobs ?verbose ?fit_params ?pre_dispatch ?method_ ~estimator ~x () =
                     Py.Module.get_function_with_keywords ns "cross_val_predict"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("cv", Wrap_utils.Option.map cv (function
| `Int x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Ndarray x -> Ndarray.to_pyobject x
)); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("fit_params", fit_params); ("pre_dispatch", Wrap_utils.Option.map pre_dispatch (function
| `Int x -> Py.Int.of_int x
| `String x -> Py.String.of_string x
)); ("method", Wrap_utils.Option.map method_ Py.String.of_string); ("estimator", Some(estimator )); ("X", Some(x |> Ndarray.to_pyobject))])
                       |> Ndarray.of_pyobject
let delayed ?check_pickle ~function_ () =
   Py.Module.get_function_with_keywords ns "delayed"
     [||]
     (Wrap_utils.keyword_args [("check_pickle", check_pickle); ("function", Some(function_ ))])

module Deprecated = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?extra () =
   Py.Module.get_function_with_keywords ns "deprecated"
     [||]
     (Wrap_utils.keyword_args [("extra", Wrap_utils.Option.map extra Py.String.of_string)])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let has_fit_parameter ~estimator ~parameter () =
   Py.Module.get_function_with_keywords ns "has_fit_parameter"
     [||]
     (Wrap_utils.keyword_args [("estimator", Some(estimator )); ("parameter", Some(parameter |> Py.String.of_string))])
     |> Py.Bool.to_bool
                  let if_delegate_has_method ~delegate () =
                     Py.Module.get_function_with_keywords ns "if_delegate_has_method"
                       [||]
                       (Wrap_utils.keyword_args [("delegate", Some(delegate |> (function
| `String x -> Py.String.of_string x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `PyObject x -> Wrap_utils.id x
)))])

let is_classifier ~estimator () =
   Py.Module.get_function_with_keywords ns "is_classifier"
     [||]
     (Wrap_utils.keyword_args [("estimator", Some(estimator ))])
     |> Py.Bool.to_bool
