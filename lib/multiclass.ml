let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.multiclass"

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
module LabelBinarizer = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?neg_label ?pos_label ?sparse_output () =
   Py.Module.get_function_with_keywords ns "LabelBinarizer"
     [||]
     (Wrap_utils.keyword_args [("neg_label", Wrap_utils.Option.map neg_label Py.Int.of_int); ("pos_label", Wrap_utils.Option.map pos_label Py.Int.of_int); ("sparse_output", Wrap_utils.Option.map sparse_output Py.Bool.of_bool)])

let fit ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Ndarray.to_pyobject))])

                  let fit_transform ~y self =
                     Py.Module.get_function_with_keywords self "fit_transform"
                       [||]
                       (Wrap_utils.keyword_args [("y", Some(y |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

                  let inverse_transform ?threshold ~y self =
                     Py.Module.get_function_with_keywords self "inverse_transform"
                       [||]
                       (Wrap_utils.keyword_args [("threshold", Wrap_utils.Option.map threshold (function
| `Float x -> Py.Float.of_float x
| `None -> Py.String.of_string "None"
)); ("Y", Some(y |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

                  let transform ~y self =
                     Py.Module.get_function_with_keywords self "transform"
                       [||]
                       (Wrap_utils.keyword_args [("y", Some(y |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let classes_ self =
  match Py.Object.get_attr_string self "classes_" with
| None -> raise (Wrap_utils.Attribute_not_found "classes_")
| Some x -> Ndarray.of_pyobject x
let y_type_ self =
  match Py.Object.get_attr_string self "y_type_" with
| None -> raise (Wrap_utils.Attribute_not_found "y_type_")
| Some x -> Py.String.to_string x
let sparse_input_ self =
  match Py.Object.get_attr_string self "sparse_input_" with
| None -> raise (Wrap_utils.Attribute_not_found "sparse_input_")
| Some x -> Py.Bool.to_bool x
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
module MultiOutputMixin = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create () =
   Py.Module.get_function_with_keywords ns "MultiOutputMixin"
     [||]
     []

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module OneVsOneClassifier = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_jobs ~estimator () =
                     Py.Module.get_function_with_keywords ns "OneVsOneClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("estimator", Some(estimator ))])

let decision_function ~x self =
   Py.Module.get_function_with_keywords self "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let fit ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("y", Some(y |> Ndarray.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let partial_fit ?classes ~x ~y self =
   Py.Module.get_function_with_keywords self "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("classes", Wrap_utils.Option.map classes Ndarray.to_pyobject); ("X", Some(x )); ("y", Some(y |> Ndarray.to_pyobject))])

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
let classes_ self =
  match Py.Object.get_attr_string self "classes_" with
| None -> raise (Wrap_utils.Attribute_not_found "classes_")
| Some x -> Ndarray.of_pyobject x
let n_classes_ self =
  match Py.Object.get_attr_string self "n_classes_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_classes_")
| Some x -> Py.Int.to_int x
let pairwise_indices_ self =
  match Py.Object.get_attr_string self "pairwise_indices_" with
| None -> raise (Wrap_utils.Attribute_not_found "pairwise_indices_")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module OneVsRestClassifier = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_jobs ~estimator () =
                     Py.Module.get_function_with_keywords ns "OneVsRestClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("estimator", Some(estimator ))])

let decision_function ~x self =
   Py.Module.get_function_with_keywords self "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let fit ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("y", Some(y ))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let partial_fit ?classes ~x ~y self =
   Py.Module.get_function_with_keywords self "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("classes", Wrap_utils.Option.map classes Ndarray.to_pyobject); ("X", Some(x )); ("y", Some(y ))])

let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
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

let estimators_ self =
  match Py.Object.get_attr_string self "estimators_" with
| None -> raise (Wrap_utils.Attribute_not_found "estimators_")
| Some x -> Wrap_utils.id x
let classes_ self =
  match Py.Object.get_attr_string self "classes_" with
| None -> raise (Wrap_utils.Attribute_not_found "classes_")
| Some x -> Ndarray.of_pyobject x
let n_classes_ self =
  match Py.Object.get_attr_string self "n_classes_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_classes_")
| Some x -> Py.Int.to_int x
let label_binarizer_ self =
  match Py.Object.get_attr_string self "label_binarizer_" with
| None -> raise (Wrap_utils.Attribute_not_found "label_binarizer_")
| Some x -> Wrap_utils.id x
let multilabel_ self =
  match Py.Object.get_attr_string self "multilabel_" with
| None -> raise (Wrap_utils.Attribute_not_found "multilabel_")
| Some x -> Py.Bool.to_bool x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module OutputCodeClassifier = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?code_size ?random_state ?n_jobs ~estimator () =
                     Py.Module.get_function_with_keywords ns "OutputCodeClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("code_size", Wrap_utils.Option.map code_size Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("estimator", Some(estimator ))])

let fit ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("y", Some(y |> Ndarray.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

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
let classes_ self =
  match Py.Object.get_attr_string self "classes_" with
| None -> raise (Wrap_utils.Attribute_not_found "classes_")
| Some x -> Ndarray.of_pyobject x
let code_book_ self =
  match Py.Object.get_attr_string self "code_book_" with
| None -> raise (Wrap_utils.Attribute_not_found "code_book_")
| Some x -> Ndarray.of_pyobject x
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

let delayed ?check_pickle ~function_ () =
   Py.Module.get_function_with_keywords ns "delayed"
     [||]
     (Wrap_utils.keyword_args [("check_pickle", check_pickle); ("function", Some(function_ ))])

                  let euclidean_distances ?y ?y_norm_squared ?squared ?x_norm_squared ~x () =
                     Py.Module.get_function_with_keywords ns "euclidean_distances"
                       [||]
                       (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)); ("Y_norm_squared", Wrap_utils.Option.map y_norm_squared Ndarray.to_pyobject); ("squared", Wrap_utils.Option.map squared Py.Bool.of_bool); ("X_norm_squared", Wrap_utils.Option.map x_norm_squared Ndarray.to_pyobject); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
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
let is_regressor ~estimator () =
   Py.Module.get_function_with_keywords ns "is_regressor"
     [||]
     (Wrap_utils.keyword_args [("estimator", Some(estimator ))])
     |> Py.Bool.to_bool
