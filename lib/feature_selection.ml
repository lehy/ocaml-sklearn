let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.feature_selection"

module GenericUnivariateSelect = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?score_func ?mode ?param () =
                     Py.Module.get_function_with_keywords ns "GenericUnivariateSelect"
                       [||]
                       (Wrap_utils.keyword_args [("score_func", score_func); ("mode", Wrap_utils.Option.map mode (function
| `Percentile -> Py.String.of_string "percentile"
| `K_best -> Py.String.of_string "k_best"
| `Fpr -> Py.String.of_string "fpr"
| `Fdr -> Py.String.of_string "fdr"
| `Fwe -> Py.String.of_string "fwe"
)); ("param", Wrap_utils.Option.map param (function
| `Float x -> Py.Float.of_float x
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

let get_support ?indices self =
   Py.Module.get_function_with_keywords self "get_support"
     [||]
     (Wrap_utils.keyword_args [("indices", Wrap_utils.Option.map indices Py.Bool.of_bool)])
     |> Ndarray.of_pyobject
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
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
let scores_ self =
  match Py.Object.get_attr_string self "scores_" with
| None -> raise (Wrap_utils.Attribute_not_found "scores_")
| Some x -> Ndarray.of_pyobject x
let pvalues_ self =
  match Py.Object.get_attr_string self "pvalues_" with
| None -> raise (Wrap_utils.Attribute_not_found "pvalues_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RFE = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_features_to_select ?step ?verbose ~estimator () =
                     Py.Module.get_function_with_keywords ns "RFE"
                       [||]
                       (Wrap_utils.keyword_args [("n_features_to_select", Wrap_utils.Option.map n_features_to_select (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("step", Wrap_utils.Option.map step (function
| `Int x -> Py.Int.of_int x
| `Float x -> Py.Float.of_float x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("estimator", Some(estimator ))])

                  let decision_function ~x self =
                     Py.Module.get_function_with_keywords self "decision_function"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
                  let fit ~x ~y self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
))); ("y", Some(y |> Ndarray.to_pyobject))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let get_support ?indices self =
   Py.Module.get_function_with_keywords self "get_support"
     [||]
     (Wrap_utils.keyword_args [("indices", Wrap_utils.Option.map indices Py.Bool.of_bool)])
     |> Ndarray.of_pyobject
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let predict_log_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_log_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
                  let predict_proba ~x self =
                     Py.Module.get_function_with_keywords self "predict_proba"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let score ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let n_features_ self =
  match Py.Object.get_attr_string self "n_features_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_features_")
| Some x -> Py.Int.to_int x
let support_ self =
  match Py.Object.get_attr_string self "support_" with
| None -> raise (Wrap_utils.Attribute_not_found "support_")
| Some x -> Ndarray.of_pyobject x
let ranking_ self =
  match Py.Object.get_attr_string self "ranking_" with
| None -> raise (Wrap_utils.Attribute_not_found "ranking_")
| Some x -> Ndarray.of_pyobject x
let estimator_ self =
  match Py.Object.get_attr_string self "estimator_" with
| None -> raise (Wrap_utils.Attribute_not_found "estimator_")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RFECV = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?step ?min_features_to_select ?cv ?scoring ?verbose ?n_jobs ~estimator () =
                     Py.Module.get_function_with_keywords ns "RFECV"
                       [||]
                       (Wrap_utils.keyword_args [("step", Wrap_utils.Option.map step (function
| `Int x -> Py.Int.of_int x
| `Float x -> Py.Float.of_float x
)); ("min_features_to_select", Wrap_utils.Option.map min_features_to_select Py.Int.of_int); ("cv", Wrap_utils.Option.map cv (function
| `Int x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Ndarray x -> Ndarray.to_pyobject x
)); ("scoring", Wrap_utils.Option.map scoring (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("estimator", Some(estimator ))])

                  let decision_function ~x self =
                     Py.Module.get_function_with_keywords self "decision_function"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
                  let fit ?groups ~x ~y self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("groups", Wrap_utils.Option.map groups (function
| `Ndarray x -> Ndarray.to_pyobject x
| `None -> Py.String.of_string "None"
)); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
))); ("y", Some(y |> Ndarray.to_pyobject))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let get_support ?indices self =
   Py.Module.get_function_with_keywords self "get_support"
     [||]
     (Wrap_utils.keyword_args [("indices", Wrap_utils.Option.map indices Py.Bool.of_bool)])
     |> Ndarray.of_pyobject
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let predict_log_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_log_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
                  let predict_proba ~x self =
                     Py.Module.get_function_with_keywords self "predict_proba"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let score ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let n_features_ self =
  match Py.Object.get_attr_string self "n_features_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_features_")
| Some x -> Py.Int.to_int x
let support_ self =
  match Py.Object.get_attr_string self "support_" with
| None -> raise (Wrap_utils.Attribute_not_found "support_")
| Some x -> Ndarray.of_pyobject x
let ranking_ self =
  match Py.Object.get_attr_string self "ranking_" with
| None -> raise (Wrap_utils.Attribute_not_found "ranking_")
| Some x -> Ndarray.of_pyobject x
let grid_scores_ self =
  match Py.Object.get_attr_string self "grid_scores_" with
| None -> raise (Wrap_utils.Attribute_not_found "grid_scores_")
| Some x -> Ndarray.of_pyobject x
let estimator_ self =
  match Py.Object.get_attr_string self "estimator_" with
| None -> raise (Wrap_utils.Attribute_not_found "estimator_")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SelectFdr = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?score_func ?alpha () =
   Py.Module.get_function_with_keywords ns "SelectFdr"
     [||]
     (Wrap_utils.keyword_args [("score_func", score_func); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float)])

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

let get_support ?indices self =
   Py.Module.get_function_with_keywords self "get_support"
     [||]
     (Wrap_utils.keyword_args [("indices", Wrap_utils.Option.map indices Py.Bool.of_bool)])
     |> Ndarray.of_pyobject
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
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
let scores_ self =
  match Py.Object.get_attr_string self "scores_" with
| None -> raise (Wrap_utils.Attribute_not_found "scores_")
| Some x -> Ndarray.of_pyobject x
let pvalues_ self =
  match Py.Object.get_attr_string self "pvalues_" with
| None -> raise (Wrap_utils.Attribute_not_found "pvalues_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SelectFpr = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?score_func ?alpha () =
   Py.Module.get_function_with_keywords ns "SelectFpr"
     [||]
     (Wrap_utils.keyword_args [("score_func", score_func); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float)])

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

let get_support ?indices self =
   Py.Module.get_function_with_keywords self "get_support"
     [||]
     (Wrap_utils.keyword_args [("indices", Wrap_utils.Option.map indices Py.Bool.of_bool)])
     |> Ndarray.of_pyobject
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
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
let scores_ self =
  match Py.Object.get_attr_string self "scores_" with
| None -> raise (Wrap_utils.Attribute_not_found "scores_")
| Some x -> Ndarray.of_pyobject x
let pvalues_ self =
  match Py.Object.get_attr_string self "pvalues_" with
| None -> raise (Wrap_utils.Attribute_not_found "pvalues_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SelectFromModel = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?threshold ?prefit ?norm_order ?max_features ~estimator () =
                     Py.Module.get_function_with_keywords ns "SelectFromModel"
                       [||]
                       (Wrap_utils.keyword_args [("threshold", Wrap_utils.Option.map threshold (function
| `String x -> Py.String.of_string x
| `Float x -> Py.Float.of_float x
)); ("prefit", Wrap_utils.Option.map prefit Py.Bool.of_bool); ("norm_order", norm_order); ("max_features", Wrap_utils.Option.map max_features (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("estimator", Some(estimator ))])

let fit ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let get_support ?indices self =
   Py.Module.get_function_with_keywords self "get_support"
     [||]
     (Wrap_utils.keyword_args [("indices", Wrap_utils.Option.map indices Py.Bool.of_bool)])
     |> Ndarray.of_pyobject
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let partial_fit ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "partial_fit"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let estimator_ self =
  match Py.Object.get_attr_string self "estimator_" with
| None -> raise (Wrap_utils.Attribute_not_found "estimator_")
| Some x -> Wrap_utils.id x
let threshold_ self =
  match Py.Object.get_attr_string self "threshold_" with
| None -> raise (Wrap_utils.Attribute_not_found "threshold_")
| Some x -> Py.Float.to_float x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SelectFwe = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?score_func ?alpha () =
   Py.Module.get_function_with_keywords ns "SelectFwe"
     [||]
     (Wrap_utils.keyword_args [("score_func", score_func); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float)])

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

let get_support ?indices self =
   Py.Module.get_function_with_keywords self "get_support"
     [||]
     (Wrap_utils.keyword_args [("indices", Wrap_utils.Option.map indices Py.Bool.of_bool)])
     |> Ndarray.of_pyobject
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
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
let scores_ self =
  match Py.Object.get_attr_string self "scores_" with
| None -> raise (Wrap_utils.Attribute_not_found "scores_")
| Some x -> Ndarray.of_pyobject x
let pvalues_ self =
  match Py.Object.get_attr_string self "pvalues_" with
| None -> raise (Wrap_utils.Attribute_not_found "pvalues_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SelectKBest = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?score_func ?k () =
                     Py.Module.get_function_with_keywords ns "SelectKBest"
                       [||]
                       (Wrap_utils.keyword_args [("score_func", score_func); ("k", Wrap_utils.Option.map k (function
| `Int x -> Py.Int.of_int x
| `All -> Py.String.of_string "all"
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

let get_support ?indices self =
   Py.Module.get_function_with_keywords self "get_support"
     [||]
     (Wrap_utils.keyword_args [("indices", Wrap_utils.Option.map indices Py.Bool.of_bool)])
     |> Ndarray.of_pyobject
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
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
let scores_ self =
  match Py.Object.get_attr_string self "scores_" with
| None -> raise (Wrap_utils.Attribute_not_found "scores_")
| Some x -> Ndarray.of_pyobject x
let pvalues_ self =
  match Py.Object.get_attr_string self "pvalues_" with
| None -> raise (Wrap_utils.Attribute_not_found "pvalues_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SelectPercentile = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?score_func ?percentile () =
   Py.Module.get_function_with_keywords ns "SelectPercentile"
     [||]
     (Wrap_utils.keyword_args [("score_func", score_func); ("percentile", Wrap_utils.Option.map percentile Py.Int.of_int)])

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

let get_support ?indices self =
   Py.Module.get_function_with_keywords self "get_support"
     [||]
     (Wrap_utils.keyword_args [("indices", Wrap_utils.Option.map indices Py.Bool.of_bool)])
     |> Ndarray.of_pyobject
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
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
let scores_ self =
  match Py.Object.get_attr_string self "scores_" with
| None -> raise (Wrap_utils.Attribute_not_found "scores_")
| Some x -> Ndarray.of_pyobject x
let pvalues_ self =
  match Py.Object.get_attr_string self "pvalues_" with
| None -> raise (Wrap_utils.Attribute_not_found "pvalues_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module VarianceThreshold = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?threshold () =
   Py.Module.get_function_with_keywords ns "VarianceThreshold"
     [||]
     (Wrap_utils.keyword_args [("threshold", Wrap_utils.Option.map threshold Py.Float.of_float)])

                  let fit ?y ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let get_support ?indices self =
   Py.Module.get_function_with_keywords self "get_support"
     [||]
     (Wrap_utils.keyword_args [("indices", Wrap_utils.Option.map indices Py.Bool.of_bool)])
     |> Ndarray.of_pyobject
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
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
let variances_ self =
  match Py.Object.get_attr_string self "variances_" with
| None -> raise (Wrap_utils.Attribute_not_found "variances_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let chi2 ~x ~y () =
                     Py.Module.get_function_with_keywords ns "chi2"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
))); ("y", Some(y |> Ndarray.to_pyobject))])
                       |> Ndarray.of_pyobject
                  let f_classif ~x ~y () =
                     Py.Module.get_function_with_keywords ns "f_classif"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
))); ("y", Some(y |> Ndarray.to_pyobject))])
                       |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
let f_oneway args =
   Py.Module.get_function_with_keywords ns "f_oneway"
     (Wrap_utils.pos_arg Wrap_utils.id args)
     []

                  let f_regression ?center ~x ~y () =
                     Py.Module.get_function_with_keywords ns "f_regression"
                       [||]
                       (Wrap_utils.keyword_args [("center", Wrap_utils.Option.map center (function
| `True -> Py.Bool.t
| `Bool x -> Py.Bool.of_bool x
)); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
))); ("y", Some(y |> Ndarray.to_pyobject))])
                       |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
                  let mutual_info_classif ?discrete_features ?n_neighbors ?copy ?random_state ~x ~y () =
                     Py.Module.get_function_with_keywords ns "mutual_info_classif"
                       [||]
                       (Wrap_utils.keyword_args [("discrete_features", Wrap_utils.Option.map discrete_features (function
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
| `Ndarray x -> Ndarray.to_pyobject x
)); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
))); ("y", Some(y |> Ndarray.to_pyobject))])
                       |> Ndarray.of_pyobject
                  let mutual_info_regression ?discrete_features ?n_neighbors ?copy ?random_state ~x ~y () =
                     Py.Module.get_function_with_keywords ns "mutual_info_regression"
                       [||]
                       (Wrap_utils.keyword_args [("discrete_features", Wrap_utils.Option.map discrete_features (function
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
| `Ndarray x -> Ndarray.to_pyobject x
)); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
))); ("y", Some(y |> Ndarray.to_pyobject))])
                       |> Ndarray.of_pyobject
