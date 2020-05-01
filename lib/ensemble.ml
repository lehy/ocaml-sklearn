let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.ensemble"

let get_py name = Py.Module.get ns name
module AdaBoostClassifier = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?base_estimator ?n_estimators ?learning_rate ?algorithm ?random_state () =
                     Py.Module.get_function_with_keywords ns "AdaBoostClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("base_estimator", base_estimator); ("n_estimators", Wrap_utils.Option.map n_estimators Py.Int.of_int); ("learning_rate", Wrap_utils.Option.map learning_rate Py.Float.of_float); ("algorithm", Wrap_utils.Option.map algorithm (function
| `SAMME -> Py.String.of_string "SAMME"
| `SAMME_R -> Py.String.of_string "SAMME.R"
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])

let get_item ~index self =
   Py.Module.get_function_with_keywords self "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let decision_function ~x self =
   Py.Module.get_function_with_keywords self "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let predict_log_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_log_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let predict_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let staged_decision_function ~x self =
   Py.Module.get_function_with_keywords self "staged_decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])

let staged_predict ~x self =
   Py.Module.get_function_with_keywords self "staged_predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])

let staged_predict_proba ~x self =
   Py.Module.get_function_with_keywords self "staged_predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])

let staged_score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "staged_score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])


let base_estimator_opt self =
  match Py.Object.get_attr_string self "base_estimator_" with
  | None -> failwith "attribute base_estimator_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let base_estimator_ self = match base_estimator_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_opt self =
  match Py.Object.get_attr_string self "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string self "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_classes_opt self =
  match Py.Object.get_attr_string self "n_classes_" with
  | None -> failwith "attribute n_classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_classes_ self = match n_classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimator_weights_opt self =
  match Py.Object.get_attr_string self "estimator_weights_" with
  | None -> failwith "attribute estimator_weights_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let estimator_weights_ self = match estimator_weights_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimator_errors_opt self =
  match Py.Object.get_attr_string self "estimator_errors_" with
  | None -> failwith "attribute estimator_errors_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let estimator_errors_ self = match estimator_errors_opt self with
  | None -> raise Not_found
  | Some x -> x

let feature_importances_opt self =
  match Py.Object.get_attr_string self "feature_importances_" with
  | None -> failwith "attribute feature_importances_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let feature_importances_ self = match feature_importances_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module AdaBoostRegressor = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?base_estimator ?n_estimators ?learning_rate ?loss ?random_state () =
                     Py.Module.get_function_with_keywords ns "AdaBoostRegressor"
                       [||]
                       (Wrap_utils.keyword_args [("base_estimator", base_estimator); ("n_estimators", Wrap_utils.Option.map n_estimators Py.Int.of_int); ("learning_rate", Wrap_utils.Option.map learning_rate Py.Float.of_float); ("loss", Wrap_utils.Option.map loss (function
| `Linear -> Py.String.of_string "linear"
| `Square -> Py.String.of_string "square"
| `Exponential -> Py.String.of_string "exponential"
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])

let get_item ~index self =
   Py.Module.get_function_with_keywords self "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let staged_predict ~x self =
   Py.Module.get_function_with_keywords self "staged_predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])

let staged_score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "staged_score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])


let base_estimator_opt self =
  match Py.Object.get_attr_string self "base_estimator_" with
  | None -> failwith "attribute base_estimator_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let base_estimator_ self = match base_estimator_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_opt self =
  match Py.Object.get_attr_string self "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimator_weights_opt self =
  match Py.Object.get_attr_string self "estimator_weights_" with
  | None -> failwith "attribute estimator_weights_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let estimator_weights_ self = match estimator_weights_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimator_errors_opt self =
  match Py.Object.get_attr_string self "estimator_errors_" with
  | None -> failwith "attribute estimator_errors_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let estimator_errors_ self = match estimator_errors_opt self with
  | None -> raise Not_found
  | Some x -> x

let feature_importances_opt self =
  match Py.Object.get_attr_string self "feature_importances_" with
  | None -> failwith "attribute feature_importances_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let feature_importances_ self = match feature_importances_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module BaggingClassifier = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?base_estimator ?n_estimators ?max_samples ?max_features ?bootstrap ?bootstrap_features ?oob_score ?warm_start ?n_jobs ?random_state ?verbose () =
                     Py.Module.get_function_with_keywords ns "BaggingClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("base_estimator", base_estimator); ("n_estimators", Wrap_utils.Option.map n_estimators Py.Int.of_int); ("max_samples", Wrap_utils.Option.map max_samples (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("max_features", Wrap_utils.Option.map max_features (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("bootstrap", Wrap_utils.Option.map bootstrap Py.Bool.of_bool); ("bootstrap_features", Wrap_utils.Option.map bootstrap_features Py.Bool.of_bool); ("oob_score", Wrap_utils.Option.map oob_score Py.Bool.of_bool); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int)])

let get_item ~index self =
   Py.Module.get_function_with_keywords self "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let predict_log_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_log_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let predict_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)


let base_estimator_opt self =
  match Py.Object.get_attr_string self "base_estimator_" with
  | None -> failwith "attribute base_estimator_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let base_estimator_ self = match base_estimator_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_features_opt self =
  match Py.Object.get_attr_string self "n_features_" with
  | None -> failwith "attribute n_features_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_features_ self = match n_features_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_opt self =
  match Py.Object.get_attr_string self "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_samples_opt self =
  match Py.Object.get_attr_string self "estimators_samples_" with
  | None -> failwith "attribute estimators_samples_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.List.of_pyobject x)

let estimators_samples_ self = match estimators_samples_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_features_opt self =
  match Py.Object.get_attr_string self "estimators_features_" with
  | None -> failwith "attribute estimators_features_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.List.of_pyobject x)

let estimators_features_ self = match estimators_features_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string self "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_classes_opt self =
  match Py.Object.get_attr_string self "n_classes_" with
  | None -> failwith "attribute n_classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun x -> if Py.Int.check x then `I (Py.Int.to_int x) else if (fun x -> (Wrap_utils.isinstance Wrap_utils.ndarray x) || (Wrap_utils.isinstance Wrap_utils.csr_matrix x)) x then `Arr (Arr.of_pyobject x) else failwith "could not identify type from Python value") x)

let n_classes_ self = match n_classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_score_opt self =
  match Py.Object.get_attr_string self "oob_score_" with
  | None -> failwith "attribute oob_score_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let oob_score_ self = match oob_score_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_decision_function_opt self =
  match Py.Object.get_attr_string self "oob_decision_function_" with
  | None -> failwith "attribute oob_decision_function_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let oob_decision_function_ self = match oob_decision_function_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module BaggingRegressor = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?base_estimator ?n_estimators ?max_samples ?max_features ?bootstrap ?bootstrap_features ?oob_score ?warm_start ?n_jobs ?random_state ?verbose () =
                     Py.Module.get_function_with_keywords ns "BaggingRegressor"
                       [||]
                       (Wrap_utils.keyword_args [("base_estimator", base_estimator); ("n_estimators", Wrap_utils.Option.map n_estimators Py.Int.of_int); ("max_samples", Wrap_utils.Option.map max_samples (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("max_features", Wrap_utils.Option.map max_features (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("bootstrap", Wrap_utils.Option.map bootstrap Py.Bool.of_bool); ("bootstrap_features", Wrap_utils.Option.map bootstrap_features Py.Bool.of_bool); ("oob_score", Wrap_utils.Option.map oob_score Py.Bool.of_bool); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int)])

let get_item ~index self =
   Py.Module.get_function_with_keywords self "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)


let base_estimator_opt self =
  match Py.Object.get_attr_string self "base_estimator_" with
  | None -> failwith "attribute base_estimator_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let base_estimator_ self = match base_estimator_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_features_opt self =
  match Py.Object.get_attr_string self "n_features_" with
  | None -> failwith "attribute n_features_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_features_ self = match n_features_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_opt self =
  match Py.Object.get_attr_string self "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_samples_opt self =
  match Py.Object.get_attr_string self "estimators_samples_" with
  | None -> failwith "attribute estimators_samples_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.List.of_pyobject x)

let estimators_samples_ self = match estimators_samples_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_features_opt self =
  match Py.Object.get_attr_string self "estimators_features_" with
  | None -> failwith "attribute estimators_features_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.List.of_pyobject x)

let estimators_features_ self = match estimators_features_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_score_opt self =
  match Py.Object.get_attr_string self "oob_score_" with
  | None -> failwith "attribute oob_score_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let oob_score_ self = match oob_score_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_prediction_opt self =
  match Py.Object.get_attr_string self "oob_prediction_" with
  | None -> failwith "attribute oob_prediction_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let oob_prediction_ self = match oob_prediction_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ExtraTreesClassifier = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_estimators ?criterion ?max_depth ?min_samples_split ?min_samples_leaf ?min_weight_fraction_leaf ?max_features ?max_leaf_nodes ?min_impurity_decrease ?min_impurity_split ?bootstrap ?oob_score ?n_jobs ?random_state ?verbose ?warm_start ?class_weight ?ccp_alpha ?max_samples () =
                     Py.Module.get_function_with_keywords ns "ExtraTreesClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("n_estimators", Wrap_utils.Option.map n_estimators Py.Int.of_int); ("criterion", Wrap_utils.Option.map criterion Py.String.of_string); ("max_depth", Wrap_utils.Option.map max_depth Py.Int.of_int); ("min_samples_split", Wrap_utils.Option.map min_samples_split (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("min_samples_leaf", Wrap_utils.Option.map min_samples_leaf (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("min_weight_fraction_leaf", Wrap_utils.Option.map min_weight_fraction_leaf Py.Float.of_float); ("max_features", Wrap_utils.Option.map max_features (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
| `S x -> Py.String.of_string x
| `None -> Py.none
)); ("max_leaf_nodes", Wrap_utils.Option.map max_leaf_nodes Py.Int.of_int); ("min_impurity_decrease", Wrap_utils.Option.map min_impurity_decrease Py.Float.of_float); ("min_impurity_split", Wrap_utils.Option.map min_impurity_split Py.Float.of_float); ("bootstrap", Wrap_utils.Option.map bootstrap Py.Bool.of_bool); ("oob_score", Wrap_utils.Option.map oob_score Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("class_weight", Wrap_utils.Option.map class_weight (function
| `DictIntToFloat x -> (Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float) x
| `List_of_dicts x -> Wrap_utils.id x
| `Balanced -> Py.String.of_string "balanced"
| `Balanced_subsample -> Py.String.of_string "balanced_subsample"
)); ("ccp_alpha", Wrap_utils.Option.map ccp_alpha Py.Float.of_float); ("max_samples", Wrap_utils.Option.map max_samples (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
))])

let get_item ~index self =
   Py.Module.get_function_with_keywords self "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let apply ~x self =
   Py.Module.get_function_with_keywords self "apply"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let decision_path ~x self =
   Py.Module.get_function_with_keywords self "decision_path"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let predict_log_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_log_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])

let predict_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)


let base_estimator_opt self =
  match Py.Object.get_attr_string self "base_estimator_" with
  | None -> failwith "attribute base_estimator_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let base_estimator_ self = match base_estimator_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_opt self =
  match Py.Object.get_attr_string self "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string self "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_classes_opt self =
  match Py.Object.get_attr_string self "n_classes_" with
  | None -> failwith "attribute n_classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun x -> if Py.Int.check x then `I (Py.Int.to_int x) else if (fun x -> (Wrap_utils.isinstance Wrap_utils.ndarray x) || (Wrap_utils.isinstance Wrap_utils.csr_matrix x)) x then `Arr (Arr.of_pyobject x) else failwith "could not identify type from Python value") x)

let n_classes_ self = match n_classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let feature_importances_opt self =
  match Py.Object.get_attr_string self "feature_importances_" with
  | None -> failwith "attribute feature_importances_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let feature_importances_ self = match feature_importances_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_features_opt self =
  match Py.Object.get_attr_string self "n_features_" with
  | None -> failwith "attribute n_features_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_features_ self = match n_features_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_outputs_opt self =
  match Py.Object.get_attr_string self "n_outputs_" with
  | None -> failwith "attribute n_outputs_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_outputs_ self = match n_outputs_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_score_opt self =
  match Py.Object.get_attr_string self "oob_score_" with
  | None -> failwith "attribute oob_score_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let oob_score_ self = match oob_score_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_decision_function_opt self =
  match Py.Object.get_attr_string self "oob_decision_function_" with
  | None -> failwith "attribute oob_decision_function_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let oob_decision_function_ self = match oob_decision_function_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ExtraTreesRegressor = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_estimators ?criterion ?max_depth ?min_samples_split ?min_samples_leaf ?min_weight_fraction_leaf ?max_features ?max_leaf_nodes ?min_impurity_decrease ?min_impurity_split ?bootstrap ?oob_score ?n_jobs ?random_state ?verbose ?warm_start ?ccp_alpha ?max_samples () =
                     Py.Module.get_function_with_keywords ns "ExtraTreesRegressor"
                       [||]
                       (Wrap_utils.keyword_args [("n_estimators", Wrap_utils.Option.map n_estimators Py.Int.of_int); ("criterion", Wrap_utils.Option.map criterion Py.String.of_string); ("max_depth", Wrap_utils.Option.map max_depth Py.Int.of_int); ("min_samples_split", Wrap_utils.Option.map min_samples_split (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("min_samples_leaf", Wrap_utils.Option.map min_samples_leaf (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("min_weight_fraction_leaf", Wrap_utils.Option.map min_weight_fraction_leaf Py.Float.of_float); ("max_features", Wrap_utils.Option.map max_features (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
| `S x -> Py.String.of_string x
| `None -> Py.none
)); ("max_leaf_nodes", Wrap_utils.Option.map max_leaf_nodes Py.Int.of_int); ("min_impurity_decrease", Wrap_utils.Option.map min_impurity_decrease Py.Float.of_float); ("min_impurity_split", Wrap_utils.Option.map min_impurity_split Py.Float.of_float); ("bootstrap", Wrap_utils.Option.map bootstrap Py.Bool.of_bool); ("oob_score", Wrap_utils.Option.map oob_score Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("ccp_alpha", Wrap_utils.Option.map ccp_alpha Py.Float.of_float); ("max_samples", Wrap_utils.Option.map max_samples (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
))])

let get_item ~index self =
   Py.Module.get_function_with_keywords self "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let apply ~x self =
   Py.Module.get_function_with_keywords self "apply"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let decision_path ~x self =
   Py.Module.get_function_with_keywords self "decision_path"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)


let base_estimator_opt self =
  match Py.Object.get_attr_string self "base_estimator_" with
  | None -> failwith "attribute base_estimator_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let base_estimator_ self = match base_estimator_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_opt self =
  match Py.Object.get_attr_string self "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let feature_importances_opt self =
  match Py.Object.get_attr_string self "feature_importances_" with
  | None -> failwith "attribute feature_importances_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let feature_importances_ self = match feature_importances_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_features_opt self =
  match Py.Object.get_attr_string self "n_features_" with
  | None -> failwith "attribute n_features_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_features_ self = match n_features_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_outputs_opt self =
  match Py.Object.get_attr_string self "n_outputs_" with
  | None -> failwith "attribute n_outputs_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_outputs_ self = match n_outputs_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_score_opt self =
  match Py.Object.get_attr_string self "oob_score_" with
  | None -> failwith "attribute oob_score_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let oob_score_ self = match oob_score_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_prediction_opt self =
  match Py.Object.get_attr_string self "oob_prediction_" with
  | None -> failwith "attribute oob_prediction_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let oob_prediction_ self = match oob_prediction_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module GradientBoostingClassifier = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?loss ?learning_rate ?n_estimators ?subsample ?criterion ?min_samples_split ?min_samples_leaf ?min_weight_fraction_leaf ?max_depth ?min_impurity_decrease ?min_impurity_split ?init ?random_state ?max_features ?verbose ?max_leaf_nodes ?warm_start ?presort ?validation_fraction ?n_iter_no_change ?tol ?ccp_alpha () =
                     Py.Module.get_function_with_keywords ns "GradientBoostingClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("loss", Wrap_utils.Option.map loss (function
| `Deviance -> Py.String.of_string "deviance"
| `Exponential -> Py.String.of_string "exponential"
)); ("learning_rate", Wrap_utils.Option.map learning_rate Py.Float.of_float); ("n_estimators", Wrap_utils.Option.map n_estimators Py.Int.of_int); ("subsample", Wrap_utils.Option.map subsample Py.Float.of_float); ("criterion", Wrap_utils.Option.map criterion Py.String.of_string); ("min_samples_split", Wrap_utils.Option.map min_samples_split (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("min_samples_leaf", Wrap_utils.Option.map min_samples_leaf (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("min_weight_fraction_leaf", Wrap_utils.Option.map min_weight_fraction_leaf Py.Float.of_float); ("max_depth", Wrap_utils.Option.map max_depth Py.Int.of_int); ("min_impurity_decrease", Wrap_utils.Option.map min_impurity_decrease Py.Float.of_float); ("min_impurity_split", Wrap_utils.Option.map min_impurity_split Py.Float.of_float); ("init", Wrap_utils.Option.map init (function
| `Estimator x -> Wrap_utils.id x
| `Zero -> Py.String.of_string "zero"
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("max_features", Wrap_utils.Option.map max_features (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
| `S x -> Py.String.of_string x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("max_leaf_nodes", Wrap_utils.Option.map max_leaf_nodes Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("presort", presort); ("validation_fraction", Wrap_utils.Option.map validation_fraction Py.Float.of_float); ("n_iter_no_change", Wrap_utils.Option.map n_iter_no_change Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("ccp_alpha", Wrap_utils.Option.map ccp_alpha Py.Float.of_float)])

let get_item ~index self =
   Py.Module.get_function_with_keywords self "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let apply ~x self =
   Py.Module.get_function_with_keywords self "apply"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let decision_function ~x self =
   Py.Module.get_function_with_keywords self "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let fit ?sample_weight ?monitor ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("monitor", monitor); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let predict_log_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_log_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let predict_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let staged_decision_function ~x self =
   Py.Module.get_function_with_keywords self "staged_decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])

let staged_predict ~x self =
   Py.Module.get_function_with_keywords self "staged_predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])

let staged_predict_proba ~x self =
   Py.Module.get_function_with_keywords self "staged_predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])


let n_estimators_opt self =
  match Py.Object.get_attr_string self "n_estimators_" with
  | None -> failwith "attribute n_estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_estimators_ self = match n_estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let feature_importances_opt self =
  match Py.Object.get_attr_string self "feature_importances_" with
  | None -> failwith "attribute feature_importances_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let feature_importances_ self = match feature_importances_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_improvement_opt self =
  match Py.Object.get_attr_string self "oob_improvement_" with
  | None -> failwith "attribute oob_improvement_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let oob_improvement_ self = match oob_improvement_opt self with
  | None -> raise Not_found
  | Some x -> x

let train_score_opt self =
  match Py.Object.get_attr_string self "train_score_" with
  | None -> failwith "attribute train_score_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let train_score_ self = match train_score_opt self with
  | None -> raise Not_found
  | Some x -> x

let loss_opt self =
  match Py.Object.get_attr_string self "loss_" with
  | None -> failwith "attribute loss_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let loss_ self = match loss_opt self with
  | None -> raise Not_found
  | Some x -> x

let init_opt self =
  match Py.Object.get_attr_string self "init_" with
  | None -> failwith "attribute init_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let init_ self = match init_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_opt self =
  match Py.Object.get_attr_string self "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string self "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module GradientBoostingRegressor = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?loss ?learning_rate ?n_estimators ?subsample ?criterion ?min_samples_split ?min_samples_leaf ?min_weight_fraction_leaf ?max_depth ?min_impurity_decrease ?min_impurity_split ?init ?random_state ?max_features ?alpha ?verbose ?max_leaf_nodes ?warm_start ?presort ?validation_fraction ?n_iter_no_change ?tol ?ccp_alpha () =
                     Py.Module.get_function_with_keywords ns "GradientBoostingRegressor"
                       [||]
                       (Wrap_utils.keyword_args [("loss", Wrap_utils.Option.map loss (function
| `Ls -> Py.String.of_string "ls"
| `Lad -> Py.String.of_string "lad"
| `Huber -> Py.String.of_string "huber"
| `Quantile -> Py.String.of_string "quantile"
)); ("learning_rate", Wrap_utils.Option.map learning_rate Py.Float.of_float); ("n_estimators", Wrap_utils.Option.map n_estimators Py.Int.of_int); ("subsample", Wrap_utils.Option.map subsample Py.Float.of_float); ("criterion", Wrap_utils.Option.map criterion Py.String.of_string); ("min_samples_split", Wrap_utils.Option.map min_samples_split (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("min_samples_leaf", Wrap_utils.Option.map min_samples_leaf (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("min_weight_fraction_leaf", Wrap_utils.Option.map min_weight_fraction_leaf Py.Float.of_float); ("max_depth", Wrap_utils.Option.map max_depth Py.Int.of_int); ("min_impurity_decrease", Wrap_utils.Option.map min_impurity_decrease Py.Float.of_float); ("min_impurity_split", Wrap_utils.Option.map min_impurity_split Py.Float.of_float); ("init", Wrap_utils.Option.map init (function
| `Estimator x -> Wrap_utils.id x
| `Zero -> Py.String.of_string "zero"
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("max_features", Wrap_utils.Option.map max_features (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
| `S x -> Py.String.of_string x
)); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("max_leaf_nodes", Wrap_utils.Option.map max_leaf_nodes Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("presort", presort); ("validation_fraction", Wrap_utils.Option.map validation_fraction Py.Float.of_float); ("n_iter_no_change", Wrap_utils.Option.map n_iter_no_change Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("ccp_alpha", Wrap_utils.Option.map ccp_alpha Py.Float.of_float)])

let get_item ~index self =
   Py.Module.get_function_with_keywords self "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let apply ~x self =
   Py.Module.get_function_with_keywords self "apply"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let fit ?sample_weight ?monitor ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("monitor", monitor); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let staged_predict ~x self =
   Py.Module.get_function_with_keywords self "staged_predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])


let feature_importances_opt self =
  match Py.Object.get_attr_string self "feature_importances_" with
  | None -> failwith "attribute feature_importances_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let feature_importances_ self = match feature_importances_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_improvement_opt self =
  match Py.Object.get_attr_string self "oob_improvement_" with
  | None -> failwith "attribute oob_improvement_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let oob_improvement_ self = match oob_improvement_opt self with
  | None -> raise Not_found
  | Some x -> x

let train_score_opt self =
  match Py.Object.get_attr_string self "train_score_" with
  | None -> failwith "attribute train_score_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let train_score_ self = match train_score_opt self with
  | None -> raise Not_found
  | Some x -> x

let loss_opt self =
  match Py.Object.get_attr_string self "loss_" with
  | None -> failwith "attribute loss_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let loss_ self = match loss_opt self with
  | None -> raise Not_found
  | Some x -> x

let init_opt self =
  match Py.Object.get_attr_string self "init_" with
  | None -> failwith "attribute init_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let init_ self = match init_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_opt self =
  match Py.Object.get_attr_string self "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module IsolationForest = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_estimators ?max_samples ?contamination ?max_features ?bootstrap ?n_jobs ?behaviour ?random_state ?verbose ?warm_start () =
                     Py.Module.get_function_with_keywords ns "IsolationForest"
                       [||]
                       (Wrap_utils.keyword_args [("n_estimators", Wrap_utils.Option.map n_estimators Py.Int.of_int); ("max_samples", Wrap_utils.Option.map max_samples (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("contamination", Wrap_utils.Option.map contamination (function
| `Auto -> Py.String.of_string "auto"
| `F x -> Py.Float.of_float x
)); ("max_features", Wrap_utils.Option.map max_features (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("bootstrap", Wrap_utils.Option.map bootstrap Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("behaviour", Wrap_utils.Option.map behaviour Py.String.of_string); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool)])

let get_item ~index self =
   Py.Module.get_function_with_keywords self "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let decision_function ~x self =
   Py.Module.get_function_with_keywords self "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let fit ?y ?sample_weight ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))])

let fit_predict ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_predict"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let score_samples ~x self =
   Py.Module.get_function_with_keywords self "score_samples"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)


let estimators_opt self =
  match Py.Object.get_attr_string self "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_samples_opt self =
  match Py.Object.get_attr_string self "estimators_samples_" with
  | None -> failwith "attribute estimators_samples_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.List.of_pyobject x)

let estimators_samples_ self = match estimators_samples_opt self with
  | None -> raise Not_found
  | Some x -> x

let max_samples_opt self =
  match Py.Object.get_attr_string self "max_samples_" with
  | None -> failwith "attribute max_samples_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let max_samples_ self = match max_samples_opt self with
  | None -> raise Not_found
  | Some x -> x

let offset_opt self =
  match Py.Object.get_attr_string self "offset_" with
  | None -> failwith "attribute offset_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let offset_ self = match offset_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RandomForestClassifier = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_estimators ?criterion ?max_depth ?min_samples_split ?min_samples_leaf ?min_weight_fraction_leaf ?max_features ?max_leaf_nodes ?min_impurity_decrease ?min_impurity_split ?bootstrap ?oob_score ?n_jobs ?random_state ?verbose ?warm_start ?class_weight ?ccp_alpha ?max_samples () =
                     Py.Module.get_function_with_keywords ns "RandomForestClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("n_estimators", Wrap_utils.Option.map n_estimators Py.Int.of_int); ("criterion", Wrap_utils.Option.map criterion Py.String.of_string); ("max_depth", Wrap_utils.Option.map max_depth Py.Int.of_int); ("min_samples_split", Wrap_utils.Option.map min_samples_split (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("min_samples_leaf", Wrap_utils.Option.map min_samples_leaf (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("min_weight_fraction_leaf", Wrap_utils.Option.map min_weight_fraction_leaf Py.Float.of_float); ("max_features", Wrap_utils.Option.map max_features (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
| `S x -> Py.String.of_string x
| `None -> Py.none
)); ("max_leaf_nodes", Wrap_utils.Option.map max_leaf_nodes Py.Int.of_int); ("min_impurity_decrease", Wrap_utils.Option.map min_impurity_decrease Py.Float.of_float); ("min_impurity_split", Wrap_utils.Option.map min_impurity_split Py.Float.of_float); ("bootstrap", Wrap_utils.Option.map bootstrap Py.Bool.of_bool); ("oob_score", Wrap_utils.Option.map oob_score Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("class_weight", Wrap_utils.Option.map class_weight (function
| `DictIntToFloat x -> (Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float) x
| `List_of_dicts x -> Wrap_utils.id x
| `Balanced -> Py.String.of_string "balanced"
| `Balanced_subsample -> Py.String.of_string "balanced_subsample"
)); ("ccp_alpha", Wrap_utils.Option.map ccp_alpha Py.Float.of_float); ("max_samples", Wrap_utils.Option.map max_samples (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
))])

let get_item ~index self =
   Py.Module.get_function_with_keywords self "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let apply ~x self =
   Py.Module.get_function_with_keywords self "apply"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let decision_path ~x self =
   Py.Module.get_function_with_keywords self "decision_path"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let predict_log_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_log_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])

let predict_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)


let base_estimator_opt self =
  match Py.Object.get_attr_string self "base_estimator_" with
  | None -> failwith "attribute base_estimator_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let base_estimator_ self = match base_estimator_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_opt self =
  match Py.Object.get_attr_string self "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string self "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_classes_opt self =
  match Py.Object.get_attr_string self "n_classes_" with
  | None -> failwith "attribute n_classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun x -> if Py.Int.check x then `I (Py.Int.to_int x) else if (fun x -> (Wrap_utils.isinstance Wrap_utils.ndarray x) || (Wrap_utils.isinstance Wrap_utils.csr_matrix x)) x then `Arr (Arr.of_pyobject x) else failwith "could not identify type from Python value") x)

let n_classes_ self = match n_classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_features_opt self =
  match Py.Object.get_attr_string self "n_features_" with
  | None -> failwith "attribute n_features_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_features_ self = match n_features_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_outputs_opt self =
  match Py.Object.get_attr_string self "n_outputs_" with
  | None -> failwith "attribute n_outputs_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_outputs_ self = match n_outputs_opt self with
  | None -> raise Not_found
  | Some x -> x

let feature_importances_opt self =
  match Py.Object.get_attr_string self "feature_importances_" with
  | None -> failwith "attribute feature_importances_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let feature_importances_ self = match feature_importances_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_score_opt self =
  match Py.Object.get_attr_string self "oob_score_" with
  | None -> failwith "attribute oob_score_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let oob_score_ self = match oob_score_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_decision_function_opt self =
  match Py.Object.get_attr_string self "oob_decision_function_" with
  | None -> failwith "attribute oob_decision_function_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let oob_decision_function_ self = match oob_decision_function_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RandomForestRegressor = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_estimators ?criterion ?max_depth ?min_samples_split ?min_samples_leaf ?min_weight_fraction_leaf ?max_features ?max_leaf_nodes ?min_impurity_decrease ?min_impurity_split ?bootstrap ?oob_score ?n_jobs ?random_state ?verbose ?warm_start ?ccp_alpha ?max_samples () =
                     Py.Module.get_function_with_keywords ns "RandomForestRegressor"
                       [||]
                       (Wrap_utils.keyword_args [("n_estimators", Wrap_utils.Option.map n_estimators Py.Int.of_int); ("criterion", Wrap_utils.Option.map criterion Py.String.of_string); ("max_depth", Wrap_utils.Option.map max_depth Py.Int.of_int); ("min_samples_split", Wrap_utils.Option.map min_samples_split (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("min_samples_leaf", Wrap_utils.Option.map min_samples_leaf (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("min_weight_fraction_leaf", Wrap_utils.Option.map min_weight_fraction_leaf Py.Float.of_float); ("max_features", Wrap_utils.Option.map max_features (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
| `S x -> Py.String.of_string x
| `None -> Py.none
)); ("max_leaf_nodes", Wrap_utils.Option.map max_leaf_nodes Py.Int.of_int); ("min_impurity_decrease", Wrap_utils.Option.map min_impurity_decrease Py.Float.of_float); ("min_impurity_split", Wrap_utils.Option.map min_impurity_split Py.Float.of_float); ("bootstrap", Wrap_utils.Option.map bootstrap Py.Bool.of_bool); ("oob_score", Wrap_utils.Option.map oob_score Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("ccp_alpha", Wrap_utils.Option.map ccp_alpha Py.Float.of_float); ("max_samples", Wrap_utils.Option.map max_samples (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
))])

let get_item ~index self =
   Py.Module.get_function_with_keywords self "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let apply ~x self =
   Py.Module.get_function_with_keywords self "apply"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let decision_path ~x self =
   Py.Module.get_function_with_keywords self "decision_path"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)


let base_estimator_opt self =
  match Py.Object.get_attr_string self "base_estimator_" with
  | None -> failwith "attribute base_estimator_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let base_estimator_ self = match base_estimator_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_opt self =
  match Py.Object.get_attr_string self "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let feature_importances_opt self =
  match Py.Object.get_attr_string self "feature_importances_" with
  | None -> failwith "attribute feature_importances_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let feature_importances_ self = match feature_importances_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_features_opt self =
  match Py.Object.get_attr_string self "n_features_" with
  | None -> failwith "attribute n_features_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_features_ self = match n_features_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_outputs_opt self =
  match Py.Object.get_attr_string self "n_outputs_" with
  | None -> failwith "attribute n_outputs_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_outputs_ self = match n_outputs_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_score_opt self =
  match Py.Object.get_attr_string self "oob_score_" with
  | None -> failwith "attribute oob_score_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let oob_score_ self = match oob_score_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_prediction_opt self =
  match Py.Object.get_attr_string self "oob_prediction_" with
  | None -> failwith "attribute oob_prediction_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let oob_prediction_ self = match oob_prediction_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RandomTreesEmbedding = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_estimators ?max_depth ?min_samples_split ?min_samples_leaf ?min_weight_fraction_leaf ?max_leaf_nodes ?min_impurity_decrease ?min_impurity_split ?sparse_output ?n_jobs ?random_state ?verbose ?warm_start () =
                     Py.Module.get_function_with_keywords ns "RandomTreesEmbedding"
                       [||]
                       (Wrap_utils.keyword_args [("n_estimators", Wrap_utils.Option.map n_estimators Py.Int.of_int); ("max_depth", Wrap_utils.Option.map max_depth Py.Int.of_int); ("min_samples_split", Wrap_utils.Option.map min_samples_split (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("min_samples_leaf", Wrap_utils.Option.map min_samples_leaf (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("min_weight_fraction_leaf", Wrap_utils.Option.map min_weight_fraction_leaf Py.Float.of_float); ("max_leaf_nodes", Wrap_utils.Option.map max_leaf_nodes Py.Int.of_int); ("min_impurity_decrease", Wrap_utils.Option.map min_impurity_decrease Py.Float.of_float); ("min_impurity_split", Wrap_utils.Option.map min_impurity_split Py.Float.of_float); ("sparse_output", Wrap_utils.Option.map sparse_output Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool)])

let get_item ~index self =
   Py.Module.get_function_with_keywords self "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let apply ~x self =
   Py.Module.get_function_with_keywords self "apply"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let decision_path ~x self =
   Py.Module.get_function_with_keywords self "decision_path"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let fit ?y ?sample_weight ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))])

let fit_transform ?y ?sample_weight ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let estimators_opt self =
  match Py.Object.get_attr_string self "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module StackingClassifier = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?final_estimator ?cv ?stack_method ?n_jobs ?passthrough ?verbose ~estimators () =
                     Py.Module.get_function_with_keywords ns "StackingClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("final_estimator", final_estimator); ("cv", Wrap_utils.Option.map cv (function
| `I x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Arr x -> Arr.to_pyobject x
)); ("stack_method", Wrap_utils.Option.map stack_method (function
| `Auto -> Py.String.of_string "auto"
| `Predict_proba -> Py.String.of_string "predict_proba"
| `Decision_function -> Py.String.of_string "decision_function"
| `Predict -> Py.String.of_string "predict"
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("passthrough", Wrap_utils.Option.map passthrough Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("estimators", Some(estimators |> (fun ml -> Py.List.of_list_map (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.String.of_string ml_0); (Wrap_utils.id ml_1)]) ml)))])

let decision_function ~x self =
   Py.Module.get_function_with_keywords self "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let predict ?predict_params ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))]) (match predict_params with None -> [] | Some x -> x))
     |> Arr.of_pyobject
let predict_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let estimators_opt self =
  match Py.Object.get_attr_string self "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let named_estimators_opt self =
  match Py.Object.get_attr_string self "named_estimators_" with
  | None -> failwith "attribute named_estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let named_estimators_ self = match named_estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let final_estimator_opt self =
  match Py.Object.get_attr_string self "final_estimator_" with
  | None -> failwith "attribute final_estimator_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let final_estimator_ self = match final_estimator_opt self with
  | None -> raise Not_found
  | Some x -> x

let stack_method_opt self =
  match Py.Object.get_attr_string self "stack_method_" with
  | None -> failwith "attribute stack_method_ not found"
  | Some x -> if Py.is_none x then None else Some ((Py.List.to_list_map Py.String.to_string) x)

let stack_method_ self = match stack_method_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module StackingRegressor = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?final_estimator ?cv ?n_jobs ?passthrough ?verbose ~estimators () =
                     Py.Module.get_function_with_keywords ns "StackingRegressor"
                       [||]
                       (Wrap_utils.keyword_args [("final_estimator", final_estimator); ("cv", Wrap_utils.Option.map cv (function
| `I x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Arr x -> Arr.to_pyobject x
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("passthrough", Wrap_utils.Option.map passthrough Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("estimators", Some(estimators |> (fun ml -> Py.List.of_list_map (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.String.of_string ml_0); (Wrap_utils.id ml_1)]) ml)))])

let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let predict ?predict_params ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))]) (match predict_params with None -> [] | Some x -> x))
     |> Arr.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let estimators_opt self =
  match Py.Object.get_attr_string self "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let named_estimators_opt self =
  match Py.Object.get_attr_string self "named_estimators_" with
  | None -> failwith "attribute named_estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let named_estimators_ self = match named_estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let final_estimator_opt self =
  match Py.Object.get_attr_string self "final_estimator_" with
  | None -> failwith "attribute final_estimator_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let final_estimator_ self = match final_estimator_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module VotingClassifier = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?voting ?weights ?n_jobs ?flatten_transform ~estimators () =
                     Py.Module.get_function_with_keywords ns "VotingClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("voting", Wrap_utils.Option.map voting (function
| `Hard -> Py.String.of_string "hard"
| `Soft -> Py.String.of_string "soft"
)); ("weights", Wrap_utils.Option.map weights Arr.to_pyobject); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("flatten_transform", Wrap_utils.Option.map flatten_transform Py.Bool.of_bool); ("estimators", Some(estimators |> (fun ml -> Py.List.of_list_map (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.String.of_string ml_0); (Wrap_utils.id ml_1)]) ml)))])

let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let estimators_opt self =
  match Py.Object.get_attr_string self "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let named_estimators_opt self =
  match Py.Object.get_attr_string self "named_estimators_" with
  | None -> failwith "attribute named_estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let named_estimators_ self = match named_estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string self "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module VotingRegressor = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?weights ?n_jobs ~estimators () =
   Py.Module.get_function_with_keywords ns "VotingRegressor"
     [||]
     (Wrap_utils.keyword_args [("weights", Wrap_utils.Option.map weights Arr.to_pyobject); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("estimators", Some(estimators |> (fun ml -> Py.List.of_list_map (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.String.of_string ml_0); (Wrap_utils.id ml_1)]) ml)))])

let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let estimators_opt self =
  match Py.Object.get_attr_string self "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let named_estimators_opt self =
  match Py.Object.get_attr_string self "named_estimators_" with
  | None -> failwith "attribute named_estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let named_estimators_ self = match named_estimators_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Partial_dependence = struct
let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.ensemble.partial_dependence"

let get_py name = Py.Module.get ns name
module Parallel = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_jobs ?backend ?verbose ?timeout ?pre_dispatch ?batch_size ?temp_folder ?max_nbytes ?mmap_mode ?prefer ?require () =
                     Py.Module.get_function_with_keywords ns "Parallel"
                       [||]
                       (Wrap_utils.keyword_args [("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("backend", backend); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("timeout", Wrap_utils.Option.map timeout Py.Float.of_float); ("pre_dispatch", Wrap_utils.Option.map pre_dispatch (function
| `All -> Py.String.of_string "all"
| `I x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)); ("batch_size", Wrap_utils.Option.map batch_size (function
| `I x -> Py.Int.of_int x
| `Auto -> Py.String.of_string "auto"
)); ("temp_folder", Wrap_utils.Option.map temp_folder Py.String.of_string); ("max_nbytes", max_nbytes); ("mmap_mode", Wrap_utils.Option.map mmap_mode (function
| `R_ -> Py.String.of_string "r+"
| `R -> Py.String.of_string "r"
| `W_ -> Py.String.of_string "w+"
| `C -> Py.String.of_string "c"
| `None -> Py.none
)); ("prefer", Wrap_utils.Option.map prefer (function
| `Processes -> Py.String.of_string "processes"
| `Threads -> Py.String.of_string "threads"
)); ("require", Wrap_utils.Option.map require Py.String.of_string)])

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
let cartesian ?out ~arrays () =
   Py.Module.get_function_with_keywords ns "cartesian"
     [||]
     (Wrap_utils.keyword_args [("out", out); ("arrays", Some(arrays ))])
     |> Arr.of_pyobject
                  let check_array ?accept_sparse ?accept_large_sparse ?dtype ?order ?copy ?force_all_finite ?ensure_2d ?allow_nd ?ensure_min_samples ?ensure_min_features ?warn_on_dtype ?estimator ~array () =
                     Py.Module.get_function_with_keywords ns "check_array"
                       [||]
                       (Wrap_utils.keyword_args [("accept_sparse", Wrap_utils.Option.map accept_sparse (function
| `S x -> Py.String.of_string x
| `Bool x -> Py.Bool.of_bool x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
)); ("accept_large_sparse", Wrap_utils.Option.map accept_large_sparse Py.Bool.of_bool); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Wrap_utils.id x
| `TypeList x -> Wrap_utils.id x
| `None -> Py.none
)); ("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("force_all_finite", Wrap_utils.Option.map force_all_finite (function
| `Bool x -> Py.Bool.of_bool x
| `Allow_nan -> Py.String.of_string "allow-nan"
)); ("ensure_2d", Wrap_utils.Option.map ensure_2d Py.Bool.of_bool); ("allow_nd", Wrap_utils.Option.map allow_nd Py.Bool.of_bool); ("ensure_min_samples", Wrap_utils.Option.map ensure_min_samples Py.Int.of_int); ("ensure_min_features", Wrap_utils.Option.map ensure_min_features Py.Int.of_int); ("warn_on_dtype", Wrap_utils.Option.map warn_on_dtype Py.Bool.of_bool); ("estimator", Wrap_utils.Option.map estimator (function
| `S x -> Py.String.of_string x
| `Estimator x -> Wrap_utils.id x
)); ("array", Some(array ))])

                  let check_is_fitted ?attributes ?msg ?all_or_any ~estimator () =
                     Py.Module.get_function_with_keywords ns "check_is_fitted"
                       [||]
                       (Wrap_utils.keyword_args [("attributes", Wrap_utils.Option.map attributes (function
| `S x -> Py.String.of_string x
| `Arr x -> Arr.to_pyobject x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
)); ("msg", Wrap_utils.Option.map msg Py.String.of_string); ("all_or_any", Wrap_utils.Option.map all_or_any (function
| `Callable x -> Wrap_utils.id x
| `PyObject x -> Wrap_utils.id x
)); ("estimator", Some(estimator ))])

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
let mquantiles ?prob ?alphap ?betap ?axis ?limit ~a () =
   Py.Module.get_function_with_keywords ns "mquantiles"
     [||]
     (Wrap_utils.keyword_args [("prob", Wrap_utils.Option.map prob Arr.to_pyobject); ("alphap", Wrap_utils.Option.map alphap Py.Float.of_float); ("betap", Wrap_utils.Option.map betap Py.Float.of_float); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("limit", Wrap_utils.Option.map limit (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.Float.of_float ml_0); (Py.Float.of_float ml_1)])); ("a", Some(a |> Arr.to_pyobject))])
     |> Arr.of_pyobject
                  let partial_dependence ?grid ?x ?percentiles ?grid_resolution ~gbrt ~target_variables () =
                     Py.Module.get_function_with_keywords ns "partial_dependence"
                       [||]
                       (Wrap_utils.keyword_args [("grid", Wrap_utils.Option.map grid Arr.to_pyobject); ("X", Wrap_utils.Option.map x Arr.to_pyobject); ("percentiles", percentiles); ("grid_resolution", Wrap_utils.Option.map grid_resolution Py.Int.of_int); ("gbrt", Some(gbrt )); ("target_variables", Some(target_variables |> (function
| `Arr x -> Arr.to_pyobject x
| `Dtype_int x -> Wrap_utils.id x
)))])
                       |> (fun x -> ((Arr.of_pyobject (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let plot_partial_dependence ?feature_names ?label ?n_cols ?grid_resolution ?percentiles ?n_jobs ?verbose ?ax ?line_kw ?contour_kw ?fig_kw ~gbrt ~x ~features () =
                     Py.Module.get_function_with_keywords ns "plot_partial_dependence"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("feature_names", feature_names); ("label", label); ("n_cols", Wrap_utils.Option.map n_cols Py.Int.of_int); ("grid_resolution", Wrap_utils.Option.map grid_resolution Py.Int.of_int); ("percentiles", percentiles); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("ax", ax); ("line_kw", Wrap_utils.Option.map line_kw Dict.to_pyobject); ("contour_kw", Wrap_utils.Option.map contour_kw Dict.to_pyobject); ("gbrt", Some(gbrt )); ("X", Some(x |> Arr.to_pyobject)); ("features", Some(features |> (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `PyObject x -> Wrap_utils.id x
)))]) (match fig_kw with None -> [] | Some x -> x))
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))

end
