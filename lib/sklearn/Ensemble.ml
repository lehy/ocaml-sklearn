let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.ensemble"

let get_py name = Py.Module.get __wrap_namespace name
module AdaBoostClassifier = struct
type tag = [`AdaBoostClassifier]
type t = [`AdaBoostClassifier | `BaseEnsemble | `BaseEstimator | `BaseWeightBoosting | `ClassifierMixin | `MetaEstimatorMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_meta_estimator x = (x :> [`MetaEstimatorMixin] Obj.t)
let as_weight_boosting x = (x :> [`BaseWeightBoosting] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_ensemble x = (x :> [`BaseEnsemble] Obj.t)
                  let create ?base_estimator ?n_estimators ?learning_rate ?algorithm ?random_state () =
                     Py.Module.get_function_with_keywords __wrap_namespace "AdaBoostClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("base_estimator", Wrap_utils.Option.map base_estimator Np.Obj.to_pyobject); ("n_estimators", Wrap_utils.Option.map n_estimators Py.Int.of_int); ("learning_rate", Wrap_utils.Option.map learning_rate Py.Float.of_float); ("algorithm", Wrap_utils.Option.map algorithm (function
| `SAMME -> Py.String.of_string "SAMME"
| `SAMME_R -> Py.String.of_string "SAMME.R"
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
                       |> of_pyobject
let get_item ~index self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []
     |> (fun py -> Py.Iter.to_seq py |> Seq.map Dict.of_pyobject)
let decision_function ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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
let predict_log_proba ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict_log_proba"
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
let staged_decision_function ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "staged_decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> Py.Iter.to_seq py |> Seq.map (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)))
let staged_predict ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "staged_predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> Py.Iter.to_seq py |> Seq.map (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)))
let staged_predict_proba ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "staged_predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> Py.Iter.to_seq py |> Seq.map (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)))
let staged_score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "staged_score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])


let base_estimator_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "base_estimator_" with
  | None -> failwith "attribute base_estimator_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`BaseEstimator|`Object] Np.Obj.t)) x)

let base_estimator_ self = match base_estimator_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_classes_" with
  | None -> failwith "attribute n_classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_classes_ self = match n_classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimator_weights_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimator_weights_" with
  | None -> failwith "attribute estimator_weights_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let estimator_weights_ self = match estimator_weights_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimator_errors_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimator_errors_" with
  | None -> failwith "attribute estimator_errors_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let estimator_errors_ self = match estimator_errors_opt self with
  | None -> raise Not_found
  | Some x -> x

let feature_importances_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "feature_importances_" with
  | None -> failwith "attribute feature_importances_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let feature_importances_ self = match feature_importances_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module AdaBoostRegressor = struct
type tag = [`AdaBoostRegressor]
type t = [`AdaBoostRegressor | `BaseEnsemble | `BaseEstimator | `BaseWeightBoosting | `MetaEstimatorMixin | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_meta_estimator x = (x :> [`MetaEstimatorMixin] Obj.t)
let as_weight_boosting x = (x :> [`BaseWeightBoosting] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_ensemble x = (x :> [`BaseEnsemble] Obj.t)
                  let create ?base_estimator ?n_estimators ?learning_rate ?loss ?random_state () =
                     Py.Module.get_function_with_keywords __wrap_namespace "AdaBoostRegressor"
                       [||]
                       (Wrap_utils.keyword_args [("base_estimator", Wrap_utils.Option.map base_estimator Np.Obj.to_pyobject); ("n_estimators", Wrap_utils.Option.map n_estimators Py.Int.of_int); ("learning_rate", Wrap_utils.Option.map learning_rate Py.Float.of_float); ("loss", Wrap_utils.Option.map loss (function
| `Linear -> Py.String.of_string "linear"
| `Square -> Py.String.of_string "square"
| `Exponential -> Py.String.of_string "exponential"
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
                       |> of_pyobject
let get_item ~index self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []
     |> (fun py -> Py.Iter.to_seq py |> Seq.map Dict.of_pyobject)
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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
let staged_predict ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "staged_predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> Py.Iter.to_seq py |> Seq.map (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)))
let staged_score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "staged_score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])


let base_estimator_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "base_estimator_" with
  | None -> failwith "attribute base_estimator_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`BaseEstimator|`Object] Np.Obj.t)) x)

let base_estimator_ self = match base_estimator_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimator_weights_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimator_weights_" with
  | None -> failwith "attribute estimator_weights_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let estimator_weights_ self = match estimator_weights_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimator_errors_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimator_errors_" with
  | None -> failwith "attribute estimator_errors_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let estimator_errors_ self = match estimator_errors_opt self with
  | None -> raise Not_found
  | Some x -> x

let feature_importances_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "feature_importances_" with
  | None -> failwith "attribute feature_importances_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let feature_importances_ self = match feature_importances_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module BaggingClassifier = struct
type tag = [`BaggingClassifier]
type t = [`BaggingClassifier | `BaseBagging | `BaseEnsemble | `BaseEstimator | `ClassifierMixin | `MetaEstimatorMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_meta_estimator x = (x :> [`MetaEstimatorMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_bagging x = (x :> [`BaseBagging] Obj.t)
let as_ensemble x = (x :> [`BaseEnsemble] Obj.t)
                  let create ?base_estimator ?n_estimators ?max_samples ?max_features ?bootstrap ?bootstrap_features ?oob_score ?warm_start ?n_jobs ?random_state ?verbose () =
                     Py.Module.get_function_with_keywords __wrap_namespace "BaggingClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("base_estimator", Wrap_utils.Option.map base_estimator Np.Obj.to_pyobject); ("n_estimators", Wrap_utils.Option.map n_estimators Py.Int.of_int); ("max_samples", Wrap_utils.Option.map max_samples (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("max_features", Wrap_utils.Option.map max_features (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("bootstrap", Wrap_utils.Option.map bootstrap Py.Bool.of_bool); ("bootstrap_features", Wrap_utils.Option.map bootstrap_features Py.Bool.of_bool); ("oob_score", Wrap_utils.Option.map oob_score Py.Bool.of_bool); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int)])
                       |> of_pyobject
let get_item ~index self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []
     |> (fun py -> Py.Iter.to_seq py |> Seq.map Dict.of_pyobject)
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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
let predict_log_proba ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict_log_proba"
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

let base_estimator_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "base_estimator_" with
  | None -> failwith "attribute base_estimator_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`BaseEstimator|`Object] Np.Obj.t)) x)

let base_estimator_ self = match base_estimator_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_features_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_features_" with
  | None -> failwith "attribute n_features_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_features_ self = match n_features_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> Py.List.to_list_map ((fun py -> (Np.Obj.of_pyobject py : [`BaseEstimator|`Object] Np.Obj.t))) py) x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_samples_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimators_samples_" with
  | None -> failwith "attribute estimators_samples_ not found"
  | Some x -> if Py.is_none x then None else Some (Np.Numpy.Ndarray.List.of_pyobject x)

let estimators_samples_ self = match estimators_samples_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_features_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimators_features_" with
  | None -> failwith "attribute estimators_features_ not found"
  | Some x -> if Py.is_none x then None else Some (Np.Numpy.Ndarray.List.of_pyobject x)

let estimators_features_ self = match estimators_features_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_classes_" with
  | None -> failwith "attribute n_classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let n_classes_ self = match n_classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_score_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "oob_score_" with
  | None -> failwith "attribute oob_score_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let oob_score_ self = match oob_score_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_decision_function_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "oob_decision_function_" with
  | None -> failwith "attribute oob_decision_function_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let oob_decision_function_ self = match oob_decision_function_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module BaggingRegressor = struct
type tag = [`BaggingRegressor]
type t = [`BaggingRegressor | `BaseBagging | `BaseEnsemble | `BaseEstimator | `MetaEstimatorMixin | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_meta_estimator x = (x :> [`MetaEstimatorMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_bagging x = (x :> [`BaseBagging] Obj.t)
let as_ensemble x = (x :> [`BaseEnsemble] Obj.t)
                  let create ?base_estimator ?n_estimators ?max_samples ?max_features ?bootstrap ?bootstrap_features ?oob_score ?warm_start ?n_jobs ?random_state ?verbose () =
                     Py.Module.get_function_with_keywords __wrap_namespace "BaggingRegressor"
                       [||]
                       (Wrap_utils.keyword_args [("base_estimator", Wrap_utils.Option.map base_estimator Np.Obj.to_pyobject); ("n_estimators", Wrap_utils.Option.map n_estimators Py.Int.of_int); ("max_samples", Wrap_utils.Option.map max_samples (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("max_features", Wrap_utils.Option.map max_features (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("bootstrap", Wrap_utils.Option.map bootstrap Py.Bool.of_bool); ("bootstrap_features", Wrap_utils.Option.map bootstrap_features Py.Bool.of_bool); ("oob_score", Wrap_utils.Option.map oob_score Py.Bool.of_bool); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int)])
                       |> of_pyobject
let get_item ~index self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []
     |> (fun py -> Py.Iter.to_seq py |> Seq.map Dict.of_pyobject)
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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

let base_estimator_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "base_estimator_" with
  | None -> failwith "attribute base_estimator_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`BaseEstimator|`Object] Np.Obj.t)) x)

let base_estimator_ self = match base_estimator_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_features_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_features_" with
  | None -> failwith "attribute n_features_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_features_ self = match n_features_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> Py.List.to_list_map ((fun py -> (Np.Obj.of_pyobject py : [`BaseEstimator|`Object] Np.Obj.t))) py) x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_samples_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimators_samples_" with
  | None -> failwith "attribute estimators_samples_ not found"
  | Some x -> if Py.is_none x then None else Some (Np.Numpy.Ndarray.List.of_pyobject x)

let estimators_samples_ self = match estimators_samples_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_features_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimators_features_" with
  | None -> failwith "attribute estimators_features_ not found"
  | Some x -> if Py.is_none x then None else Some (Np.Numpy.Ndarray.List.of_pyobject x)

let estimators_features_ self = match estimators_features_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_score_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "oob_score_" with
  | None -> failwith "attribute oob_score_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let oob_score_ self = match oob_score_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_prediction_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "oob_prediction_" with
  | None -> failwith "attribute oob_prediction_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let oob_prediction_ self = match oob_prediction_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module BaseEnsemble = struct
type tag = [`BaseEnsemble]
type t = [`BaseEnsemble | `BaseEstimator | `MetaEstimatorMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_meta_estimator x = (x :> [`MetaEstimatorMixin] Obj.t)
let get_item ~index self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []
     |> (fun py -> Py.Iter.to_seq py |> Seq.map Dict.of_pyobject)
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

let base_estimator_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "base_estimator_" with
  | None -> failwith "attribute base_estimator_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`BaseEstimator|`Object] Np.Obj.t)) x)

let base_estimator_ self = match base_estimator_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> Py.List.to_list_map ((fun py -> (Np.Obj.of_pyobject py : [`BaseEstimator|`Object] Np.Obj.t))) py) x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ExtraTreesClassifier = struct
type tag = [`ExtraTreesClassifier]
type t = [`BaseEnsemble | `BaseEstimator | `BaseForest | `ClassifierMixin | `ExtraTreesClassifier | `MetaEstimatorMixin | `MultiOutputMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
let as_meta_estimator x = (x :> [`MetaEstimatorMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_ensemble x = (x :> [`BaseEnsemble] Obj.t)
let as_forest x = (x :> [`BaseForest] Obj.t)
                  let create ?n_estimators ?criterion ?max_depth ?min_samples_split ?min_samples_leaf ?min_weight_fraction_leaf ?max_features ?max_leaf_nodes ?min_impurity_decrease ?min_impurity_split ?bootstrap ?oob_score ?n_jobs ?random_state ?verbose ?warm_start ?class_weight ?ccp_alpha ?max_samples () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ExtraTreesClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("n_estimators", Wrap_utils.Option.map n_estimators Py.Int.of_int); ("criterion", Wrap_utils.Option.map criterion Py.String.of_string); ("max_depth", Wrap_utils.Option.map max_depth Py.Int.of_int); ("min_samples_split", Wrap_utils.Option.map min_samples_split (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("min_samples_leaf", Wrap_utils.Option.map min_samples_leaf (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("min_weight_fraction_leaf", Wrap_utils.Option.map min_weight_fraction_leaf Py.Float.of_float); ("max_features", Wrap_utils.Option.map max_features (function
| `F x -> Py.Float.of_float x
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("max_leaf_nodes", Wrap_utils.Option.map max_leaf_nodes Py.Int.of_int); ("min_impurity_decrease", Wrap_utils.Option.map min_impurity_decrease Py.Float.of_float); ("min_impurity_split", Wrap_utils.Option.map min_impurity_split Py.Float.of_float); ("bootstrap", Wrap_utils.Option.map bootstrap Py.Bool.of_bool); ("oob_score", Wrap_utils.Option.map oob_score Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("class_weight", Wrap_utils.Option.map class_weight (function
| `List_of_dicts x -> Wrap_utils.id x
| `Balanced_subsample -> Py.String.of_string "balanced_subsample"
| `DictIntToFloat x -> (Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float) x
| `Balanced -> Py.String.of_string "balanced"
)); ("ccp_alpha", Wrap_utils.Option.map ccp_alpha Py.Float.of_float); ("max_samples", Wrap_utils.Option.map max_samples (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
))])
                       |> of_pyobject
let get_item ~index self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []
     |> (fun py -> Py.Iter.to_seq py |> Seq.map Dict.of_pyobject)
let apply ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "apply"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let decision_path ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decision_path"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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
let predict_log_proba ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict_log_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])

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

let base_estimator_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "base_estimator_" with
  | None -> failwith "attribute base_estimator_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let base_estimator_ self = match base_estimator_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_classes_" with
  | None -> failwith "attribute n_classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let n_classes_ self = match n_classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let feature_importances_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "feature_importances_" with
  | None -> failwith "attribute feature_importances_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let feature_importances_ self = match feature_importances_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_features_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_features_" with
  | None -> failwith "attribute n_features_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_features_ self = match n_features_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_outputs_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_outputs_" with
  | None -> failwith "attribute n_outputs_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_outputs_ self = match n_outputs_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_score_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "oob_score_" with
  | None -> failwith "attribute oob_score_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let oob_score_ self = match oob_score_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_decision_function_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "oob_decision_function_" with
  | None -> failwith "attribute oob_decision_function_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let oob_decision_function_ self = match oob_decision_function_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ExtraTreesRegressor = struct
type tag = [`ExtraTreesRegressor]
type t = [`BaseEnsemble | `BaseEstimator | `BaseForest | `ExtraTreesRegressor | `MetaEstimatorMixin | `MultiOutputMixin | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
let as_meta_estimator x = (x :> [`MetaEstimatorMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_ensemble x = (x :> [`BaseEnsemble] Obj.t)
let as_forest x = (x :> [`BaseForest] Obj.t)
                  let create ?n_estimators ?criterion ?max_depth ?min_samples_split ?min_samples_leaf ?min_weight_fraction_leaf ?max_features ?max_leaf_nodes ?min_impurity_decrease ?min_impurity_split ?bootstrap ?oob_score ?n_jobs ?random_state ?verbose ?warm_start ?ccp_alpha ?max_samples () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ExtraTreesRegressor"
                       [||]
                       (Wrap_utils.keyword_args [("n_estimators", Wrap_utils.Option.map n_estimators Py.Int.of_int); ("criterion", Wrap_utils.Option.map criterion Py.String.of_string); ("max_depth", Wrap_utils.Option.map max_depth Py.Int.of_int); ("min_samples_split", Wrap_utils.Option.map min_samples_split (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("min_samples_leaf", Wrap_utils.Option.map min_samples_leaf (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("min_weight_fraction_leaf", Wrap_utils.Option.map min_weight_fraction_leaf Py.Float.of_float); ("max_features", Wrap_utils.Option.map max_features (function
| `F x -> Py.Float.of_float x
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("max_leaf_nodes", Wrap_utils.Option.map max_leaf_nodes Py.Int.of_int); ("min_impurity_decrease", Wrap_utils.Option.map min_impurity_decrease Py.Float.of_float); ("min_impurity_split", Wrap_utils.Option.map min_impurity_split Py.Float.of_float); ("bootstrap", Wrap_utils.Option.map bootstrap Py.Bool.of_bool); ("oob_score", Wrap_utils.Option.map oob_score Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("ccp_alpha", Wrap_utils.Option.map ccp_alpha Py.Float.of_float); ("max_samples", Wrap_utils.Option.map max_samples (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
))])
                       |> of_pyobject
let get_item ~index self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []
     |> (fun py -> Py.Iter.to_seq py |> Seq.map Dict.of_pyobject)
let apply ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "apply"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let decision_path ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decision_path"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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

let base_estimator_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "base_estimator_" with
  | None -> failwith "attribute base_estimator_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let base_estimator_ self = match base_estimator_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let feature_importances_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "feature_importances_" with
  | None -> failwith "attribute feature_importances_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let feature_importances_ self = match feature_importances_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_features_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_features_" with
  | None -> failwith "attribute n_features_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_features_ self = match n_features_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_outputs_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_outputs_" with
  | None -> failwith "attribute n_outputs_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_outputs_ self = match n_outputs_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_score_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "oob_score_" with
  | None -> failwith "attribute oob_score_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let oob_score_ self = match oob_score_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_prediction_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "oob_prediction_" with
  | None -> failwith "attribute oob_prediction_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let oob_prediction_ self = match oob_prediction_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module GradientBoostingClassifier = struct
type tag = [`GradientBoostingClassifier]
type t = [`BaseEnsemble | `BaseEstimator | `BaseGradientBoosting | `ClassifierMixin | `GradientBoostingClassifier | `MetaEstimatorMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_gradient_boosting x = (x :> [`BaseGradientBoosting] Obj.t)
let as_meta_estimator x = (x :> [`MetaEstimatorMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_ensemble x = (x :> [`BaseEnsemble] Obj.t)
                  let create ?loss ?learning_rate ?n_estimators ?subsample ?criterion ?min_samples_split ?min_samples_leaf ?min_weight_fraction_leaf ?max_depth ?min_impurity_decrease ?min_impurity_split ?init ?random_state ?max_features ?verbose ?max_leaf_nodes ?warm_start ?presort ?validation_fraction ?n_iter_no_change ?tol ?ccp_alpha () =
                     Py.Module.get_function_with_keywords __wrap_namespace "GradientBoostingClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("loss", Wrap_utils.Option.map loss (function
| `Deviance -> Py.String.of_string "deviance"
| `Exponential -> Py.String.of_string "exponential"
)); ("learning_rate", Wrap_utils.Option.map learning_rate Py.Float.of_float); ("n_estimators", Wrap_utils.Option.map n_estimators Py.Int.of_int); ("subsample", Wrap_utils.Option.map subsample Py.Float.of_float); ("criterion", Wrap_utils.Option.map criterion Py.String.of_string); ("min_samples_split", Wrap_utils.Option.map min_samples_split (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("min_samples_leaf", Wrap_utils.Option.map min_samples_leaf (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("min_weight_fraction_leaf", Wrap_utils.Option.map min_weight_fraction_leaf Py.Float.of_float); ("max_depth", Wrap_utils.Option.map max_depth Py.Int.of_int); ("min_impurity_decrease", Wrap_utils.Option.map min_impurity_decrease Py.Float.of_float); ("min_impurity_split", Wrap_utils.Option.map min_impurity_split Py.Float.of_float); ("init", Wrap_utils.Option.map init (function
| `BaseEstimator x -> Np.Obj.to_pyobject x
| `Zero -> Py.String.of_string "zero"
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("max_features", Wrap_utils.Option.map max_features (function
| `F x -> Py.Float.of_float x
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("max_leaf_nodes", Wrap_utils.Option.map max_leaf_nodes Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("presort", presort); ("validation_fraction", Wrap_utils.Option.map validation_fraction Py.Float.of_float); ("n_iter_no_change", Wrap_utils.Option.map n_iter_no_change Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("ccp_alpha", Wrap_utils.Option.map ccp_alpha Py.Float.of_float)])
                       |> of_pyobject
let get_item ~index self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []
     |> (fun py -> Py.Iter.to_seq py |> Seq.map Dict.of_pyobject)
let apply ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "apply"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let decision_function ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let fit ?sample_weight ?monitor ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("monitor", monitor); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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
let predict_log_proba ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict_log_proba"
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
let staged_decision_function ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "staged_decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> Py.Iter.to_seq py |> Seq.map (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)))
let staged_predict ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "staged_predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> Py.Iter.to_seq py |> Seq.map (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)))
let staged_predict_proba ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "staged_predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> Py.Iter.to_seq py |> Seq.map (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)))

let n_estimators_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_estimators_" with
  | None -> failwith "attribute n_estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_estimators_ self = match n_estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let feature_importances_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "feature_importances_" with
  | None -> failwith "attribute feature_importances_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let feature_importances_ self = match feature_importances_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_improvement_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "oob_improvement_" with
  | None -> failwith "attribute oob_improvement_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let oob_improvement_ self = match oob_improvement_opt self with
  | None -> raise Not_found
  | Some x -> x

let train_score_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "train_score_" with
  | None -> failwith "attribute train_score_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let train_score_ self = match train_score_opt self with
  | None -> raise Not_found
  | Some x -> x

let loss_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "loss_" with
  | None -> failwith "attribute loss_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> fun x y -> Py.Callable.to_function py
       [|Np.NumpyRaw.Ndarray.to_pyobject x; Np.NumpyRaw.Ndarray.to_pyobject y|] |> Py.Float.to_float) x)

let loss_ self = match loss_opt self with
  | None -> raise Not_found
  | Some x -> x

let init_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "init_" with
  | None -> failwith "attribute init_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`BaseEstimator|`Object] Np.Obj.t)) x)

let init_ self = match init_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
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
module GradientBoostingRegressor = struct
type tag = [`GradientBoostingRegressor]
type t = [`BaseEnsemble | `BaseEstimator | `BaseGradientBoosting | `GradientBoostingRegressor | `MetaEstimatorMixin | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_gradient_boosting x = (x :> [`BaseGradientBoosting] Obj.t)
let as_meta_estimator x = (x :> [`MetaEstimatorMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_ensemble x = (x :> [`BaseEnsemble] Obj.t)
                  let create ?loss ?learning_rate ?n_estimators ?subsample ?criterion ?min_samples_split ?min_samples_leaf ?min_weight_fraction_leaf ?max_depth ?min_impurity_decrease ?min_impurity_split ?init ?random_state ?max_features ?alpha ?verbose ?max_leaf_nodes ?warm_start ?presort ?validation_fraction ?n_iter_no_change ?tol ?ccp_alpha () =
                     Py.Module.get_function_with_keywords __wrap_namespace "GradientBoostingRegressor"
                       [||]
                       (Wrap_utils.keyword_args [("loss", Wrap_utils.Option.map loss (function
| `Ls -> Py.String.of_string "ls"
| `Lad -> Py.String.of_string "lad"
| `Huber -> Py.String.of_string "huber"
| `Quantile -> Py.String.of_string "quantile"
)); ("learning_rate", Wrap_utils.Option.map learning_rate Py.Float.of_float); ("n_estimators", Wrap_utils.Option.map n_estimators Py.Int.of_int); ("subsample", Wrap_utils.Option.map subsample Py.Float.of_float); ("criterion", Wrap_utils.Option.map criterion Py.String.of_string); ("min_samples_split", Wrap_utils.Option.map min_samples_split (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("min_samples_leaf", Wrap_utils.Option.map min_samples_leaf (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("min_weight_fraction_leaf", Wrap_utils.Option.map min_weight_fraction_leaf Py.Float.of_float); ("max_depth", Wrap_utils.Option.map max_depth Py.Int.of_int); ("min_impurity_decrease", Wrap_utils.Option.map min_impurity_decrease Py.Float.of_float); ("min_impurity_split", Wrap_utils.Option.map min_impurity_split Py.Float.of_float); ("init", Wrap_utils.Option.map init (function
| `BaseEstimator x -> Np.Obj.to_pyobject x
| `Zero -> Py.String.of_string "zero"
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("max_features", Wrap_utils.Option.map max_features (function
| `F x -> Py.Float.of_float x
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("max_leaf_nodes", Wrap_utils.Option.map max_leaf_nodes Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("presort", presort); ("validation_fraction", Wrap_utils.Option.map validation_fraction Py.Float.of_float); ("n_iter_no_change", Wrap_utils.Option.map n_iter_no_change Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("ccp_alpha", Wrap_utils.Option.map ccp_alpha Py.Float.of_float)])
                       |> of_pyobject
let get_item ~index self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []
     |> (fun py -> Py.Iter.to_seq py |> Seq.map Dict.of_pyobject)
let apply ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "apply"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let fit ?sample_weight ?monitor ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("monitor", monitor); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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
let staged_predict ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "staged_predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> Py.Iter.to_seq py |> Seq.map (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)))

let feature_importances_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "feature_importances_" with
  | None -> failwith "attribute feature_importances_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let feature_importances_ self = match feature_importances_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_improvement_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "oob_improvement_" with
  | None -> failwith "attribute oob_improvement_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let oob_improvement_ self = match oob_improvement_opt self with
  | None -> raise Not_found
  | Some x -> x

let train_score_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "train_score_" with
  | None -> failwith "attribute train_score_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let train_score_ self = match train_score_opt self with
  | None -> raise Not_found
  | Some x -> x

let loss_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "loss_" with
  | None -> failwith "attribute loss_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> fun x y -> Py.Callable.to_function py
       [|Np.NumpyRaw.Ndarray.to_pyobject x; Np.NumpyRaw.Ndarray.to_pyobject y|] |> Py.Float.to_float) x)

let loss_ self = match loss_opt self with
  | None -> raise Not_found
  | Some x -> x

let init_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "init_" with
  | None -> failwith "attribute init_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`BaseEstimator|`Object] Np.Obj.t)) x)

let init_ self = match init_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module IsolationForest = struct
type tag = [`IsolationForest]
type t = [`BaseBagging | `BaseEnsemble | `BaseEstimator | `IsolationForest | `MetaEstimatorMixin | `Object | `OutlierMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_meta_estimator x = (x :> [`MetaEstimatorMixin] Obj.t)
let as_outlier x = (x :> [`OutlierMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_bagging x = (x :> [`BaseBagging] Obj.t)
let as_ensemble x = (x :> [`BaseEnsemble] Obj.t)
                  let create ?n_estimators ?max_samples ?contamination ?max_features ?bootstrap ?n_jobs ?behaviour ?random_state ?verbose ?warm_start () =
                     Py.Module.get_function_with_keywords __wrap_namespace "IsolationForest"
                       [||]
                       (Wrap_utils.keyword_args [("n_estimators", Wrap_utils.Option.map n_estimators Py.Int.of_int); ("max_samples", Wrap_utils.Option.map max_samples (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("contamination", Wrap_utils.Option.map contamination (function
| `F x -> Py.Float.of_float x
| `Auto -> Py.String.of_string "auto"
)); ("max_features", Wrap_utils.Option.map max_features (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("bootstrap", Wrap_utils.Option.map bootstrap Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("behaviour", Wrap_utils.Option.map behaviour Py.String.of_string); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool)])
                       |> of_pyobject
let get_item ~index self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []
     |> (fun py -> Py.Iter.to_seq py |> Seq.map Dict.of_pyobject)
let decision_function ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let fit ?y ?sample_weight ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
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

let estimators_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_samples_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimators_samples_" with
  | None -> failwith "attribute estimators_samples_ not found"
  | Some x -> if Py.is_none x then None else Some (Np.Numpy.Ndarray.List.of_pyobject x)

let estimators_samples_ self = match estimators_samples_opt self with
  | None -> raise Not_found
  | Some x -> x

let max_samples_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "max_samples_" with
  | None -> failwith "attribute max_samples_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let max_samples_ self = match max_samples_opt self with
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
module RandomForestClassifier = struct
type tag = [`RandomForestClassifier]
type t = [`BaseEnsemble | `BaseEstimator | `BaseForest | `ClassifierMixin | `MetaEstimatorMixin | `MultiOutputMixin | `Object | `RandomForestClassifier] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
let as_meta_estimator x = (x :> [`MetaEstimatorMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_ensemble x = (x :> [`BaseEnsemble] Obj.t)
let as_forest x = (x :> [`BaseForest] Obj.t)
                  let create ?n_estimators ?criterion ?max_depth ?min_samples_split ?min_samples_leaf ?min_weight_fraction_leaf ?max_features ?max_leaf_nodes ?min_impurity_decrease ?min_impurity_split ?bootstrap ?oob_score ?n_jobs ?random_state ?verbose ?warm_start ?class_weight ?ccp_alpha ?max_samples () =
                     Py.Module.get_function_with_keywords __wrap_namespace "RandomForestClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("n_estimators", Wrap_utils.Option.map n_estimators Py.Int.of_int); ("criterion", Wrap_utils.Option.map criterion Py.String.of_string); ("max_depth", Wrap_utils.Option.map max_depth Py.Int.of_int); ("min_samples_split", Wrap_utils.Option.map min_samples_split (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("min_samples_leaf", Wrap_utils.Option.map min_samples_leaf (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("min_weight_fraction_leaf", Wrap_utils.Option.map min_weight_fraction_leaf Py.Float.of_float); ("max_features", Wrap_utils.Option.map max_features (function
| `F x -> Py.Float.of_float x
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("max_leaf_nodes", Wrap_utils.Option.map max_leaf_nodes Py.Int.of_int); ("min_impurity_decrease", Wrap_utils.Option.map min_impurity_decrease Py.Float.of_float); ("min_impurity_split", Wrap_utils.Option.map min_impurity_split Py.Float.of_float); ("bootstrap", Wrap_utils.Option.map bootstrap Py.Bool.of_bool); ("oob_score", Wrap_utils.Option.map oob_score Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("class_weight", Wrap_utils.Option.map class_weight (function
| `List_of_dicts x -> Wrap_utils.id x
| `Balanced_subsample -> Py.String.of_string "balanced_subsample"
| `DictIntToFloat x -> (Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float) x
| `Balanced -> Py.String.of_string "balanced"
)); ("ccp_alpha", Wrap_utils.Option.map ccp_alpha Py.Float.of_float); ("max_samples", Wrap_utils.Option.map max_samples (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
))])
                       |> of_pyobject
let get_item ~index self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []
     |> (fun py -> Py.Iter.to_seq py |> Seq.map Dict.of_pyobject)
let apply ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "apply"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let decision_path ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decision_path"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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
let predict_log_proba ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict_log_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])

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

let base_estimator_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "base_estimator_" with
  | None -> failwith "attribute base_estimator_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let base_estimator_ self = match base_estimator_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_classes_" with
  | None -> failwith "attribute n_classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let n_classes_ self = match n_classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_features_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_features_" with
  | None -> failwith "attribute n_features_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_features_ self = match n_features_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_outputs_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_outputs_" with
  | None -> failwith "attribute n_outputs_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_outputs_ self = match n_outputs_opt self with
  | None -> raise Not_found
  | Some x -> x

let feature_importances_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "feature_importances_" with
  | None -> failwith "attribute feature_importances_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let feature_importances_ self = match feature_importances_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_score_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "oob_score_" with
  | None -> failwith "attribute oob_score_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let oob_score_ self = match oob_score_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_decision_function_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "oob_decision_function_" with
  | None -> failwith "attribute oob_decision_function_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let oob_decision_function_ self = match oob_decision_function_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RandomForestRegressor = struct
type tag = [`RandomForestRegressor]
type t = [`BaseEnsemble | `BaseEstimator | `BaseForest | `MetaEstimatorMixin | `MultiOutputMixin | `Object | `RandomForestRegressor | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
let as_meta_estimator x = (x :> [`MetaEstimatorMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_ensemble x = (x :> [`BaseEnsemble] Obj.t)
let as_forest x = (x :> [`BaseForest] Obj.t)
                  let create ?n_estimators ?criterion ?max_depth ?min_samples_split ?min_samples_leaf ?min_weight_fraction_leaf ?max_features ?max_leaf_nodes ?min_impurity_decrease ?min_impurity_split ?bootstrap ?oob_score ?n_jobs ?random_state ?verbose ?warm_start ?ccp_alpha ?max_samples () =
                     Py.Module.get_function_with_keywords __wrap_namespace "RandomForestRegressor"
                       [||]
                       (Wrap_utils.keyword_args [("n_estimators", Wrap_utils.Option.map n_estimators Py.Int.of_int); ("criterion", Wrap_utils.Option.map criterion Py.String.of_string); ("max_depth", Wrap_utils.Option.map max_depth Py.Int.of_int); ("min_samples_split", Wrap_utils.Option.map min_samples_split (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("min_samples_leaf", Wrap_utils.Option.map min_samples_leaf (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("min_weight_fraction_leaf", Wrap_utils.Option.map min_weight_fraction_leaf Py.Float.of_float); ("max_features", Wrap_utils.Option.map max_features (function
| `F x -> Py.Float.of_float x
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("max_leaf_nodes", Wrap_utils.Option.map max_leaf_nodes Py.Int.of_int); ("min_impurity_decrease", Wrap_utils.Option.map min_impurity_decrease Py.Float.of_float); ("min_impurity_split", Wrap_utils.Option.map min_impurity_split Py.Float.of_float); ("bootstrap", Wrap_utils.Option.map bootstrap Py.Bool.of_bool); ("oob_score", Wrap_utils.Option.map oob_score Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("ccp_alpha", Wrap_utils.Option.map ccp_alpha Py.Float.of_float); ("max_samples", Wrap_utils.Option.map max_samples (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
))])
                       |> of_pyobject
let get_item ~index self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []
     |> (fun py -> Py.Iter.to_seq py |> Seq.map Dict.of_pyobject)
let apply ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "apply"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let decision_path ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decision_path"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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

let base_estimator_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "base_estimator_" with
  | None -> failwith "attribute base_estimator_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let base_estimator_ self = match base_estimator_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimators_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let feature_importances_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "feature_importances_" with
  | None -> failwith "attribute feature_importances_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let feature_importances_ self = match feature_importances_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_features_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_features_" with
  | None -> failwith "attribute n_features_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_features_ self = match n_features_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_outputs_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_outputs_" with
  | None -> failwith "attribute n_outputs_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_outputs_ self = match n_outputs_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_score_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "oob_score_" with
  | None -> failwith "attribute oob_score_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let oob_score_ self = match oob_score_opt self with
  | None -> raise Not_found
  | Some x -> x

let oob_prediction_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "oob_prediction_" with
  | None -> failwith "attribute oob_prediction_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let oob_prediction_ self = match oob_prediction_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RandomTreesEmbedding = struct
type tag = [`RandomTreesEmbedding]
type t = [`BaseEnsemble | `BaseEstimator | `BaseForest | `MetaEstimatorMixin | `MultiOutputMixin | `Object | `RandomTreesEmbedding] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
let as_meta_estimator x = (x :> [`MetaEstimatorMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_ensemble x = (x :> [`BaseEnsemble] Obj.t)
let as_forest x = (x :> [`BaseForest] Obj.t)
                  let create ?n_estimators ?max_depth ?min_samples_split ?min_samples_leaf ?min_weight_fraction_leaf ?max_leaf_nodes ?min_impurity_decrease ?min_impurity_split ?sparse_output ?n_jobs ?random_state ?verbose ?warm_start () =
                     Py.Module.get_function_with_keywords __wrap_namespace "RandomTreesEmbedding"
                       [||]
                       (Wrap_utils.keyword_args [("n_estimators", Wrap_utils.Option.map n_estimators Py.Int.of_int); ("max_depth", Wrap_utils.Option.map max_depth Py.Int.of_int); ("min_samples_split", Wrap_utils.Option.map min_samples_split (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("min_samples_leaf", Wrap_utils.Option.map min_samples_leaf (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("min_weight_fraction_leaf", Wrap_utils.Option.map min_weight_fraction_leaf Py.Float.of_float); ("max_leaf_nodes", Wrap_utils.Option.map max_leaf_nodes Py.Int.of_int); ("min_impurity_decrease", Wrap_utils.Option.map min_impurity_decrease Py.Float.of_float); ("min_impurity_split", Wrap_utils.Option.map min_impurity_split Py.Float.of_float); ("sparse_output", Wrap_utils.Option.map sparse_output Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool)])
                       |> of_pyobject
let get_item ~index self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []
     |> (fun py -> Py.Iter.to_seq py |> Seq.map Dict.of_pyobject)
let apply ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "apply"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let decision_path ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decision_path"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let fit ?y ?sample_weight ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> of_pyobject
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

let estimators_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module StackingClassifier = struct
type tag = [`StackingClassifier]
type t = [`BaseEstimator | `ClassifierMixin | `MetaEstimatorMixin | `Object | `StackingClassifier | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_meta_estimator x = (x :> [`MetaEstimatorMixin] Obj.t)
                  let create ?final_estimator ?cv ?stack_method ?n_jobs ?passthrough ?verbose ~estimators () =
                     Py.Module.get_function_with_keywords __wrap_namespace "StackingClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("final_estimator", Wrap_utils.Option.map final_estimator Np.Obj.to_pyobject); ("cv", Wrap_utils.Option.map cv (function
| `Arr x -> Np.Obj.to_pyobject x
| `BaseCrossValidator x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("stack_method", Wrap_utils.Option.map stack_method (function
| `Auto -> Py.String.of_string "auto"
| `Predict_proba -> Py.String.of_string "predict_proba"
| `Decision_function -> Py.String.of_string "decision_function"
| `Predict -> Py.String.of_string "predict"
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("passthrough", Wrap_utils.Option.map passthrough Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("estimators", Some(estimators |> (fun ml -> Py.List.of_list_map (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.String.of_string ml_0); (Np.Obj.to_pyobject ml_1)]) ml)))])
                       |> of_pyobject
let decision_function ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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

let predict ?predict_params ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))]) (match predict_params with None -> [] | Some x -> x))
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
let transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

let estimators_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> Py.List.to_list_map ((fun py -> (Np.Obj.of_pyobject py : [`BaseEstimator|`Object] Np.Obj.t))) py) x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let named_estimators_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "named_estimators_" with
  | None -> failwith "attribute named_estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let named_estimators_ self = match named_estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let final_estimator_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "final_estimator_" with
  | None -> failwith "attribute final_estimator_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`BaseEstimator|`Object] Np.Obj.t)) x)

let final_estimator_ self = match final_estimator_opt self with
  | None -> raise Not_found
  | Some x -> x

let stack_method_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "stack_method_" with
  | None -> failwith "attribute stack_method_ not found"
  | Some x -> if Py.is_none x then None else Some ((Py.List.to_list_map Py.String.to_string) x)

let stack_method_ self = match stack_method_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module StackingRegressor = struct
type tag = [`StackingRegressor]
type t = [`BaseEstimator | `MetaEstimatorMixin | `Object | `RegressorMixin | `StackingRegressor | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_meta_estimator x = (x :> [`MetaEstimatorMixin] Obj.t)
                  let create ?final_estimator ?cv ?n_jobs ?passthrough ?verbose ~estimators () =
                     Py.Module.get_function_with_keywords __wrap_namespace "StackingRegressor"
                       [||]
                       (Wrap_utils.keyword_args [("final_estimator", Wrap_utils.Option.map final_estimator Np.Obj.to_pyobject); ("cv", Wrap_utils.Option.map cv (function
| `Arr x -> Np.Obj.to_pyobject x
| `BaseCrossValidator x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("passthrough", Wrap_utils.Option.map passthrough Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("estimators", Some(estimators |> (fun ml -> Py.List.of_list_map (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.String.of_string ml_0); (Np.Obj.to_pyobject ml_1)]) ml)))])
                       |> of_pyobject
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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

let predict ?predict_params ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))]) (match predict_params with None -> [] | Some x -> x))
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
let transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

let estimators_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> Py.List.to_list_map ((fun py -> (Np.Obj.of_pyobject py : [`BaseEstimator|`Object] Np.Obj.t))) py) x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let named_estimators_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "named_estimators_" with
  | None -> failwith "attribute named_estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let named_estimators_ self = match named_estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let final_estimator_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "final_estimator_" with
  | None -> failwith "attribute final_estimator_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`BaseEstimator|`Object] Np.Obj.t)) x)

let final_estimator_ self = match final_estimator_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module VotingClassifier = struct
type tag = [`VotingClassifier]
type t = [`BaseEstimator | `ClassifierMixin | `MetaEstimatorMixin | `Object | `TransformerMixin | `VotingClassifier] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_meta_estimator x = (x :> [`MetaEstimatorMixin] Obj.t)
                  let create ?voting ?weights ?n_jobs ?flatten_transform ~estimators () =
                     Py.Module.get_function_with_keywords __wrap_namespace "VotingClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("voting", Wrap_utils.Option.map voting (function
| `Hard -> Py.String.of_string "hard"
| `Soft -> Py.String.of_string "soft"
)); ("weights", Wrap_utils.Option.map weights Np.Obj.to_pyobject); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("flatten_transform", Wrap_utils.Option.map flatten_transform Py.Bool.of_bool); ("estimators", Some(estimators |> (fun ml -> Py.List.of_list_map (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.String.of_string ml_0); (Np.Obj.to_pyobject ml_1)]) ml)))])
                       |> of_pyobject
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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
let transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

let estimators_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let named_estimators_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "named_estimators_" with
  | None -> failwith "attribute named_estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let named_estimators_ self = match named_estimators_opt self with
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
module VotingRegressor = struct
type tag = [`VotingRegressor]
type t = [`BaseEstimator | `MetaEstimatorMixin | `Object | `RegressorMixin | `TransformerMixin | `VotingRegressor] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_meta_estimator x = (x :> [`MetaEstimatorMixin] Obj.t)
let create ?weights ?n_jobs ~estimators () =
   Py.Module.get_function_with_keywords __wrap_namespace "VotingRegressor"
     [||]
     (Wrap_utils.keyword_args [("weights", Wrap_utils.Option.map weights Np.Obj.to_pyobject); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("estimators", Some(estimators |> (fun ml -> Py.List.of_list_map (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.String.of_string ml_0); (Np.Obj.to_pyobject ml_1)]) ml)))])
     |> of_pyobject
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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
let transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

let estimators_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimators_" with
  | None -> failwith "attribute estimators_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> Py.List.to_list_map ((fun py -> (Np.Obj.of_pyobject py : [`Object|`RegressorMixin] Np.Obj.t))) py) x)

let estimators_ self = match estimators_opt self with
  | None -> raise Not_found
  | Some x -> x

let named_estimators_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "named_estimators_" with
  | None -> failwith "attribute named_estimators_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let named_estimators_ self = match named_estimators_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Partial_dependence = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.ensemble.partial_dependence"

let get_py name = Py.Module.get __wrap_namespace name
module BaseGradientBoosting = struct
type tag = [`BaseGradientBoosting]
type t = [`BaseEnsemble | `BaseEstimator | `BaseGradientBoosting | `MetaEstimatorMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_ensemble x = (x :> [`BaseEnsemble] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_meta_estimator x = (x :> [`MetaEstimatorMixin] Obj.t)
let get_item ~index self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []
     |> (fun py -> Py.Iter.to_seq py |> Seq.map Dict.of_pyobject)
let apply ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "apply"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let fit ?sample_weight ?monitor ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("monitor", monitor); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
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
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module DTYPE = struct
type tag = [`Float32]
type t = [`Float32 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let get_item ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Np.Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Count = struct
type tag = [`Count]
type t = [`Count | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?start ?step () =
   Py.Module.get_function_with_keywords __wrap_namespace "count"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("step", step)])
     |> of_pyobject
let iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []
     |> (fun py -> Py.Iter.to_seq py |> Seq.map Dict.of_pyobject)
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let cartesian ?out ~arrays () =
   Py.Module.get_function_with_keywords __wrap_namespace "cartesian"
     [||]
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("arrays", Some(arrays |> Np.Numpy.Ndarray.List.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let check_array ?accept_sparse ?accept_large_sparse ?dtype ?order ?copy ?force_all_finite ?ensure_2d ?allow_nd ?ensure_min_samples ?ensure_min_features ?warn_on_dtype ?estimator ~array () =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_array"
                       [||]
                       (Wrap_utils.keyword_args [("accept_sparse", Wrap_utils.Option.map accept_sparse (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `S x -> Py.String.of_string x
| `Bool x -> Py.Bool.of_bool x
)); ("accept_large_sparse", Wrap_utils.Option.map accept_large_sparse Py.Bool.of_bool); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
| `Dtypes x -> (fun ml -> Py.List.of_list_map Np.Dtype.to_pyobject ml) x
| `None -> Py.none
)); ("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("force_all_finite", Wrap_utils.Option.map force_all_finite (function
| `Allow_nan -> Py.String.of_string "allow-nan"
| `Bool x -> Py.Bool.of_bool x
)); ("ensure_2d", Wrap_utils.Option.map ensure_2d Py.Bool.of_bool); ("allow_nd", Wrap_utils.Option.map allow_nd Py.Bool.of_bool); ("ensure_min_samples", Wrap_utils.Option.map ensure_min_samples Py.Int.of_int); ("ensure_min_features", Wrap_utils.Option.map ensure_min_features Py.Int.of_int); ("warn_on_dtype", Wrap_utils.Option.map warn_on_dtype Py.Bool.of_bool); ("estimator", Wrap_utils.Option.map estimator Np.Obj.to_pyobject); ("array", Some(array ))])

                  let check_is_fitted ?attributes ?msg ?all_or_any ~estimator () =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_is_fitted"
                       [||]
                       (Wrap_utils.keyword_args [("attributes", Wrap_utils.Option.map attributes (function
| `S x -> Py.String.of_string x
| `Arr x -> Np.Obj.to_pyobject x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
)); ("msg", Wrap_utils.Option.map msg Py.String.of_string); ("all_or_any", Wrap_utils.Option.map all_or_any (function
| `Callable x -> Wrap_utils.id x
| `PyObject x -> Wrap_utils.id x
)); ("estimator", Some(estimator |> Np.Obj.to_pyobject))])

let delayed ?check_pickle ~function_ () =
   Py.Module.get_function_with_keywords __wrap_namespace "delayed"
     [||]
     (Wrap_utils.keyword_args [("check_pickle", check_pickle); ("function", Some(function_ ))])

let mquantiles ?prob ?alphap ?betap ?axis ?limit ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "mquantiles"
     [||]
     (Wrap_utils.keyword_args [("prob", Wrap_utils.Option.map prob Np.Obj.to_pyobject); ("alphap", Wrap_utils.Option.map alphap Py.Float.of_float); ("betap", Wrap_utils.Option.map betap Py.Float.of_float); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("limit", Wrap_utils.Option.map limit (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.Float.of_float ml_0); (Py.Float.of_float ml_1)])); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let partial_dependence ?grid ?x ?percentiles ?grid_resolution ~gbrt ~target_variables () =
                     Py.Module.get_function_with_keywords __wrap_namespace "partial_dependence"
                       [||]
                       (Wrap_utils.keyword_args [("grid", Wrap_utils.Option.map grid Np.Obj.to_pyobject); ("X", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("percentiles", percentiles); ("grid_resolution", Wrap_utils.Option.map grid_resolution Py.Int.of_int); ("gbrt", Some(gbrt )); ("target_variables", Some(target_variables |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `Dtype_int x -> Wrap_utils.id x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> if Py.is_none py then None else Some (Wrap_utils.id py)) (Py.Tuple.get x 1))))
let plot_partial_dependence ?feature_names ?label ?n_cols ?grid_resolution ?percentiles ?n_jobs ?verbose ?ax ?line_kw ?contour_kw ?fig_kw ~gbrt ~x ~features () =
   Py.Module.get_function_with_keywords __wrap_namespace "plot_partial_dependence"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("feature_names", feature_names); ("label", label); ("n_cols", Wrap_utils.Option.map n_cols Py.Int.of_int); ("grid_resolution", Wrap_utils.Option.map grid_resolution Py.Int.of_int); ("percentiles", percentiles); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("ax", ax); ("line_kw", Wrap_utils.Option.map line_kw Dict.to_pyobject); ("contour_kw", Wrap_utils.Option.map contour_kw Dict.to_pyobject); ("gbrt", Some(gbrt )); ("X", Some(x |> Np.Obj.to_pyobject)); ("features", Some(features ))]) (match fig_kw with None -> [] | Some x -> x))
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))

end
