let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.feature_selection"

let get_py name = Py.Module.get __wrap_namespace name
module GenericUnivariateSelect = struct
type tag = [`GenericUnivariateSelect]
type t = [`BaseEstimator | `GenericUnivariateSelect | `Object | `SelectorMixin | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_selector x = (x :> [`SelectorMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
                  let create ?score_func ?mode ?param () =
                     Py.Module.get_function_with_keywords __wrap_namespace "GenericUnivariateSelect"
                       [||]
                       (Wrap_utils.keyword_args [("score_func", score_func); ("mode", Wrap_utils.Option.map mode (function
| `Percentile -> Py.String.of_string "percentile"
| `K_best -> Py.String.of_string "k_best"
| `Fpr -> Py.String.of_string "fpr"
| `Fdr -> Py.String.of_string "fdr"
| `Fwe -> Py.String.of_string "fwe"
)); ("param", Wrap_utils.Option.map param (function
| `Int_depending_on_the_feature_selection_mode x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
))])
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
let get_support ?indices self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_support"
     [||]
     (Wrap_utils.keyword_args [("indices", Wrap_utils.Option.map indices Py.Bool.of_bool)])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "inverse_transform"
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

let scores_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "scores_" with
  | None -> failwith "attribute scores_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let scores_ self = match scores_opt self with
  | None -> raise Not_found
  | Some x -> x

let pvalues_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "pvalues_" with
  | None -> failwith "attribute pvalues_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let pvalues_ self = match pvalues_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RFE = struct
type tag = [`RFE]
type t = [`BaseEstimator | `MetaEstimatorMixin | `Object | `RFE | `SelectorMixin | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_selector x = (x :> [`SelectorMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_meta_estimator x = (x :> [`MetaEstimatorMixin] Obj.t)
                  let create ?n_features_to_select ?step ?verbose ~estimator () =
                     Py.Module.get_function_with_keywords __wrap_namespace "RFE"
                       [||]
                       (Wrap_utils.keyword_args [("n_features_to_select", Wrap_utils.Option.map n_features_to_select Py.Int.of_int); ("step", Wrap_utils.Option.map step (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("estimator", Some(estimator |> Np.Obj.to_pyobject))])
                       |> of_pyobject
let decision_function ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
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
let get_support ?indices self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_support"
     [||]
     (Wrap_utils.keyword_args [("indices", Wrap_utils.Option.map indices Py.Bool.of_bool)])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
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
let score ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])

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

let n_features_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_features_" with
  | None -> failwith "attribute n_features_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_features_ self = match n_features_opt self with
  | None -> raise Not_found
  | Some x -> x

let support_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "support_" with
  | None -> failwith "attribute support_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let support_ self = match support_opt self with
  | None -> raise Not_found
  | Some x -> x

let ranking_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "ranking_" with
  | None -> failwith "attribute ranking_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let ranking_ self = match ranking_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimator_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimator_" with
  | None -> failwith "attribute estimator_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimator_ self = match estimator_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RFECV = struct
type tag = [`RFECV]
type t = [`BaseEstimator | `MetaEstimatorMixin | `Object | `RFECV | `SelectorMixin | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_selector x = (x :> [`SelectorMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_meta_estimator x = (x :> [`MetaEstimatorMixin] Obj.t)
                  let create ?step ?min_features_to_select ?cv ?scoring ?verbose ?n_jobs ~estimator () =
                     Py.Module.get_function_with_keywords __wrap_namespace "RFECV"
                       [||]
                       (Wrap_utils.keyword_args [("step", Wrap_utils.Option.map step (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("min_features_to_select", Wrap_utils.Option.map min_features_to_select Py.Int.of_int); ("cv", Wrap_utils.Option.map cv (function
| `Arr x -> Np.Obj.to_pyobject x
| `BaseCrossValidator x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("scoring", Wrap_utils.Option.map scoring (function
| `Callable x -> Wrap_utils.id x
| `Score x -> (function
| `Explained_variance -> Py.String.of_string "explained_variance"
| `R2 -> Py.String.of_string "r2"
| `Max_error -> Py.String.of_string "max_error"
| `Neg_median_absolute_error -> Py.String.of_string "neg_median_absolute_error"
| `Neg_mean_absolute_error -> Py.String.of_string "neg_mean_absolute_error"
| `Neg_mean_squared_error -> Py.String.of_string "neg_mean_squared_error"
| `Neg_mean_squared_log_error -> Py.String.of_string "neg_mean_squared_log_error"
| `Neg_root_mean_squared_error -> Py.String.of_string "neg_root_mean_squared_error"
| `Neg_mean_poisson_deviance -> Py.String.of_string "neg_mean_poisson_deviance"
| `Neg_mean_gamma_deviance -> Py.String.of_string "neg_mean_gamma_deviance"
| `Accuracy -> Py.String.of_string "accuracy"
| `Roc_auc -> Py.String.of_string "roc_auc"
| `Roc_auc_ovr -> Py.String.of_string "roc_auc_ovr"
| `Roc_auc_ovo -> Py.String.of_string "roc_auc_ovo"
| `Roc_auc_ovr_weighted -> Py.String.of_string "roc_auc_ovr_weighted"
| `Roc_auc_ovo_weighted -> Py.String.of_string "roc_auc_ovo_weighted"
| `Balanced_accuracy -> Py.String.of_string "balanced_accuracy"
| `Average_precision -> Py.String.of_string "average_precision"
| `Neg_log_loss -> Py.String.of_string "neg_log_loss"
| `Neg_brier_score -> Py.String.of_string "neg_brier_score"
| `Adjusted_rand_score -> Py.String.of_string "adjusted_rand_score"
| `Homogeneity_score -> Py.String.of_string "homogeneity_score"
| `Completeness_score -> Py.String.of_string "completeness_score"
| `V_measure_score -> Py.String.of_string "v_measure_score"
| `Mutual_info_score -> Py.String.of_string "mutual_info_score"
| `Adjusted_mutual_info_score -> Py.String.of_string "adjusted_mutual_info_score"
| `Normalized_mutual_info_score -> Py.String.of_string "normalized_mutual_info_score"
| `Fowlkes_mallows_score -> Py.String.of_string "fowlkes_mallows_score"
| `Precision -> Py.String.of_string "precision"
| `Precision_macro -> Py.String.of_string "precision_macro"
| `Precision_micro -> Py.String.of_string "precision_micro"
| `Precision_samples -> Py.String.of_string "precision_samples"
| `Precision_weighted -> Py.String.of_string "precision_weighted"
| `Recall -> Py.String.of_string "recall"
| `Recall_macro -> Py.String.of_string "recall_macro"
| `Recall_micro -> Py.String.of_string "recall_micro"
| `Recall_samples -> Py.String.of_string "recall_samples"
| `Recall_weighted -> Py.String.of_string "recall_weighted"
| `F1 -> Py.String.of_string "f1"
| `F1_macro -> Py.String.of_string "f1_macro"
| `F1_micro -> Py.String.of_string "f1_micro"
| `F1_samples -> Py.String.of_string "f1_samples"
| `F1_weighted -> Py.String.of_string "f1_weighted"
| `Jaccard -> Py.String.of_string "jaccard"
| `Jaccard_macro -> Py.String.of_string "jaccard_macro"
| `Jaccard_micro -> Py.String.of_string "jaccard_micro"
| `Jaccard_samples -> Py.String.of_string "jaccard_samples"
| `Jaccard_weighted -> Py.String.of_string "jaccard_weighted"
) x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("estimator", Some(estimator |> Np.Obj.to_pyobject))])
                       |> of_pyobject
let decision_function ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let fit ?groups ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("groups", Wrap_utils.Option.map groups Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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
let get_support ?indices self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_support"
     [||]
     (Wrap_utils.keyword_args [("indices", Wrap_utils.Option.map indices Py.Bool.of_bool)])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
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
let score ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])

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

let n_features_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_features_" with
  | None -> failwith "attribute n_features_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_features_ self = match n_features_opt self with
  | None -> raise Not_found
  | Some x -> x

let support_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "support_" with
  | None -> failwith "attribute support_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let support_ self = match support_opt self with
  | None -> raise Not_found
  | Some x -> x

let ranking_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "ranking_" with
  | None -> failwith "attribute ranking_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let ranking_ self = match ranking_opt self with
  | None -> raise Not_found
  | Some x -> x

let grid_scores_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "grid_scores_" with
  | None -> failwith "attribute grid_scores_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let grid_scores_ self = match grid_scores_opt self with
  | None -> raise Not_found
  | Some x -> x

let estimator_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimator_" with
  | None -> failwith "attribute estimator_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimator_ self = match estimator_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SelectFdr = struct
type tag = [`SelectFdr]
type t = [`BaseEstimator | `Object | `SelectFdr | `SelectorMixin | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_selector x = (x :> [`SelectorMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let create ?score_func ?alpha () =
   Py.Module.get_function_with_keywords __wrap_namespace "SelectFdr"
     [||]
     (Wrap_utils.keyword_args [("score_func", score_func); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float)])
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
let get_support ?indices self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_support"
     [||]
     (Wrap_utils.keyword_args [("indices", Wrap_utils.Option.map indices Py.Bool.of_bool)])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "inverse_transform"
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

let scores_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "scores_" with
  | None -> failwith "attribute scores_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let scores_ self = match scores_opt self with
  | None -> raise Not_found
  | Some x -> x

let pvalues_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "pvalues_" with
  | None -> failwith "attribute pvalues_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let pvalues_ self = match pvalues_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SelectFpr = struct
type tag = [`SelectFpr]
type t = [`BaseEstimator | `Object | `SelectFpr | `SelectorMixin | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_selector x = (x :> [`SelectorMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let create ?score_func ?alpha () =
   Py.Module.get_function_with_keywords __wrap_namespace "SelectFpr"
     [||]
     (Wrap_utils.keyword_args [("score_func", score_func); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float)])
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
let get_support ?indices self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_support"
     [||]
     (Wrap_utils.keyword_args [("indices", Wrap_utils.Option.map indices Py.Bool.of_bool)])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "inverse_transform"
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

let scores_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "scores_" with
  | None -> failwith "attribute scores_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let scores_ self = match scores_opt self with
  | None -> raise Not_found
  | Some x -> x

let pvalues_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "pvalues_" with
  | None -> failwith "attribute pvalues_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let pvalues_ self = match pvalues_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SelectFromModel = struct
type tag = [`SelectFromModel]
type t = [`BaseEstimator | `MetaEstimatorMixin | `Object | `SelectFromModel | `SelectorMixin | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_selector x = (x :> [`SelectorMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_meta_estimator x = (x :> [`MetaEstimatorMixin] Obj.t)
                  let create ?threshold ?prefit ?norm_order ?max_features ~estimator () =
                     Py.Module.get_function_with_keywords __wrap_namespace "SelectFromModel"
                       [||]
                       (Wrap_utils.keyword_args [("threshold", Wrap_utils.Option.map threshold (function
| `F x -> Py.Float.of_float x
| `S x -> Py.String.of_string x
)); ("prefit", Wrap_utils.Option.map prefit Py.Bool.of_bool); ("norm_order", norm_order); ("max_features", Wrap_utils.Option.map max_features Py.Int.of_int); ("estimator", Some(estimator |> Np.Obj.to_pyobject))])
                       |> of_pyobject
let fit ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
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
let get_support ?indices self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_support"
     [||]
     (Wrap_utils.keyword_args [("indices", Wrap_utils.Option.map indices Py.Bool.of_bool)])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let partial_fit ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "partial_fit"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> of_pyobject
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

let estimator_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimator_" with
  | None -> failwith "attribute estimator_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimator_ self = match estimator_opt self with
  | None -> raise Not_found
  | Some x -> x

let threshold_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "threshold_" with
  | None -> failwith "attribute threshold_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let threshold_ self = match threshold_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SelectFwe = struct
type tag = [`SelectFwe]
type t = [`BaseEstimator | `Object | `SelectFwe | `SelectorMixin | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_selector x = (x :> [`SelectorMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let create ?score_func ?alpha () =
   Py.Module.get_function_with_keywords __wrap_namespace "SelectFwe"
     [||]
     (Wrap_utils.keyword_args [("score_func", score_func); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float)])
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
let get_support ?indices self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_support"
     [||]
     (Wrap_utils.keyword_args [("indices", Wrap_utils.Option.map indices Py.Bool.of_bool)])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "inverse_transform"
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

let scores_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "scores_" with
  | None -> failwith "attribute scores_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let scores_ self = match scores_opt self with
  | None -> raise Not_found
  | Some x -> x

let pvalues_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "pvalues_" with
  | None -> failwith "attribute pvalues_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let pvalues_ self = match pvalues_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SelectKBest = struct
type tag = [`SelectKBest]
type t = [`BaseEstimator | `Object | `SelectKBest | `SelectorMixin | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_selector x = (x :> [`SelectorMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
                  let create ?score_func ?k () =
                     Py.Module.get_function_with_keywords __wrap_namespace "SelectKBest"
                       [||]
                       (Wrap_utils.keyword_args [("score_func", score_func); ("k", Wrap_utils.Option.map k (function
| `All -> Py.String.of_string "all"
| `I x -> Py.Int.of_int x
))])
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
let get_support ?indices self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_support"
     [||]
     (Wrap_utils.keyword_args [("indices", Wrap_utils.Option.map indices Py.Bool.of_bool)])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "inverse_transform"
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

let scores_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "scores_" with
  | None -> failwith "attribute scores_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let scores_ self = match scores_opt self with
  | None -> raise Not_found
  | Some x -> x

let pvalues_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "pvalues_" with
  | None -> failwith "attribute pvalues_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let pvalues_ self = match pvalues_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SelectPercentile = struct
type tag = [`SelectPercentile]
type t = [`BaseEstimator | `Object | `SelectPercentile | `SelectorMixin | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_selector x = (x :> [`SelectorMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let create ?score_func ?percentile () =
   Py.Module.get_function_with_keywords __wrap_namespace "SelectPercentile"
     [||]
     (Wrap_utils.keyword_args [("score_func", score_func); ("percentile", Wrap_utils.Option.map percentile Py.Int.of_int)])
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
let get_support ?indices self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_support"
     [||]
     (Wrap_utils.keyword_args [("indices", Wrap_utils.Option.map indices Py.Bool.of_bool)])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "inverse_transform"
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

let scores_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "scores_" with
  | None -> failwith "attribute scores_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let scores_ self = match scores_opt self with
  | None -> raise Not_found
  | Some x -> x

let pvalues_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "pvalues_" with
  | None -> failwith "attribute pvalues_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let pvalues_ self = match pvalues_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module VarianceThreshold = struct
type tag = [`VarianceThreshold]
type t = [`BaseEstimator | `Object | `SelectorMixin | `TransformerMixin | `VarianceThreshold] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_selector x = (x :> [`SelectorMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let create ?threshold () =
   Py.Module.get_function_with_keywords __wrap_namespace "VarianceThreshold"
     [||]
     (Wrap_utils.keyword_args [("threshold", Wrap_utils.Option.map threshold Py.Float.of_float)])
     |> of_pyobject
let fit ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
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
let get_support ?indices self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_support"
     [||]
     (Wrap_utils.keyword_args [("indices", Wrap_utils.Option.map indices Py.Bool.of_bool)])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "inverse_transform"
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

let variances_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "variances_" with
  | None -> failwith "attribute variances_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let variances_ self = match variances_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let chi2 ~x ~y () =
   Py.Module.get_function_with_keywords __wrap_namespace "chi2"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
let f_classif ~x ~y () =
   Py.Module.get_function_with_keywords __wrap_namespace "f_classif"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
let f_oneway args =
   Py.Module.get_function_with_keywords __wrap_namespace "f_oneway"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

                  let f_regression ?center ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "f_regression"
                       [||]
                       (Wrap_utils.keyword_args [("center", Wrap_utils.Option.map center (function
| `True -> Py.Bool.t
| `Bool x -> Py.Bool.of_bool x
)); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let mutual_info_classif ?discrete_features ?n_neighbors ?copy ?random_state ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "mutual_info_classif"
                       [||]
                       (Wrap_utils.keyword_args [("discrete_features", Wrap_utils.Option.map discrete_features (function
| `Arr x -> Np.Obj.to_pyobject x
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let mutual_info_regression ?discrete_features ?n_neighbors ?copy ?random_state ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "mutual_info_regression"
                       [||]
                       (Wrap_utils.keyword_args [("discrete_features", Wrap_utils.Option.map discrete_features (function
| `Arr x -> Np.Obj.to_pyobject x
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
