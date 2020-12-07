let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.model_selection"

let get_py name = Py.Module.get __wrap_namespace name
module BaseCrossValidator = struct
type tag = [`BaseCrossValidator]
type t = [`BaseCrossValidator | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let get_n_splits ?x ?y ?groups self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", groups)])
     |> Py.Int.to_int
let split ?y ?groups ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "split"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("groups", Wrap_utils.Option.map groups Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> Py.Iter.to_seq py |> Seq.map (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)))))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module GridSearchCV = struct
type tag = [`GridSearchCV]
type t = [`BaseEstimator | `BaseSearchCV | `GridSearchCV | `MetaEstimatorMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_meta_estimator x = (x :> [`MetaEstimatorMixin] Obj.t)
let as_search_cv x = (x :> [`BaseSearchCV] Obj.t)
                  let create ?scoring ?n_jobs ?iid ?refit ?cv ?verbose ?pre_dispatch ?error_score ?return_train_score ~estimator ~param_grid () =
                     Py.Module.get_function_with_keywords __wrap_namespace "GridSearchCV"
                       [||]
                       (Wrap_utils.keyword_args [("scoring", Wrap_utils.Option.map scoring (function
| `Neg_mean_absolute_error -> Py.String.of_string "neg_mean_absolute_error"
| `Completeness_score -> Py.String.of_string "completeness_score"
| `Roc_auc_ovr -> Py.String.of_string "roc_auc_ovr"
| `Neg_mean_squared_log_error -> Py.String.of_string "neg_mean_squared_log_error"
| `Neg_mean_gamma_deviance -> Py.String.of_string "neg_mean_gamma_deviance"
| `Precision_macro -> Py.String.of_string "precision_macro"
| `R2 -> Py.String.of_string "r2"
| `Precision_micro -> Py.String.of_string "precision_micro"
| `F1_weighted -> Py.String.of_string "f1_weighted"
| `Balanced_accuracy -> Py.String.of_string "balanced_accuracy"
| `Neg_mean_squared_error -> Py.String.of_string "neg_mean_squared_error"
| `F1_samples -> Py.String.of_string "f1_samples"
| `Jaccard_micro -> Py.String.of_string "jaccard_micro"
| `Scores x -> (fun ml -> Py.List.of_list_map (function
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
) ml) x
| `Normalized_mutual_info_score -> Py.String.of_string "normalized_mutual_info_score"
| `F1_micro -> Py.String.of_string "f1_micro"
| `Roc_auc -> Py.String.of_string "roc_auc"
| `Mutual_info_score -> Py.String.of_string "mutual_info_score"
| `Adjusted_rand_score -> Py.String.of_string "adjusted_rand_score"
| `Average_precision -> Py.String.of_string "average_precision"
| `Jaccard -> Py.String.of_string "jaccard"
| `Homogeneity_score -> Py.String.of_string "homogeneity_score"
| `Accuracy -> Py.String.of_string "accuracy"
| `Jaccard_macro -> Py.String.of_string "jaccard_macro"
| `Jaccard_weighted -> Py.String.of_string "jaccard_weighted"
| `Recall_micro -> Py.String.of_string "recall_micro"
| `Explained_variance -> Py.String.of_string "explained_variance"
| `Precision -> Py.String.of_string "precision"
| `Callable x -> Wrap_utils.id x
| `V_measure_score -> Py.String.of_string "v_measure_score"
| `F1 -> Py.String.of_string "f1"
| `Roc_auc_ovo -> Py.String.of_string "roc_auc_ovo"
| `Neg_mean_poisson_deviance -> Py.String.of_string "neg_mean_poisson_deviance"
| `Recall_samples -> Py.String.of_string "recall_samples"
| `Adjusted_mutual_info_score -> Py.String.of_string "adjusted_mutual_info_score"
| `Neg_brier_score -> Py.String.of_string "neg_brier_score"
| `Roc_auc_ovo_weighted -> Py.String.of_string "roc_auc_ovo_weighted"
| `Recall -> Py.String.of_string "recall"
| `Dict x -> Dict.to_pyobject x
| `Fowlkes_mallows_score -> Py.String.of_string "fowlkes_mallows_score"
| `Neg_log_loss -> Py.String.of_string "neg_log_loss"
| `Neg_root_mean_squared_error -> Py.String.of_string "neg_root_mean_squared_error"
| `Precision_samples -> Py.String.of_string "precision_samples"
| `F1_macro -> Py.String.of_string "f1_macro"
| `Roc_auc_ovr_weighted -> Py.String.of_string "roc_auc_ovr_weighted"
| `Recall_weighted -> Py.String.of_string "recall_weighted"
| `Neg_median_absolute_error -> Py.String.of_string "neg_median_absolute_error"
| `Jaccard_samples -> Py.String.of_string "jaccard_samples"
| `Precision_weighted -> Py.String.of_string "precision_weighted"
| `Max_error -> Py.String.of_string "max_error"
| `Recall_macro -> Py.String.of_string "recall_macro"
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("iid", Wrap_utils.Option.map iid Py.Bool.of_bool); ("refit", Wrap_utils.Option.map refit (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
| `Bool x -> Py.Bool.of_bool x
)); ("cv", Wrap_utils.Option.map cv (function
| `BaseCrossValidator x -> Np.Obj.to_pyobject x
| `Arr x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("pre_dispatch", Wrap_utils.Option.map pre_dispatch (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("error_score", Wrap_utils.Option.map error_score (function
| `Raise -> Py.String.of_string "raise"
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("return_train_score", Wrap_utils.Option.map return_train_score Py.Bool.of_bool); ("estimator", Some(estimator |> Np.Obj.to_pyobject)); ("param_grid", Some(param_grid |> (function
| `Grid x -> (fun x -> Dict.(of_param_grid_alist x |> to_pyobject)) x
| `Grids x -> (fun ml -> Py.List.of_list_map (fun x -> Dict.(of_param_grid_alist x |> to_pyobject)) ml) x
)))])
                       |> of_pyobject
let decision_function ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let fit ?y ?groups ?fit_params ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("groups", Wrap_utils.Option.map groups Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
                  let inverse_transform ~xt self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "inverse_transform"
                       [||]
                       (Wrap_utils.keyword_args [("Xt", Some(xt |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `Length_n_samples x -> Wrap_utils.id x
)))])

let predict ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let predict_log_proba ~x self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "predict_log_proba"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `Length_n_samples x -> Wrap_utils.id x
)))])

let predict_proba ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let score ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
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

let cv_results_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "cv_results_" with
  | None -> failwith "attribute cv_results_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let cv_results_ self = match cv_results_opt self with
  | None -> raise Not_found
  | Some x -> x

let best_estimator_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "best_estimator_" with
  | None -> failwith "attribute best_estimator_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`BaseEstimator|`Object] Np.Obj.t)) x)

let best_estimator_ self = match best_estimator_opt self with
  | None -> raise Not_found
  | Some x -> x

let best_score_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "best_score_" with
  | None -> failwith "attribute best_score_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let best_score_ self = match best_score_opt self with
  | None -> raise Not_found
  | Some x -> x

let best_params_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "best_params_" with
  | None -> failwith "attribute best_params_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let best_params_ self = match best_params_opt self with
  | None -> raise Not_found
  | Some x -> x

let best_index_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "best_index_" with
  | None -> failwith "attribute best_index_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let best_index_ self = match best_index_opt self with
  | None -> raise Not_found
  | Some x -> x

let scorer_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "scorer_" with
  | None -> failwith "attribute scorer_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let scorer_ self = match scorer_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_splits_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_splits_" with
  | None -> failwith "attribute n_splits_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_splits_ self = match n_splits_opt self with
  | None -> raise Not_found
  | Some x -> x

let refit_time_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "refit_time_" with
  | None -> failwith "attribute refit_time_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let refit_time_ self = match refit_time_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module GroupKFold = struct
type tag = [`GroupKFold]
type t = [`BaseCrossValidator | `GroupKFold | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_cross_validator x = (x :> [`BaseCrossValidator] Obj.t)
let create ?n_splits () =
   Py.Module.get_function_with_keywords __wrap_namespace "GroupKFold"
     [||]
     (Wrap_utils.keyword_args [("n_splits", Wrap_utils.Option.map n_splits Py.Int.of_int)])
     |> of_pyobject
let get_n_splits ?x ?y ?groups self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", groups)])
     |> Py.Int.to_int
let split ?y ?groups ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "split"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("groups", Wrap_utils.Option.map groups Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> Py.Iter.to_seq py |> Seq.map (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)))))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module GroupShuffleSplit = struct
type tag = [`GroupShuffleSplit]
type t = [`BaseShuffleSplit | `GroupShuffleSplit | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_shuffle_split x = (x :> [`BaseShuffleSplit] Obj.t)
                  let create ?n_splits ?test_size ?train_size ?random_state () =
                     Py.Module.get_function_with_keywords __wrap_namespace "GroupShuffleSplit"
                       [||]
                       (Wrap_utils.keyword_args [("n_splits", Wrap_utils.Option.map n_splits Py.Int.of_int); ("test_size", Wrap_utils.Option.map test_size (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("train_size", Wrap_utils.Option.map train_size (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
                       |> of_pyobject
let get_n_splits ?x ?y ?groups self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", groups)])
     |> Py.Int.to_int
let split ?y ?groups ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "split"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("groups", Wrap_utils.Option.map groups Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> Py.Iter.to_seq py |> Seq.map (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)))))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module KFold = struct
type tag = [`KFold]
type t = [`BaseCrossValidator | `KFold | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_cross_validator x = (x :> [`BaseCrossValidator] Obj.t)
let create ?n_splits ?shuffle ?random_state () =
   Py.Module.get_function_with_keywords __wrap_namespace "KFold"
     [||]
     (Wrap_utils.keyword_args [("n_splits", Wrap_utils.Option.map n_splits Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
     |> of_pyobject
let get_n_splits ?x ?y ?groups self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", groups)])
     |> Py.Int.to_int
let split ?y ?groups ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "split"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("groups", Wrap_utils.Option.map groups Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> Py.Iter.to_seq py |> Seq.map (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)))))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LeaveOneGroupOut = struct
type tag = [`LeaveOneGroupOut]
type t = [`BaseCrossValidator | `LeaveOneGroupOut | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_cross_validator x = (x :> [`BaseCrossValidator] Obj.t)
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "LeaveOneGroupOut"
     [||]
     []
     |> of_pyobject
let get_n_splits ?x ?y ?groups self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", Wrap_utils.Option.map groups Np.Obj.to_pyobject)])
     |> Py.Int.to_int
let split ?y ?groups ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "split"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("groups", Wrap_utils.Option.map groups Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> Py.Iter.to_seq py |> Seq.map (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)))))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LeaveOneOut = struct
type tag = [`LeaveOneOut]
type t = [`BaseCrossValidator | `LeaveOneOut | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_cross_validator x = (x :> [`BaseCrossValidator] Obj.t)
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "LeaveOneOut"
     [||]
     []
     |> of_pyobject
let get_n_splits ?y ?groups ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("groups", groups); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> Py.Int.to_int
let split ?y ?groups ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "split"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("groups", Wrap_utils.Option.map groups Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> Py.Iter.to_seq py |> Seq.map (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)))))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LeavePGroupsOut = struct
type tag = [`LeavePGroupsOut]
type t = [`BaseCrossValidator | `LeavePGroupsOut | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_cross_validator x = (x :> [`BaseCrossValidator] Obj.t)
let create n_groups =
   Py.Module.get_function_with_keywords __wrap_namespace "LeavePGroupsOut"
     [||]
     (Wrap_utils.keyword_args [("n_groups", Some(n_groups |> Py.Int.of_int))])
     |> of_pyobject
let get_n_splits ?x ?y ?groups self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", Wrap_utils.Option.map groups Np.Obj.to_pyobject)])
     |> Py.Int.to_int
let split ?y ?groups ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "split"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("groups", Wrap_utils.Option.map groups Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> Py.Iter.to_seq py |> Seq.map (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)))))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LeavePOut = struct
type tag = [`LeavePOut]
type t = [`BaseCrossValidator | `LeavePOut | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_cross_validator x = (x :> [`BaseCrossValidator] Obj.t)
let create p =
   Py.Module.get_function_with_keywords __wrap_namespace "LeavePOut"
     [||]
     (Wrap_utils.keyword_args [("p", Some(p |> Py.Int.of_int))])
     |> of_pyobject
let get_n_splits ?y ?groups ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("groups", groups); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> Py.Int.to_int
let split ?y ?groups ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "split"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("groups", Wrap_utils.Option.map groups Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> Py.Iter.to_seq py |> Seq.map (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)))))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ParameterGrid = struct
type tag = [`ParameterGrid]
type t = [`Object | `ParameterGrid] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create param_grid =
                     Py.Module.get_function_with_keywords __wrap_namespace "ParameterGrid"
                       [||]
                       (Wrap_utils.keyword_args [("param_grid", Some(param_grid |> (function
| `Grid x -> (fun x -> Dict.(of_param_grid_alist x |> to_pyobject)) x
| `Grids x -> (fun ml -> Py.List.of_list_map (fun x -> Dict.(of_param_grid_alist x |> to_pyobject)) ml) x
)))])
                       |> of_pyobject
let get_item ~ind self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("ind", Some(ind |> Py.Int.of_int))])

let iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []
     |> (fun py -> Py.Iter.to_seq py |> Seq.map Dict.of_pyobject)
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ParameterSampler = struct
type tag = [`ParameterSampler]
type t = [`Object | `ParameterSampler] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?random_state ~param_distributions ~n_iter () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ParameterSampler"
                       [||]
                       (Wrap_utils.keyword_args [("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("param_distributions", Some(param_distributions |> (function
| `Grid x -> (fun x -> Dict.(of_param_distributions_alist x |> to_pyobject)) x
| `Grids x -> (fun ml -> Py.List.of_list_map (fun x -> Dict.(of_param_distributions_alist x |> to_pyobject)) ml) x
))); ("n_iter", Some(n_iter |> Py.Int.of_int))])
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
module PredefinedSplit = struct
type tag = [`PredefinedSplit]
type t = [`BaseCrossValidator | `Object | `PredefinedSplit] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_cross_validator x = (x :> [`BaseCrossValidator] Obj.t)
let create test_fold =
   Py.Module.get_function_with_keywords __wrap_namespace "PredefinedSplit"
     [||]
     (Wrap_utils.keyword_args [("test_fold", Some(test_fold |> Np.Obj.to_pyobject))])
     |> of_pyobject
let get_n_splits ?x ?y ?groups self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", groups)])
     |> Py.Int.to_int
let split ?x ?y ?groups self =
   Py.Module.get_function_with_keywords (to_pyobject self) "split"
     [||]
     (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", groups)])
     |> (fun py -> Py.Iter.to_seq py |> Seq.map (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)))))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RandomizedSearchCV = struct
type tag = [`RandomizedSearchCV]
type t = [`BaseEstimator | `BaseSearchCV | `MetaEstimatorMixin | `Object | `RandomizedSearchCV] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_meta_estimator x = (x :> [`MetaEstimatorMixin] Obj.t)
let as_search_cv x = (x :> [`BaseSearchCV] Obj.t)
                  let create ?n_iter ?scoring ?n_jobs ?iid ?refit ?cv ?verbose ?pre_dispatch ?random_state ?error_score ?return_train_score ~estimator ~param_distributions () =
                     Py.Module.get_function_with_keywords __wrap_namespace "RandomizedSearchCV"
                       [||]
                       (Wrap_utils.keyword_args [("n_iter", Wrap_utils.Option.map n_iter Py.Int.of_int); ("scoring", Wrap_utils.Option.map scoring (function
| `Neg_mean_absolute_error -> Py.String.of_string "neg_mean_absolute_error"
| `Completeness_score -> Py.String.of_string "completeness_score"
| `Roc_auc_ovr -> Py.String.of_string "roc_auc_ovr"
| `Neg_mean_squared_log_error -> Py.String.of_string "neg_mean_squared_log_error"
| `Neg_mean_gamma_deviance -> Py.String.of_string "neg_mean_gamma_deviance"
| `Precision_macro -> Py.String.of_string "precision_macro"
| `R2 -> Py.String.of_string "r2"
| `Precision_micro -> Py.String.of_string "precision_micro"
| `F1_weighted -> Py.String.of_string "f1_weighted"
| `Balanced_accuracy -> Py.String.of_string "balanced_accuracy"
| `Neg_mean_squared_error -> Py.String.of_string "neg_mean_squared_error"
| `F1_samples -> Py.String.of_string "f1_samples"
| `Jaccard_micro -> Py.String.of_string "jaccard_micro"
| `Normalized_mutual_info_score -> Py.String.of_string "normalized_mutual_info_score"
| `Scores x -> (fun ml -> Py.List.of_list_map (function
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
) ml) x
| `F1_micro -> Py.String.of_string "f1_micro"
| `Roc_auc -> Py.String.of_string "roc_auc"
| `Mutual_info_score -> Py.String.of_string "mutual_info_score"
| `Adjusted_rand_score -> Py.String.of_string "adjusted_rand_score"
| `Average_precision -> Py.String.of_string "average_precision"
| `Jaccard -> Py.String.of_string "jaccard"
| `Homogeneity_score -> Py.String.of_string "homogeneity_score"
| `Accuracy -> Py.String.of_string "accuracy"
| `Jaccard_macro -> Py.String.of_string "jaccard_macro"
| `Jaccard_weighted -> Py.String.of_string "jaccard_weighted"
| `Recall_micro -> Py.String.of_string "recall_micro"
| `Explained_variance -> Py.String.of_string "explained_variance"
| `Precision -> Py.String.of_string "precision"
| `Callable x -> Wrap_utils.id x
| `V_measure_score -> Py.String.of_string "v_measure_score"
| `F1 -> Py.String.of_string "f1"
| `Roc_auc_ovo -> Py.String.of_string "roc_auc_ovo"
| `Neg_mean_poisson_deviance -> Py.String.of_string "neg_mean_poisson_deviance"
| `Recall_samples -> Py.String.of_string "recall_samples"
| `Adjusted_mutual_info_score -> Py.String.of_string "adjusted_mutual_info_score"
| `Neg_brier_score -> Py.String.of_string "neg_brier_score"
| `Roc_auc_ovo_weighted -> Py.String.of_string "roc_auc_ovo_weighted"
| `Recall -> Py.String.of_string "recall"
| `Dict x -> Dict.to_pyobject x
| `Fowlkes_mallows_score -> Py.String.of_string "fowlkes_mallows_score"
| `Neg_log_loss -> Py.String.of_string "neg_log_loss"
| `Neg_root_mean_squared_error -> Py.String.of_string "neg_root_mean_squared_error"
| `Precision_samples -> Py.String.of_string "precision_samples"
| `F1_macro -> Py.String.of_string "f1_macro"
| `Roc_auc_ovr_weighted -> Py.String.of_string "roc_auc_ovr_weighted"
| `Recall_weighted -> Py.String.of_string "recall_weighted"
| `Neg_median_absolute_error -> Py.String.of_string "neg_median_absolute_error"
| `Jaccard_samples -> Py.String.of_string "jaccard_samples"
| `Precision_weighted -> Py.String.of_string "precision_weighted"
| `Max_error -> Py.String.of_string "max_error"
| `Recall_macro -> Py.String.of_string "recall_macro"
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("iid", Wrap_utils.Option.map iid Py.Bool.of_bool); ("refit", Wrap_utils.Option.map refit (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
| `Bool x -> Py.Bool.of_bool x
)); ("cv", Wrap_utils.Option.map cv (function
| `BaseCrossValidator x -> Np.Obj.to_pyobject x
| `Arr x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("pre_dispatch", Wrap_utils.Option.map pre_dispatch (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("error_score", Wrap_utils.Option.map error_score (function
| `Raise -> Py.String.of_string "raise"
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("return_train_score", Wrap_utils.Option.map return_train_score Py.Bool.of_bool); ("estimator", Some(estimator |> Np.Obj.to_pyobject)); ("param_distributions", Some(param_distributions |> (function
| `Grid x -> (fun x -> Dict.(of_param_distributions_alist x |> to_pyobject)) x
| `Grids x -> (fun ml -> Py.List.of_list_map (fun x -> Dict.(of_param_distributions_alist x |> to_pyobject)) ml) x
)))])
                       |> of_pyobject
let decision_function ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let fit ?y ?groups ?fit_params ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("groups", Wrap_utils.Option.map groups Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
                  let inverse_transform ~xt self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "inverse_transform"
                       [||]
                       (Wrap_utils.keyword_args [("Xt", Some(xt |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `Length_n_samples x -> Wrap_utils.id x
)))])

let predict ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let predict_log_proba ~x self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "predict_log_proba"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `Length_n_samples x -> Wrap_utils.id x
)))])

let predict_proba ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let score ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
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

let cv_results_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "cv_results_" with
  | None -> failwith "attribute cv_results_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let cv_results_ self = match cv_results_opt self with
  | None -> raise Not_found
  | Some x -> x

let best_estimator_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "best_estimator_" with
  | None -> failwith "attribute best_estimator_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`BaseEstimator|`Object] Np.Obj.t)) x)

let best_estimator_ self = match best_estimator_opt self with
  | None -> raise Not_found
  | Some x -> x

let best_score_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "best_score_" with
  | None -> failwith "attribute best_score_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let best_score_ self = match best_score_opt self with
  | None -> raise Not_found
  | Some x -> x

let best_params_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "best_params_" with
  | None -> failwith "attribute best_params_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let best_params_ self = match best_params_opt self with
  | None -> raise Not_found
  | Some x -> x

let best_index_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "best_index_" with
  | None -> failwith "attribute best_index_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let best_index_ self = match best_index_opt self with
  | None -> raise Not_found
  | Some x -> x

let scorer_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "scorer_" with
  | None -> failwith "attribute scorer_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let scorer_ self = match scorer_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_splits_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_splits_" with
  | None -> failwith "attribute n_splits_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_splits_ self = match n_splits_opt self with
  | None -> raise Not_found
  | Some x -> x

let refit_time_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "refit_time_" with
  | None -> failwith "attribute refit_time_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let refit_time_ self = match refit_time_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RepeatedKFold = struct
type tag = [`RepeatedKFold]
type t = [`Object | `RepeatedKFold] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?n_splits ?n_repeats ?random_state () =
   Py.Module.get_function_with_keywords __wrap_namespace "RepeatedKFold"
     [||]
     (Wrap_utils.keyword_args [("n_splits", Wrap_utils.Option.map n_splits Py.Int.of_int); ("n_repeats", Wrap_utils.Option.map n_repeats Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
     |> of_pyobject
let get_n_splits ?x ?y ?groups self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", Wrap_utils.Option.map groups Np.Obj.to_pyobject)])
     |> Py.Int.to_int
let split ?y ?groups ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "split"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("groups", Wrap_utils.Option.map groups Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> Py.Iter.to_seq py |> Seq.map (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)))))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RepeatedStratifiedKFold = struct
type tag = [`RepeatedStratifiedKFold]
type t = [`Object | `RepeatedStratifiedKFold] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?n_splits ?n_repeats ?random_state () =
   Py.Module.get_function_with_keywords __wrap_namespace "RepeatedStratifiedKFold"
     [||]
     (Wrap_utils.keyword_args [("n_splits", Wrap_utils.Option.map n_splits Py.Int.of_int); ("n_repeats", Wrap_utils.Option.map n_repeats Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
     |> of_pyobject
let get_n_splits ?x ?y ?groups self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", Wrap_utils.Option.map groups Np.Obj.to_pyobject)])
     |> Py.Int.to_int
let split ?y ?groups ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "split"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("groups", Wrap_utils.Option.map groups Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> Py.Iter.to_seq py |> Seq.map (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)))))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ShuffleSplit = struct
type tag = [`ShuffleSplit]
type t = [`BaseShuffleSplit | `Object | `ShuffleSplit] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_shuffle_split x = (x :> [`BaseShuffleSplit] Obj.t)
                  let create ?n_splits ?test_size ?train_size ?random_state () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ShuffleSplit"
                       [||]
                       (Wrap_utils.keyword_args [("n_splits", Wrap_utils.Option.map n_splits Py.Int.of_int); ("test_size", Wrap_utils.Option.map test_size (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("train_size", Wrap_utils.Option.map train_size (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
                       |> of_pyobject
let get_n_splits ?x ?y ?groups self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", groups)])
     |> Py.Int.to_int
let split ?y ?groups ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "split"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("groups", Wrap_utils.Option.map groups Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> Py.Iter.to_seq py |> Seq.map (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)))))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module StratifiedKFold = struct
type tag = [`StratifiedKFold]
type t = [`BaseCrossValidator | `Object | `StratifiedKFold] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_cross_validator x = (x :> [`BaseCrossValidator] Obj.t)
let create ?n_splits ?shuffle ?random_state () =
   Py.Module.get_function_with_keywords __wrap_namespace "StratifiedKFold"
     [||]
     (Wrap_utils.keyword_args [("n_splits", Wrap_utils.Option.map n_splits Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
     |> of_pyobject
let get_n_splits ?x ?y ?groups self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", groups)])
     |> Py.Int.to_int
let split ?groups ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "split"
     [||]
     (Wrap_utils.keyword_args [("groups", groups); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun py -> Py.Iter.to_seq py |> Seq.map (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)))))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module StratifiedShuffleSplit = struct
type tag = [`StratifiedShuffleSplit]
type t = [`BaseShuffleSplit | `Object | `StratifiedShuffleSplit] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_shuffle_split x = (x :> [`BaseShuffleSplit] Obj.t)
                  let create ?n_splits ?test_size ?train_size ?random_state () =
                     Py.Module.get_function_with_keywords __wrap_namespace "StratifiedShuffleSplit"
                       [||]
                       (Wrap_utils.keyword_args [("n_splits", Wrap_utils.Option.map n_splits Py.Int.of_int); ("test_size", Wrap_utils.Option.map test_size (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("train_size", Wrap_utils.Option.map train_size (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
                       |> of_pyobject
let get_n_splits ?x ?y ?groups self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", groups)])
     |> Py.Int.to_int
let split ?groups ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "split"
     [||]
     (Wrap_utils.keyword_args [("groups", groups); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun py -> Py.Iter.to_seq py |> Seq.map (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)))))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module TimeSeriesSplit = struct
type tag = [`TimeSeriesSplit]
type t = [`BaseCrossValidator | `Object | `TimeSeriesSplit] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_cross_validator x = (x :> [`BaseCrossValidator] Obj.t)
let create ?n_splits ?max_train_size () =
   Py.Module.get_function_with_keywords __wrap_namespace "TimeSeriesSplit"
     [||]
     (Wrap_utils.keyword_args [("n_splits", Wrap_utils.Option.map n_splits Py.Int.of_int); ("max_train_size", Wrap_utils.Option.map max_train_size Py.Int.of_int)])
     |> of_pyobject
let get_n_splits ?x ?y ?groups self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", groups)])
     |> Py.Int.to_int
let split ?y ?groups ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "split"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("groups", Wrap_utils.Option.map groups Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> Py.Iter.to_seq py |> Seq.map (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)))))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let check_cv ?cv ?y ?classifier () =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_cv"
                       [||]
                       (Wrap_utils.keyword_args [("cv", Wrap_utils.Option.map cv (function
| `BaseCrossValidator x -> Np.Obj.to_pyobject x
| `Arr x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("classifier", Wrap_utils.Option.map classifier Py.Bool.of_bool)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`BaseCrossValidator|`Object] Np.Obj.t))
                  let cross_val_predict ?y ?groups ?cv ?n_jobs ?verbose ?fit_params ?pre_dispatch ?method_ ~estimator ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cross_val_predict"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("groups", Wrap_utils.Option.map groups Np.Obj.to_pyobject); ("cv", Wrap_utils.Option.map cv (function
| `BaseCrossValidator x -> Np.Obj.to_pyobject x
| `Arr x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("fit_params", Wrap_utils.Option.map fit_params (function
| `Defualt_None x -> Wrap_utils.id x
| `Dict x -> Dict.to_pyobject x
)); ("pre_dispatch", Wrap_utils.Option.map pre_dispatch (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("method", Wrap_utils.Option.map method_ Py.String.of_string); ("estimator", Some(estimator |> Np.Obj.to_pyobject)); ("X", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let cross_val_score ?y ?groups ?scoring ?cv ?n_jobs ?verbose ?fit_params ?pre_dispatch ?error_score ~estimator ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cross_val_score"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("groups", Wrap_utils.Option.map groups Np.Obj.to_pyobject); ("scoring", Wrap_utils.Option.map scoring (function
| `Neg_mean_absolute_error -> Py.String.of_string "neg_mean_absolute_error"
| `Completeness_score -> Py.String.of_string "completeness_score"
| `Roc_auc_ovr -> Py.String.of_string "roc_auc_ovr"
| `Neg_mean_squared_log_error -> Py.String.of_string "neg_mean_squared_log_error"
| `Neg_mean_gamma_deviance -> Py.String.of_string "neg_mean_gamma_deviance"
| `Precision_macro -> Py.String.of_string "precision_macro"
| `R2 -> Py.String.of_string "r2"
| `Precision_micro -> Py.String.of_string "precision_micro"
| `F1_weighted -> Py.String.of_string "f1_weighted"
| `Balanced_accuracy -> Py.String.of_string "balanced_accuracy"
| `Neg_mean_squared_error -> Py.String.of_string "neg_mean_squared_error"
| `F1_samples -> Py.String.of_string "f1_samples"
| `Jaccard_micro -> Py.String.of_string "jaccard_micro"
| `Normalized_mutual_info_score -> Py.String.of_string "normalized_mutual_info_score"
| `F1_micro -> Py.String.of_string "f1_micro"
| `Roc_auc -> Py.String.of_string "roc_auc"
| `Mutual_info_score -> Py.String.of_string "mutual_info_score"
| `Adjusted_rand_score -> Py.String.of_string "adjusted_rand_score"
| `Average_precision -> Py.String.of_string "average_precision"
| `Jaccard -> Py.String.of_string "jaccard"
| `Homogeneity_score -> Py.String.of_string "homogeneity_score"
| `Accuracy -> Py.String.of_string "accuracy"
| `Jaccard_macro -> Py.String.of_string "jaccard_macro"
| `Jaccard_weighted -> Py.String.of_string "jaccard_weighted"
| `Recall_micro -> Py.String.of_string "recall_micro"
| `Explained_variance -> Py.String.of_string "explained_variance"
| `Precision -> Py.String.of_string "precision"
| `Callable x -> Wrap_utils.id x
| `V_measure_score -> Py.String.of_string "v_measure_score"
| `F1 -> Py.String.of_string "f1"
| `Roc_auc_ovo -> Py.String.of_string "roc_auc_ovo"
| `Neg_mean_poisson_deviance -> Py.String.of_string "neg_mean_poisson_deviance"
| `Recall_samples -> Py.String.of_string "recall_samples"
| `Adjusted_mutual_info_score -> Py.String.of_string "adjusted_mutual_info_score"
| `Neg_brier_score -> Py.String.of_string "neg_brier_score"
| `Roc_auc_ovo_weighted -> Py.String.of_string "roc_auc_ovo_weighted"
| `Recall -> Py.String.of_string "recall"
| `Fowlkes_mallows_score -> Py.String.of_string "fowlkes_mallows_score"
| `Neg_log_loss -> Py.String.of_string "neg_log_loss"
| `Neg_root_mean_squared_error -> Py.String.of_string "neg_root_mean_squared_error"
| `Precision_samples -> Py.String.of_string "precision_samples"
| `F1_macro -> Py.String.of_string "f1_macro"
| `Roc_auc_ovr_weighted -> Py.String.of_string "roc_auc_ovr_weighted"
| `Recall_weighted -> Py.String.of_string "recall_weighted"
| `Neg_median_absolute_error -> Py.String.of_string "neg_median_absolute_error"
| `Jaccard_samples -> Py.String.of_string "jaccard_samples"
| `Precision_weighted -> Py.String.of_string "precision_weighted"
| `Max_error -> Py.String.of_string "max_error"
| `Recall_macro -> Py.String.of_string "recall_macro"
)); ("cv", Wrap_utils.Option.map cv (function
| `BaseCrossValidator x -> Np.Obj.to_pyobject x
| `Arr x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("fit_params", Wrap_utils.Option.map fit_params Dict.to_pyobject); ("pre_dispatch", Wrap_utils.Option.map pre_dispatch (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("error_score", Wrap_utils.Option.map error_score (function
| `Raise -> Py.String.of_string "raise"
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("estimator", Some(estimator |> Np.Obj.to_pyobject)); ("X", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let cross_validate ?y ?groups ?scoring ?cv ?n_jobs ?verbose ?fit_params ?pre_dispatch ?return_train_score ?return_estimator ?error_score ~estimator ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cross_validate"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("groups", Wrap_utils.Option.map groups Np.Obj.to_pyobject); ("scoring", Wrap_utils.Option.map scoring (function
| `Neg_mean_absolute_error -> Py.String.of_string "neg_mean_absolute_error"
| `Completeness_score -> Py.String.of_string "completeness_score"
| `Roc_auc_ovr -> Py.String.of_string "roc_auc_ovr"
| `Neg_mean_squared_log_error -> Py.String.of_string "neg_mean_squared_log_error"
| `Neg_mean_gamma_deviance -> Py.String.of_string "neg_mean_gamma_deviance"
| `Precision_macro -> Py.String.of_string "precision_macro"
| `R2 -> Py.String.of_string "r2"
| `Precision_micro -> Py.String.of_string "precision_micro"
| `F1_weighted -> Py.String.of_string "f1_weighted"
| `Balanced_accuracy -> Py.String.of_string "balanced_accuracy"
| `Neg_mean_squared_error -> Py.String.of_string "neg_mean_squared_error"
| `Scores x -> (fun ml -> Py.List.of_list_map (function
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
) ml) x
| `F1_samples -> Py.String.of_string "f1_samples"
| `Jaccard_micro -> Py.String.of_string "jaccard_micro"
| `Normalized_mutual_info_score -> Py.String.of_string "normalized_mutual_info_score"
| `F1_micro -> Py.String.of_string "f1_micro"
| `Roc_auc -> Py.String.of_string "roc_auc"
| `Mutual_info_score -> Py.String.of_string "mutual_info_score"
| `Adjusted_rand_score -> Py.String.of_string "adjusted_rand_score"
| `Average_precision -> Py.String.of_string "average_precision"
| `Jaccard -> Py.String.of_string "jaccard"
| `Homogeneity_score -> Py.String.of_string "homogeneity_score"
| `Accuracy -> Py.String.of_string "accuracy"
| `Jaccard_macro -> Py.String.of_string "jaccard_macro"
| `Jaccard_weighted -> Py.String.of_string "jaccard_weighted"
| `Recall_micro -> Py.String.of_string "recall_micro"
| `Explained_variance -> Py.String.of_string "explained_variance"
| `Precision -> Py.String.of_string "precision"
| `Callable x -> Wrap_utils.id x
| `V_measure_score -> Py.String.of_string "v_measure_score"
| `F1 -> Py.String.of_string "f1"
| `Roc_auc_ovo -> Py.String.of_string "roc_auc_ovo"
| `Neg_mean_poisson_deviance -> Py.String.of_string "neg_mean_poisson_deviance"
| `Recall_samples -> Py.String.of_string "recall_samples"
| `Adjusted_mutual_info_score -> Py.String.of_string "adjusted_mutual_info_score"
| `Neg_brier_score -> Py.String.of_string "neg_brier_score"
| `Roc_auc_ovo_weighted -> Py.String.of_string "roc_auc_ovo_weighted"
| `Recall -> Py.String.of_string "recall"
| `Dict x -> Dict.to_pyobject x
| `Fowlkes_mallows_score -> Py.String.of_string "fowlkes_mallows_score"
| `Neg_log_loss -> Py.String.of_string "neg_log_loss"
| `Neg_root_mean_squared_error -> Py.String.of_string "neg_root_mean_squared_error"
| `Precision_samples -> Py.String.of_string "precision_samples"
| `F1_macro -> Py.String.of_string "f1_macro"
| `Roc_auc_ovr_weighted -> Py.String.of_string "roc_auc_ovr_weighted"
| `Recall_weighted -> Py.String.of_string "recall_weighted"
| `Neg_median_absolute_error -> Py.String.of_string "neg_median_absolute_error"
| `Jaccard_samples -> Py.String.of_string "jaccard_samples"
| `Precision_weighted -> Py.String.of_string "precision_weighted"
| `Max_error -> Py.String.of_string "max_error"
| `Recall_macro -> Py.String.of_string "recall_macro"
)); ("cv", Wrap_utils.Option.map cv (function
| `BaseCrossValidator x -> Np.Obj.to_pyobject x
| `Arr x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("fit_params", Wrap_utils.Option.map fit_params Dict.to_pyobject); ("pre_dispatch", Wrap_utils.Option.map pre_dispatch (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("return_train_score", Wrap_utils.Option.map return_train_score Py.Bool.of_bool); ("return_estimator", Wrap_utils.Option.map return_estimator Py.Bool.of_bool); ("error_score", Wrap_utils.Option.map error_score (function
| `Raise -> Py.String.of_string "raise"
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("estimator", Some(estimator |> Np.Obj.to_pyobject)); ("X", Some(x |> Np.Obj.to_pyobject))])
                       |> Dict.of_pyobject
                  let fit_grid_point ?error_score ?fit_params ~x ~y ~estimator ~parameters ~train ~test ~scorer ~verbose () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fit_grid_point"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("error_score", Wrap_utils.Option.map error_score (function
| `Raise -> Py.String.of_string "raise"
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `None -> Py.none
))); ("estimator", Some(estimator |> Np.Obj.to_pyobject)); ("parameters", Some(parameters |> Dict.to_pyobject)); ("train", Some(train |> (function
| `Bool x -> Py.Bool.of_bool x
| `Arr x -> Np.Obj.to_pyobject x
| `Dtype_int x -> Wrap_utils.id x
))); ("test", Some(test |> (function
| `Bool x -> Py.Bool.of_bool x
| `Arr x -> Np.Obj.to_pyobject x
| `Dtype_int x -> Wrap_utils.id x
))); ("scorer", Some(scorer |> (function
| `Callable x -> Wrap_utils.id x
| `None -> Py.none
))); ("verbose", Some(verbose |> Py.Int.of_int))]) (match fit_params with None -> [] | Some x -> x))
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Dict.of_pyobject (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2))))
                  let learning_curve ?groups ?train_sizes ?cv ?scoring ?exploit_incremental_learning ?n_jobs ?pre_dispatch ?verbose ?shuffle ?random_state ?error_score ?return_times ~estimator ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "learning_curve"
                       [||]
                       (Wrap_utils.keyword_args [("groups", Wrap_utils.Option.map groups Np.Obj.to_pyobject); ("train_sizes", Wrap_utils.Option.map train_sizes Np.Obj.to_pyobject); ("cv", Wrap_utils.Option.map cv (function
| `BaseCrossValidator x -> Np.Obj.to_pyobject x
| `Arr x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("scoring", Wrap_utils.Option.map scoring (function
| `Neg_mean_absolute_error -> Py.String.of_string "neg_mean_absolute_error"
| `Completeness_score -> Py.String.of_string "completeness_score"
| `Roc_auc_ovr -> Py.String.of_string "roc_auc_ovr"
| `Neg_mean_squared_log_error -> Py.String.of_string "neg_mean_squared_log_error"
| `Neg_mean_gamma_deviance -> Py.String.of_string "neg_mean_gamma_deviance"
| `Precision_macro -> Py.String.of_string "precision_macro"
| `R2 -> Py.String.of_string "r2"
| `Precision_micro -> Py.String.of_string "precision_micro"
| `F1_weighted -> Py.String.of_string "f1_weighted"
| `Balanced_accuracy -> Py.String.of_string "balanced_accuracy"
| `Neg_mean_squared_error -> Py.String.of_string "neg_mean_squared_error"
| `F1_samples -> Py.String.of_string "f1_samples"
| `Jaccard_micro -> Py.String.of_string "jaccard_micro"
| `Normalized_mutual_info_score -> Py.String.of_string "normalized_mutual_info_score"
| `F1_micro -> Py.String.of_string "f1_micro"
| `Roc_auc -> Py.String.of_string "roc_auc"
| `Mutual_info_score -> Py.String.of_string "mutual_info_score"
| `Adjusted_rand_score -> Py.String.of_string "adjusted_rand_score"
| `Average_precision -> Py.String.of_string "average_precision"
| `Jaccard -> Py.String.of_string "jaccard"
| `Homogeneity_score -> Py.String.of_string "homogeneity_score"
| `Accuracy -> Py.String.of_string "accuracy"
| `Jaccard_macro -> Py.String.of_string "jaccard_macro"
| `Jaccard_weighted -> Py.String.of_string "jaccard_weighted"
| `Recall_micro -> Py.String.of_string "recall_micro"
| `Explained_variance -> Py.String.of_string "explained_variance"
| `Precision -> Py.String.of_string "precision"
| `Callable x -> Wrap_utils.id x
| `V_measure_score -> Py.String.of_string "v_measure_score"
| `F1 -> Py.String.of_string "f1"
| `Roc_auc_ovo -> Py.String.of_string "roc_auc_ovo"
| `Neg_mean_poisson_deviance -> Py.String.of_string "neg_mean_poisson_deviance"
| `Recall_samples -> Py.String.of_string "recall_samples"
| `Adjusted_mutual_info_score -> Py.String.of_string "adjusted_mutual_info_score"
| `Neg_brier_score -> Py.String.of_string "neg_brier_score"
| `Roc_auc_ovo_weighted -> Py.String.of_string "roc_auc_ovo_weighted"
| `Recall -> Py.String.of_string "recall"
| `Fowlkes_mallows_score -> Py.String.of_string "fowlkes_mallows_score"
| `Neg_log_loss -> Py.String.of_string "neg_log_loss"
| `Neg_root_mean_squared_error -> Py.String.of_string "neg_root_mean_squared_error"
| `Precision_samples -> Py.String.of_string "precision_samples"
| `F1_macro -> Py.String.of_string "f1_macro"
| `Roc_auc_ovr_weighted -> Py.String.of_string "roc_auc_ovr_weighted"
| `Recall_weighted -> Py.String.of_string "recall_weighted"
| `Neg_median_absolute_error -> Py.String.of_string "neg_median_absolute_error"
| `Jaccard_samples -> Py.String.of_string "jaccard_samples"
| `Precision_weighted -> Py.String.of_string "precision_weighted"
| `Max_error -> Py.String.of_string "max_error"
| `Recall_macro -> Py.String.of_string "recall_macro"
)); ("exploit_incremental_learning", Wrap_utils.Option.map exploit_incremental_learning Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("pre_dispatch", Wrap_utils.Option.map pre_dispatch (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("error_score", Wrap_utils.Option.map error_score (function
| `Raise -> Py.String.of_string "raise"
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("return_times", Wrap_utils.Option.map return_times Py.Bool.of_bool); ("estimator", Some(estimator |> Np.Obj.to_pyobject)); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 2)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 3)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 4))))
                  let permutation_test_score ?groups ?cv ?n_permutations ?n_jobs ?random_state ?verbose ?scoring ~estimator ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "permutation_test_score"
                       [||]
                       (Wrap_utils.keyword_args [("groups", Wrap_utils.Option.map groups Np.Obj.to_pyobject); ("cv", Wrap_utils.Option.map cv (function
| `BaseCrossValidator x -> Np.Obj.to_pyobject x
| `Arr x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("n_permutations", Wrap_utils.Option.map n_permutations Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("scoring", Wrap_utils.Option.map scoring (function
| `Neg_mean_absolute_error -> Py.String.of_string "neg_mean_absolute_error"
| `Completeness_score -> Py.String.of_string "completeness_score"
| `Roc_auc_ovr -> Py.String.of_string "roc_auc_ovr"
| `Neg_mean_squared_log_error -> Py.String.of_string "neg_mean_squared_log_error"
| `Neg_mean_gamma_deviance -> Py.String.of_string "neg_mean_gamma_deviance"
| `Precision_macro -> Py.String.of_string "precision_macro"
| `R2 -> Py.String.of_string "r2"
| `Precision_micro -> Py.String.of_string "precision_micro"
| `F1_weighted -> Py.String.of_string "f1_weighted"
| `Balanced_accuracy -> Py.String.of_string "balanced_accuracy"
| `Neg_mean_squared_error -> Py.String.of_string "neg_mean_squared_error"
| `F1_samples -> Py.String.of_string "f1_samples"
| `Jaccard_micro -> Py.String.of_string "jaccard_micro"
| `Normalized_mutual_info_score -> Py.String.of_string "normalized_mutual_info_score"
| `F1_micro -> Py.String.of_string "f1_micro"
| `Roc_auc -> Py.String.of_string "roc_auc"
| `Mutual_info_score -> Py.String.of_string "mutual_info_score"
| `Adjusted_rand_score -> Py.String.of_string "adjusted_rand_score"
| `Average_precision -> Py.String.of_string "average_precision"
| `Jaccard -> Py.String.of_string "jaccard"
| `Homogeneity_score -> Py.String.of_string "homogeneity_score"
| `Accuracy -> Py.String.of_string "accuracy"
| `Jaccard_macro -> Py.String.of_string "jaccard_macro"
| `Jaccard_weighted -> Py.String.of_string "jaccard_weighted"
| `Recall_micro -> Py.String.of_string "recall_micro"
| `Explained_variance -> Py.String.of_string "explained_variance"
| `Precision -> Py.String.of_string "precision"
| `Callable x -> Wrap_utils.id x
| `V_measure_score -> Py.String.of_string "v_measure_score"
| `F1 -> Py.String.of_string "f1"
| `Roc_auc_ovo -> Py.String.of_string "roc_auc_ovo"
| `Neg_mean_poisson_deviance -> Py.String.of_string "neg_mean_poisson_deviance"
| `Recall_samples -> Py.String.of_string "recall_samples"
| `Adjusted_mutual_info_score -> Py.String.of_string "adjusted_mutual_info_score"
| `Neg_brier_score -> Py.String.of_string "neg_brier_score"
| `Roc_auc_ovo_weighted -> Py.String.of_string "roc_auc_ovo_weighted"
| `Recall -> Py.String.of_string "recall"
| `Fowlkes_mallows_score -> Py.String.of_string "fowlkes_mallows_score"
| `Neg_log_loss -> Py.String.of_string "neg_log_loss"
| `Neg_root_mean_squared_error -> Py.String.of_string "neg_root_mean_squared_error"
| `Precision_samples -> Py.String.of_string "precision_samples"
| `F1_macro -> Py.String.of_string "f1_macro"
| `Roc_auc_ovr_weighted -> Py.String.of_string "roc_auc_ovr_weighted"
| `Recall_weighted -> Py.String.of_string "recall_weighted"
| `Neg_median_absolute_error -> Py.String.of_string "neg_median_absolute_error"
| `Jaccard_samples -> Py.String.of_string "jaccard_samples"
| `Precision_weighted -> Py.String.of_string "precision_weighted"
| `Max_error -> Py.String.of_string "max_error"
| `Recall_macro -> Py.String.of_string "recall_macro"
)); ("estimator", Some(estimator |> Np.Obj.to_pyobject)); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `None -> Py.none
)))])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
                  let train_test_split ?test_size ?train_size ?random_state ?shuffle ?stratify arrays =
                     Py.Module.get_function_with_keywords __wrap_namespace "train_test_split"
                       (Array.of_list @@ List.concat [(List.map Np.Obj.to_pyobject arrays)])
                       (Wrap_utils.keyword_args [("test_size", Wrap_utils.Option.map test_size (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("train_size", Wrap_utils.Option.map train_size (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("stratify", Wrap_utils.Option.map stratify Np.Obj.to_pyobject)])
                       |> (fun py -> Py.List.to_list_map ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))) py)
                  let validation_curve ?groups ?cv ?scoring ?n_jobs ?pre_dispatch ?verbose ?error_score ~estimator ~x ~y ~param_name ~param_range () =
                     Py.Module.get_function_with_keywords __wrap_namespace "validation_curve"
                       [||]
                       (Wrap_utils.keyword_args [("groups", Wrap_utils.Option.map groups Np.Obj.to_pyobject); ("cv", Wrap_utils.Option.map cv (function
| `BaseCrossValidator x -> Np.Obj.to_pyobject x
| `Arr x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("scoring", Wrap_utils.Option.map scoring (function
| `Neg_mean_absolute_error -> Py.String.of_string "neg_mean_absolute_error"
| `Completeness_score -> Py.String.of_string "completeness_score"
| `Roc_auc_ovr -> Py.String.of_string "roc_auc_ovr"
| `Neg_mean_squared_log_error -> Py.String.of_string "neg_mean_squared_log_error"
| `Neg_mean_gamma_deviance -> Py.String.of_string "neg_mean_gamma_deviance"
| `Precision_macro -> Py.String.of_string "precision_macro"
| `R2 -> Py.String.of_string "r2"
| `Precision_micro -> Py.String.of_string "precision_micro"
| `F1_weighted -> Py.String.of_string "f1_weighted"
| `Balanced_accuracy -> Py.String.of_string "balanced_accuracy"
| `Neg_mean_squared_error -> Py.String.of_string "neg_mean_squared_error"
| `F1_samples -> Py.String.of_string "f1_samples"
| `Jaccard_micro -> Py.String.of_string "jaccard_micro"
| `Normalized_mutual_info_score -> Py.String.of_string "normalized_mutual_info_score"
| `F1_micro -> Py.String.of_string "f1_micro"
| `Roc_auc -> Py.String.of_string "roc_auc"
| `Mutual_info_score -> Py.String.of_string "mutual_info_score"
| `Adjusted_rand_score -> Py.String.of_string "adjusted_rand_score"
| `Average_precision -> Py.String.of_string "average_precision"
| `Jaccard -> Py.String.of_string "jaccard"
| `Homogeneity_score -> Py.String.of_string "homogeneity_score"
| `Accuracy -> Py.String.of_string "accuracy"
| `Jaccard_macro -> Py.String.of_string "jaccard_macro"
| `Jaccard_weighted -> Py.String.of_string "jaccard_weighted"
| `Recall_micro -> Py.String.of_string "recall_micro"
| `Explained_variance -> Py.String.of_string "explained_variance"
| `Precision -> Py.String.of_string "precision"
| `Callable x -> Wrap_utils.id x
| `V_measure_score -> Py.String.of_string "v_measure_score"
| `F1 -> Py.String.of_string "f1"
| `Roc_auc_ovo -> Py.String.of_string "roc_auc_ovo"
| `Neg_mean_poisson_deviance -> Py.String.of_string "neg_mean_poisson_deviance"
| `Recall_samples -> Py.String.of_string "recall_samples"
| `Adjusted_mutual_info_score -> Py.String.of_string "adjusted_mutual_info_score"
| `Neg_brier_score -> Py.String.of_string "neg_brier_score"
| `Roc_auc_ovo_weighted -> Py.String.of_string "roc_auc_ovo_weighted"
| `Recall -> Py.String.of_string "recall"
| `Fowlkes_mallows_score -> Py.String.of_string "fowlkes_mallows_score"
| `Neg_log_loss -> Py.String.of_string "neg_log_loss"
| `Neg_root_mean_squared_error -> Py.String.of_string "neg_root_mean_squared_error"
| `Precision_samples -> Py.String.of_string "precision_samples"
| `F1_macro -> Py.String.of_string "f1_macro"
| `Roc_auc_ovr_weighted -> Py.String.of_string "roc_auc_ovr_weighted"
| `Recall_weighted -> Py.String.of_string "recall_weighted"
| `Neg_median_absolute_error -> Py.String.of_string "neg_median_absolute_error"
| `Jaccard_samples -> Py.String.of_string "jaccard_samples"
| `Precision_weighted -> Py.String.of_string "precision_weighted"
| `Max_error -> Py.String.of_string "max_error"
| `Recall_macro -> Py.String.of_string "recall_macro"
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("pre_dispatch", Wrap_utils.Option.map pre_dispatch (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("error_score", Wrap_utils.Option.map error_score (function
| `Raise -> Py.String.of_string "raise"
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("estimator", Some(estimator |> Np.Obj.to_pyobject)); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `None -> Py.none
))); ("param_name", Some(param_name |> Py.String.of_string)); ("param_range", Some(param_range |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
