let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.inspection"

let get_py name = Py.Module.get __wrap_namespace name
module PartialDependenceDisplay = struct
type tag = [`PartialDependenceDisplay]
type t = [`Object | `PartialDependenceDisplay] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ~pd_results ~features ~feature_names ~target_idx ~pdp_lim ~deciles () =
                     Py.Module.get_function_with_keywords __wrap_namespace "PartialDependenceDisplay"
                       [||]
                       (Wrap_utils.keyword_args [("pd_results", Some(pd_results |> Np.Obj.to_pyobject)); ("features", Some(features |> (function
| `Tuples x -> (fun ml -> Py.List.of_list_map (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.Int.of_int ml_0); (Py.Int.of_int ml_1)]) ml) x
| `List_of_int_ x -> Wrap_utils.id x
))); ("feature_names", Some(feature_names |> (Py.List.of_list_map Py.String.of_string))); ("target_idx", Some(target_idx |> Py.Int.of_int)); ("pdp_lim", Some(pdp_lim |> Dict.to_pyobject)); ("deciles", Some(deciles |> Dict.to_pyobject))])
                       |> of_pyobject
let plot ?ax ?n_cols ?line_kw ?contour_kw self =
   Py.Module.get_function_with_keywords (to_pyobject self) "plot"
     [||]
     (Wrap_utils.keyword_args [("ax", ax); ("n_cols", Wrap_utils.Option.map n_cols Py.Int.of_int); ("line_kw", Wrap_utils.Option.map line_kw Dict.to_pyobject); ("contour_kw", Wrap_utils.Option.map contour_kw Dict.to_pyobject)])


let bounding_ax_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "bounding_ax_" with
  | None -> failwith "attribute bounding_ax_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let bounding_ax_ self = match bounding_ax_opt self with
  | None -> raise Not_found
  | Some x -> x

let axes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "axes_" with
  | None -> failwith "attribute axes_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let axes_ self = match axes_opt self with
  | None -> raise Not_found
  | Some x -> x

let lines_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "lines_" with
  | None -> failwith "attribute lines_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let lines_ self = match lines_opt self with
  | None -> raise Not_found
  | Some x -> x

let deciles_vlines_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "deciles_vlines_" with
  | None -> failwith "attribute deciles_vlines_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let deciles_vlines_ self = match deciles_vlines_opt self with
  | None -> raise Not_found
  | Some x -> x

let deciles_hlines_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "deciles_hlines_" with
  | None -> failwith "attribute deciles_hlines_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let deciles_hlines_ self = match deciles_hlines_opt self with
  | None -> raise Not_found
  | Some x -> x

let contours_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "contours_" with
  | None -> failwith "attribute contours_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let contours_ self = match contours_opt self with
  | None -> raise Not_found
  | Some x -> x

let figure_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "figure_" with
  | None -> failwith "attribute figure_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let figure_ self = match figure_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let partial_dependence ?response_method ?percentiles ?grid_resolution ?method_ ~estimator ~x ~features () =
                     Py.Module.get_function_with_keywords __wrap_namespace "partial_dependence"
                       [||]
                       (Wrap_utils.keyword_args [("response_method", Wrap_utils.Option.map response_method (function
| `Auto -> Py.String.of_string "auto"
| `Predict_proba -> Py.String.of_string "predict_proba"
| `Decision_function -> Py.String.of_string "decision_function"
)); ("percentiles", percentiles); ("grid_resolution", Wrap_utils.Option.map grid_resolution Py.Int.of_int); ("method", Wrap_utils.Option.map method_ Py.String.of_string); ("estimator", Some(estimator |> Np.Obj.to_pyobject)); ("X", Some(x )); ("features", Some(features ))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let permutation_importance ?scoring ?n_repeats ?n_jobs ?random_state ~estimator ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "permutation_importance"
                       [||]
                       (Wrap_utils.keyword_args [("scoring", Wrap_utils.Option.map scoring (function
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
| `Callable x -> Wrap_utils.id x
)); ("n_repeats", Wrap_utils.Option.map n_repeats Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("estimator", Some(estimator |> Np.Obj.to_pyobject)); ("X", Some(x |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `DataFrame x -> Wrap_utils.id x
))); ("y", Some(y |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `None -> Py.none
)))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 2)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 3))))
                  let plot_partial_dependence ?feature_names ?target ?response_method ?n_cols ?grid_resolution ?percentiles ?method_ ?n_jobs ?verbose ?fig ?line_kw ?contour_kw ?ax ~estimator ~x ~features () =
                     Py.Module.get_function_with_keywords __wrap_namespace "plot_partial_dependence"
                       [||]
                       (Wrap_utils.keyword_args [("feature_names", Wrap_utils.Option.map feature_names (function
| `Dtype_str x -> Wrap_utils.id x
| `Ss x -> (fun ml -> Py.List.of_list_map Py.String.of_string ml) x
)); ("target", Wrap_utils.Option.map target Py.Int.of_int); ("response_method", Wrap_utils.Option.map response_method (function
| `Auto -> Py.String.of_string "auto"
| `Predict_proba -> Py.String.of_string "predict_proba"
| `Decision_function -> Py.String.of_string "decision_function"
)); ("n_cols", Wrap_utils.Option.map n_cols Py.Int.of_int); ("grid_resolution", Wrap_utils.Option.map grid_resolution Py.Int.of_int); ("percentiles", percentiles); ("method", Wrap_utils.Option.map method_ Py.String.of_string); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("fig", fig); ("line_kw", Wrap_utils.Option.map line_kw Dict.to_pyobject); ("contour_kw", Wrap_utils.Option.map contour_kw Dict.to_pyobject); ("ax", ax); ("estimator", Some(estimator |> Np.Obj.to_pyobject)); ("X", Some(x )); ("features", Some(features |> (function
| `S x -> Py.String.of_string x
| `PyObject x -> Wrap_utils.id x
)))])

