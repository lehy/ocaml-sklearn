let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.metrics"

let get_py name = Py.Module.get ns name
module ConfusionMatrixDisplay = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ~confusion_matrix ~display_labels () =
   Py.Module.get_function_with_keywords ns "ConfusionMatrixDisplay"
     [||]
     (Wrap_utils.keyword_args [("confusion_matrix", Some(confusion_matrix |> Arr.to_pyobject)); ("display_labels", Some(display_labels |> Arr.to_pyobject))])

                  let plot ?include_values ?cmap ?xticks_rotation ?values_format ?ax self =
                     Py.Module.get_function_with_keywords self "plot"
                       [||]
                       (Wrap_utils.keyword_args [("include_values", Wrap_utils.Option.map include_values Py.Bool.of_bool); ("cmap", Wrap_utils.Option.map cmap (function
| `S x -> Py.String.of_string x
| `Matplotlib_Colormap x -> Wrap_utils.id x
)); ("xticks_rotation", Wrap_utils.Option.map xticks_rotation (function
| `Vertical -> Py.String.of_string "vertical"
| `Horizontal -> Py.String.of_string "horizontal"
| `F x -> Py.Float.of_float x
)); ("values_format", Wrap_utils.Option.map values_format Py.String.of_string); ("ax", ax)])


let im_opt self =
  match Py.Object.get_attr_string self "im_" with
  | None -> failwith "attribute im_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let im_ self = match im_opt self with
  | None -> raise Not_found
  | Some x -> x

let text_opt self =
  match Py.Object.get_attr_string self "text_" with
  | None -> failwith "attribute text_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let text_ self = match text_opt self with
  | None -> raise Not_found
  | Some x -> x

let ax_opt self =
  match Py.Object.get_attr_string self "ax_" with
  | None -> failwith "attribute ax_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let ax_ self = match ax_opt self with
  | None -> raise Not_found
  | Some x -> x

let figure_opt self =
  match Py.Object.get_attr_string self "figure_" with
  | None -> failwith "attribute figure_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let figure_ self = match figure_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module PrecisionRecallDisplay = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ~precision ~recall ~average_precision ~estimator_name () =
   Py.Module.get_function_with_keywords ns "PrecisionRecallDisplay"
     [||]
     (Wrap_utils.keyword_args [("precision", Some(precision |> Arr.to_pyobject)); ("recall", Some(recall |> Arr.to_pyobject)); ("average_precision", Some(average_precision |> Py.Float.of_float)); ("estimator_name", Some(estimator_name |> Py.String.of_string))])

let plot ?ax ?name ?kwargs self =
   Py.Module.get_function_with_keywords self "plot"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("ax", ax); ("name", Wrap_utils.Option.map name Py.String.of_string)]) (match kwargs with None -> [] | Some x -> x))


let line_opt self =
  match Py.Object.get_attr_string self "line_" with
  | None -> failwith "attribute line_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let line_ self = match line_opt self with
  | None -> raise Not_found
  | Some x -> x

let ax_opt self =
  match Py.Object.get_attr_string self "ax_" with
  | None -> failwith "attribute ax_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let ax_ self = match ax_opt self with
  | None -> raise Not_found
  | Some x -> x

let figure_opt self =
  match Py.Object.get_attr_string self "figure_" with
  | None -> failwith "attribute figure_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let figure_ self = match figure_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RocCurveDisplay = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ~fpr ~tpr ~roc_auc ~estimator_name () =
   Py.Module.get_function_with_keywords ns "RocCurveDisplay"
     [||]
     (Wrap_utils.keyword_args [("fpr", Some(fpr |> Arr.to_pyobject)); ("tpr", Some(tpr |> Arr.to_pyobject)); ("roc_auc", Some(roc_auc |> Py.Float.of_float)); ("estimator_name", Some(estimator_name |> Py.String.of_string))])

let plot ?ax ?name ?kwargs self =
   Py.Module.get_function_with_keywords self "plot"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("ax", ax); ("name", Wrap_utils.Option.map name Py.String.of_string)]) (match kwargs with None -> [] | Some x -> x))


let line_opt self =
  match Py.Object.get_attr_string self "line_" with
  | None -> failwith "attribute line_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let line_ self = match line_opt self with
  | None -> raise Not_found
  | Some x -> x

let ax_opt self =
  match Py.Object.get_attr_string self "ax_" with
  | None -> failwith "attribute ax_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let ax_ self = match ax_opt self with
  | None -> raise Not_found
  | Some x -> x

let figure_opt self =
  match Py.Object.get_attr_string self "figure_" with
  | None -> failwith "attribute figure_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let figure_ self = match figure_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let accuracy_score ?normalize ?sample_weight ~y_true ~y_pred () =
   Py.Module.get_function_with_keywords ns "accuracy_score"
     [||]
     (Wrap_utils.keyword_args [("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_pred", Some(y_pred |> Arr.to_pyobject))])
     |> Py.Float.to_float
let adjusted_mutual_info_score ?average_method ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords ns "adjusted_mutual_info_score"
     [||]
     (Wrap_utils.keyword_args [("average_method", Wrap_utils.Option.map average_method Py.String.of_string); ("labels_true", Some(labels_true |> Arr.to_pyobject)); ("labels_pred", Some(labels_pred |> Arr.to_pyobject))])
     |> Py.Float.to_float
let adjusted_rand_score ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords ns "adjusted_rand_score"
     [||]
     (Wrap_utils.keyword_args [("labels_true", Some(labels_true |> Arr.to_pyobject)); ("labels_pred", Some(labels_pred |> Arr.to_pyobject))])
     |> Py.Float.to_float
let auc ~x ~y () =
   Py.Module.get_function_with_keywords ns "auc"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
                  let average_precision_score ?average ?pos_label ?sample_weight ~y_true ~y_score () =
                     Py.Module.get_function_with_keywords ns "average_precision_score"
                       [||]
                       (Wrap_utils.keyword_args [("average", Wrap_utils.Option.map average (function
| `Micro -> Py.String.of_string "micro"
| `Macro -> Py.String.of_string "macro"
| `Samples -> Py.String.of_string "samples"
| `Weighted -> Py.String.of_string "weighted"
| `None -> Py.none
)); ("pos_label", Wrap_utils.Option.map pos_label (function
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
)); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_score", Some(y_score |> Arr.to_pyobject))])
                       |> Py.Float.to_float
let balanced_accuracy_score ?sample_weight ?adjusted ~y_true ~y_pred () =
   Py.Module.get_function_with_keywords ns "balanced_accuracy_score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("adjusted", Wrap_utils.Option.map adjusted Py.Bool.of_bool); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_pred", Some(y_pred |> Arr.to_pyobject))])
     |> Py.Float.to_float
                  let brier_score_loss ?sample_weight ?pos_label ~y_true ~y_prob () =
                     Py.Module.get_function_with_keywords ns "brier_score_loss"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("pos_label", Wrap_utils.Option.map pos_label (function
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
)); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_prob", Some(y_prob |> Arr.to_pyobject))])
                       |> Py.Float.to_float
let calinski_harabasz_score ~x ~labels () =
   Py.Module.get_function_with_keywords ns "calinski_harabasz_score"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject)); ("labels", Some(labels |> Arr.to_pyobject))])
     |> Py.Float.to_float
let calinski_harabaz_score ~x ~labels () =
   Py.Module.get_function_with_keywords ns "calinski_harabaz_score"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("labels", Some(labels ))])

                  let check_scoring ?scoring ?allow_none ~estimator () =
                     Py.Module.get_function_with_keywords ns "check_scoring"
                       [||]
                       (Wrap_utils.keyword_args [("scoring", Wrap_utils.Option.map scoring (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("allow_none", Wrap_utils.Option.map allow_none Py.Bool.of_bool); ("estimator", Some(estimator ))])

                  let classification_report ?labels ?target_names ?sample_weight ?digits ?output_dict ?zero_division ~y_true ~y_pred () =
                     Py.Module.get_function_with_keywords ns "classification_report"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Arr.to_pyobject); ("target_names", Wrap_utils.Option.map target_names Arr.to_pyobject); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("digits", Wrap_utils.Option.map digits Py.Int.of_int); ("output_dict", Wrap_utils.Option.map output_dict Py.Bool.of_bool); ("zero_division", Wrap_utils.Option.map zero_division (function
| `Warn -> Py.String.of_string "warn"
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
)); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_pred", Some(y_pred |> Arr.to_pyobject))])
                       |> (fun x -> if Wrap_utils.isinstance Wrap_utils.string x then `S (Py.String.to_string x) else if Wrap_utils.isinstance Wrap_utils.dict x then `Dict ((fun py -> Py.Dict.fold (fun kpy vpy acc -> ((Py.String.to_string kpy), object
      method precision = Py.Dict.get_item_string vpy "precision" |> Wrap_utils.Option.get |> Py.Float.to_float
      method recall = Py.Dict.get_item_string vpy "recall" |> Wrap_utils.Option.get |> Py.Float.to_float
      method f1_score = Py.Dict.get_item_string vpy "f1-score" |> Wrap_utils.Option.get |> Py.Float.to_float
      method support = Py.Dict.get_item_string vpy "support" |> Wrap_utils.Option.get |> Py.Float.to_float
    end)::acc) py [])
     x) else failwith "could not identify type from Python value")
module Cluster = struct
let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.metrics.cluster"

let get_py name = Py.Module.get ns name
let adjusted_mutual_info_score ?average_method ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords ns "adjusted_mutual_info_score"
     [||]
     (Wrap_utils.keyword_args [("average_method", Wrap_utils.Option.map average_method Py.String.of_string); ("labels_true", Some(labels_true |> Arr.to_pyobject)); ("labels_pred", Some(labels_pred |> Arr.to_pyobject))])
     |> Py.Float.to_float
let adjusted_rand_score ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords ns "adjusted_rand_score"
     [||]
     (Wrap_utils.keyword_args [("labels_true", Some(labels_true |> Arr.to_pyobject)); ("labels_pred", Some(labels_pred |> Arr.to_pyobject))])
     |> Py.Float.to_float
let calinski_harabasz_score ~x ~labels () =
   Py.Module.get_function_with_keywords ns "calinski_harabasz_score"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject)); ("labels", Some(labels |> Arr.to_pyobject))])
     |> Py.Float.to_float
let calinski_harabaz_score ~x ~labels () =
   Py.Module.get_function_with_keywords ns "calinski_harabaz_score"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("labels", Some(labels ))])

let completeness_score ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords ns "completeness_score"
     [||]
     (Wrap_utils.keyword_args [("labels_true", Some(labels_true |> Arr.to_pyobject)); ("labels_pred", Some(labels_pred |> Arr.to_pyobject))])
     |> Py.Float.to_float
                  let consensus_score ?similarity ~a ~b () =
                     Py.Module.get_function_with_keywords ns "consensus_score"
                       [||]
                       (Wrap_utils.keyword_args [("similarity", Wrap_utils.Option.map similarity (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("a", Some(a )); ("b", Some(b ))])

let contingency_matrix ?eps ?sparse ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords ns "contingency_matrix"
     [||]
     (Wrap_utils.keyword_args [("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("sparse", Wrap_utils.Option.map sparse Py.Bool.of_bool); ("labels_true", Some(labels_true |> Arr.to_pyobject)); ("labels_pred", Some(labels_pred |> Arr.to_pyobject))])

let davies_bouldin_score ~x ~labels () =
   Py.Module.get_function_with_keywords ns "davies_bouldin_score"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject)); ("labels", Some(labels |> Arr.to_pyobject))])
     |> Py.Float.to_float
let entropy ~labels () =
   Py.Module.get_function_with_keywords ns "entropy"
     [||]
     (Wrap_utils.keyword_args [("labels", Some(labels |> Arr.to_pyobject))])

let fowlkes_mallows_score ?sparse ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords ns "fowlkes_mallows_score"
     [||]
     (Wrap_utils.keyword_args [("sparse", Wrap_utils.Option.map sparse Py.Bool.of_bool); ("labels_true", Some(labels_true |> Arr.to_pyobject)); ("labels_pred", Some(labels_pred |> Arr.to_pyobject))])
     |> Py.Float.to_float
let homogeneity_completeness_v_measure ?beta ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords ns "homogeneity_completeness_v_measure"
     [||]
     (Wrap_utils.keyword_args [("beta", Wrap_utils.Option.map beta Py.Float.of_float); ("labels_true", Some(labels_true |> Arr.to_pyobject)); ("labels_pred", Some(labels_pred |> Arr.to_pyobject))])
     |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let homogeneity_score ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords ns "homogeneity_score"
     [||]
     (Wrap_utils.keyword_args [("labels_true", Some(labels_true |> Arr.to_pyobject)); ("labels_pred", Some(labels_pred |> Arr.to_pyobject))])
     |> Py.Float.to_float
let mutual_info_score ?contingency ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords ns "mutual_info_score"
     [||]
     (Wrap_utils.keyword_args [("contingency", Wrap_utils.Option.map contingency Arr.to_pyobject); ("labels_true", Some(labels_true |> Arr.to_pyobject)); ("labels_pred", Some(labels_pred |> Arr.to_pyobject))])
     |> Py.Float.to_float
let normalized_mutual_info_score ?average_method ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords ns "normalized_mutual_info_score"
     [||]
     (Wrap_utils.keyword_args [("average_method", Wrap_utils.Option.map average_method Py.String.of_string); ("labels_true", Some(labels_true |> Arr.to_pyobject)); ("labels_pred", Some(labels_pred |> Arr.to_pyobject))])
     |> Py.Float.to_float
                  let silhouette_samples ?metric ?kwds ~x ~labels () =
                     Py.Module.get_function_with_keywords ns "silhouette_samples"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("metric", Wrap_utils.Option.map metric (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("X", Some(x |> (function
| `Arr x -> Arr.to_pyobject x
| `Otherwise x -> Wrap_utils.id x
))); ("labels", Some(labels |> Arr.to_pyobject))]) (match kwds with None -> [] | Some x -> x))
                       |> Arr.of_pyobject
                  let silhouette_score ?metric ?sample_size ?random_state ?kwds ~x ~labels () =
                     Py.Module.get_function_with_keywords ns "silhouette_score"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("metric", Wrap_utils.Option.map metric (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("sample_size", Wrap_utils.Option.map sample_size Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("X", Some(x |> (function
| `Arr x -> Arr.to_pyobject x
| `Otherwise x -> Wrap_utils.id x
))); ("labels", Some(labels |> Arr.to_pyobject))]) (match kwds with None -> [] | Some x -> x))
                       |> Py.Float.to_float
let v_measure_score ?beta ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords ns "v_measure_score"
     [||]
     (Wrap_utils.keyword_args [("beta", Wrap_utils.Option.map beta Py.Float.of_float); ("labels_true", Some(labels_true |> Arr.to_pyobject)); ("labels_pred", Some(labels_pred |> Arr.to_pyobject))])
     |> Py.Float.to_float

end
let cohen_kappa_score ?labels ?weights ?sample_weight ~y1 ~y2 () =
   Py.Module.get_function_with_keywords ns "cohen_kappa_score"
     [||]
     (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Arr.to_pyobject); ("weights", Wrap_utils.Option.map weights Py.String.of_string); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("y1", Some(y1 |> Arr.to_pyobject)); ("y2", Some(y2 |> Arr.to_pyobject))])
     |> Py.Float.to_float
let completeness_score ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords ns "completeness_score"
     [||]
     (Wrap_utils.keyword_args [("labels_true", Some(labels_true |> Arr.to_pyobject)); ("labels_pred", Some(labels_pred |> Arr.to_pyobject))])
     |> Py.Float.to_float
                  let confusion_matrix ?labels ?sample_weight ?normalize ~y_true ~y_pred () =
                     Py.Module.get_function_with_keywords ns "confusion_matrix"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Arr.to_pyobject); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("normalize", Wrap_utils.Option.map normalize (function
| `True -> Py.String.of_string "true"
| `Pred -> Py.String.of_string "pred"
| `All -> Py.String.of_string "all"
)); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_pred", Some(y_pred |> Arr.to_pyobject))])
                       |> Arr.of_pyobject
                  let consensus_score ?similarity ~a ~b () =
                     Py.Module.get_function_with_keywords ns "consensus_score"
                       [||]
                       (Wrap_utils.keyword_args [("similarity", Wrap_utils.Option.map similarity (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("a", Some(a )); ("b", Some(b ))])

let coverage_error ?sample_weight ~y_true ~y_score () =
   Py.Module.get_function_with_keywords ns "coverage_error"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_score", Some(y_score |> Arr.to_pyobject))])
     |> Py.Float.to_float
let davies_bouldin_score ~x ~labels () =
   Py.Module.get_function_with_keywords ns "davies_bouldin_score"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject)); ("labels", Some(labels |> Arr.to_pyobject))])
     |> Py.Float.to_float
let dcg_score ?k ?log_base ?sample_weight ?ignore_ties ~y_true ~y_score () =
   Py.Module.get_function_with_keywords ns "dcg_score"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("log_base", Wrap_utils.Option.map log_base Py.Float.of_float); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("ignore_ties", Wrap_utils.Option.map ignore_ties Py.Bool.of_bool); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_score", Some(y_score |> Arr.to_pyobject))])
     |> Py.Float.to_float
let euclidean_distances ?y ?y_norm_squared ?squared ?x_norm_squared ~x () =
   Py.Module.get_function_with_keywords ns "euclidean_distances"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Arr.to_pyobject); ("Y_norm_squared", Wrap_utils.Option.map y_norm_squared Arr.to_pyobject); ("squared", Wrap_utils.Option.map squared Py.Bool.of_bool); ("X_norm_squared", Wrap_utils.Option.map x_norm_squared Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
                  let explained_variance_score ?sample_weight ?multioutput ~y_true ~y_pred () =
                     Py.Module.get_function_with_keywords ns "explained_variance_score"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("multioutput", Wrap_utils.Option.map multioutput (function
| `Raw_values -> Py.String.of_string "raw_values"
| `Uniform_average -> Py.String.of_string "uniform_average"
| `Variance_weighted -> Py.String.of_string "variance_weighted"
| `Arr x -> Arr.to_pyobject x
)); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_pred", Some(y_pred |> Arr.to_pyobject))])
                       |> (fun x -> if Py.Float.check x then `F (Py.Float.to_float x) else if (fun x -> (Wrap_utils.isinstance Wrap_utils.ndarray x) || (Wrap_utils.isinstance Wrap_utils.csr_matrix x)) x then `Arr (Arr.of_pyobject x) else failwith "could not identify type from Python value")
                  let f1_score ?labels ?pos_label ?average ?sample_weight ?zero_division ~y_true ~y_pred () =
                     Py.Module.get_function_with_keywords ns "f1_score"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Arr.to_pyobject); ("pos_label", Wrap_utils.Option.map pos_label (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("average", Wrap_utils.Option.map average (function
| `Binary -> Py.String.of_string "binary"
| `Micro -> Py.String.of_string "micro"
| `Macro -> Py.String.of_string "macro"
| `Samples -> Py.String.of_string "samples"
| `Weighted -> Py.String.of_string "weighted"
| `None -> Py.none
)); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("zero_division", Wrap_utils.Option.map zero_division (function
| `Warn -> Py.String.of_string "warn"
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
)); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_pred", Some(y_pred |> Arr.to_pyobject))])
                       |> (fun x -> if Py.Float.check x then `F (Py.Float.to_float x) else if (fun x -> (Wrap_utils.isinstance Wrap_utils.ndarray x) || (Wrap_utils.isinstance Wrap_utils.csr_matrix x)) x then `Arr (Arr.of_pyobject x) else failwith "could not identify type from Python value")
                  let fbeta_score ?labels ?pos_label ?average ?sample_weight ?zero_division ~y_true ~y_pred ~beta () =
                     Py.Module.get_function_with_keywords ns "fbeta_score"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Arr.to_pyobject); ("pos_label", Wrap_utils.Option.map pos_label (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("average", Wrap_utils.Option.map average (function
| `Binary -> Py.String.of_string "binary"
| `Micro -> Py.String.of_string "micro"
| `Macro -> Py.String.of_string "macro"
| `Samples -> Py.String.of_string "samples"
| `Weighted -> Py.String.of_string "weighted"
| `None -> Py.none
)); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("zero_division", Wrap_utils.Option.map zero_division (function
| `Warn -> Py.String.of_string "warn"
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
)); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_pred", Some(y_pred |> Arr.to_pyobject)); ("beta", Some(beta |> Py.Float.of_float))])
                       |> (fun x -> if Py.Float.check x then `F (Py.Float.to_float x) else if (fun x -> (Wrap_utils.isinstance Wrap_utils.ndarray x) || (Wrap_utils.isinstance Wrap_utils.csr_matrix x)) x then `Arr (Arr.of_pyobject x) else failwith "could not identify type from Python value")
let fowlkes_mallows_score ?sparse ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords ns "fowlkes_mallows_score"
     [||]
     (Wrap_utils.keyword_args [("sparse", Wrap_utils.Option.map sparse Py.Bool.of_bool); ("labels_true", Some(labels_true |> Arr.to_pyobject)); ("labels_pred", Some(labels_pred |> Arr.to_pyobject))])
     |> Py.Float.to_float
                  let get_scorer ~scoring () =
                     Py.Module.get_function_with_keywords ns "get_scorer"
                       [||]
                       (Wrap_utils.keyword_args [("scoring", Some(scoring |> (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)))])

let hamming_loss ?labels ?sample_weight ~y_true ~y_pred () =
   Py.Module.get_function_with_keywords ns "hamming_loss"
     [||]
     (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Arr.to_pyobject); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_pred", Some(y_pred |> Arr.to_pyobject))])
     |> Py.Float.to_float
let hinge_loss ?labels ?sample_weight ~y_true ~pred_decision () =
   Py.Module.get_function_with_keywords ns "hinge_loss"
     [||]
     (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Arr.to_pyobject); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("y_true", Some(y_true |> Arr.to_pyobject)); ("pred_decision", Some(pred_decision |> Arr.to_pyobject))])
     |> Py.Float.to_float
let homogeneity_completeness_v_measure ?beta ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords ns "homogeneity_completeness_v_measure"
     [||]
     (Wrap_utils.keyword_args [("beta", Wrap_utils.Option.map beta Py.Float.of_float); ("labels_true", Some(labels_true |> Arr.to_pyobject)); ("labels_pred", Some(labels_pred |> Arr.to_pyobject))])
     |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let homogeneity_score ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords ns "homogeneity_score"
     [||]
     (Wrap_utils.keyword_args [("labels_true", Some(labels_true |> Arr.to_pyobject)); ("labels_pred", Some(labels_pred |> Arr.to_pyobject))])
     |> Py.Float.to_float
                  let jaccard_score ?labels ?pos_label ?average ?sample_weight ~y_true ~y_pred () =
                     Py.Module.get_function_with_keywords ns "jaccard_score"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Arr.to_pyobject); ("pos_label", Wrap_utils.Option.map pos_label (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("average", Wrap_utils.Option.map average (function
| `Binary -> Py.String.of_string "binary"
| `Micro -> Py.String.of_string "micro"
| `Macro -> Py.String.of_string "macro"
| `Samples -> Py.String.of_string "samples"
| `Weighted -> Py.String.of_string "weighted"
| `None -> Py.none
)); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_pred", Some(y_pred |> Arr.to_pyobject))])
                       |> (fun x -> if Py.Float.check x then `F (Py.Float.to_float x) else if (fun x -> (Wrap_utils.isinstance Wrap_utils.ndarray x) || (Wrap_utils.isinstance Wrap_utils.csr_matrix x)) x then `Arr (Arr.of_pyobject x) else failwith "could not identify type from Python value")
let jaccard_similarity_score ?normalize ?sample_weight ~y_true ~y_pred () =
   Py.Module.get_function_with_keywords ns "jaccard_similarity_score"
     [||]
     (Wrap_utils.keyword_args [("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_pred", Some(y_pred |> Arr.to_pyobject))])
     |> Py.Float.to_float
let label_ranking_average_precision_score ?sample_weight ~y_true ~y_score () =
   Py.Module.get_function_with_keywords ns "label_ranking_average_precision_score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_score", Some(y_score |> Arr.to_pyobject))])
     |> Py.Float.to_float
let label_ranking_loss ?sample_weight ~y_true ~y_score () =
   Py.Module.get_function_with_keywords ns "label_ranking_loss"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_score", Some(y_score |> Arr.to_pyobject))])
     |> Py.Float.to_float
let log_loss ?eps ?normalize ?sample_weight ?labels ~y_true ~y_pred () =
   Py.Module.get_function_with_keywords ns "log_loss"
     [||]
     (Wrap_utils.keyword_args [("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("labels", Wrap_utils.Option.map labels Arr.to_pyobject); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_pred", Some(y_pred |> Arr.to_pyobject))])
     |> Py.Float.to_float
let make_scorer ?greater_is_better ?needs_proba ?needs_threshold ?kwargs ~score_func () =
   Py.Module.get_function_with_keywords ns "make_scorer"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("greater_is_better", Wrap_utils.Option.map greater_is_better Py.Bool.of_bool); ("needs_proba", Wrap_utils.Option.map needs_proba Py.Bool.of_bool); ("needs_threshold", Wrap_utils.Option.map needs_threshold Py.Bool.of_bool); ("score_func", Some(score_func ))]) (match kwargs with None -> [] | Some x -> x))

let matthews_corrcoef ?sample_weight ~y_true ~y_pred () =
   Py.Module.get_function_with_keywords ns "matthews_corrcoef"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_pred", Some(y_pred |> Arr.to_pyobject))])
     |> Py.Float.to_float
let max_error ~y_true ~y_pred () =
   Py.Module.get_function_with_keywords ns "max_error"
     [||]
     (Wrap_utils.keyword_args [("y_true", Some(y_true |> Arr.to_pyobject)); ("y_pred", Some(y_pred |> Arr.to_pyobject))])
     |> Py.Float.to_float
                  let mean_absolute_error ?sample_weight ?multioutput ~y_true ~y_pred () =
                     Py.Module.get_function_with_keywords ns "mean_absolute_error"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("multioutput", Wrap_utils.Option.map multioutput (function
| `Raw_values -> Py.String.of_string "raw_values"
| `Uniform_average -> Py.String.of_string "uniform_average"
| `Arr x -> Arr.to_pyobject x
)); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_pred", Some(y_pred |> Arr.to_pyobject))])
                       |> (fun x -> if Py.Float.check x then `F (Py.Float.to_float x) else if (fun x -> (Wrap_utils.isinstance Wrap_utils.ndarray x) || (Wrap_utils.isinstance Wrap_utils.csr_matrix x)) x then `Arr (Arr.of_pyobject x) else failwith "could not identify type from Python value")
let mean_gamma_deviance ?sample_weight ~y_true ~y_pred () =
   Py.Module.get_function_with_keywords ns "mean_gamma_deviance"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_pred", Some(y_pred |> Arr.to_pyobject))])
     |> Py.Float.to_float
let mean_poisson_deviance ?sample_weight ~y_true ~y_pred () =
   Py.Module.get_function_with_keywords ns "mean_poisson_deviance"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_pred", Some(y_pred |> Arr.to_pyobject))])
     |> Py.Float.to_float
                  let mean_squared_error ?sample_weight ?multioutput ?squared ~y_true ~y_pred () =
                     Py.Module.get_function_with_keywords ns "mean_squared_error"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("multioutput", Wrap_utils.Option.map multioutput (function
| `Raw_values -> Py.String.of_string "raw_values"
| `Uniform_average -> Py.String.of_string "uniform_average"
| `Arr x -> Arr.to_pyobject x
)); ("squared", Wrap_utils.Option.map squared Py.Bool.of_bool); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_pred", Some(y_pred |> Arr.to_pyobject))])
                       |> (fun x -> if Py.Float.check x then `F (Py.Float.to_float x) else if (fun x -> (Wrap_utils.isinstance Wrap_utils.ndarray x) || (Wrap_utils.isinstance Wrap_utils.csr_matrix x)) x then `Arr (Arr.of_pyobject x) else failwith "could not identify type from Python value")
                  let mean_squared_log_error ?sample_weight ?multioutput ~y_true ~y_pred () =
                     Py.Module.get_function_with_keywords ns "mean_squared_log_error"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("multioutput", Wrap_utils.Option.map multioutput (function
| `Raw_values -> Py.String.of_string "raw_values"
| `Uniform_average -> Py.String.of_string "uniform_average"
| `Arr x -> Arr.to_pyobject x
)); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_pred", Some(y_pred |> Arr.to_pyobject))])
                       |> (fun x -> if Py.Float.check x then `F (Py.Float.to_float x) else if (fun x -> (Wrap_utils.isinstance Wrap_utils.ndarray x) || (Wrap_utils.isinstance Wrap_utils.csr_matrix x)) x then `Arr (Arr.of_pyobject x) else failwith "could not identify type from Python value")
let mean_tweedie_deviance ?sample_weight ?power ~y_true ~y_pred () =
   Py.Module.get_function_with_keywords ns "mean_tweedie_deviance"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("power", Wrap_utils.Option.map power Py.Float.of_float); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_pred", Some(y_pred |> Arr.to_pyobject))])
     |> Py.Float.to_float
                  let median_absolute_error ?multioutput ~y_true ~y_pred () =
                     Py.Module.get_function_with_keywords ns "median_absolute_error"
                       [||]
                       (Wrap_utils.keyword_args [("multioutput", Wrap_utils.Option.map multioutput (function
| `Raw_values -> Py.String.of_string "raw_values"
| `Uniform_average -> Py.String.of_string "uniform_average"
| `Array_like_of_shape x -> Wrap_utils.id x
)); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_pred", Some(y_pred |> Arr.to_pyobject))])
                       |> (fun x -> if Py.Float.check x then `F (Py.Float.to_float x) else if (fun x -> (Wrap_utils.isinstance Wrap_utils.ndarray x) || (Wrap_utils.isinstance Wrap_utils.csr_matrix x)) x then `Arr (Arr.of_pyobject x) else failwith "could not identify type from Python value")
let multilabel_confusion_matrix ?sample_weight ?labels ?samplewise ~y_true ~y_pred () =
   Py.Module.get_function_with_keywords ns "multilabel_confusion_matrix"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("labels", Wrap_utils.Option.map labels Arr.to_pyobject); ("samplewise", Wrap_utils.Option.map samplewise Py.Bool.of_bool); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_pred", Some(y_pred |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let mutual_info_score ?contingency ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords ns "mutual_info_score"
     [||]
     (Wrap_utils.keyword_args [("contingency", Wrap_utils.Option.map contingency Arr.to_pyobject); ("labels_true", Some(labels_true |> Arr.to_pyobject)); ("labels_pred", Some(labels_pred |> Arr.to_pyobject))])
     |> Py.Float.to_float
                  let nan_euclidean_distances ?y ?squared ?missing_values ?copy ~x () =
                     Py.Module.get_function_with_keywords ns "nan_euclidean_distances"
                       [||]
                       (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Arr.to_pyobject); ("squared", Wrap_utils.Option.map squared Py.Bool.of_bool); ("missing_values", Wrap_utils.Option.map missing_values (function
| `Np_nan x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])
                       |> Arr.of_pyobject
let ndcg_score ?k ?sample_weight ?ignore_ties ~y_true ~y_score () =
   Py.Module.get_function_with_keywords ns "ndcg_score"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("ignore_ties", Wrap_utils.Option.map ignore_ties Py.Bool.of_bool); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_score", Some(y_score |> Arr.to_pyobject))])

let normalized_mutual_info_score ?average_method ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords ns "normalized_mutual_info_score"
     [||]
     (Wrap_utils.keyword_args [("average_method", Wrap_utils.Option.map average_method Py.String.of_string); ("labels_true", Some(labels_true |> Arr.to_pyobject)); ("labels_pred", Some(labels_pred |> Arr.to_pyobject))])
     |> Py.Float.to_float
module Pairwise = struct
let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.metrics.pairwise"

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
let additive_chi2_kernel ?y ~x () =
   Py.Module.get_function_with_keywords ns "additive_chi2_kernel"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))])
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

let check_non_negative ~x ~whom () =
   Py.Module.get_function_with_keywords ns "check_non_negative"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject)); ("whom", Some(whom |> Py.String.of_string))])

let check_paired_arrays ~x ~y () =
   Py.Module.get_function_with_keywords ns "check_paired_arrays"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject)); ("Y", Some(y |> Arr.to_pyobject))])
     |> (fun x -> ((Arr.of_pyobject (Py.Tuple.get x 0)), (Arr.of_pyobject (Py.Tuple.get x 1))))
                  let check_pairwise_arrays ?precomputed ?dtype ?accept_sparse ?force_all_finite ?copy ~x ~y () =
                     Py.Module.get_function_with_keywords ns "check_pairwise_arrays"
                       [||]
                       (Wrap_utils.keyword_args [("precomputed", Wrap_utils.Option.map precomputed Py.Bool.of_bool); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Wrap_utils.id x
| `TypeList x -> Wrap_utils.id x
)); ("accept_sparse", Wrap_utils.Option.map accept_sparse (function
| `S x -> Py.String.of_string x
| `Bool x -> Py.Bool.of_bool x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
)); ("force_all_finite", Wrap_utils.Option.map force_all_finite (function
| `Bool x -> Py.Bool.of_bool x
| `Allow_nan -> Py.String.of_string "allow-nan"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject)); ("Y", Some(y |> Arr.to_pyobject))])
                       |> (fun x -> ((Arr.of_pyobject (Py.Tuple.get x 0)), (Arr.of_pyobject (Py.Tuple.get x 1))))
let chi2_kernel ?y ?gamma ~x () =
   Py.Module.get_function_with_keywords ns "chi2_kernel"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Arr.to_pyobject); ("gamma", Wrap_utils.Option.map gamma Py.Float.of_float); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let cosine_distances ?y ~x () =
   Py.Module.get_function_with_keywords ns "cosine_distances"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))])

                  let cosine_similarity ?y ?dense_output ~x () =
                     Py.Module.get_function_with_keywords ns "cosine_similarity"
                       [||]
                       (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y (function
| `Arr x -> Arr.to_pyobject x
| `Sparse_array x -> Wrap_utils.id x
)); ("dense_output", Wrap_utils.Option.map dense_output Py.Bool.of_bool); ("X", Some(x |> (function
| `Arr x -> Arr.to_pyobject x
| `Sparse_array x -> Wrap_utils.id x
)))])

let delayed ?check_pickle ~function_ () =
   Py.Module.get_function_with_keywords ns "delayed"
     [||]
     (Wrap_utils.keyword_args [("check_pickle", check_pickle); ("function", Some(function_ ))])

let distance_metrics () =
   Py.Module.get_function_with_keywords ns "distance_metrics"
     [||]
     []

let effective_n_jobs ?n_jobs () =
   Py.Module.get_function_with_keywords ns "effective_n_jobs"
     [||]
     (Wrap_utils.keyword_args [("n_jobs", n_jobs)])

let euclidean_distances ?y ?y_norm_squared ?squared ?x_norm_squared ~x () =
   Py.Module.get_function_with_keywords ns "euclidean_distances"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Arr.to_pyobject); ("Y_norm_squared", Wrap_utils.Option.map y_norm_squared Arr.to_pyobject); ("squared", Wrap_utils.Option.map squared Py.Bool.of_bool); ("X_norm_squared", Wrap_utils.Option.map x_norm_squared Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let gen_batches ?min_batch_size ~n ~batch_size () =
   Py.Module.get_function_with_keywords ns "gen_batches"
     [||]
     (Wrap_utils.keyword_args [("min_batch_size", min_batch_size); ("n", Some(n |> Py.Int.of_int)); ("batch_size", Some(batch_size ))])

let gen_even_slices ?n_samples ~n ~n_packs () =
   Py.Module.get_function_with_keywords ns "gen_even_slices"
     [||]
     (Wrap_utils.keyword_args [("n_samples", n_samples); ("n", Some(n |> Py.Int.of_int)); ("n_packs", Some(n_packs ))])

let get_chunk_n_rows ?max_n_rows ?working_memory ~row_bytes () =
   Py.Module.get_function_with_keywords ns "get_chunk_n_rows"
     [||]
     (Wrap_utils.keyword_args [("max_n_rows", max_n_rows); ("working_memory", working_memory); ("row_bytes", Some(row_bytes |> Py.Int.of_int))])

let haversine_distances ?y ~x () =
   Py.Module.get_function_with_keywords ns "haversine_distances"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let is_scalar_nan ~x () =
   Py.Module.get_function_with_keywords ns "is_scalar_nan"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let issparse ~x () =
   Py.Module.get_function_with_keywords ns "issparse"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let kernel_metrics () =
   Py.Module.get_function_with_keywords ns "kernel_metrics"
     [||]
     []

let laplacian_kernel ?y ?gamma ~x () =
   Py.Module.get_function_with_keywords ns "laplacian_kernel"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Arr.to_pyobject); ("gamma", Wrap_utils.Option.map gamma Py.Float.of_float); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let linear_kernel ?y ?dense_output ~x () =
   Py.Module.get_function_with_keywords ns "linear_kernel"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Arr.to_pyobject); ("dense_output", Wrap_utils.Option.map dense_output Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])

let manhattan_distances ?y ?sum_over_features ~x () =
   Py.Module.get_function_with_keywords ns "manhattan_distances"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Arr.to_pyobject); ("sum_over_features", Wrap_utils.Option.map sum_over_features Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
                  let nan_euclidean_distances ?y ?squared ?missing_values ?copy ~x () =
                     Py.Module.get_function_with_keywords ns "nan_euclidean_distances"
                       [||]
                       (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Arr.to_pyobject); ("squared", Wrap_utils.Option.map squared Py.Bool.of_bool); ("missing_values", Wrap_utils.Option.map missing_values (function
| `Np_nan x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])
                       |> Arr.of_pyobject
                  let normalize ?norm ?axis ?copy ?return_norm ~x () =
                     Py.Module.get_function_with_keywords ns "normalize"
                       [||]
                       (Wrap_utils.keyword_args [("norm", Wrap_utils.Option.map norm (function
| `L1 -> Py.String.of_string "l1"
| `L2 -> Py.String.of_string "l2"
| `Max -> Py.String.of_string "max"
| `T_l2_by x -> Wrap_utils.id x
)); ("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `T_1_by x -> Wrap_utils.id x
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("return_norm", Wrap_utils.Option.map return_norm Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])
                       |> (fun x -> ((Arr.of_pyobject (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let paired_cosine_distances ~x ~y () =
   Py.Module.get_function_with_keywords ns "paired_cosine_distances"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject)); ("Y", Some(y |> Arr.to_pyobject))])
     |> Arr.of_pyobject
                  let paired_distances ?metric ?kwds ~x ~y () =
                     Py.Module.get_function_with_keywords ns "paired_distances"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("metric", Wrap_utils.Option.map metric (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("X", Some(x )); ("Y", Some(y ))]) (match kwds with None -> [] | Some x -> x))

let paired_euclidean_distances ~x ~y () =
   Py.Module.get_function_with_keywords ns "paired_euclidean_distances"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject)); ("Y", Some(y |> Arr.to_pyobject))])

let paired_manhattan_distances ~x ~y () =
   Py.Module.get_function_with_keywords ns "paired_manhattan_distances"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject)); ("Y", Some(y |> Arr.to_pyobject))])

                  let pairwise_distances ?y ?metric ?n_jobs ?force_all_finite ?kwds ~x () =
                     Py.Module.get_function_with_keywords ns "pairwise_distances"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Arr.to_pyobject); ("metric", Wrap_utils.Option.map metric (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("force_all_finite", Wrap_utils.Option.map force_all_finite (function
| `Bool x -> Py.Bool.of_bool x
| `Allow_nan -> Py.String.of_string "allow-nan"
)); ("X", Some(x |> (function
| `Arr x -> Arr.to_pyobject x
| `Otherwise x -> Wrap_utils.id x
)))]) (match kwds with None -> [] | Some x -> x))

                  let pairwise_distances_argmin ?axis ?metric ?metric_kwargs ~x ~y () =
                     Py.Module.get_function_with_keywords ns "pairwise_distances_argmin"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("metric_kwargs", Wrap_utils.Option.map metric_kwargs Dict.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("Y", Some(y |> Arr.to_pyobject))])
                       |> Arr.of_pyobject
                  let pairwise_distances_argmin_min ?axis ?metric ?metric_kwargs ~x ~y () =
                     Py.Module.get_function_with_keywords ns "pairwise_distances_argmin_min"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("metric_kwargs", Wrap_utils.Option.map metric_kwargs Dict.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("Y", Some(y |> Arr.to_pyobject))])
                       |> (fun x -> ((Arr.of_pyobject (Py.Tuple.get x 0)), (Arr.of_pyobject (Py.Tuple.get x 1))))
                  let pairwise_distances_chunked ?y ?reduce_func ?metric ?n_jobs ?working_memory ?kwds ~x () =
                     Py.Module.get_function_with_keywords ns "pairwise_distances_chunked"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Arr.to_pyobject); ("reduce_func", reduce_func); ("metric", Wrap_utils.Option.map metric (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("working_memory", Wrap_utils.Option.map working_memory Py.Int.of_int); ("X", Some(x ))]) (match kwds with None -> [] | Some x -> x))

                  let pairwise_kernels ?y ?metric ?filter_params ?n_jobs ?kwds ~x () =
                     Py.Module.get_function_with_keywords ns "pairwise_kernels"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Arr.to_pyobject); ("metric", Wrap_utils.Option.map metric (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("filter_params", Wrap_utils.Option.map filter_params Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("X", Some(x |> (function
| `Arr x -> Arr.to_pyobject x
| `Otherwise x -> Wrap_utils.id x
)))]) (match kwds with None -> [] | Some x -> x))

let polynomial_kernel ?y ?degree ?gamma ?coef0 ~x () =
   Py.Module.get_function_with_keywords ns "polynomial_kernel"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Arr.to_pyobject); ("degree", Wrap_utils.Option.map degree Py.Int.of_int); ("gamma", Wrap_utils.Option.map gamma Py.Float.of_float); ("coef0", Wrap_utils.Option.map coef0 Py.Float.of_float); ("X", Some(x |> Arr.to_pyobject))])

let rbf_kernel ?y ?gamma ~x () =
   Py.Module.get_function_with_keywords ns "rbf_kernel"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Arr.to_pyobject); ("gamma", Wrap_utils.Option.map gamma Py.Float.of_float); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let row_norms ?squared ~x () =
   Py.Module.get_function_with_keywords ns "row_norms"
     [||]
     (Wrap_utils.keyword_args [("squared", squared); ("X", Some(x |> Arr.to_pyobject))])

let safe_sparse_dot ?dense_output ~a ~b () =
   Py.Module.get_function_with_keywords ns "safe_sparse_dot"
     [||]
     (Wrap_utils.keyword_args [("dense_output", dense_output); ("a", Some(a |> Arr.to_pyobject)); ("b", Some(b ))])
     |> Arr.of_pyobject
let sigmoid_kernel ?y ?gamma ?coef0 ~x () =
   Py.Module.get_function_with_keywords ns "sigmoid_kernel"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Arr.to_pyobject); ("gamma", Wrap_utils.Option.map gamma Py.Float.of_float); ("coef0", Wrap_utils.Option.map coef0 Py.Float.of_float); ("X", Some(x |> Arr.to_pyobject))])


end
                  let pairwise_distances ?y ?metric ?n_jobs ?force_all_finite ?kwds ~x () =
                     Py.Module.get_function_with_keywords ns "pairwise_distances"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Arr.to_pyobject); ("metric", Wrap_utils.Option.map metric (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("force_all_finite", Wrap_utils.Option.map force_all_finite (function
| `Bool x -> Py.Bool.of_bool x
| `Allow_nan -> Py.String.of_string "allow-nan"
)); ("X", Some(x |> (function
| `Arr x -> Arr.to_pyobject x
| `Otherwise x -> Wrap_utils.id x
)))]) (match kwds with None -> [] | Some x -> x))

                  let pairwise_distances_argmin ?axis ?metric ?metric_kwargs ~x ~y () =
                     Py.Module.get_function_with_keywords ns "pairwise_distances_argmin"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("metric_kwargs", Wrap_utils.Option.map metric_kwargs Dict.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("Y", Some(y |> Arr.to_pyobject))])
                       |> Arr.of_pyobject
                  let pairwise_distances_argmin_min ?axis ?metric ?metric_kwargs ~x ~y () =
                     Py.Module.get_function_with_keywords ns "pairwise_distances_argmin_min"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("metric_kwargs", Wrap_utils.Option.map metric_kwargs Dict.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("Y", Some(y |> Arr.to_pyobject))])
                       |> (fun x -> ((Arr.of_pyobject (Py.Tuple.get x 0)), (Arr.of_pyobject (Py.Tuple.get x 1))))
                  let pairwise_distances_chunked ?y ?reduce_func ?metric ?n_jobs ?working_memory ?kwds ~x () =
                     Py.Module.get_function_with_keywords ns "pairwise_distances_chunked"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Arr.to_pyobject); ("reduce_func", reduce_func); ("metric", Wrap_utils.Option.map metric (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("working_memory", Wrap_utils.Option.map working_memory Py.Int.of_int); ("X", Some(x ))]) (match kwds with None -> [] | Some x -> x))

                  let pairwise_kernels ?y ?metric ?filter_params ?n_jobs ?kwds ~x () =
                     Py.Module.get_function_with_keywords ns "pairwise_kernels"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Arr.to_pyobject); ("metric", Wrap_utils.Option.map metric (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("filter_params", Wrap_utils.Option.map filter_params Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("X", Some(x |> (function
| `Arr x -> Arr.to_pyobject x
| `Otherwise x -> Wrap_utils.id x
)))]) (match kwds with None -> [] | Some x -> x))

                  let plot_confusion_matrix ?labels ?sample_weight ?normalize ?display_labels ?include_values ?xticks_rotation ?values_format ?cmap ?ax ~estimator ~x ~y_true () =
                     Py.Module.get_function_with_keywords ns "plot_confusion_matrix"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Arr.to_pyobject); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("normalize", Wrap_utils.Option.map normalize (function
| `True -> Py.String.of_string "true"
| `Pred -> Py.String.of_string "pred"
| `All -> Py.String.of_string "all"
)); ("display_labels", Wrap_utils.Option.map display_labels Arr.to_pyobject); ("include_values", Wrap_utils.Option.map include_values Py.Bool.of_bool); ("xticks_rotation", Wrap_utils.Option.map xticks_rotation (function
| `Vertical -> Py.String.of_string "vertical"
| `Horizontal -> Py.String.of_string "horizontal"
| `F x -> Py.Float.of_float x
)); ("values_format", Wrap_utils.Option.map values_format Py.String.of_string); ("cmap", Wrap_utils.Option.map cmap (function
| `S x -> Py.String.of_string x
| `Matplotlib_Colormap x -> Wrap_utils.id x
)); ("ax", ax); ("estimator", Some(estimator )); ("X", Some(x |> Arr.to_pyobject)); ("y_true", Some(y_true ))])

                  let plot_precision_recall_curve ?sample_weight ?response_method ?name ?ax ?kwargs ~estimator ~x ~y () =
                     Py.Module.get_function_with_keywords ns "plot_precision_recall_curve"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("response_method", Wrap_utils.Option.map response_method (function
| `Predict_proba -> Py.String.of_string "predict_proba"
| `Decision_function -> Py.String.of_string "decision_function"
| `Auto -> Py.String.of_string "auto"
)); ("name", Wrap_utils.Option.map name Py.String.of_string); ("ax", ax); ("estimator", Some(estimator )); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))]) (match kwargs with None -> [] | Some x -> x))

                  let plot_roc_curve ?sample_weight ?drop_intermediate ?response_method ?name ?ax ?kwargs ~estimator ~x ~y () =
                     Py.Module.get_function_with_keywords ns "plot_roc_curve"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("drop_intermediate", Wrap_utils.Option.map drop_intermediate Py.Bool.of_bool); ("response_method", Wrap_utils.Option.map response_method (function
| `Predict_proba -> Py.String.of_string "predict_proba"
| `Decision_function -> Py.String.of_string "decision_function"
| `Auto -> Py.String.of_string "auto"
)); ("name", Wrap_utils.Option.map name Py.String.of_string); ("ax", ax); ("estimator", Some(estimator )); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))]) (match kwargs with None -> [] | Some x -> x))

                  let precision_recall_curve ?pos_label ?sample_weight ~y_true ~probas_pred () =
                     Py.Module.get_function_with_keywords ns "precision_recall_curve"
                       [||]
                       (Wrap_utils.keyword_args [("pos_label", Wrap_utils.Option.map pos_label (function
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
)); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("y_true", Some(y_true |> Arr.to_pyobject)); ("probas_pred", Some(probas_pred |> Arr.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
                  let precision_recall_fscore_support ?beta ?labels ?pos_label ?average ?warn_for ?sample_weight ?zero_division ~y_true ~y_pred () =
                     Py.Module.get_function_with_keywords ns "precision_recall_fscore_support"
                       [||]
                       (Wrap_utils.keyword_args [("beta", Wrap_utils.Option.map beta Py.Float.of_float); ("labels", Wrap_utils.Option.map labels Arr.to_pyobject); ("pos_label", Wrap_utils.Option.map pos_label (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("average", Wrap_utils.Option.map average (function
| `Binary -> Py.String.of_string "binary"
| `Micro -> Py.String.of_string "micro"
| `Macro -> Py.String.of_string "macro"
| `Samples -> Py.String.of_string "samples"
| `Weighted -> Py.String.of_string "weighted"
)); ("warn_for", warn_for); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("zero_division", Wrap_utils.Option.map zero_division (function
| `Warn -> Py.String.of_string "warn"
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
)); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_pred", Some(y_pred |> Arr.to_pyobject))])
                       |> (fun x -> (((fun x -> if Py.Float.check x then `F (Py.Float.to_float x) else if (fun x -> (Wrap_utils.isinstance Wrap_utils.ndarray x) || (Wrap_utils.isinstance Wrap_utils.csr_matrix x)) x then `Arr (Arr.of_pyobject x) else failwith "could not identify type from Python value") (Py.Tuple.get x 0)), ((fun x -> if Py.Float.check x then `F (Py.Float.to_float x) else if (fun x -> (Wrap_utils.isinstance Wrap_utils.ndarray x) || (Wrap_utils.isinstance Wrap_utils.csr_matrix x)) x then `Arr (Arr.of_pyobject x) else failwith "could not identify type from Python value") (Py.Tuple.get x 1)), ((fun x -> if Py.Float.check x then `F (Py.Float.to_float x) else if (fun x -> (Wrap_utils.isinstance Wrap_utils.ndarray x) || (Wrap_utils.isinstance Wrap_utils.csr_matrix x)) x then `Arr (Arr.of_pyobject x) else failwith "could not identify type from Python value") (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3))))
                  let precision_score ?labels ?pos_label ?average ?sample_weight ?zero_division ~y_true ~y_pred () =
                     Py.Module.get_function_with_keywords ns "precision_score"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Arr.to_pyobject); ("pos_label", Wrap_utils.Option.map pos_label (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("average", Wrap_utils.Option.map average (function
| `Binary -> Py.String.of_string "binary"
| `Micro -> Py.String.of_string "micro"
| `Macro -> Py.String.of_string "macro"
| `Samples -> Py.String.of_string "samples"
| `Weighted -> Py.String.of_string "weighted"
| `None -> Py.none
)); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("zero_division", Wrap_utils.Option.map zero_division (function
| `Warn -> Py.String.of_string "warn"
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
)); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_pred", Some(y_pred |> Arr.to_pyobject))])
                       |> (fun x -> if Py.Float.check x then `F (Py.Float.to_float x) else if (fun x -> (Wrap_utils.isinstance Wrap_utils.ndarray x) || (Wrap_utils.isinstance Wrap_utils.csr_matrix x)) x then `Arr (Arr.of_pyobject x) else failwith "could not identify type from Python value")
                  let r2_score ?sample_weight ?multioutput ~y_true ~y_pred () =
                     Py.Module.get_function_with_keywords ns "r2_score"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("multioutput", Wrap_utils.Option.map multioutput (function
| `Raw_values -> Py.String.of_string "raw_values"
| `Uniform_average -> Py.String.of_string "uniform_average"
| `Variance_weighted -> Py.String.of_string "variance_weighted"
| `Arr x -> Arr.to_pyobject x
| `None -> Py.none
)); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_pred", Some(y_pred |> Arr.to_pyobject))])
                       |> (fun x -> if Py.Float.check x then `F (Py.Float.to_float x) else if (fun x -> (Wrap_utils.isinstance Wrap_utils.ndarray x) || (Wrap_utils.isinstance Wrap_utils.csr_matrix x)) x then `Arr (Arr.of_pyobject x) else failwith "could not identify type from Python value")
                  let recall_score ?labels ?pos_label ?average ?sample_weight ?zero_division ~y_true ~y_pred () =
                     Py.Module.get_function_with_keywords ns "recall_score"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Arr.to_pyobject); ("pos_label", Wrap_utils.Option.map pos_label (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("average", Wrap_utils.Option.map average (function
| `Binary -> Py.String.of_string "binary"
| `Micro -> Py.String.of_string "micro"
| `Macro -> Py.String.of_string "macro"
| `Samples -> Py.String.of_string "samples"
| `Weighted -> Py.String.of_string "weighted"
| `None -> Py.none
)); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("zero_division", Wrap_utils.Option.map zero_division (function
| `Warn -> Py.String.of_string "warn"
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
)); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_pred", Some(y_pred |> Arr.to_pyobject))])
                       |> (fun x -> if Py.Float.check x then `F (Py.Float.to_float x) else if (fun x -> (Wrap_utils.isinstance Wrap_utils.ndarray x) || (Wrap_utils.isinstance Wrap_utils.csr_matrix x)) x then `Arr (Arr.of_pyobject x) else failwith "could not identify type from Python value")
                  let roc_auc_score ?average ?sample_weight ?max_fpr ?multi_class ?labels ~y_true ~y_score () =
                     Py.Module.get_function_with_keywords ns "roc_auc_score"
                       [||]
                       (Wrap_utils.keyword_args [("average", Wrap_utils.Option.map average (function
| `Micro -> Py.String.of_string "micro"
| `Macro -> Py.String.of_string "macro"
| `Samples -> Py.String.of_string "samples"
| `Weighted -> Py.String.of_string "weighted"
| `None -> Py.none
)); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("max_fpr", max_fpr); ("multi_class", Wrap_utils.Option.map multi_class (function
| `Raise -> Py.String.of_string "raise"
| `Ovr -> Py.String.of_string "ovr"
| `Ovo -> Py.String.of_string "ovo"
)); ("labels", Wrap_utils.Option.map labels Arr.to_pyobject); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_score", Some(y_score |> Arr.to_pyobject))])
                       |> Py.Float.to_float
                  let roc_curve ?pos_label ?sample_weight ?drop_intermediate ~y_true ~y_score () =
                     Py.Module.get_function_with_keywords ns "roc_curve"
                       [||]
                       (Wrap_utils.keyword_args [("pos_label", Wrap_utils.Option.map pos_label (function
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
)); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("drop_intermediate", Wrap_utils.Option.map drop_intermediate Py.Bool.of_bool); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_score", Some(y_score |> Arr.to_pyobject))])
                       |> (fun x -> ((Arr.of_pyobject (Py.Tuple.get x 0)), (Arr.of_pyobject (Py.Tuple.get x 1)), (Arr.of_pyobject (Py.Tuple.get x 2))))
                  let silhouette_samples ?metric ?kwds ~x ~labels () =
                     Py.Module.get_function_with_keywords ns "silhouette_samples"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("metric", Wrap_utils.Option.map metric (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("X", Some(x |> (function
| `Arr x -> Arr.to_pyobject x
| `Otherwise x -> Wrap_utils.id x
))); ("labels", Some(labels |> Arr.to_pyobject))]) (match kwds with None -> [] | Some x -> x))
                       |> Arr.of_pyobject
                  let silhouette_score ?metric ?sample_size ?random_state ?kwds ~x ~labels () =
                     Py.Module.get_function_with_keywords ns "silhouette_score"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("metric", Wrap_utils.Option.map metric (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("sample_size", Wrap_utils.Option.map sample_size Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("X", Some(x |> (function
| `Arr x -> Arr.to_pyobject x
| `Otherwise x -> Wrap_utils.id x
))); ("labels", Some(labels |> Arr.to_pyobject))]) (match kwds with None -> [] | Some x -> x))
                       |> Py.Float.to_float
let v_measure_score ?beta ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords ns "v_measure_score"
     [||]
     (Wrap_utils.keyword_args [("beta", Wrap_utils.Option.map beta Py.Float.of_float); ("labels_true", Some(labels_true |> Arr.to_pyobject)); ("labels_pred", Some(labels_pred |> Arr.to_pyobject))])
     |> Py.Float.to_float
let zero_one_loss ?normalize ?sample_weight ~y_true ~y_pred () =
   Py.Module.get_function_with_keywords ns "zero_one_loss"
     [||]
     (Wrap_utils.keyword_args [("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("y_true", Some(y_true |> Arr.to_pyobject)); ("y_pred", Some(y_pred |> Arr.to_pyobject))])
     |> (fun x -> if Py.Float.check x then `F (Py.Float.to_float x) else if Py.Int.check x then `I (Py.Int.to_int x) else failwith "could not identify type from Python value")
