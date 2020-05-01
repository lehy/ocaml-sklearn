let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.inspection"

let get_py name = Py.Module.get ns name
module PartialDependenceDisplay = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ~pd_results ~features ~feature_names ~target_idx ~pdp_lim ~deciles () =
   Py.Module.get_function_with_keywords ns "PartialDependenceDisplay"
     [||]
     (Wrap_utils.keyword_args [("pd_results", Some(pd_results )); ("features", Some(features )); ("feature_names", Some(feature_names |> (Py.List.of_list_map Py.String.of_string))); ("target_idx", Some(target_idx |> Py.Int.of_int)); ("pdp_lim", Some(pdp_lim |> Dict.to_pyobject)); ("deciles", Some(deciles |> Dict.to_pyobject))])

let plot ?ax ?n_cols ?line_kw ?contour_kw self =
   Py.Module.get_function_with_keywords self "plot"
     [||]
     (Wrap_utils.keyword_args [("ax", ax); ("n_cols", Wrap_utils.Option.map n_cols Py.Int.of_int); ("line_kw", Wrap_utils.Option.map line_kw Dict.to_pyobject); ("contour_kw", Wrap_utils.Option.map contour_kw Dict.to_pyobject)])


let bounding_ax_opt self =
  match Py.Object.get_attr_string self "bounding_ax_" with
  | None -> failwith "attribute bounding_ax_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let bounding_ax_ self = match bounding_ax_opt self with
  | None -> raise Not_found
  | Some x -> x

let axes_opt self =
  match Py.Object.get_attr_string self "axes_" with
  | None -> failwith "attribute axes_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let axes_ self = match axes_opt self with
  | None -> raise Not_found
  | Some x -> x

let lines_opt self =
  match Py.Object.get_attr_string self "lines_" with
  | None -> failwith "attribute lines_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let lines_ self = match lines_opt self with
  | None -> raise Not_found
  | Some x -> x

let contours_opt self =
  match Py.Object.get_attr_string self "contours_" with
  | None -> failwith "attribute contours_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let contours_ self = match contours_opt self with
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
                  let partial_dependence ?response_method ?percentiles ?grid_resolution ?method_ ~estimator ~x ~features () =
                     Py.Module.get_function_with_keywords ns "partial_dependence"
                       [||]
                       (Wrap_utils.keyword_args [("response_method", Wrap_utils.Option.map response_method (function
| `Auto -> Py.String.of_string "auto"
| `Predict_proba -> Py.String.of_string "predict_proba"
| `Decision_function -> Py.String.of_string "decision_function"
)); ("percentiles", percentiles); ("grid_resolution", Wrap_utils.Option.map grid_resolution Py.Int.of_int); ("method", Wrap_utils.Option.map method_ Py.String.of_string); ("estimator", Some(estimator )); ("X", Some(x |> (function
| `Arr x -> Arr.to_pyobject x
| `Dataframe x -> Wrap_utils.id x
))); ("features", Some(features ))])
                       |> (fun x -> ((Arr.of_pyobject (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let permutation_importance ?scoring ?n_repeats ?n_jobs ?random_state ~estimator ~x ~y () =
                     Py.Module.get_function_with_keywords ns "permutation_importance"
                       [||]
                       (Wrap_utils.keyword_args [("scoring", Wrap_utils.Option.map scoring (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("n_repeats", Wrap_utils.Option.map n_repeats Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("estimator", Some(estimator )); ("X", Some(x |> (function
| `Arr x -> Arr.to_pyobject x
| `DataFrame x -> Wrap_utils.id x
))); ("y", Some(y |> (function
| `Arr x -> Arr.to_pyobject x
| `None -> Py.none
)))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Arr.of_pyobject (Py.Tuple.get x 1))))
                  let plot_partial_dependence ?feature_names ?target ?response_method ?n_cols ?grid_resolution ?percentiles ?method_ ?n_jobs ?verbose ?fig ?line_kw ?contour_kw ?ax ~estimator ~x ~features () =
                     Py.Module.get_function_with_keywords ns "plot_partial_dependence"
                       [||]
                       (Wrap_utils.keyword_args [("feature_names", Wrap_utils.Option.map feature_names (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `Dtype_str x -> Wrap_utils.id x
)); ("target", Wrap_utils.Option.map target Py.Int.of_int); ("response_method", Wrap_utils.Option.map response_method (function
| `Auto -> Py.String.of_string "auto"
| `Predict_proba -> Py.String.of_string "predict_proba"
| `Decision_function -> Py.String.of_string "decision_function"
)); ("n_cols", Wrap_utils.Option.map n_cols Py.Int.of_int); ("grid_resolution", Wrap_utils.Option.map grid_resolution Py.Int.of_int); ("percentiles", percentiles); ("method", Wrap_utils.Option.map method_ Py.String.of_string); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("fig", fig); ("line_kw", Wrap_utils.Option.map line_kw Dict.to_pyobject); ("contour_kw", Wrap_utils.Option.map contour_kw Dict.to_pyobject); ("ax", ax); ("estimator", Some(estimator )); ("X", Some(x |> (function
| `Arr x -> Arr.to_pyobject x
| `Dataframe x -> Wrap_utils.id x
))); ("features", Some(features |> (function
| `S x -> Py.String.of_string x
| `PyObject x -> Wrap_utils.id x
)))])

