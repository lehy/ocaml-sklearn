let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.inspection"

module PartialDependenceDisplay = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ~pd_results ~features ~feature_names ~target_idx ~pdp_lim ~deciles () =
   Py.Module.get_function_with_keywords ns "PartialDependenceDisplay"
     [||]
     (Wrap_utils.keyword_args [("pd_results", Some(pd_results )); ("features", Some(features )); ("feature_names", Some(feature_names |> (Py.List.of_list_map Py.String.of_string))); ("target_idx", Some(target_idx |> Py.Int.of_int)); ("pdp_lim", Some(pdp_lim )); ("deciles", Some(deciles ))])

let plot ?ax ?n_cols ?line_kw ?contour_kw self =
   Py.Module.get_function_with_keywords self "plot"
     [||]
     (Wrap_utils.keyword_args [("ax", ax); ("n_cols", Wrap_utils.Option.map n_cols Py.Int.of_int); ("line_kw", line_kw); ("contour_kw", contour_kw)])

let bounding_ax_ self =
  match Py.Object.get_attr_string self "bounding_ax_" with
| None -> raise (Wrap_utils.Attribute_not_found "bounding_ax_")
| Some x -> Wrap_utils.id x
let axes_ self =
  match Py.Object.get_attr_string self "axes_" with
| None -> raise (Wrap_utils.Attribute_not_found "axes_")
| Some x -> Wrap_utils.id x
let lines_ self =
  match Py.Object.get_attr_string self "lines_" with
| None -> raise (Wrap_utils.Attribute_not_found "lines_")
| Some x -> Wrap_utils.id x
let contours_ self =
  match Py.Object.get_attr_string self "contours_" with
| None -> raise (Wrap_utils.Attribute_not_found "contours_")
| Some x -> Wrap_utils.id x
let figure_ self =
  match Py.Object.get_attr_string self "figure_" with
| None -> raise (Wrap_utils.Attribute_not_found "figure_")
| Some x -> Wrap_utils.id x
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
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("features", Some(features ))])
                       |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let permutation_importance ?scoring ?n_repeats ?n_jobs ?random_state ~estimator ~x ~y () =
                     Py.Module.get_function_with_keywords ns "permutation_importance"
                       [||]
                       (Wrap_utils.keyword_args [("scoring", Wrap_utils.Option.map scoring (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("n_repeats", Wrap_utils.Option.map n_repeats Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("estimator", Some(estimator )); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("y", Some(y |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `None -> Py.String.of_string "None"
)))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
                  let plot_partial_dependence ?feature_names ?target ?response_method ?n_cols ?grid_resolution ?percentiles ?method_ ?n_jobs ?verbose ?fig ?line_kw ?contour_kw ?ax ~estimator ~x ~features () =
                     Py.Module.get_function_with_keywords ns "plot_partial_dependence"
                       [||]
                       (Wrap_utils.keyword_args [("feature_names", Wrap_utils.Option.map feature_names (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `PyObject x -> Wrap_utils.id x
)); ("target", Wrap_utils.Option.map target Py.Int.of_int); ("response_method", Wrap_utils.Option.map response_method (function
| `Auto -> Py.String.of_string "auto"
| `Predict_proba -> Py.String.of_string "predict_proba"
| `Decision_function -> Py.String.of_string "decision_function"
)); ("n_cols", Wrap_utils.Option.map n_cols Py.Int.of_int); ("grid_resolution", Wrap_utils.Option.map grid_resolution Py.Int.of_int); ("percentiles", percentiles); ("method", Wrap_utils.Option.map method_ Py.String.of_string); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("fig", fig); ("line_kw", line_kw); ("contour_kw", contour_kw); ("ax", ax); ("estimator", Some(estimator )); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("features", Some(features |> (function
| `String x -> Py.String.of_string x
| `PyObject x -> Wrap_utils.id x
)))])

