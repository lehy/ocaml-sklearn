let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.compose"

module ColumnTransformer = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?remainder ?sparse_threshold ?n_jobs ?transformer_weights ?verbose ~transformers () =
                     Py.Module.get_function_with_keywords ns "ColumnTransformer"
                       [||]
                       (Wrap_utils.keyword_args [("remainder", Wrap_utils.Option.map remainder (function
| `Drop -> Py.String.of_string "drop"
| `Passthrough -> Py.String.of_string "passthrough"
| `Estimator x -> Wrap_utils.id x
)); ("sparse_threshold", Wrap_utils.Option.map sparse_threshold Py.Float.of_float); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("transformer_weights", transformer_weights); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("transformers", Some(transformers ))])

                  let fit ?y ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])

                  let fit_transform ?y ~x self =
                     Py.Module.get_function_with_keywords self "fit_transform"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])
                       |> Ndarray.of_pyobject
let get_feature_names self =
   Py.Module.get_function_with_keywords self "get_feature_names"
     [||]
     []
     |> (Py.List.to_list_map Py.String.to_string)
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let set_params ?kwargs self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match kwargs with None -> [] | Some x -> x)

                  let transform ~x self =
                     Py.Module.get_function_with_keywords self "transform"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])
                       |> Ndarray.of_pyobject
let transformers_ self =
  match Py.Object.get_attr_string self "transformers_" with
| None -> raise (Wrap_utils.Attribute_not_found "transformers_")
| Some x -> Wrap_utils.id x
let named_transformers_ self =
  match Py.Object.get_attr_string self "named_transformers_" with
| None -> raise (Wrap_utils.Attribute_not_found "named_transformers_")
| Some x -> Wrap_utils.id x
let sparse_output_ self =
  match Py.Object.get_attr_string self "sparse_output_" with
| None -> raise (Wrap_utils.Attribute_not_found "sparse_output_")
| Some x -> Py.Bool.to_bool x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module TransformedTargetRegressor = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?regressor ?transformer ?func ?inverse_func ?check_inverse () =
   Py.Module.get_function_with_keywords ns "TransformedTargetRegressor"
     [||]
     (Wrap_utils.keyword_args [("regressor", regressor); ("transformer", transformer); ("func", func); ("inverse_func", inverse_func); ("check_inverse", Wrap_utils.Option.map check_inverse Py.Bool.of_bool)])

                  let fit ?fit_params ~x ~y self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
))); ("y", Some(y |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

                  let predict ~x self =
                     Py.Module.get_function_with_keywords self "predict"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
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

let regressor_ self =
  match Py.Object.get_attr_string self "regressor_" with
| None -> raise (Wrap_utils.Attribute_not_found "regressor_")
| Some x -> Wrap_utils.id x
let transformer_ self =
  match Py.Object.get_attr_string self "transformer_" with
| None -> raise (Wrap_utils.Attribute_not_found "transformer_")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Make_column_selector = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?pattern ?dtype_include ?dtype_exclude () =
   Py.Module.get_function_with_keywords ns "make_column_selector"
     [||]
     (Wrap_utils.keyword_args [("pattern", Wrap_utils.Option.map pattern Py.String.of_string); ("dtype_include", dtype_include); ("dtype_exclude", dtype_exclude)])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let make_column_transformer ?kwargs transformers =
   Py.Module.get_function_with_keywords ns "make_column_transformer"
     (Wrap_utils.pos_arg Wrap_utils.id transformers)
     (match kwargs with None -> [] | Some x -> x)

