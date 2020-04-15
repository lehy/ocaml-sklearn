let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.impute"

module KNNImputer = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?missing_values ?n_neighbors ?weights ?metric ?copy ?add_indicator () =
                     Py.Module.get_function_with_keywords ns "KNNImputer"
                       [||]
                       (Wrap_utils.keyword_args [("missing_values", Wrap_utils.Option.map missing_values (function
| `String x -> Py.String.of_string x
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
)); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("weights", Wrap_utils.Option.map weights (function
| `Uniform -> Py.String.of_string "uniform"
| `Distance -> Py.String.of_string "distance"
| `Callable x -> Wrap_utils.id x
)); ("metric", Wrap_utils.Option.map metric (function
| `Nan_euclidean -> Py.String.of_string "nan_euclidean"
| `Callable x -> Wrap_utils.id x
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("add_indicator", Wrap_utils.Option.map add_indicator Py.Bool.of_bool)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x ))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let indicator_ self =
  match Py.Object.get_attr_string self "indicator_" with
| None -> raise (Wrap_utils.Attribute_not_found "indicator_")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MissingIndicator = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?missing_values ?features ?sparse ?error_on_new () =
                     Py.Module.get_function_with_keywords ns "MissingIndicator"
                       [||]
                       (Wrap_utils.keyword_args [("missing_values", Wrap_utils.Option.map missing_values (function
| `String x -> Py.String.of_string x
| `PyObject x -> Wrap_utils.id x
)); ("features", Wrap_utils.Option.map features Py.String.of_string); ("sparse", Wrap_utils.Option.map sparse (function
| `Bool x -> Py.Bool.of_bool x
| `Auto -> Py.String.of_string "auto"
)); ("error_on_new", Wrap_utils.Option.map error_on_new Py.Bool.of_bool)])

                  let fit ?y ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

                  let fit_transform ?y ~x self =
                     Py.Module.get_function_with_keywords self "fit_transform"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

                  let transform ~x self =
                     Py.Module.get_function_with_keywords self "transform"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let features_ self =
  match Py.Object.get_attr_string self "features_" with
| None -> raise (Wrap_utils.Attribute_not_found "features_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SimpleImputer = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?missing_values ?strategy ?fill_value ?verbose ?copy ?add_indicator () =
                     Py.Module.get_function_with_keywords ns "SimpleImputer"
                       [||]
                       (Wrap_utils.keyword_args [("missing_values", Wrap_utils.Option.map missing_values (function
| `String x -> Py.String.of_string x
| `PyObject x -> Wrap_utils.id x
)); ("strategy", Wrap_utils.Option.map strategy Py.String.of_string); ("fill_value", Wrap_utils.Option.map fill_value (function
| `String x -> Py.String.of_string x
| `PyObject x -> Wrap_utils.id x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("add_indicator", Wrap_utils.Option.map add_indicator Py.Bool.of_bool)])

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

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

                  let transform ~x self =
                     Py.Module.get_function_with_keywords self "transform"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let statistics_ self =
  match Py.Object.get_attr_string self "statistics_" with
| None -> raise (Wrap_utils.Attribute_not_found "statistics_")
| Some x -> Ndarray.of_pyobject x
let indicator_ self =
  match Py.Object.get_attr_string self "indicator_" with
| None -> raise (Wrap_utils.Attribute_not_found "indicator_")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
