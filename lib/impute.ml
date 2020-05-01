let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.impute"

let get_py name = Py.Module.get ns name
module KNNImputer = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?missing_values ?n_neighbors ?weights ?metric ?copy ?add_indicator () =
                     Py.Module.get_function_with_keywords ns "KNNImputer"
                       [||]
                       (Wrap_utils.keyword_args [("missing_values", Wrap_utils.Option.map missing_values (function
| `S x -> Py.String.of_string x
| `None -> Py.none
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
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
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

let indicator_opt self =
  match Py.Object.get_attr_string self "indicator_" with
  | None -> failwith "attribute indicator_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let indicator_ self = match indicator_opt self with
  | None -> raise Not_found
  | Some x -> x
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
| `S x -> Py.String.of_string x
| `None -> Py.none
| `PyObject x -> Wrap_utils.id x
)); ("features", Wrap_utils.Option.map features Py.String.of_string); ("sparse", Wrap_utils.Option.map sparse (function
| `Bool x -> Py.Bool.of_bool x
| `Auto -> Py.String.of_string "auto"
)); ("error_on_new", Wrap_utils.Option.map error_on_new Py.Bool.of_bool)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])

let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])
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

let features_opt self =
  match Py.Object.get_attr_string self "features_" with
  | None -> failwith "attribute features_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let features_ self = match features_opt self with
  | None -> raise Not_found
  | Some x -> x
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
| `S x -> Py.String.of_string x
| `None -> Py.none
| `PyObject x -> Wrap_utils.id x
)); ("strategy", Wrap_utils.Option.map strategy Py.String.of_string); ("fill_value", Wrap_utils.Option.map fill_value (function
| `S x -> Py.String.of_string x
| `Numerical_value x -> Wrap_utils.id x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("add_indicator", Wrap_utils.Option.map add_indicator Py.Bool.of_bool)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
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

let statistics_opt self =
  match Py.Object.get_attr_string self "statistics_" with
  | None -> failwith "attribute statistics_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let statistics_ self = match statistics_opt self with
  | None -> raise Not_found
  | Some x -> x

let indicator_opt self =
  match Py.Object.get_attr_string self "indicator_" with
  | None -> failwith "attribute indicator_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let indicator_ self = match indicator_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
