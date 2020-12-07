let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.impute"

let get_py name = Py.Module.get __wrap_namespace name
module KNNImputer = struct
type tag = [`KNNImputer]
type t = [`BaseEstimator | `KNNImputer | `Object | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
                  let create ?missing_values ?n_neighbors ?weights ?metric ?copy ?add_indicator () =
                     Py.Module.get_function_with_keywords __wrap_namespace "KNNImputer"
                       [||]
                       (Wrap_utils.keyword_args [("missing_values", Wrap_utils.Option.map missing_values (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
| `Np_nan x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
| `None -> Py.none
)); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("weights", Wrap_utils.Option.map weights (function
| `Uniform -> Py.String.of_string "uniform"
| `Callable x -> Wrap_utils.id x
| `Distance -> Py.String.of_string "distance"
)); ("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `Nan_euclidean -> Py.String.of_string "nan_euclidean"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("add_indicator", Wrap_utils.Option.map add_indicator Py.Bool.of_bool)])
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

let indicator_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "indicator_" with
  | None -> failwith "attribute indicator_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let indicator_ self = match indicator_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MissingIndicator = struct
type tag = [`MissingIndicator]
type t = [`BaseEstimator | `MissingIndicator | `Object | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
                  let create ?missing_values ?features ?sparse ?error_on_new () =
                     Py.Module.get_function_with_keywords __wrap_namespace "MissingIndicator"
                       [||]
                       (Wrap_utils.keyword_args [("missing_values", Wrap_utils.Option.map missing_values (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
| `Np_nan x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
| `None -> Py.none
)); ("features", Wrap_utils.Option.map features Py.String.of_string); ("sparse", Wrap_utils.Option.map sparse (function
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("error_on_new", Wrap_utils.Option.map error_on_new Py.Bool.of_bool)])
                       |> of_pyobject
let fit ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> of_pyobject
let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
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

let features_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "features_" with
  | None -> failwith "attribute features_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let features_ self = match features_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SimpleImputer = struct
type tag = [`SimpleImputer]
type t = [`BaseEstimator | `Object | `SimpleImputer | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
                  let create ?missing_values ?strategy ?fill_value ?verbose ?copy ?add_indicator () =
                     Py.Module.get_function_with_keywords __wrap_namespace "SimpleImputer"
                       [||]
                       (Wrap_utils.keyword_args [("missing_values", Wrap_utils.Option.map missing_values (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
| `Np_nan x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
| `None -> Py.none
)); ("strategy", Wrap_utils.Option.map strategy (function
| `Mean -> Py.String.of_string "mean"
| `Median -> Py.String.of_string "median"
| `Most_frequent -> Py.String.of_string "most_frequent"
| `Constant -> Py.String.of_string "constant"
)); ("fill_value", Wrap_utils.Option.map fill_value (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("add_indicator", Wrap_utils.Option.map add_indicator Py.Bool.of_bool)])
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

let statistics_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "statistics_" with
  | None -> failwith "attribute statistics_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let statistics_ self = match statistics_opt self with
  | None -> raise Not_found
  | Some x -> x

let indicator_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "indicator_" with
  | None -> failwith "attribute indicator_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let indicator_ self = match indicator_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
