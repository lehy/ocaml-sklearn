let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.compose"

let get_py name = Py.Module.get __wrap_namespace name
module ColumnTransformer = struct
type tag = [`ColumnTransformer]
type t = [`BaseEstimator | `ColumnTransformer | `Object | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
                  let create ?remainder ?sparse_threshold ?n_jobs ?transformer_weights ?verbose ~transformers () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ColumnTransformer"
                       [||]
                       (Wrap_utils.keyword_args [("remainder", Wrap_utils.Option.map remainder (function
| `Passthrough -> Py.String.of_string "passthrough"
| `Drop -> Py.String.of_string "drop"
| `BaseEstimator x -> Np.Obj.to_pyobject x
)); ("sparse_threshold", Wrap_utils.Option.map sparse_threshold Py.Float.of_float); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("transformer_weights", Wrap_utils.Option.map transformer_weights Dict.to_pyobject); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("transformers", Some(transformers |> (fun ml -> Py.List.of_list_map (fun (ml_0, ml_1, ml_2) -> Py.Tuple.of_list [(Py.String.of_string ml_0); (Np.Obj.to_pyobject ml_1); ((function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
| `Ss x -> (fun ml -> Py.List.of_list_map Py.String.of_string ml) x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
| `Slice x -> Np.Wrap_utils.Slice.to_pyobject x
| `Arr x -> Np.Obj.to_pyobject x
| `Callable x -> Wrap_utils.id x
) ml_2)]) ml)))])
                       |> of_pyobject
                  let fit ?y ~x self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("X", Some(x |> (function
| `DataFrame x -> Wrap_utils.id x
| `Arr x -> Np.Obj.to_pyobject x
)))])
                       |> of_pyobject
let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_feature_names self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_feature_names"
     [||]
     []
     |> (Py.List.to_list_map Py.String.to_string)
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let set_params ?kwargs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

let transformers_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "transformers_" with
  | None -> failwith "attribute transformers_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let transformers_ self = match transformers_opt self with
  | None -> raise Not_found
  | Some x -> x

let named_transformers_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "named_transformers_" with
  | None -> failwith "attribute named_transformers_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let named_transformers_ self = match named_transformers_opt self with
  | None -> raise Not_found
  | Some x -> x

let sparse_output_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "sparse_output_" with
  | None -> failwith "attribute sparse_output_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Bool.to_bool x)

let sparse_output_ self = match sparse_output_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module TransformedTargetRegressor = struct
type tag = [`TransformedTargetRegressor]
type t = [`BaseEstimator | `Object | `RegressorMixin | `TransformedTargetRegressor] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let create ?regressor ?transformer ?func ?inverse_func ?check_inverse () =
   Py.Module.get_function_with_keywords __wrap_namespace "TransformedTargetRegressor"
     [||]
     (Wrap_utils.keyword_args [("regressor", Wrap_utils.Option.map regressor Np.Obj.to_pyobject); ("transformer", Wrap_utils.Option.map transformer Np.Obj.to_pyobject); ("func", func); ("inverse_func", inverse_func); ("check_inverse", Wrap_utils.Option.map check_inverse Py.Bool.of_bool)])
     |> of_pyobject
let fit ?fit_params ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let predict ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject

let regressor_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "regressor_" with
  | None -> failwith "attribute regressor_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let regressor_ self = match regressor_opt self with
  | None -> raise Not_found
  | Some x -> x

let transformer_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "transformer_" with
  | None -> failwith "attribute transformer_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let transformer_ self = match transformer_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Make_column_selector = struct
type tag = [`Make_column_selector]
type t = [`Make_column_selector | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?pattern ?dtype_include ?dtype_exclude () =
                     Py.Module.get_function_with_keywords __wrap_namespace "make_column_selector"
                       [||]
                       (Wrap_utils.keyword_args [("pattern", Wrap_utils.Option.map pattern Py.String.of_string); ("dtype_include", Wrap_utils.Option.map dtype_include (function
| `Dtype x -> Np.Dtype.to_pyobject x
| `Dtypes x -> (fun ml -> Py.List.of_list_map Np.Dtype.to_pyobject ml) x
)); ("dtype_exclude", Wrap_utils.Option.map dtype_exclude (function
| `Dtype x -> Np.Dtype.to_pyobject x
| `Dtypes x -> (fun ml -> Py.List.of_list_map Np.Dtype.to_pyobject ml) x
))])
                       |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let make_column_transformer ?kwargs transformers =
                     Py.Module.get_function_with_keywords __wrap_namespace "make_column_transformer"
                       (Array.of_list @@ List.concat [(List.map (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Np.Obj.to_pyobject ml_0); ((function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
| `Ss x -> (fun ml -> Py.List.of_list_map Py.String.of_string ml) x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
| `Slice x -> Np.Wrap_utils.Slice.to_pyobject x
| `Arr x -> Np.Obj.to_pyobject x
| `Callable x -> Wrap_utils.id x
) ml_1)]) transformers)])
                       (match kwargs with None -> [] | Some x -> x)
                       |> ColumnTransformer.of_pyobject
