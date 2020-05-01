let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.isotonic"

let get_py name = Py.Module.get ns name
module BaseEstimator = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create () =
   Py.Module.get_function_with_keywords ns "BaseEstimator"
     [||]
     []

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module IsotonicRegression = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?y_min ?y_max ?increasing ?out_of_bounds () =
                     Py.Module.get_function_with_keywords ns "IsotonicRegression"
                       [||]
                       (Wrap_utils.keyword_args [("y_min", y_min); ("y_max", y_max); ("increasing", Wrap_utils.Option.map increasing (function
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("out_of_bounds", Wrap_utils.Option.map out_of_bounds Py.String.of_string)])

let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

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
let predict ~t self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("T", Some(t |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~t self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("T", Some(t |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let x_min_opt self =
  match Py.Object.get_attr_string self "X_min_" with
  | None -> failwith "attribute X_min_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let x_min_ self = match x_min_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_max_opt self =
  match Py.Object.get_attr_string self "X_max_" with
  | None -> failwith "attribute X_max_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let x_max_ self = match x_max_opt self with
  | None -> raise Not_found
  | Some x -> x

let f_opt self =
  match Py.Object.get_attr_string self "f_" with
  | None -> failwith "attribute f_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let f_ self = match f_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RegressorMixin = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create () =
   Py.Module.get_function_with_keywords ns "RegressorMixin"
     [||]
     []

let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module TransformerMixin = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create () =
   Py.Module.get_function_with_keywords ns "TransformerMixin"
     [||]
     []

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Arr.of_pyobject
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
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

let check_consistent_length arrays =
   Py.Module.get_function_with_keywords ns "check_consistent_length"
     (Wrap_utils.pos_arg Wrap_utils.id arrays)
     []

let check_increasing ~x ~y () =
   Py.Module.get_function_with_keywords ns "check_increasing"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Bool.to_bool
let isotonic_regression ?sample_weight ?y_min ?y_max ?increasing ~y () =
   Py.Module.get_function_with_keywords ns "isotonic_regression"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", sample_weight); ("y_min", y_min); ("y_max", y_max); ("increasing", Wrap_utils.Option.map increasing Py.Bool.of_bool); ("y", Some(y ))])

let spearmanr ?b ?axis ?nan_policy ~a () =
   Py.Module.get_function_with_keywords ns "spearmanr"
     [||]
     (Wrap_utils.keyword_args [("b", b); ("axis", axis); ("nan_policy", nan_policy); ("a", Some(a ))])

