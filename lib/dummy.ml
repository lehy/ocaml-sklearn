let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.dummy"

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
module ClassifierMixin = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create () =
   Py.Module.get_function_with_keywords ns "ClassifierMixin"
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
module DummyClassifier = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?strategy ?random_state ?constant () =
                     Py.Module.get_function_with_keywords ns "DummyClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("strategy", Wrap_utils.Option.map strategy Py.String.of_string); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("constant", Wrap_utils.Option.map constant (function
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `Arr x -> Arr.to_pyobject x
))])

                  let fit ?sample_weight ~x ~y self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> (function
| `Arr x -> Arr.to_pyobject x
| `Object_with_finite_shape x -> Wrap_utils.id x
))); ("y", Some(y |> Arr.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
                  let predict_log_proba ~x self =
                     Py.Module.get_function_with_keywords self "predict_log_proba"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Arr x -> Arr.to_pyobject x
| `Object_with_finite_shape x -> Wrap_utils.id x
)))])

let predict_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
                  let score ?sample_weight ~x ~y self =
                     Py.Module.get_function_with_keywords self "score"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> (function
| `Arr x -> Arr.to_pyobject x
| `None -> Py.none
))); ("y", Some(y |> Arr.to_pyobject))])
                       |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)


let classes_opt self =
  match Py.Object.get_attr_string self "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_classes_opt self =
  match Py.Object.get_attr_string self "n_classes_" with
  | None -> failwith "attribute n_classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let n_classes_ self = match n_classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let class_prior_opt self =
  match Py.Object.get_attr_string self "class_prior_" with
  | None -> failwith "attribute class_prior_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let class_prior_ self = match class_prior_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_outputs_opt self =
  match Py.Object.get_attr_string self "n_outputs_" with
  | None -> failwith "attribute n_outputs_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_outputs_ self = match n_outputs_opt self with
  | None -> raise Not_found
  | Some x -> x

let sparse_output_opt self =
  match Py.Object.get_attr_string self "sparse_output_" with
  | None -> failwith "attribute sparse_output_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Bool.to_bool x)

let sparse_output_ self = match sparse_output_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module DummyRegressor = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?strategy ?constant ?quantile () =
                     Py.Module.get_function_with_keywords ns "DummyRegressor"
                       [||]
                       (Wrap_utils.keyword_args [("strategy", Wrap_utils.Option.map strategy Py.String.of_string); ("constant", Wrap_utils.Option.map constant (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
| `Arr x -> Arr.to_pyobject x
)); ("quantile", quantile)])

                  let fit ?sample_weight ~x ~y self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> (function
| `Arr x -> Arr.to_pyobject x
| `Object_with_finite_shape x -> Wrap_utils.id x
))); ("y", Some(y |> Arr.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let predict ?return_std ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("return_std", Wrap_utils.Option.map return_std Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
                  let score ?sample_weight ~x ~y self =
                     Py.Module.get_function_with_keywords self "score"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> (function
| `Arr x -> Arr.to_pyobject x
| `None -> Py.none
))); ("y", Some(y |> Arr.to_pyobject))])
                       |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)


let constant_opt self =
  match Py.Object.get_attr_string self "constant_" with
  | None -> failwith "attribute constant_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let constant_ self = match constant_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_outputs_opt self =
  match Py.Object.get_attr_string self "n_outputs_" with
  | None -> failwith "attribute n_outputs_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_outputs_ self = match n_outputs_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MultiOutputMixin = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create () =
   Py.Module.get_function_with_keywords ns "MultiOutputMixin"
     [||]
     []

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

                  let check_is_fitted ?attributes ?msg ?all_or_any ~estimator () =
                     Py.Module.get_function_with_keywords ns "check_is_fitted"
                       [||]
                       (Wrap_utils.keyword_args [("attributes", Wrap_utils.Option.map attributes (function
| `S x -> Py.String.of_string x
| `Arr x -> Arr.to_pyobject x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
)); ("msg", Wrap_utils.Option.map msg Py.String.of_string); ("all_or_any", Wrap_utils.Option.map all_or_any (function
| `Callable x -> Wrap_utils.id x
| `PyObject x -> Wrap_utils.id x
)); ("estimator", Some(estimator ))])

                  let check_random_state ~seed () =
                     Py.Module.get_function_with_keywords ns "check_random_state"
                       [||]
                       (Wrap_utils.keyword_args [("seed", Some(seed |> (function
| `I x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.none
)))])

                  let class_distribution ?sample_weight ~y () =
                     Py.Module.get_function_with_keywords ns "class_distribution"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("y", Some(y |> (function
| `Arr x -> Arr.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
module Deprecated = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?extra () =
   Py.Module.get_function_with_keywords ns "deprecated"
     [||]
     (Wrap_utils.keyword_args [("extra", Wrap_utils.Option.map extra Py.String.of_string)])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
