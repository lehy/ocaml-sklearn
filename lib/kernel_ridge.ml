let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.kernel_ridge"

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
module KernelRidge = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?alpha ?kernel ?gamma ?degree ?coef0 ?kernel_params () =
                     Py.Module.get_function_with_keywords ns "KernelRidge"
                       [||]
                       (Wrap_utils.keyword_args [("alpha", Wrap_utils.Option.map alpha (function
| `F x -> Py.Float.of_float x
| `Arr x -> Arr.to_pyobject x
)); ("kernel", Wrap_utils.Option.map kernel (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("gamma", Wrap_utils.Option.map gamma Py.Float.of_float); ("degree", Wrap_utils.Option.map degree Py.Float.of_float); ("coef0", Wrap_utils.Option.map coef0 Py.Float.of_float); ("kernel_params", Wrap_utils.Option.map kernel_params Dict.to_pyobject)])

                  let fit ?y ?sample_weight ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("sample_weight", Wrap_utils.Option.map sample_weight (function
| `F x -> Py.Float.of_float x
| `Arr x -> Arr.to_pyobject x
)); ("X", Some(x |> Arr.to_pyobject))])

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
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)


let dual_coef_opt self =
  match Py.Object.get_attr_string self "dual_coef_" with
  | None -> failwith "attribute dual_coef_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let dual_coef_ self = match dual_coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_fit_opt self =
  match Py.Object.get_attr_string self "X_fit_" with
  | None -> failwith "attribute X_fit_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let x_fit_ self = match x_fit_opt self with
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
                  let check_X_y ?accept_sparse ?accept_large_sparse ?dtype ?order ?copy ?force_all_finite ?ensure_2d ?allow_nd ?multi_output ?ensure_min_samples ?ensure_min_features ?y_numeric ?warn_on_dtype ?estimator ~x ~y () =
                     Py.Module.get_function_with_keywords ns "check_X_y"
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
)); ("ensure_2d", Wrap_utils.Option.map ensure_2d Py.Bool.of_bool); ("allow_nd", Wrap_utils.Option.map allow_nd Py.Bool.of_bool); ("multi_output", Wrap_utils.Option.map multi_output Py.Bool.of_bool); ("ensure_min_samples", Wrap_utils.Option.map ensure_min_samples Py.Int.of_int); ("ensure_min_features", Wrap_utils.Option.map ensure_min_features Py.Int.of_int); ("y_numeric", Wrap_utils.Option.map y_numeric Py.Bool.of_bool); ("warn_on_dtype", Wrap_utils.Option.map warn_on_dtype Py.Bool.of_bool); ("estimator", Wrap_utils.Option.map estimator (function
| `S x -> Py.String.of_string x
| `Estimator x -> Wrap_utils.id x
)); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
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

