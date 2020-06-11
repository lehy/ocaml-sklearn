let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.isotonic"

let get_py name = Py.Module.get __wrap_namespace name
module IsotonicRegression = struct
type tag = [`IsotonicRegression]
type t = [`BaseEstimator | `IsotonicRegression | `Object | `RegressorMixin | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
                  let create ?y_min ?y_max ?increasing ?out_of_bounds () =
                     Py.Module.get_function_with_keywords __wrap_namespace "IsotonicRegression"
                       [||]
                       (Wrap_utils.keyword_args [("y_min", y_min); ("y_max", y_max); ("increasing", Wrap_utils.Option.map increasing (function
| `S x -> Py.String.of_string x
| `Bool x -> Py.Bool.of_bool x
)); ("out_of_bounds", Wrap_utils.Option.map out_of_bounds Py.String.of_string)])
                       |> of_pyobject
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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
let predict ~t self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict"
     [||]
     (Wrap_utils.keyword_args [("T", Some(t |> Np.Obj.to_pyobject))])
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
let transform ~t self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("T", Some(t |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

let x_min_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "X_min_" with
  | None -> failwith "attribute X_min_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let x_min_ self = match x_min_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_max_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "X_max_" with
  | None -> failwith "attribute X_max_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let x_max_ self = match x_max_opt self with
  | None -> raise Not_found
  | Some x -> x

let f_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "f_" with
  | None -> failwith "attribute f_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let f_ self = match f_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let check_array ?accept_sparse ?accept_large_sparse ?dtype ?order ?copy ?force_all_finite ?ensure_2d ?allow_nd ?ensure_min_samples ?ensure_min_features ?warn_on_dtype ?estimator ~array () =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_array"
                       [||]
                       (Wrap_utils.keyword_args [("accept_sparse", Wrap_utils.Option.map accept_sparse (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `S x -> Py.String.of_string x
| `Bool x -> Py.Bool.of_bool x
)); ("accept_large_sparse", Wrap_utils.Option.map accept_large_sparse Py.Bool.of_bool); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
| `Dtypes x -> (fun ml -> Py.List.of_list_map Np.Dtype.to_pyobject ml) x
| `None -> Py.none
)); ("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("force_all_finite", Wrap_utils.Option.map force_all_finite (function
| `Allow_nan -> Py.String.of_string "allow-nan"
| `Bool x -> Py.Bool.of_bool x
)); ("ensure_2d", Wrap_utils.Option.map ensure_2d Py.Bool.of_bool); ("allow_nd", Wrap_utils.Option.map allow_nd Py.Bool.of_bool); ("ensure_min_samples", Wrap_utils.Option.map ensure_min_samples Py.Int.of_int); ("ensure_min_features", Wrap_utils.Option.map ensure_min_features Py.Int.of_int); ("warn_on_dtype", Wrap_utils.Option.map warn_on_dtype Py.Bool.of_bool); ("estimator", Wrap_utils.Option.map estimator Np.Obj.to_pyobject); ("array", Some(array ))])

let check_consistent_length arrays =
   Py.Module.get_function_with_keywords __wrap_namespace "check_consistent_length"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arrays)])
     []

let check_increasing ~x ~y () =
   Py.Module.get_function_with_keywords __wrap_namespace "check_increasing"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> Py.Bool.to_bool
let isotonic_regression ?sample_weight ?y_min ?y_max ?increasing ~y () =
   Py.Module.get_function_with_keywords __wrap_namespace "isotonic_regression"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", sample_weight); ("y_min", y_min); ("y_max", y_max); ("increasing", Wrap_utils.Option.map increasing Py.Bool.of_bool); ("y", Some(y ))])
     |> (fun py -> Py.List.to_list_map (Py.Float.to_float) py)
                  let spearmanr ?b ?axis ?nan_policy ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "spearmanr"
                       [||]
                       (Wrap_utils.keyword_args [("b", b); ("axis", Wrap_utils.Option.map axis (function
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("nan_policy", Wrap_utils.Option.map nan_policy (function
| `Propagate -> Py.String.of_string "propagate"
| `Raise -> Py.String.of_string "raise"
| `Omit -> Py.String.of_string "omit"
)); ("a", Some(a ))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))
