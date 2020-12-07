let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.dummy"

let get_py name = Py.Module.get __wrap_namespace name
module DummyClassifier = struct
type tag = [`DummyClassifier]
type t = [`BaseEstimator | `ClassifierMixin | `DummyClassifier | `MultiOutputMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
                  let create ?strategy ?random_state ?constant () =
                     Py.Module.get_function_with_keywords __wrap_namespace "DummyClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("strategy", Wrap_utils.Option.map strategy Py.String.of_string); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("constant", Wrap_utils.Option.map constant (function
| `S x -> Py.String.of_string x
| `Arr x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
))])
                       |> of_pyobject
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x )); ("y", Some(y |> Np.Obj.to_pyobject))])
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
let predict_log_proba ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict_log_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])

let predict_proba ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let score ?sample_weight ~x ~y self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "score"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `None -> Py.none
))); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject

let classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_classes_" with
  | None -> failwith "attribute n_classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let n_classes_ self = match n_classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let class_prior_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "class_prior_" with
  | None -> failwith "attribute class_prior_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let class_prior_ self = match class_prior_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_outputs_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_outputs_" with
  | None -> failwith "attribute n_outputs_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_outputs_ self = match n_outputs_opt self with
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
module DummyRegressor = struct
type tag = [`DummyRegressor]
type t = [`BaseEstimator | `DummyRegressor | `MultiOutputMixin | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
                  let create ?strategy ?constant ?quantile () =
                     Py.Module.get_function_with_keywords __wrap_namespace "DummyRegressor"
                       [||]
                       (Wrap_utils.keyword_args [("strategy", Wrap_utils.Option.map strategy Py.String.of_string); ("constant", Wrap_utils.Option.map constant (function
| `F x -> Py.Float.of_float x
| `Arr x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("quantile", Wrap_utils.Option.map quantile Py.Float.of_float)])
                       |> of_pyobject
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x )); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let predict ?return_std ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict"
     [||]
     (Wrap_utils.keyword_args [("return_std", Wrap_utils.Option.map return_std Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let score ?sample_weight ~x ~y self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "score"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `None -> Py.none
))); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject

let constant_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "constant_" with
  | None -> failwith "attribute constant_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let constant_ self = match constant_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_outputs_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_outputs_" with
  | None -> failwith "attribute n_outputs_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_outputs_ self = match n_outputs_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let check_array ?accept_sparse ?accept_large_sparse ?dtype ?order ?copy ?force_all_finite ?ensure_2d ?allow_nd ?ensure_min_samples ?ensure_min_features ?estimator ~array () =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_array"
                       [||]
                       (Wrap_utils.keyword_args [("accept_sparse", Wrap_utils.Option.map accept_sparse (function
| `S x -> Py.String.of_string x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `Bool x -> Py.Bool.of_bool x
)); ("accept_large_sparse", Wrap_utils.Option.map accept_large_sparse Py.Bool.of_bool); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
| `Dtypes x -> (fun ml -> Py.List.of_list_map Np.Dtype.to_pyobject ml) x
| `None -> Py.none
)); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("force_all_finite", Wrap_utils.Option.map force_all_finite (function
| `Allow_nan -> Py.String.of_string "allow-nan"
| `Bool x -> Py.Bool.of_bool x
)); ("ensure_2d", Wrap_utils.Option.map ensure_2d Py.Bool.of_bool); ("allow_nd", Wrap_utils.Option.map allow_nd Py.Bool.of_bool); ("ensure_min_samples", Wrap_utils.Option.map ensure_min_samples Py.Int.of_int); ("ensure_min_features", Wrap_utils.Option.map ensure_min_features Py.Int.of_int); ("estimator", Wrap_utils.Option.map estimator Np.Obj.to_pyobject); ("array", Some(array ))])

let check_consistent_length arrays =
   Py.Module.get_function_with_keywords __wrap_namespace "check_consistent_length"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arrays)])
     []

                  let check_is_fitted ?attributes ?msg ?all_or_any ~estimator () =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_is_fitted"
                       [||]
                       (Wrap_utils.keyword_args [("attributes", Wrap_utils.Option.map attributes (function
| `S x -> Py.String.of_string x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `Arr x -> Np.Obj.to_pyobject x
)); ("msg", Wrap_utils.Option.map msg Py.String.of_string); ("all_or_any", Wrap_utils.Option.map all_or_any (function
| `Callable x -> Wrap_utils.id x
| `PyObject x -> Wrap_utils.id x
)); ("estimator", Some(estimator |> Np.Obj.to_pyobject))])

                  let check_random_state seed =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_random_state"
                       [||]
                       (Wrap_utils.keyword_args [("seed", Some(seed |> (function
| `Optional x -> (function
| `I x -> Py.Int.of_int x
| `None -> Py.none
) x
| `RandomState x -> Wrap_utils.id x
)))])

                  let class_distribution ?sample_weight ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "class_distribution"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("y", Some(y |> (function
| `Sparse_matrix_of_size x -> Wrap_utils.id x
| `Arr x -> Np.Obj.to_pyobject x
)))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
