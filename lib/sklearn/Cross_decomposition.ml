let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.cross_decomposition"

let get_py name = Py.Module.get __wrap_namespace name
module CCA = struct
type tag = [`CCA]
type t = [`BaseEstimator | `CCA | `MultiOutputMixin | `Object | `RegressorMixin | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
let create ?n_components ?scale ?max_iter ?tol ?copy () =
   Py.Module.get_function_with_keywords __wrap_namespace "CCA"
     [||]
     (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("scale", Wrap_utils.Option.map scale Py.Bool.of_bool); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])
     |> of_pyobject
let fit ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("Y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let predict ?copy ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])
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
let transform ?y ?copy ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

let x_weights_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "x_weights_" with
  | None -> failwith "attribute x_weights_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let x_weights_ self = match x_weights_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_weights_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "y_weights_" with
  | None -> failwith "attribute y_weights_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let y_weights_ self = match y_weights_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_loadings_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "x_loadings_" with
  | None -> failwith "attribute x_loadings_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let x_loadings_ self = match x_loadings_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_loadings_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "y_loadings_" with
  | None -> failwith "attribute y_loadings_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let y_loadings_ self = match y_loadings_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_scores_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "x_scores_" with
  | None -> failwith "attribute x_scores_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let x_scores_ self = match x_scores_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_scores_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "y_scores_" with
  | None -> failwith "attribute y_scores_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let y_scores_ self = match y_scores_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_rotations_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "x_rotations_" with
  | None -> failwith "attribute x_rotations_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let x_rotations_ self = match x_rotations_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_rotations_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "y_rotations_" with
  | None -> failwith "attribute y_rotations_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let y_rotations_ self = match y_rotations_opt self with
  | None -> raise Not_found
  | Some x -> x

let coef_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "coef_" with
  | None -> failwith "attribute coef_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let coef_ self = match coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module PLSCanonical = struct
type tag = [`PLSCanonical]
type t = [`BaseEstimator | `MultiOutputMixin | `Object | `PLSCanonical | `RegressorMixin | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
                  let create ?n_components ?scale ?algorithm ?max_iter ?tol ?copy () =
                     Py.Module.get_function_with_keywords __wrap_namespace "PLSCanonical"
                       [||]
                       (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("scale", Wrap_utils.Option.map scale Py.Bool.of_bool); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Svd -> Py.String.of_string "svd"
| `Nipals -> Py.String.of_string "nipals"
)); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])
                       |> of_pyobject
let fit ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("Y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let predict ?copy ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])
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
let transform ?y ?copy ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

let x_weights_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "x_weights_" with
  | None -> failwith "attribute x_weights_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let x_weights_ self = match x_weights_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_weights_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "y_weights_" with
  | None -> failwith "attribute y_weights_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let y_weights_ self = match y_weights_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_loadings_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "x_loadings_" with
  | None -> failwith "attribute x_loadings_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let x_loadings_ self = match x_loadings_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_loadings_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "y_loadings_" with
  | None -> failwith "attribute y_loadings_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let y_loadings_ self = match y_loadings_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_scores_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "x_scores_" with
  | None -> failwith "attribute x_scores_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let x_scores_ self = match x_scores_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_scores_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "y_scores_" with
  | None -> failwith "attribute y_scores_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let y_scores_ self = match y_scores_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_rotations_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "x_rotations_" with
  | None -> failwith "attribute x_rotations_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let x_rotations_ self = match x_rotations_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_rotations_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "y_rotations_" with
  | None -> failwith "attribute y_rotations_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let y_rotations_ self = match y_rotations_opt self with
  | None -> raise Not_found
  | Some x -> x

let coef_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "coef_" with
  | None -> failwith "attribute coef_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let coef_ self = match coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module PLSRegression = struct
type tag = [`PLSRegression]
type t = [`BaseEstimator | `MultiOutputMixin | `Object | `PLSRegression | `RegressorMixin | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
let create ?n_components ?scale ?max_iter ?tol ?copy () =
   Py.Module.get_function_with_keywords __wrap_namespace "PLSRegression"
     [||]
     (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("scale", Wrap_utils.Option.map scale Py.Bool.of_bool); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])
     |> of_pyobject
let fit ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("Y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let predict ?copy ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])
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
let transform ?y ?copy ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

let x_weights_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "x_weights_" with
  | None -> failwith "attribute x_weights_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let x_weights_ self = match x_weights_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_weights_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "y_weights_" with
  | None -> failwith "attribute y_weights_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let y_weights_ self = match y_weights_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_loadings_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "x_loadings_" with
  | None -> failwith "attribute x_loadings_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let x_loadings_ self = match x_loadings_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_loadings_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "y_loadings_" with
  | None -> failwith "attribute y_loadings_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let y_loadings_ self = match y_loadings_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_scores_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "x_scores_" with
  | None -> failwith "attribute x_scores_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let x_scores_ self = match x_scores_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_scores_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "y_scores_" with
  | None -> failwith "attribute y_scores_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let y_scores_ self = match y_scores_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_rotations_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "x_rotations_" with
  | None -> failwith "attribute x_rotations_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let x_rotations_ self = match x_rotations_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_rotations_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "y_rotations_" with
  | None -> failwith "attribute y_rotations_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let y_rotations_ self = match y_rotations_opt self with
  | None -> raise Not_found
  | Some x -> x

let coef_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "coef_" with
  | None -> failwith "attribute coef_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let coef_ self = match coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module PLSSVD = struct
type tag = [`PLSSVD]
type t = [`BaseEstimator | `Object | `PLSSVD | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let create ?n_components ?scale ?copy () =
   Py.Module.get_function_with_keywords __wrap_namespace "PLSSVD"
     [||]
     (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("scale", Wrap_utils.Option.map scale Py.Bool.of_bool); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])
     |> of_pyobject
let fit ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("Y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
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
let transform ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

let x_weights_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "x_weights_" with
  | None -> failwith "attribute x_weights_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let x_weights_ self = match x_weights_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_weights_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "y_weights_" with
  | None -> failwith "attribute y_weights_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let y_weights_ self = match y_weights_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_scores_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "x_scores_" with
  | None -> failwith "attribute x_scores_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let x_scores_ self = match x_scores_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_scores_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "y_scores_" with
  | None -> failwith "attribute y_scores_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let y_scores_ self = match y_scores_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
