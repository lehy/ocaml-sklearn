let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.semi_supervised"

let get_py name = Py.Module.get __wrap_namespace name
module LabelPropagation = struct
type tag = [`LabelPropagation]
type t = [`BaseEstimator | `BaseLabelPropagation | `ClassifierMixin | `LabelPropagation | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_label_propagation x = (x :> [`BaseLabelPropagation] Obj.t)
                  let create ?kernel ?gamma ?n_neighbors ?max_iter ?tol ?n_jobs () =
                     Py.Module.get_function_with_keywords __wrap_namespace "LabelPropagation"
                       [||]
                       (Wrap_utils.keyword_args [("kernel", Wrap_utils.Option.map kernel (function
| `Knn -> Py.String.of_string "knn"
| `Callable x -> Wrap_utils.id x
| `Rbf -> Py.String.of_string "rbf"
)); ("gamma", Wrap_utils.Option.map gamma Py.Float.of_float); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol (function
| `T1e_3 x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int)])
                       |> of_pyobject
let fit ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("y", Some(y ))])
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
let predict_proba ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict_proba"
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

let x_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "X_" with
  | None -> failwith "attribute X_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let x_ self = match x_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let label_distributions_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "label_distributions_" with
  | None -> failwith "attribute label_distributions_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let label_distributions_ self = match label_distributions_opt self with
  | None -> raise Not_found
  | Some x -> x

let transduction_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "transduction_" with
  | None -> failwith "attribute transduction_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let transduction_ self = match transduction_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LabelSpreading = struct
type tag = [`LabelSpreading]
type t = [`BaseEstimator | `BaseLabelPropagation | `ClassifierMixin | `LabelSpreading | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_label_propagation x = (x :> [`BaseLabelPropagation] Obj.t)
                  let create ?kernel ?gamma ?n_neighbors ?alpha ?max_iter ?tol ?n_jobs () =
                     Py.Module.get_function_with_keywords __wrap_namespace "LabelSpreading"
                       [||]
                       (Wrap_utils.keyword_args [("kernel", Wrap_utils.Option.map kernel (function
| `Knn -> Py.String.of_string "knn"
| `Callable x -> Wrap_utils.id x
| `Rbf -> Py.String.of_string "rbf"
)); ("gamma", Wrap_utils.Option.map gamma Py.Float.of_float); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int)])
                       |> of_pyobject
let fit ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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
let predict_proba ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict_proba"
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

let x_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "X_" with
  | None -> failwith "attribute X_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let x_ self = match x_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let label_distributions_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "label_distributions_" with
  | None -> failwith "attribute label_distributions_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let label_distributions_ self = match label_distributions_opt self with
  | None -> raise Not_found
  | Some x -> x

let transduction_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "transduction_" with
  | None -> failwith "attribute transduction_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let transduction_ self = match transduction_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
