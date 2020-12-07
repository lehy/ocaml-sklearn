let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.svm"

let get_py name = Py.Module.get __wrap_namespace name
module LinearSVC = struct
type tag = [`LinearSVC]
type t = [`BaseEstimator | `ClassifierMixin | `LinearClassifierMixin | `LinearSVC | `Object | `SparseCoefMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_linear_classifier x = (x :> [`LinearClassifierMixin] Obj.t)
let as_sparse_coef x = (x :> [`SparseCoefMixin] Obj.t)
                  let create ?penalty ?loss ?dual ?tol ?c ?multi_class ?fit_intercept ?intercept_scaling ?class_weight ?verbose ?random_state ?max_iter () =
                     Py.Module.get_function_with_keywords __wrap_namespace "LinearSVC"
                       [||]
                       (Wrap_utils.keyword_args [("penalty", Wrap_utils.Option.map penalty (function
| `L1 -> Py.String.of_string "l1"
| `L2 -> Py.String.of_string "l2"
)); ("loss", Wrap_utils.Option.map loss (function
| `Hinge -> Py.String.of_string "hinge"
| `Squared_hinge -> Py.String.of_string "squared_hinge"
)); ("dual", Wrap_utils.Option.map dual Py.Bool.of_bool); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("C", Wrap_utils.Option.map c Py.Float.of_float); ("multi_class", Wrap_utils.Option.map multi_class (function
| `Ovr -> Py.String.of_string "ovr"
| `Crammer_singer -> Py.String.of_string "crammer_singer"
)); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("intercept_scaling", Wrap_utils.Option.map intercept_scaling Py.Float.of_float); ("class_weight", Wrap_utils.Option.map class_weight (function
| `Balanced -> Py.String.of_string "balanced"
| `DictIntToFloat x -> (Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float) x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int)])
                       |> of_pyobject
let decision_function ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let densify self =
   Py.Module.get_function_with_keywords (to_pyobject self) "densify"
     [||]
     []
     |> of_pyobject
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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
let sparsify self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sparsify"
     [||]
     []
     |> of_pyobject

let coef_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "coef_" with
  | None -> failwith "attribute coef_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let coef_ self = match coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let intercept_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "intercept_" with
  | None -> failwith "attribute intercept_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let intercept_ self = match intercept_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let classes_ self = match classes_opt self with
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
module LinearSVR = struct
type tag = [`LinearSVR]
type t = [`BaseEstimator | `LinearSVR | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
                  let create ?epsilon ?tol ?c ?loss ?fit_intercept ?intercept_scaling ?dual ?verbose ?random_state ?max_iter () =
                     Py.Module.get_function_with_keywords __wrap_namespace "LinearSVR"
                       [||]
                       (Wrap_utils.keyword_args [("epsilon", Wrap_utils.Option.map epsilon Py.Float.of_float); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("C", Wrap_utils.Option.map c Py.Float.of_float); ("loss", Wrap_utils.Option.map loss (function
| `Epsilon_insensitive -> Py.String.of_string "epsilon_insensitive"
| `Squared_epsilon_insensitive -> Py.String.of_string "squared_epsilon_insensitive"
)); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("intercept_scaling", Wrap_utils.Option.map intercept_scaling Py.Float.of_float); ("dual", Wrap_utils.Option.map dual Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int)])
                       |> of_pyobject
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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

let coef_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "coef_" with
  | None -> failwith "attribute coef_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let coef_ self = match coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let intercept_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "intercept_" with
  | None -> failwith "attribute intercept_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let intercept_ self = match intercept_opt self with
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
module NuSVC = struct
type tag = [`NuSVC]
type t = [`BaseEstimator | `BaseLibSVM | `BaseSVC | `ClassifierMixin | `NuSVC | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_lib_svm x = (x :> [`BaseLibSVM] Obj.t)
let as_svc x = (x :> [`BaseSVC] Obj.t)
                  let create ?nu ?kernel ?degree ?gamma ?coef0 ?shrinking ?probability ?tol ?cache_size ?class_weight ?verbose ?max_iter ?decision_function_shape ?break_ties ?random_state () =
                     Py.Module.get_function_with_keywords __wrap_namespace "NuSVC"
                       [||]
                       (Wrap_utils.keyword_args [("nu", Wrap_utils.Option.map nu Py.Float.of_float); ("kernel", Wrap_utils.Option.map kernel (function
| `Linear -> Py.String.of_string "linear"
| `Poly -> Py.String.of_string "poly"
| `Rbf -> Py.String.of_string "rbf"
| `Sigmoid -> Py.String.of_string "sigmoid"
| `Precomputed -> Py.String.of_string "precomputed"
)); ("degree", Wrap_utils.Option.map degree Py.Int.of_int); ("gamma", Wrap_utils.Option.map gamma (function
| `Scale -> Py.String.of_string "scale"
| `Auto -> Py.String.of_string "auto"
| `F x -> Py.Float.of_float x
)); ("coef0", Wrap_utils.Option.map coef0 Py.Float.of_float); ("shrinking", Wrap_utils.Option.map shrinking Py.Bool.of_bool); ("probability", Wrap_utils.Option.map probability Py.Bool.of_bool); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("cache_size", Wrap_utils.Option.map cache_size Py.Float.of_float); ("class_weight", Wrap_utils.Option.map class_weight (function
| `Balanced -> Py.String.of_string "balanced"
| `DictIntToFloat x -> (Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float) x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("decision_function_shape", Wrap_utils.Option.map decision_function_shape (function
| `Ovo -> Py.String.of_string "ovo"
| `Ovr -> Py.String.of_string "ovr"
)); ("break_ties", Wrap_utils.Option.map break_ties Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
                       |> of_pyobject
let decision_function ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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

let support_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "support_" with
  | None -> failwith "attribute support_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let support_ self = match support_opt self with
  | None -> raise Not_found
  | Some x -> x

let support_vectors_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "support_vectors_" with
  | None -> failwith "attribute support_vectors_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let support_vectors_ self = match support_vectors_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_support_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_support_" with
  | None -> failwith "attribute n_support_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let n_support_ self = match n_support_opt self with
  | None -> raise Not_found
  | Some x -> x

let dual_coef_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dual_coef_" with
  | None -> failwith "attribute dual_coef_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let dual_coef_ self = match dual_coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let coef_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "coef_" with
  | None -> failwith "attribute coef_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let coef_ self = match coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let intercept_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "intercept_" with
  | None -> failwith "attribute intercept_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let intercept_ self = match intercept_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let fit_status_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "fit_status_" with
  | None -> failwith "attribute fit_status_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let fit_status_ self = match fit_status_opt self with
  | None -> raise Not_found
  | Some x -> x

let probA_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "probA_" with
  | None -> failwith "attribute probA_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let probA_ self = match probA_opt self with
  | None -> raise Not_found
  | Some x -> x

let class_weight_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "class_weight_" with
  | None -> failwith "attribute class_weight_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let class_weight_ self = match class_weight_opt self with
  | None -> raise Not_found
  | Some x -> x

let shape_fit_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "shape_fit_" with
  | None -> failwith "attribute shape_fit_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let shape_fit_ self = match shape_fit_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module NuSVR = struct
type tag = [`NuSVR]
type t = [`BaseEstimator | `BaseLibSVM | `NuSVR | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_lib_svm x = (x :> [`BaseLibSVM] Obj.t)
                  let create ?nu ?c ?kernel ?degree ?gamma ?coef0 ?shrinking ?tol ?cache_size ?verbose ?max_iter () =
                     Py.Module.get_function_with_keywords __wrap_namespace "NuSVR"
                       [||]
                       (Wrap_utils.keyword_args [("nu", Wrap_utils.Option.map nu Py.Float.of_float); ("C", Wrap_utils.Option.map c Py.Float.of_float); ("kernel", Wrap_utils.Option.map kernel (function
| `Linear -> Py.String.of_string "linear"
| `Poly -> Py.String.of_string "poly"
| `Rbf -> Py.String.of_string "rbf"
| `Sigmoid -> Py.String.of_string "sigmoid"
| `Precomputed -> Py.String.of_string "precomputed"
)); ("degree", Wrap_utils.Option.map degree Py.Int.of_int); ("gamma", Wrap_utils.Option.map gamma (function
| `Scale -> Py.String.of_string "scale"
| `Auto -> Py.String.of_string "auto"
| `F x -> Py.Float.of_float x
)); ("coef0", Wrap_utils.Option.map coef0 Py.Float.of_float); ("shrinking", Wrap_utils.Option.map shrinking Py.Bool.of_bool); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("cache_size", Wrap_utils.Option.map cache_size Py.Float.of_float); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int)])
                       |> of_pyobject
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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

let support_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "support_" with
  | None -> failwith "attribute support_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let support_ self = match support_opt self with
  | None -> raise Not_found
  | Some x -> x

let support_vectors_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "support_vectors_" with
  | None -> failwith "attribute support_vectors_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let support_vectors_ self = match support_vectors_opt self with
  | None -> raise Not_found
  | Some x -> x

let dual_coef_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dual_coef_" with
  | None -> failwith "attribute dual_coef_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let dual_coef_ self = match dual_coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let coef_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "coef_" with
  | None -> failwith "attribute coef_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let coef_ self = match coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let intercept_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "intercept_" with
  | None -> failwith "attribute intercept_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let intercept_ self = match intercept_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module OneClassSVM = struct
type tag = [`OneClassSVM]
type t = [`BaseEstimator | `BaseLibSVM | `Object | `OneClassSVM | `OutlierMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_lib_svm x = (x :> [`BaseLibSVM] Obj.t)
let as_outlier x = (x :> [`OutlierMixin] Obj.t)
                  let create ?kernel ?degree ?gamma ?coef0 ?tol ?nu ?shrinking ?cache_size ?verbose ?max_iter () =
                     Py.Module.get_function_with_keywords __wrap_namespace "OneClassSVM"
                       [||]
                       (Wrap_utils.keyword_args [("kernel", Wrap_utils.Option.map kernel (function
| `Linear -> Py.String.of_string "linear"
| `Poly -> Py.String.of_string "poly"
| `Rbf -> Py.String.of_string "rbf"
| `Sigmoid -> Py.String.of_string "sigmoid"
| `Precomputed -> Py.String.of_string "precomputed"
)); ("degree", Wrap_utils.Option.map degree Py.Int.of_int); ("gamma", Wrap_utils.Option.map gamma (function
| `Scale -> Py.String.of_string "scale"
| `Auto -> Py.String.of_string "auto"
| `F x -> Py.Float.of_float x
)); ("coef0", Wrap_utils.Option.map coef0 Py.Float.of_float); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("nu", Wrap_utils.Option.map nu Py.Float.of_float); ("shrinking", Wrap_utils.Option.map shrinking Py.Bool.of_bool); ("cache_size", Wrap_utils.Option.map cache_size Py.Float.of_float); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int)])
                       |> of_pyobject
let decision_function ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let fit ?y ?sample_weight ?params ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))]) (match params with None -> [] | Some x -> x))
     |> of_pyobject
let fit_predict ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_predict"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
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
let score_samples ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score_samples"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject

let support_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "support_" with
  | None -> failwith "attribute support_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let support_ self = match support_opt self with
  | None -> raise Not_found
  | Some x -> x

let support_vectors_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "support_vectors_" with
  | None -> failwith "attribute support_vectors_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let support_vectors_ self = match support_vectors_opt self with
  | None -> raise Not_found
  | Some x -> x

let dual_coef_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dual_coef_" with
  | None -> failwith "attribute dual_coef_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let dual_coef_ self = match dual_coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let coef_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "coef_" with
  | None -> failwith "attribute coef_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let coef_ self = match coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let intercept_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "intercept_" with
  | None -> failwith "attribute intercept_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let intercept_ self = match intercept_opt self with
  | None -> raise Not_found
  | Some x -> x

let offset_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "offset_" with
  | None -> failwith "attribute offset_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let offset_ self = match offset_opt self with
  | None -> raise Not_found
  | Some x -> x

let fit_status_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "fit_status_" with
  | None -> failwith "attribute fit_status_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let fit_status_ self = match fit_status_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SVC = struct
type tag = [`SVC]
type t = [`BaseEstimator | `BaseLibSVM | `BaseSVC | `ClassifierMixin | `Object | `SVC] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_lib_svm x = (x :> [`BaseLibSVM] Obj.t)
let as_svc x = (x :> [`BaseSVC] Obj.t)
                  let create ?c ?kernel ?degree ?gamma ?coef0 ?shrinking ?probability ?tol ?cache_size ?class_weight ?verbose ?max_iter ?decision_function_shape ?break_ties ?random_state () =
                     Py.Module.get_function_with_keywords __wrap_namespace "SVC"
                       [||]
                       (Wrap_utils.keyword_args [("C", Wrap_utils.Option.map c Py.Float.of_float); ("kernel", Wrap_utils.Option.map kernel (function
| `Linear -> Py.String.of_string "linear"
| `Poly -> Py.String.of_string "poly"
| `Rbf -> Py.String.of_string "rbf"
| `Sigmoid -> Py.String.of_string "sigmoid"
| `Precomputed -> Py.String.of_string "precomputed"
)); ("degree", Wrap_utils.Option.map degree Py.Int.of_int); ("gamma", Wrap_utils.Option.map gamma (function
| `Scale -> Py.String.of_string "scale"
| `Auto -> Py.String.of_string "auto"
| `F x -> Py.Float.of_float x
)); ("coef0", Wrap_utils.Option.map coef0 Py.Float.of_float); ("shrinking", Wrap_utils.Option.map shrinking Py.Bool.of_bool); ("probability", Wrap_utils.Option.map probability Py.Bool.of_bool); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("cache_size", Wrap_utils.Option.map cache_size Py.Float.of_float); ("class_weight", Wrap_utils.Option.map class_weight (function
| `Balanced -> Py.String.of_string "balanced"
| `DictIntToFloat x -> (Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float) x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("decision_function_shape", Wrap_utils.Option.map decision_function_shape (function
| `Ovo -> Py.String.of_string "ovo"
| `Ovr -> Py.String.of_string "ovr"
)); ("break_ties", Wrap_utils.Option.map break_ties Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
                       |> of_pyobject
let decision_function ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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

let support_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "support_" with
  | None -> failwith "attribute support_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let support_ self = match support_opt self with
  | None -> raise Not_found
  | Some x -> x

let support_vectors_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "support_vectors_" with
  | None -> failwith "attribute support_vectors_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let support_vectors_ self = match support_vectors_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_support_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_support_" with
  | None -> failwith "attribute n_support_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let n_support_ self = match n_support_opt self with
  | None -> raise Not_found
  | Some x -> x

let dual_coef_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dual_coef_" with
  | None -> failwith "attribute dual_coef_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let dual_coef_ self = match dual_coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let coef_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "coef_" with
  | None -> failwith "attribute coef_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let coef_ self = match coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let intercept_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "intercept_" with
  | None -> failwith "attribute intercept_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let intercept_ self = match intercept_opt self with
  | None -> raise Not_found
  | Some x -> x

let fit_status_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "fit_status_" with
  | None -> failwith "attribute fit_status_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let fit_status_ self = match fit_status_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let probA_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "probA_" with
  | None -> failwith "attribute probA_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let probA_ self = match probA_opt self with
  | None -> raise Not_found
  | Some x -> x

let class_weight_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "class_weight_" with
  | None -> failwith "attribute class_weight_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let class_weight_ self = match class_weight_opt self with
  | None -> raise Not_found
  | Some x -> x

let shape_fit_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "shape_fit_" with
  | None -> failwith "attribute shape_fit_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let shape_fit_ self = match shape_fit_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SVR = struct
type tag = [`SVR]
type t = [`BaseEstimator | `BaseLibSVM | `Object | `RegressorMixin | `SVR] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_lib_svm x = (x :> [`BaseLibSVM] Obj.t)
                  let create ?kernel ?degree ?gamma ?coef0 ?tol ?c ?epsilon ?shrinking ?cache_size ?verbose ?max_iter () =
                     Py.Module.get_function_with_keywords __wrap_namespace "SVR"
                       [||]
                       (Wrap_utils.keyword_args [("kernel", Wrap_utils.Option.map kernel (function
| `Linear -> Py.String.of_string "linear"
| `Poly -> Py.String.of_string "poly"
| `Rbf -> Py.String.of_string "rbf"
| `Sigmoid -> Py.String.of_string "sigmoid"
| `Precomputed -> Py.String.of_string "precomputed"
)); ("degree", Wrap_utils.Option.map degree Py.Int.of_int); ("gamma", Wrap_utils.Option.map gamma (function
| `Scale -> Py.String.of_string "scale"
| `Auto -> Py.String.of_string "auto"
| `F x -> Py.Float.of_float x
)); ("coef0", Wrap_utils.Option.map coef0 Py.Float.of_float); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("C", Wrap_utils.Option.map c Py.Float.of_float); ("epsilon", Wrap_utils.Option.map epsilon Py.Float.of_float); ("shrinking", Wrap_utils.Option.map shrinking Py.Bool.of_bool); ("cache_size", Wrap_utils.Option.map cache_size Py.Float.of_float); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int)])
                       |> of_pyobject
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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

let support_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "support_" with
  | None -> failwith "attribute support_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let support_ self = match support_opt self with
  | None -> raise Not_found
  | Some x -> x

let support_vectors_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "support_vectors_" with
  | None -> failwith "attribute support_vectors_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let support_vectors_ self = match support_vectors_opt self with
  | None -> raise Not_found
  | Some x -> x

let dual_coef_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dual_coef_" with
  | None -> failwith "attribute dual_coef_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let dual_coef_ self = match dual_coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let coef_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "coef_" with
  | None -> failwith "attribute coef_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let coef_ self = match coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let fit_status_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "fit_status_" with
  | None -> failwith "attribute fit_status_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let fit_status_ self = match fit_status_opt self with
  | None -> raise Not_found
  | Some x -> x

let intercept_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "intercept_" with
  | None -> failwith "attribute intercept_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let intercept_ self = match intercept_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let l1_min_c ?loss ?fit_intercept ?intercept_scaling ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "l1_min_c"
                       [||]
                       (Wrap_utils.keyword_args [("loss", Wrap_utils.Option.map loss (function
| `Squared_hinge -> Py.String.of_string "squared_hinge"
| `Log -> Py.String.of_string "log"
)); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("intercept_scaling", Wrap_utils.Option.map intercept_scaling Py.Float.of_float); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> Py.Float.to_float
