let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.discriminant_analysis"

let get_py name = Py.Module.get __wrap_namespace name
module LinearClassifierMixin = struct
type tag = [`LinearClassifierMixin]
type t = [`ClassifierMixin | `LinearClassifierMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "LinearClassifierMixin"
     [||]
     []
     |> of_pyobject
let decision_function ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
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
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LinearDiscriminantAnalysis = struct
type tag = [`LinearDiscriminantAnalysis]
type t = [`BaseEstimator | `ClassifierMixin | `LinearClassifierMixin | `LinearDiscriminantAnalysis | `Object | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_linear_classifier x = (x :> [`LinearClassifierMixin] Obj.t)
                  let create ?solver ?shrinkage ?priors ?n_components ?store_covariance ?tol () =
                     Py.Module.get_function_with_keywords __wrap_namespace "LinearDiscriminantAnalysis"
                       [||]
                       (Wrap_utils.keyword_args [("solver", Wrap_utils.Option.map solver (function
| `Svd -> Py.String.of_string "svd"
| `Lsqr -> Py.String.of_string "lsqr"
| `Eigen -> Py.String.of_string "eigen"
)); ("shrinkage", Wrap_utils.Option.map shrinkage (function
| `Auto -> Py.String.of_string "auto"
| `F x -> Py.Float.of_float x
)); ("priors", Wrap_utils.Option.map priors Np.Obj.to_pyobject); ("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("store_covariance", Wrap_utils.Option.map store_covariance Py.Bool.of_bool); ("tol", Wrap_utils.Option.map tol Py.Float.of_float)])
                       |> of_pyobject
let decision_function ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let fit ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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
let predict ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let predict_log_proba ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict_log_proba"
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
let transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

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

let covariance_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "covariance_" with
  | None -> failwith "attribute covariance_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let covariance_ self = match covariance_opt self with
  | None -> raise Not_found
  | Some x -> x

let explained_variance_ratio_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "explained_variance_ratio_" with
  | None -> failwith "attribute explained_variance_ratio_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let explained_variance_ratio_ self = match explained_variance_ratio_opt self with
  | None -> raise Not_found
  | Some x -> x

let means_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "means_" with
  | None -> failwith "attribute means_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let means_ self = match means_opt self with
  | None -> raise Not_found
  | Some x -> x

let priors_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "priors_" with
  | None -> failwith "attribute priors_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let priors_ self = match priors_opt self with
  | None -> raise Not_found
  | Some x -> x

let scalings_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "scalings_" with
  | None -> failwith "attribute scalings_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let scalings_ self = match scalings_opt self with
  | None -> raise Not_found
  | Some x -> x

let xbar_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "xbar_" with
  | None -> failwith "attribute xbar_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let xbar_ self = match xbar_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module QuadraticDiscriminantAnalysis = struct
type tag = [`QuadraticDiscriminantAnalysis]
type t = [`BaseEstimator | `ClassifierMixin | `Object | `QuadraticDiscriminantAnalysis] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let create ?priors ?reg_param ?store_covariance ?tol () =
   Py.Module.get_function_with_keywords __wrap_namespace "QuadraticDiscriminantAnalysis"
     [||]
     (Wrap_utils.keyword_args [("priors", Wrap_utils.Option.map priors Np.Obj.to_pyobject); ("reg_param", Wrap_utils.Option.map reg_param Py.Float.of_float); ("store_covariance", Wrap_utils.Option.map store_covariance Py.Bool.of_bool); ("tol", Wrap_utils.Option.map tol Py.Float.of_float)])
     |> of_pyobject
let decision_function ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
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
let predict_log_proba ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict_log_proba"
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

let covariance_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "covariance_" with
  | None -> failwith "attribute covariance_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let covariance_ self = match covariance_opt self with
  | None -> raise Not_found
  | Some x -> x

let means_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "means_" with
  | None -> failwith "attribute means_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let means_ self = match means_opt self with
  | None -> raise Not_found
  | Some x -> x

let priors_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "priors_" with
  | None -> failwith "attribute priors_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let priors_ self = match priors_opt self with
  | None -> raise Not_found
  | Some x -> x

let rotations_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "rotations_" with
  | None -> failwith "attribute rotations_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let rotations_ self = match rotations_opt self with
  | None -> raise Not_found
  | Some x -> x

let scalings_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "scalings_" with
  | None -> failwith "attribute scalings_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let scalings_ self = match scalings_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module StandardScaler = struct
type tag = [`StandardScaler]
type t = [`BaseEstimator | `Object | `StandardScaler | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let create ?copy ?with_mean ?with_std () =
   Py.Module.get_function_with_keywords __wrap_namespace "StandardScaler"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("with_mean", Wrap_utils.Option.map with_mean Py.Bool.of_bool); ("with_std", Wrap_utils.Option.map with_std Py.Bool.of_bool)])
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
let inverse_transform ?copy ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let partial_fit ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let transform ?copy ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

let scale_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "scale_" with
  | None -> failwith "attribute scale_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let scale_ self = match scale_opt self with
  | None -> raise Not_found
  | Some x -> x

let mean_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "mean_" with
  | None -> failwith "attribute mean_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let mean_ self = match mean_opt self with
  | None -> raise Not_found
  | Some x -> x

let var_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "var_" with
  | None -> failwith "attribute var_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let var_ self = match var_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_samples_seen_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_samples_seen_" with
  | None -> failwith "attribute n_samples_seen_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let n_samples_seen_ self = match n_samples_seen_opt self with
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

let check_classification_targets y =
   Py.Module.get_function_with_keywords __wrap_namespace "check_classification_targets"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Np.Obj.to_pyobject))])

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

let empirical_covariance ?assume_centered ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "empirical_covariance"
     [||]
     (Wrap_utils.keyword_args [("assume_centered", Wrap_utils.Option.map assume_centered Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let expit ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "expit"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let ledoit_wolf ?assume_centered ?block_size ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ledoit_wolf"
     [||]
     (Wrap_utils.keyword_args [("assume_centered", Wrap_utils.Option.map assume_centered Py.Bool.of_bool); ("block_size", Wrap_utils.Option.map block_size Py.Int.of_int); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))
let shrunk_covariance ?shrinkage ~emp_cov () =
   Py.Module.get_function_with_keywords __wrap_namespace "shrunk_covariance"
     [||]
     (Wrap_utils.keyword_args [("shrinkage", Wrap_utils.Option.map shrinkage Py.Float.of_float); ("emp_cov", Some(emp_cov |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let softmax ?copy ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "softmax"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let unique_labels ys =
   Py.Module.get_function_with_keywords __wrap_namespace "unique_labels"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id ys)])
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
