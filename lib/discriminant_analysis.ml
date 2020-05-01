let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.discriminant_analysis"

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
module LinearClassifierMixin = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create () =
   Py.Module.get_function_with_keywords ns "LinearClassifierMixin"
     [||]
     []

let decision_function ~x self =
   Py.Module.get_function_with_keywords self "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
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
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LinearDiscriminantAnalysis = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?solver ?shrinkage ?priors ?n_components ?store_covariance ?tol () =
                     Py.Module.get_function_with_keywords ns "LinearDiscriminantAnalysis"
                       [||]
                       (Wrap_utils.keyword_args [("solver", Wrap_utils.Option.map solver Py.String.of_string); ("shrinkage", Wrap_utils.Option.map shrinkage (function
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
)); ("priors", Wrap_utils.Option.map priors Arr.to_pyobject); ("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("store_covariance", Wrap_utils.Option.map store_covariance Py.Bool.of_bool); ("tol", Wrap_utils.Option.map tol Py.Float.of_float)])

let decision_function ~x self =
   Py.Module.get_function_with_keywords self "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let fit ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

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
let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let predict_log_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_log_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let predict_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_proba"
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

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let coef_opt self =
  match Py.Object.get_attr_string self "coef_" with
  | None -> failwith "attribute coef_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let coef_ self = match coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let intercept_opt self =
  match Py.Object.get_attr_string self "intercept_" with
  | None -> failwith "attribute intercept_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let intercept_ self = match intercept_opt self with
  | None -> raise Not_found
  | Some x -> x

let covariance_opt self =
  match Py.Object.get_attr_string self "covariance_" with
  | None -> failwith "attribute covariance_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let covariance_ self = match covariance_opt self with
  | None -> raise Not_found
  | Some x -> x

let explained_variance_ratio_opt self =
  match Py.Object.get_attr_string self "explained_variance_ratio_" with
  | None -> failwith "attribute explained_variance_ratio_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let explained_variance_ratio_ self = match explained_variance_ratio_opt self with
  | None -> raise Not_found
  | Some x -> x

let means_opt self =
  match Py.Object.get_attr_string self "means_" with
  | None -> failwith "attribute means_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let means_ self = match means_opt self with
  | None -> raise Not_found
  | Some x -> x

let priors_opt self =
  match Py.Object.get_attr_string self "priors_" with
  | None -> failwith "attribute priors_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let priors_ self = match priors_opt self with
  | None -> raise Not_found
  | Some x -> x

let scalings_opt self =
  match Py.Object.get_attr_string self "scalings_" with
  | None -> failwith "attribute scalings_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let scalings_ self = match scalings_opt self with
  | None -> raise Not_found
  | Some x -> x

let xbar_opt self =
  match Py.Object.get_attr_string self "xbar_" with
  | None -> failwith "attribute xbar_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let xbar_ self = match xbar_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string self "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module QuadraticDiscriminantAnalysis = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?priors ?reg_param ?store_covariance ?tol () =
   Py.Module.get_function_with_keywords ns "QuadraticDiscriminantAnalysis"
     [||]
     (Wrap_utils.keyword_args [("priors", Wrap_utils.Option.map priors Arr.to_pyobject); ("reg_param", Wrap_utils.Option.map reg_param Py.Float.of_float); ("store_covariance", Wrap_utils.Option.map store_covariance Py.Bool.of_bool); ("tol", Wrap_utils.Option.map tol Py.Float.of_float)])

let decision_function ~x self =
   Py.Module.get_function_with_keywords self "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let fit ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

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
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let predict_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_proba"
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


let covariance_opt self =
  match Py.Object.get_attr_string self "covariance_" with
  | None -> failwith "attribute covariance_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let covariance_ self = match covariance_opt self with
  | None -> raise Not_found
  | Some x -> x

let means_opt self =
  match Py.Object.get_attr_string self "means_" with
  | None -> failwith "attribute means_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let means_ self = match means_opt self with
  | None -> raise Not_found
  | Some x -> x

let priors_opt self =
  match Py.Object.get_attr_string self "priors_" with
  | None -> failwith "attribute priors_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let priors_ self = match priors_opt self with
  | None -> raise Not_found
  | Some x -> x

let rotations_opt self =
  match Py.Object.get_attr_string self "rotations_" with
  | None -> failwith "attribute rotations_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.List.of_pyobject x)

let rotations_ self = match rotations_opt self with
  | None -> raise Not_found
  | Some x -> x

let scalings_opt self =
  match Py.Object.get_attr_string self "scalings_" with
  | None -> failwith "attribute scalings_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.List.of_pyobject x)

let scalings_ self = match scalings_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string self "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module StandardScaler = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?copy ?with_mean ?with_std () =
   Py.Module.get_function_with_keywords ns "StandardScaler"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("with_mean", Wrap_utils.Option.map with_mean Py.Bool.of_bool); ("with_std", Wrap_utils.Option.map with_std Py.Bool.of_bool)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])

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
let inverse_transform ?copy ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let partial_fit ?y ~x self =
   Py.Module.get_function_with_keywords self "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ?copy ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let scale_opt self =
  match Py.Object.get_attr_string self "scale_" with
  | None -> failwith "attribute scale_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let scale_ self = match scale_opt self with
  | None -> raise Not_found
  | Some x -> x

let mean_opt self =
  match Py.Object.get_attr_string self "mean_" with
  | None -> failwith "attribute mean_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let mean_ self = match mean_opt self with
  | None -> raise Not_found
  | Some x -> x

let var_opt self =
  match Py.Object.get_attr_string self "var_" with
  | None -> failwith "attribute var_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let var_ self = match var_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_samples_seen_opt self =
  match Py.Object.get_attr_string self "n_samples_seen_" with
  | None -> failwith "attribute n_samples_seen_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun x -> if Py.Int.check x then `I (Py.Int.to_int x) else if (fun x -> (Wrap_utils.isinstance Wrap_utils.ndarray x) || (Wrap_utils.isinstance Wrap_utils.csr_matrix x)) x then `Arr (Arr.of_pyobject x) else failwith "could not identify type from Python value") x)

let n_samples_seen_ self = match n_samples_seen_opt self with
  | None -> raise Not_found
  | Some x -> x
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

let check_classification_targets ~y () =
   Py.Module.get_function_with_keywords ns "check_classification_targets"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Arr.to_pyobject))])

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

let empirical_covariance ?assume_centered ~x () =
   Py.Module.get_function_with_keywords ns "empirical_covariance"
     [||]
     (Wrap_utils.keyword_args [("assume_centered", Wrap_utils.Option.map assume_centered Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])

let ledoit_wolf ?assume_centered ?block_size ~x () =
   Py.Module.get_function_with_keywords ns "ledoit_wolf"
     [||]
     (Wrap_utils.keyword_args [("assume_centered", Wrap_utils.Option.map assume_centered Py.Bool.of_bool); ("block_size", Wrap_utils.Option.map block_size Py.Int.of_int); ("X", Some(x |> Arr.to_pyobject))])
     |> (fun x -> ((Arr.of_pyobject (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))
                  let shrunk_covariance ?shrinkage ~emp_cov () =
                     Py.Module.get_function_with_keywords ns "shrunk_covariance"
                       [||]
                       (Wrap_utils.keyword_args [("shrinkage", Wrap_utils.Option.map shrinkage (function
| `F x -> Py.Float.of_float x
| `T0_shrinkage_1 x -> Wrap_utils.id x
)); ("emp_cov", Some(emp_cov |> Arr.to_pyobject))])
                       |> Arr.of_pyobject
let softmax ?copy ~x () =
   Py.Module.get_function_with_keywords ns "softmax"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x ))])
     |> Arr.of_pyobject
let unique_labels ys =
   Py.Module.get_function_with_keywords ns "unique_labels"
     (Wrap_utils.pos_arg Wrap_utils.id ys)
     []
     |> Arr.of_pyobject
