let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.linear_model"

let get_py name = Py.Module.get __wrap_namespace name
module ARDRegression = struct
type tag = [`ARDRegression]
type t = [`ARDRegression | `BaseEstimator | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let create ?n_iter ?tol ?alpha_1 ?alpha_2 ?lambda_1 ?lambda_2 ?compute_score ?threshold_lambda ?fit_intercept ?normalize ?copy_X ?verbose () =
   Py.Module.get_function_with_keywords __wrap_namespace "ARDRegression"
     [||]
     (Wrap_utils.keyword_args [("n_iter", Wrap_utils.Option.map n_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("alpha_1", Wrap_utils.Option.map alpha_1 Py.Float.of_float); ("alpha_2", Wrap_utils.Option.map alpha_2 Py.Float.of_float); ("lambda_1", Wrap_utils.Option.map lambda_1 Py.Float.of_float); ("lambda_2", Wrap_utils.Option.map lambda_2 Py.Float.of_float); ("compute_score", Wrap_utils.Option.map compute_score Py.Bool.of_bool); ("threshold_lambda", Wrap_utils.Option.map threshold_lambda Py.Float.of_float); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("copy_X", Wrap_utils.Option.map copy_X Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int)])
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
let predict ?return_std ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict"
     [||]
     (Wrap_utils.keyword_args [("return_std", Wrap_utils.Option.map return_std Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])
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

let alpha_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "alpha_" with
  | None -> failwith "attribute alpha_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let alpha_ self = match alpha_opt self with
  | None -> raise Not_found
  | Some x -> x

let lambda_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "lambda_" with
  | None -> failwith "attribute lambda_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let lambda_ self = match lambda_opt self with
  | None -> raise Not_found
  | Some x -> x

let sigma_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "sigma_" with
  | None -> failwith "attribute sigma_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let sigma_ self = match sigma_opt self with
  | None -> raise Not_found
  | Some x -> x

let scores_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "scores_" with
  | None -> failwith "attribute scores_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let scores_ self = match scores_opt self with
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
module BayesianRidge = struct
type tag = [`BayesianRidge]
type t = [`BaseEstimator | `BayesianRidge | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let create ?n_iter ?tol ?alpha_1 ?alpha_2 ?lambda_1 ?lambda_2 ?alpha_init ?lambda_init ?compute_score ?fit_intercept ?normalize ?copy_X ?verbose () =
   Py.Module.get_function_with_keywords __wrap_namespace "BayesianRidge"
     [||]
     (Wrap_utils.keyword_args [("n_iter", Wrap_utils.Option.map n_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("alpha_1", Wrap_utils.Option.map alpha_1 Py.Float.of_float); ("alpha_2", Wrap_utils.Option.map alpha_2 Py.Float.of_float); ("lambda_1", Wrap_utils.Option.map lambda_1 Py.Float.of_float); ("lambda_2", Wrap_utils.Option.map lambda_2 Py.Float.of_float); ("alpha_init", Wrap_utils.Option.map alpha_init Py.Float.of_float); ("lambda_init", Wrap_utils.Option.map lambda_init Py.Float.of_float); ("compute_score", Wrap_utils.Option.map compute_score Py.Bool.of_bool); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("copy_X", Wrap_utils.Option.map copy_X Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int)])
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
let predict ?return_std ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict"
     [||]
     (Wrap_utils.keyword_args [("return_std", Wrap_utils.Option.map return_std Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])
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

let alpha_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "alpha_" with
  | None -> failwith "attribute alpha_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let alpha_ self = match alpha_opt self with
  | None -> raise Not_found
  | Some x -> x

let lambda_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "lambda_" with
  | None -> failwith "attribute lambda_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let lambda_ self = match lambda_opt self with
  | None -> raise Not_found
  | Some x -> x

let sigma_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "sigma_" with
  | None -> failwith "attribute sigma_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let sigma_ self = match sigma_opt self with
  | None -> raise Not_found
  | Some x -> x

let scores_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "scores_" with
  | None -> failwith "attribute scores_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let scores_ self = match scores_opt self with
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
module ElasticNet = struct
type tag = [`ElasticNet]
type t = [`BaseEstimator | `ElasticNet | `MultiOutputMixin | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
                  let create ?alpha ?l1_ratio ?fit_intercept ?normalize ?precompute ?max_iter ?copy_X ?tol ?warm_start ?positive ?random_state ?selection () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ElasticNet"
                       [||]
                       (Wrap_utils.keyword_args [("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("l1_ratio", Wrap_utils.Option.map l1_ratio Py.Float.of_float); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("precompute", Wrap_utils.Option.map precompute (function
| `Arr x -> Np.Obj.to_pyobject x
| `Bool x -> Py.Bool.of_bool x
)); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("copy_X", Wrap_utils.Option.map copy_X Py.Bool.of_bool); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("positive", Wrap_utils.Option.map positive Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("selection", Wrap_utils.Option.map selection (function
| `Cyclic -> Py.String.of_string "cyclic"
| `Random -> Py.String.of_string "random"
))])
                       |> of_pyobject
let fit ?sample_weight ?check_input ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("check_input", Wrap_utils.Option.map check_input Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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

let sparse_coef_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "sparse_coef_" with
  | None -> failwith "attribute sparse_coef_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t)) x)

let sparse_coef_ self = match sparse_coef_opt self with
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
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ElasticNetCV = struct
type tag = [`ElasticNetCV]
type t = [`BaseEstimator | `ElasticNetCV | `MultiOutputMixin | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
                  let create ?l1_ratio ?eps ?n_alphas ?alphas ?fit_intercept ?normalize ?precompute ?max_iter ?tol ?cv ?copy_X ?verbose ?n_jobs ?positive ?random_state ?selection () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ElasticNetCV"
                       [||]
                       (Wrap_utils.keyword_args [("l1_ratio", Wrap_utils.Option.map l1_ratio (function
| `Fs x -> (fun ml -> Py.List.of_list_map Py.Float.of_float ml) x
| `F x -> Py.Float.of_float x
)); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("n_alphas", Wrap_utils.Option.map n_alphas Py.Int.of_int); ("alphas", Wrap_utils.Option.map alphas Np.Obj.to_pyobject); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("precompute", Wrap_utils.Option.map precompute (function
| `Arr x -> Np.Obj.to_pyobject x
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("cv", Wrap_utils.Option.map cv (function
| `BaseCrossValidator x -> Np.Obj.to_pyobject x
| `Arr x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("copy_X", Wrap_utils.Option.map copy_X Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("positive", Wrap_utils.Option.map positive Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("selection", Wrap_utils.Option.map selection (function
| `Cyclic -> Py.String.of_string "cyclic"
| `Random -> Py.String.of_string "random"
))])
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

let alpha_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "alpha_" with
  | None -> failwith "attribute alpha_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let alpha_ self = match alpha_opt self with
  | None -> raise Not_found
  | Some x -> x

let l1_ratio_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "l1_ratio_" with
  | None -> failwith "attribute l1_ratio_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let l1_ratio_ self = match l1_ratio_opt self with
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

let mse_path_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "mse_path_" with
  | None -> failwith "attribute mse_path_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let mse_path_ self = match mse_path_opt self with
  | None -> raise Not_found
  | Some x -> x

let alphas_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "alphas_" with
  | None -> failwith "attribute alphas_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let alphas_ self = match alphas_opt self with
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
module GammaRegressor = struct
type tag = [`GammaRegressor]
type t = [`BaseEstimator | `GammaRegressor | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let create ?alpha ?fit_intercept ?max_iter ?tol ?warm_start ?verbose () =
   Py.Module.get_function_with_keywords __wrap_namespace "GammaRegressor"
     [||]
     (Wrap_utils.keyword_args [("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int)])
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
module Hinge = struct
type tag = [`Hinge]
type t = [`Hinge | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Huber = struct
type tag = [`Huber]
type t = [`Huber | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module HuberRegressor = struct
type tag = [`HuberRegressor]
type t = [`BaseEstimator | `HuberRegressor | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let create ?epsilon ?max_iter ?alpha ?warm_start ?fit_intercept ?tol () =
   Py.Module.get_function_with_keywords __wrap_namespace "HuberRegressor"
     [||]
     (Wrap_utils.keyword_args [("epsilon", Wrap_utils.Option.map epsilon Py.Float.of_float); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("tol", Wrap_utils.Option.map tol Py.Float.of_float)])
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

let scale_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "scale_" with
  | None -> failwith "attribute scale_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let scale_ self = match scale_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x

let outliers_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "outliers_" with
  | None -> failwith "attribute outliers_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let outliers_ self = match outliers_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Lars = struct
type tag = [`Lars]
type t = [`BaseEstimator | `Lars | `MultiOutputMixin | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
                  let create ?fit_intercept ?verbose ?normalize ?precompute ?n_nonzero_coefs ?eps ?copy_X ?fit_path ?jitter ?random_state () =
                     Py.Module.get_function_with_keywords __wrap_namespace "Lars"
                       [||]
                       (Wrap_utils.keyword_args [("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("precompute", Wrap_utils.Option.map precompute (function
| `Arr x -> Np.Obj.to_pyobject x
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("n_nonzero_coefs", Wrap_utils.Option.map n_nonzero_coefs Py.Int.of_int); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("copy_X", Wrap_utils.Option.map copy_X Py.Bool.of_bool); ("fit_path", Wrap_utils.Option.map fit_path Py.Bool.of_bool); ("jitter", Wrap_utils.Option.map jitter Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
                       |> of_pyobject
let fit ?xy ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("Xy", Wrap_utils.Option.map xy Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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

let alphas_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "alphas_" with
  | None -> failwith "attribute alphas_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let alphas_ self = match alphas_opt self with
  | None -> raise Not_found
  | Some x -> x

let active_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "active_" with
  | None -> failwith "attribute active_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let active_ self = match active_opt self with
  | None -> raise Not_found
  | Some x -> x

let coef_path_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "coef_path_" with
  | None -> failwith "attribute coef_path_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let coef_path_ self = match coef_path_opt self with
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

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LarsCV = struct
type tag = [`LarsCV]
type t = [`BaseEstimator | `LarsCV | `MultiOutputMixin | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
                  let create ?fit_intercept ?verbose ?max_iter ?normalize ?precompute ?cv ?max_n_alphas ?n_jobs ?eps ?copy_X () =
                     Py.Module.get_function_with_keywords __wrap_namespace "LarsCV"
                       [||]
                       (Wrap_utils.keyword_args [("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("precompute", Wrap_utils.Option.map precompute (function
| `Arr x -> Np.Obj.to_pyobject x
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("cv", Wrap_utils.Option.map cv (function
| `BaseCrossValidator x -> Np.Obj.to_pyobject x
| `Arr x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("max_n_alphas", Wrap_utils.Option.map max_n_alphas Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("copy_X", Wrap_utils.Option.map copy_X Py.Bool.of_bool)])
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

let coef_path_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "coef_path_" with
  | None -> failwith "attribute coef_path_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let coef_path_ self = match coef_path_opt self with
  | None -> raise Not_found
  | Some x -> x

let alpha_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "alpha_" with
  | None -> failwith "attribute alpha_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let alpha_ self = match alpha_opt self with
  | None -> raise Not_found
  | Some x -> x

let alphas_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "alphas_" with
  | None -> failwith "attribute alphas_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let alphas_ self = match alphas_opt self with
  | None -> raise Not_found
  | Some x -> x

let cv_alphas_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "cv_alphas_" with
  | None -> failwith "attribute cv_alphas_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let cv_alphas_ self = match cv_alphas_opt self with
  | None -> raise Not_found
  | Some x -> x

let mse_path_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "mse_path_" with
  | None -> failwith "attribute mse_path_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let mse_path_ self = match mse_path_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Lasso = struct
type tag = [`Lasso]
type t = [`BaseEstimator | `Lasso | `MultiOutputMixin | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
                  let create ?alpha ?fit_intercept ?normalize ?precompute ?copy_X ?max_iter ?tol ?warm_start ?positive ?random_state ?selection () =
                     Py.Module.get_function_with_keywords __wrap_namespace "Lasso"
                       [||]
                       (Wrap_utils.keyword_args [("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("precompute", Wrap_utils.Option.map precompute (function
| `Arr x -> Np.Obj.to_pyobject x
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("copy_X", Wrap_utils.Option.map copy_X Py.Bool.of_bool); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("positive", Wrap_utils.Option.map positive Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("selection", Wrap_utils.Option.map selection (function
| `Cyclic -> Py.String.of_string "cyclic"
| `Random -> Py.String.of_string "random"
))])
                       |> of_pyobject
let fit ?sample_weight ?check_input ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("check_input", Wrap_utils.Option.map check_input Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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

let sparse_coef_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "sparse_coef_" with
  | None -> failwith "attribute sparse_coef_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t)) x)

let sparse_coef_ self = match sparse_coef_opt self with
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
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LassoCV = struct
type tag = [`LassoCV]
type t = [`BaseEstimator | `LassoCV | `MultiOutputMixin | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
                  let create ?eps ?n_alphas ?alphas ?fit_intercept ?normalize ?precompute ?max_iter ?tol ?copy_X ?cv ?verbose ?n_jobs ?positive ?random_state ?selection () =
                     Py.Module.get_function_with_keywords __wrap_namespace "LassoCV"
                       [||]
                       (Wrap_utils.keyword_args [("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("n_alphas", Wrap_utils.Option.map n_alphas Py.Int.of_int); ("alphas", Wrap_utils.Option.map alphas Np.Obj.to_pyobject); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("precompute", Wrap_utils.Option.map precompute (function
| `Arr x -> Np.Obj.to_pyobject x
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("copy_X", Wrap_utils.Option.map copy_X Py.Bool.of_bool); ("cv", Wrap_utils.Option.map cv (function
| `BaseCrossValidator x -> Np.Obj.to_pyobject x
| `Arr x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("positive", Wrap_utils.Option.map positive Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("selection", Wrap_utils.Option.map selection (function
| `Cyclic -> Py.String.of_string "cyclic"
| `Random -> Py.String.of_string "random"
))])
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

let alpha_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "alpha_" with
  | None -> failwith "attribute alpha_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let alpha_ self = match alpha_opt self with
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

let mse_path_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "mse_path_" with
  | None -> failwith "attribute mse_path_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let mse_path_ self = match mse_path_opt self with
  | None -> raise Not_found
  | Some x -> x

let alphas_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "alphas_" with
  | None -> failwith "attribute alphas_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let alphas_ self = match alphas_opt self with
  | None -> raise Not_found
  | Some x -> x

let dual_gap_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dual_gap_" with
  | None -> failwith "attribute dual_gap_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let dual_gap_ self = match dual_gap_opt self with
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
module LassoLars = struct
type tag = [`LassoLars]
type t = [`BaseEstimator | `LassoLars | `MultiOutputMixin | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
                  let create ?alpha ?fit_intercept ?verbose ?normalize ?precompute ?max_iter ?eps ?copy_X ?fit_path ?positive ?jitter ?random_state () =
                     Py.Module.get_function_with_keywords __wrap_namespace "LassoLars"
                       [||]
                       (Wrap_utils.keyword_args [("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("precompute", Wrap_utils.Option.map precompute (function
| `Arr x -> Np.Obj.to_pyobject x
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("copy_X", Wrap_utils.Option.map copy_X Py.Bool.of_bool); ("fit_path", Wrap_utils.Option.map fit_path Py.Bool.of_bool); ("positive", Wrap_utils.Option.map positive Py.Bool.of_bool); ("jitter", Wrap_utils.Option.map jitter Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
                       |> of_pyobject
let fit ?xy ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("Xy", Wrap_utils.Option.map xy Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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

let alphas_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "alphas_" with
  | None -> failwith "attribute alphas_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let alphas_ self = match alphas_opt self with
  | None -> raise Not_found
  | Some x -> x

let active_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "active_" with
  | None -> failwith "attribute active_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let active_ self = match active_opt self with
  | None -> raise Not_found
  | Some x -> x

let coef_path_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "coef_path_" with
  | None -> failwith "attribute coef_path_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let coef_path_ self = match coef_path_opt self with
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

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LassoLarsCV = struct
type tag = [`LassoLarsCV]
type t = [`BaseEstimator | `LassoLarsCV | `MultiOutputMixin | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
                  let create ?fit_intercept ?verbose ?max_iter ?normalize ?precompute ?cv ?max_n_alphas ?n_jobs ?eps ?copy_X ?positive () =
                     Py.Module.get_function_with_keywords __wrap_namespace "LassoLarsCV"
                       [||]
                       (Wrap_utils.keyword_args [("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("precompute", Wrap_utils.Option.map precompute (function
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("cv", Wrap_utils.Option.map cv (function
| `BaseCrossValidator x -> Np.Obj.to_pyobject x
| `Arr x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("max_n_alphas", Wrap_utils.Option.map max_n_alphas Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("copy_X", Wrap_utils.Option.map copy_X Py.Bool.of_bool); ("positive", Wrap_utils.Option.map positive Py.Bool.of_bool)])
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

let coef_path_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "coef_path_" with
  | None -> failwith "attribute coef_path_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let coef_path_ self = match coef_path_opt self with
  | None -> raise Not_found
  | Some x -> x

let alpha_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "alpha_" with
  | None -> failwith "attribute alpha_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let alpha_ self = match alpha_opt self with
  | None -> raise Not_found
  | Some x -> x

let alphas_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "alphas_" with
  | None -> failwith "attribute alphas_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let alphas_ self = match alphas_opt self with
  | None -> raise Not_found
  | Some x -> x

let cv_alphas_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "cv_alphas_" with
  | None -> failwith "attribute cv_alphas_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let cv_alphas_ self = match cv_alphas_opt self with
  | None -> raise Not_found
  | Some x -> x

let mse_path_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "mse_path_" with
  | None -> failwith "attribute mse_path_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let mse_path_ self = match mse_path_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LassoLarsIC = struct
type tag = [`LassoLarsIC]
type t = [`BaseEstimator | `LassoLarsIC | `MultiOutputMixin | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
                  let create ?criterion ?fit_intercept ?verbose ?normalize ?precompute ?max_iter ?eps ?copy_X ?positive () =
                     Py.Module.get_function_with_keywords __wrap_namespace "LassoLarsIC"
                       [||]
                       (Wrap_utils.keyword_args [("criterion", Wrap_utils.Option.map criterion (function
| `Bic -> Py.String.of_string "bic"
| `Aic -> Py.String.of_string "aic"
)); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("precompute", Wrap_utils.Option.map precompute (function
| `Arr x -> Np.Obj.to_pyobject x
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("copy_X", Wrap_utils.Option.map copy_X Py.Bool.of_bool); ("positive", Wrap_utils.Option.map positive Py.Bool.of_bool)])
                       |> of_pyobject
let fit ?copy_X ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("copy_X", Wrap_utils.Option.map copy_X Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
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

let alpha_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "alpha_" with
  | None -> failwith "attribute alpha_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let alpha_ self = match alpha_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x

let criterion_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "criterion_" with
  | None -> failwith "attribute criterion_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let criterion_ self = match criterion_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LinearRegression = struct
type tag = [`LinearRegression]
type t = [`BaseEstimator | `LinearRegression | `MultiOutputMixin | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
let create ?fit_intercept ?normalize ?copy_X ?n_jobs () =
   Py.Module.get_function_with_keywords __wrap_namespace "LinearRegression"
     [||]
     (Wrap_utils.keyword_args [("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("copy_X", Wrap_utils.Option.map copy_X Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int)])
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

let rank_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "rank_" with
  | None -> failwith "attribute rank_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let rank_ self = match rank_opt self with
  | None -> raise Not_found
  | Some x -> x

let singular_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "singular_" with
  | None -> failwith "attribute singular_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let singular_ self = match singular_opt self with
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
module Log = struct
type tag = [`Log]
type t = [`Log | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LogisticRegression = struct
type tag = [`LogisticRegression]
type t = [`BaseEstimator | `ClassifierMixin | `LinearClassifierMixin | `LogisticRegression | `Object | `SparseCoefMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_linear_classifier x = (x :> [`LinearClassifierMixin] Obj.t)
let as_sparse_coef x = (x :> [`SparseCoefMixin] Obj.t)
                  let create ?penalty ?dual ?tol ?c ?fit_intercept ?intercept_scaling ?class_weight ?random_state ?solver ?max_iter ?multi_class ?verbose ?warm_start ?n_jobs ?l1_ratio () =
                     Py.Module.get_function_with_keywords __wrap_namespace "LogisticRegression"
                       [||]
                       (Wrap_utils.keyword_args [("penalty", Wrap_utils.Option.map penalty (function
| `L1 -> Py.String.of_string "l1"
| `L2 -> Py.String.of_string "l2"
| `Elasticnet -> Py.String.of_string "elasticnet"
| `None -> Py.String.of_string "none"
)); ("dual", Wrap_utils.Option.map dual Py.Bool.of_bool); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("C", Wrap_utils.Option.map c Py.Float.of_float); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("intercept_scaling", Wrap_utils.Option.map intercept_scaling Py.Float.of_float); ("class_weight", Wrap_utils.Option.map class_weight (function
| `Balanced -> Py.String.of_string "balanced"
| `DictIntToFloat x -> (Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float) x
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("solver", Wrap_utils.Option.map solver (function
| `Newton_cg -> Py.String.of_string "newton-cg"
| `Lbfgs -> Py.String.of_string "lbfgs"
| `Liblinear -> Py.String.of_string "liblinear"
| `Sag -> Py.String.of_string "sag"
| `Saga -> Py.String.of_string "saga"
)); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("multi_class", Wrap_utils.Option.map multi_class (function
| `Auto -> Py.String.of_string "auto"
| `Ovr -> Py.String.of_string "ovr"
| `Multinomial -> Py.String.of_string "multinomial"
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("l1_ratio", Wrap_utils.Option.map l1_ratio Py.Float.of_float)])
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
let sparsify self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sparsify"
     [||]
     []
     |> of_pyobject

let classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let classes_ self = match classes_opt self with
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
module LogisticRegressionCV = struct
type tag = [`LogisticRegressionCV]
type t = [`BaseEstimator | `ClassifierMixin | `LinearClassifierMixin | `LogisticRegressionCV | `Object | `SparseCoefMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_linear_classifier x = (x :> [`LinearClassifierMixin] Obj.t)
let as_sparse_coef x = (x :> [`SparseCoefMixin] Obj.t)
                  let create ?cs ?fit_intercept ?cv ?dual ?penalty ?scoring ?solver ?tol ?max_iter ?class_weight ?n_jobs ?verbose ?refit ?intercept_scaling ?multi_class ?random_state ?l1_ratios () =
                     Py.Module.get_function_with_keywords __wrap_namespace "LogisticRegressionCV"
                       [||]
                       (Wrap_utils.keyword_args [("Cs", Wrap_utils.Option.map cs (function
| `Fs x -> (fun ml -> Py.List.of_list_map Py.Float.of_float ml) x
| `I x -> Py.Int.of_int x
)); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("cv", Wrap_utils.Option.map cv (function
| `BaseCrossValidator x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("dual", Wrap_utils.Option.map dual Py.Bool.of_bool); ("penalty", Wrap_utils.Option.map penalty (function
| `L1 -> Py.String.of_string "l1"
| `L2 -> Py.String.of_string "l2"
| `Elasticnet -> Py.String.of_string "elasticnet"
)); ("scoring", Wrap_utils.Option.map scoring (function
| `Neg_mean_absolute_error -> Py.String.of_string "neg_mean_absolute_error"
| `Completeness_score -> Py.String.of_string "completeness_score"
| `Roc_auc_ovr -> Py.String.of_string "roc_auc_ovr"
| `Neg_mean_squared_log_error -> Py.String.of_string "neg_mean_squared_log_error"
| `Neg_mean_gamma_deviance -> Py.String.of_string "neg_mean_gamma_deviance"
| `Precision_macro -> Py.String.of_string "precision_macro"
| `R2 -> Py.String.of_string "r2"
| `Precision_micro -> Py.String.of_string "precision_micro"
| `F1_weighted -> Py.String.of_string "f1_weighted"
| `Balanced_accuracy -> Py.String.of_string "balanced_accuracy"
| `Neg_mean_squared_error -> Py.String.of_string "neg_mean_squared_error"
| `F1_samples -> Py.String.of_string "f1_samples"
| `Jaccard_micro -> Py.String.of_string "jaccard_micro"
| `Normalized_mutual_info_score -> Py.String.of_string "normalized_mutual_info_score"
| `F1_micro -> Py.String.of_string "f1_micro"
| `Roc_auc -> Py.String.of_string "roc_auc"
| `Mutual_info_score -> Py.String.of_string "mutual_info_score"
| `Adjusted_rand_score -> Py.String.of_string "adjusted_rand_score"
| `Average_precision -> Py.String.of_string "average_precision"
| `Jaccard -> Py.String.of_string "jaccard"
| `Homogeneity_score -> Py.String.of_string "homogeneity_score"
| `Accuracy -> Py.String.of_string "accuracy"
| `Jaccard_macro -> Py.String.of_string "jaccard_macro"
| `Jaccard_weighted -> Py.String.of_string "jaccard_weighted"
| `Recall_micro -> Py.String.of_string "recall_micro"
| `Explained_variance -> Py.String.of_string "explained_variance"
| `Precision -> Py.String.of_string "precision"
| `Callable x -> Wrap_utils.id x
| `V_measure_score -> Py.String.of_string "v_measure_score"
| `F1 -> Py.String.of_string "f1"
| `Roc_auc_ovo -> Py.String.of_string "roc_auc_ovo"
| `Neg_mean_poisson_deviance -> Py.String.of_string "neg_mean_poisson_deviance"
| `Recall_samples -> Py.String.of_string "recall_samples"
| `Adjusted_mutual_info_score -> Py.String.of_string "adjusted_mutual_info_score"
| `Neg_brier_score -> Py.String.of_string "neg_brier_score"
| `Roc_auc_ovo_weighted -> Py.String.of_string "roc_auc_ovo_weighted"
| `Recall -> Py.String.of_string "recall"
| `Fowlkes_mallows_score -> Py.String.of_string "fowlkes_mallows_score"
| `Neg_log_loss -> Py.String.of_string "neg_log_loss"
| `Neg_root_mean_squared_error -> Py.String.of_string "neg_root_mean_squared_error"
| `Precision_samples -> Py.String.of_string "precision_samples"
| `F1_macro -> Py.String.of_string "f1_macro"
| `Roc_auc_ovr_weighted -> Py.String.of_string "roc_auc_ovr_weighted"
| `Recall_weighted -> Py.String.of_string "recall_weighted"
| `Neg_median_absolute_error -> Py.String.of_string "neg_median_absolute_error"
| `Jaccard_samples -> Py.String.of_string "jaccard_samples"
| `Precision_weighted -> Py.String.of_string "precision_weighted"
| `Max_error -> Py.String.of_string "max_error"
| `Recall_macro -> Py.String.of_string "recall_macro"
)); ("solver", Wrap_utils.Option.map solver (function
| `Newton_cg -> Py.String.of_string "newton-cg"
| `Lbfgs -> Py.String.of_string "lbfgs"
| `Liblinear -> Py.String.of_string "liblinear"
| `Sag -> Py.String.of_string "sag"
| `Saga -> Py.String.of_string "saga"
)); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("class_weight", Wrap_utils.Option.map class_weight (function
| `Balanced -> Py.String.of_string "balanced"
| `DictIntToFloat x -> (Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float) x
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("refit", Wrap_utils.Option.map refit Py.Bool.of_bool); ("intercept_scaling", Wrap_utils.Option.map intercept_scaling Py.Float.of_float); ("multi_class", Wrap_utils.Option.map multi_class (function
| `Ovr -> Py.String.of_string "ovr"
| `T_auto x -> Wrap_utils.id x
| `Multinomial -> Py.String.of_string "multinomial"
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("l1_ratios", Wrap_utils.Option.map l1_ratios (fun ml -> Py.List.of_list_map Py.Float.of_float ml))])
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
let sparsify self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sparsify"
     [||]
     []
     |> of_pyobject

let classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let classes_ self = match classes_opt self with
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

let cs_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "Cs_" with
  | None -> failwith "attribute Cs_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let cs_ self = match cs_opt self with
  | None -> raise Not_found
  | Some x -> x

let l1_ratios_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "l1_ratios_" with
  | None -> failwith "attribute l1_ratios_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let l1_ratios_ self = match l1_ratios_opt self with
  | None -> raise Not_found
  | Some x -> x

let coefs_paths_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "coefs_paths_" with
  | None -> failwith "attribute coefs_paths_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let coefs_paths_ self = match coefs_paths_opt self with
  | None -> raise Not_found
  | Some x -> x

let scores_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "scores_" with
  | None -> failwith "attribute scores_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let scores_ self = match scores_opt self with
  | None -> raise Not_found
  | Some x -> x

let c_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "C_" with
  | None -> failwith "attribute C_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let c_ self = match c_opt self with
  | None -> raise Not_found
  | Some x -> x

let l1_ratio_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "l1_ratio_" with
  | None -> failwith "attribute l1_ratio_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let l1_ratio_ self = match l1_ratio_opt self with
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
module ModifiedHuber = struct
type tag = [`ModifiedHuber]
type t = [`ModifiedHuber | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MultiTaskElasticNet = struct
type tag = [`MultiTaskElasticNet]
type t = [`BaseEstimator | `MultiOutputMixin | `MultiTaskElasticNet | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
                  let create ?alpha ?l1_ratio ?fit_intercept ?normalize ?copy_X ?max_iter ?tol ?warm_start ?random_state ?selection () =
                     Py.Module.get_function_with_keywords __wrap_namespace "MultiTaskElasticNet"
                       [||]
                       (Wrap_utils.keyword_args [("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("l1_ratio", Wrap_utils.Option.map l1_ratio Py.Float.of_float); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("copy_X", Wrap_utils.Option.map copy_X Py.Bool.of_bool); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("selection", Wrap_utils.Option.map selection (function
| `Cyclic -> Py.String.of_string "cyclic"
| `Random -> Py.String.of_string "random"
))])
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

let intercept_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "intercept_" with
  | None -> failwith "attribute intercept_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let intercept_ self = match intercept_opt self with
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
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MultiTaskElasticNetCV = struct
type tag = [`MultiTaskElasticNetCV]
type t = [`BaseEstimator | `MultiOutputMixin | `MultiTaskElasticNetCV | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
                  let create ?l1_ratio ?eps ?n_alphas ?alphas ?fit_intercept ?normalize ?max_iter ?tol ?cv ?copy_X ?verbose ?n_jobs ?random_state ?selection () =
                     Py.Module.get_function_with_keywords __wrap_namespace "MultiTaskElasticNetCV"
                       [||]
                       (Wrap_utils.keyword_args [("l1_ratio", Wrap_utils.Option.map l1_ratio (function
| `Fs x -> (fun ml -> Py.List.of_list_map Py.Float.of_float ml) x
| `F x -> Py.Float.of_float x
)); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("n_alphas", Wrap_utils.Option.map n_alphas Py.Int.of_int); ("alphas", Wrap_utils.Option.map alphas Np.Obj.to_pyobject); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("cv", Wrap_utils.Option.map cv (function
| `BaseCrossValidator x -> Np.Obj.to_pyobject x
| `Arr x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("copy_X", Wrap_utils.Option.map copy_X Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("selection", Wrap_utils.Option.map selection (function
| `Cyclic -> Py.String.of_string "cyclic"
| `Random -> Py.String.of_string "random"
))])
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

let intercept_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "intercept_" with
  | None -> failwith "attribute intercept_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let intercept_ self = match intercept_opt self with
  | None -> raise Not_found
  | Some x -> x

let coef_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "coef_" with
  | None -> failwith "attribute coef_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let coef_ self = match coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let alpha_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "alpha_" with
  | None -> failwith "attribute alpha_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let alpha_ self = match alpha_opt self with
  | None -> raise Not_found
  | Some x -> x

let mse_path_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "mse_path_" with
  | None -> failwith "attribute mse_path_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let mse_path_ self = match mse_path_opt self with
  | None -> raise Not_found
  | Some x -> x

let alphas_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "alphas_" with
  | None -> failwith "attribute alphas_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let alphas_ self = match alphas_opt self with
  | None -> raise Not_found
  | Some x -> x

let l1_ratio_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "l1_ratio_" with
  | None -> failwith "attribute l1_ratio_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let l1_ratio_ self = match l1_ratio_opt self with
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
module MultiTaskLasso = struct
type tag = [`MultiTaskLasso]
type t = [`BaseEstimator | `MultiOutputMixin | `MultiTaskLasso | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
                  let create ?alpha ?fit_intercept ?normalize ?copy_X ?max_iter ?tol ?warm_start ?random_state ?selection () =
                     Py.Module.get_function_with_keywords __wrap_namespace "MultiTaskLasso"
                       [||]
                       (Wrap_utils.keyword_args [("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("copy_X", Wrap_utils.Option.map copy_X Py.Bool.of_bool); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("selection", Wrap_utils.Option.map selection (function
| `Cyclic -> Py.String.of_string "cyclic"
| `Random -> Py.String.of_string "random"
))])
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
module MultiTaskLassoCV = struct
type tag = [`MultiTaskLassoCV]
type t = [`BaseEstimator | `MultiOutputMixin | `MultiTaskLassoCV | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
                  let create ?eps ?n_alphas ?alphas ?fit_intercept ?normalize ?max_iter ?tol ?copy_X ?cv ?verbose ?n_jobs ?random_state ?selection () =
                     Py.Module.get_function_with_keywords __wrap_namespace "MultiTaskLassoCV"
                       [||]
                       (Wrap_utils.keyword_args [("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("n_alphas", Wrap_utils.Option.map n_alphas Py.Int.of_int); ("alphas", Wrap_utils.Option.map alphas Np.Obj.to_pyobject); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("copy_X", Wrap_utils.Option.map copy_X Py.Bool.of_bool); ("cv", Wrap_utils.Option.map cv (function
| `BaseCrossValidator x -> Np.Obj.to_pyobject x
| `Arr x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("selection", Wrap_utils.Option.map selection (function
| `Cyclic -> Py.String.of_string "cyclic"
| `Random -> Py.String.of_string "random"
))])
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

let intercept_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "intercept_" with
  | None -> failwith "attribute intercept_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let intercept_ self = match intercept_opt self with
  | None -> raise Not_found
  | Some x -> x

let coef_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "coef_" with
  | None -> failwith "attribute coef_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let coef_ self = match coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let alpha_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "alpha_" with
  | None -> failwith "attribute alpha_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let alpha_ self = match alpha_opt self with
  | None -> raise Not_found
  | Some x -> x

let mse_path_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "mse_path_" with
  | None -> failwith "attribute mse_path_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let mse_path_ self = match mse_path_opt self with
  | None -> raise Not_found
  | Some x -> x

let alphas_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "alphas_" with
  | None -> failwith "attribute alphas_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let alphas_ self = match alphas_opt self with
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
module OrthogonalMatchingPursuit = struct
type tag = [`OrthogonalMatchingPursuit]
type t = [`BaseEstimator | `MultiOutputMixin | `Object | `OrthogonalMatchingPursuit | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
                  let create ?n_nonzero_coefs ?tol ?fit_intercept ?normalize ?precompute () =
                     Py.Module.get_function_with_keywords __wrap_namespace "OrthogonalMatchingPursuit"
                       [||]
                       (Wrap_utils.keyword_args [("n_nonzero_coefs", Wrap_utils.Option.map n_nonzero_coefs Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("precompute", Wrap_utils.Option.map precompute (function
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
))])
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
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module OrthogonalMatchingPursuitCV = struct
type tag = [`OrthogonalMatchingPursuitCV]
type t = [`BaseEstimator | `Object | `OrthogonalMatchingPursuitCV | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
                  let create ?copy ?fit_intercept ?normalize ?max_iter ?cv ?n_jobs ?verbose () =
                     Py.Module.get_function_with_keywords __wrap_namespace "OrthogonalMatchingPursuitCV"
                       [||]
                       (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("cv", Wrap_utils.Option.map cv (function
| `BaseCrossValidator x -> Np.Obj.to_pyobject x
| `Arr x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int)])
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

let intercept_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "intercept_" with
  | None -> failwith "attribute intercept_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let intercept_ self = match intercept_opt self with
  | None -> raise Not_found
  | Some x -> x

let coef_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "coef_" with
  | None -> failwith "attribute coef_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let coef_ self = match coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_nonzero_coefs_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_nonzero_coefs_" with
  | None -> failwith "attribute n_nonzero_coefs_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_nonzero_coefs_ self = match n_nonzero_coefs_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module PassiveAggressiveClassifier = struct
type tag = [`PassiveAggressiveClassifier]
type t = [`BaseEstimator | `BaseSGD | `BaseSGDClassifier | `ClassifierMixin | `LinearClassifierMixin | `Object | `PassiveAggressiveClassifier | `SparseCoefMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_sgd x = (x :> [`BaseSGD] Obj.t)
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_sparse_coef x = (x :> [`SparseCoefMixin] Obj.t)
let as_linear_classifier x = (x :> [`LinearClassifierMixin] Obj.t)
let as_sgd_classifier x = (x :> [`BaseSGDClassifier] Obj.t)
                  let create ?c ?fit_intercept ?max_iter ?tol ?early_stopping ?validation_fraction ?n_iter_no_change ?shuffle ?verbose ?loss ?n_jobs ?random_state ?warm_start ?class_weight ?average () =
                     Py.Module.get_function_with_keywords __wrap_namespace "PassiveAggressiveClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("C", Wrap_utils.Option.map c Py.Float.of_float); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol (function
| `F x -> Py.Float.of_float x
| `None -> Py.none
)); ("early_stopping", Wrap_utils.Option.map early_stopping Py.Bool.of_bool); ("validation_fraction", Wrap_utils.Option.map validation_fraction Py.Float.of_float); ("n_iter_no_change", Wrap_utils.Option.map n_iter_no_change Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("loss", Wrap_utils.Option.map loss Py.String.of_string); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("class_weight", class_weight); ("average", Wrap_utils.Option.map average (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
))])
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
let fit ?coef_init ?intercept_init ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("coef_init", Wrap_utils.Option.map coef_init Np.Obj.to_pyobject); ("intercept_init", Wrap_utils.Option.map intercept_init Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let partial_fit ?classes ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("classes", Wrap_utils.Option.map classes Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
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
let set_params ?kwargs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match kwargs with None -> [] | Some x -> x)
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

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let t_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t_" with
  | None -> failwith "attribute t_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let t_ self = match t_opt self with
  | None -> raise Not_found
  | Some x -> x

let loss_function_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "loss_function_" with
  | None -> failwith "attribute loss_function_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let loss_function_ self = match loss_function_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module PassiveAggressiveRegressor = struct
type tag = [`PassiveAggressiveRegressor]
type t = [`BaseEstimator | `BaseSGD | `BaseSGDRegressor | `Object | `PassiveAggressiveRegressor | `RegressorMixin | `SparseCoefMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_sgd x = (x :> [`BaseSGD] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_sparse_coef x = (x :> [`SparseCoefMixin] Obj.t)
let as_sgd_regressor x = (x :> [`BaseSGDRegressor] Obj.t)
                  let create ?c ?fit_intercept ?max_iter ?tol ?early_stopping ?validation_fraction ?n_iter_no_change ?shuffle ?verbose ?loss ?epsilon ?random_state ?warm_start ?average () =
                     Py.Module.get_function_with_keywords __wrap_namespace "PassiveAggressiveRegressor"
                       [||]
                       (Wrap_utils.keyword_args [("C", Wrap_utils.Option.map c Py.Float.of_float); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol (function
| `F x -> Py.Float.of_float x
| `None -> Py.none
)); ("early_stopping", Wrap_utils.Option.map early_stopping Py.Bool.of_bool); ("validation_fraction", Wrap_utils.Option.map validation_fraction Py.Float.of_float); ("n_iter_no_change", Wrap_utils.Option.map n_iter_no_change Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("loss", Wrap_utils.Option.map loss Py.String.of_string); ("epsilon", Wrap_utils.Option.map epsilon Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("average", Wrap_utils.Option.map average (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
))])
                       |> of_pyobject
let densify self =
   Py.Module.get_function_with_keywords (to_pyobject self) "densify"
     [||]
     []
     |> of_pyobject
let fit ?coef_init ?intercept_init ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("coef_init", Wrap_utils.Option.map coef_init Np.Obj.to_pyobject); ("intercept_init", Wrap_utils.Option.map intercept_init Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let partial_fit ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
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
let set_params ?kwargs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match kwargs with None -> [] | Some x -> x)
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

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x

let t_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t_" with
  | None -> failwith "attribute t_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let t_ self = match t_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Perceptron = struct
type tag = [`Perceptron]
type t = [`BaseEstimator | `BaseSGD | `BaseSGDClassifier | `ClassifierMixin | `LinearClassifierMixin | `Object | `Perceptron | `SparseCoefMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_sgd x = (x :> [`BaseSGD] Obj.t)
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_sparse_coef x = (x :> [`SparseCoefMixin] Obj.t)
let as_linear_classifier x = (x :> [`LinearClassifierMixin] Obj.t)
let as_sgd_classifier x = (x :> [`BaseSGDClassifier] Obj.t)
                  let create ?penalty ?alpha ?fit_intercept ?max_iter ?tol ?shuffle ?verbose ?eta0 ?n_jobs ?random_state ?early_stopping ?validation_fraction ?n_iter_no_change ?class_weight ?warm_start () =
                     Py.Module.get_function_with_keywords __wrap_namespace "Perceptron"
                       [||]
                       (Wrap_utils.keyword_args [("penalty", Wrap_utils.Option.map penalty (function
| `L1 -> Py.String.of_string "l1"
| `L2 -> Py.String.of_string "l2"
| `Elasticnet -> Py.String.of_string "elasticnet"
)); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("eta0", Wrap_utils.Option.map eta0 Py.Float.of_float); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("early_stopping", Wrap_utils.Option.map early_stopping Py.Bool.of_bool); ("validation_fraction", Wrap_utils.Option.map validation_fraction Py.Float.of_float); ("n_iter_no_change", Wrap_utils.Option.map n_iter_no_change Py.Int.of_int); ("class_weight", Wrap_utils.Option.map class_weight (function
| `T_class_label_weight_ x -> Wrap_utils.id x
| `Balanced -> Py.String.of_string "balanced"
| `DictIntToFloat x -> (Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float) x
)); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool)])
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
let fit ?coef_init ?intercept_init ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("coef_init", Wrap_utils.Option.map coef_init Np.Obj.to_pyobject); ("intercept_init", Wrap_utils.Option.map intercept_init Np.Obj.to_pyobject); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let partial_fit ?classes ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("classes", Wrap_utils.Option.map classes Np.Obj.to_pyobject); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
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
let set_params ?kwargs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match kwargs with None -> [] | Some x -> x)
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

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let t_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t_" with
  | None -> failwith "attribute t_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let t_ self = match t_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module PoissonRegressor = struct
type tag = [`PoissonRegressor]
type t = [`BaseEstimator | `Object | `PoissonRegressor | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let create ?alpha ?fit_intercept ?max_iter ?tol ?warm_start ?verbose () =
   Py.Module.get_function_with_keywords __wrap_namespace "PoissonRegressor"
     [||]
     (Wrap_utils.keyword_args [("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int)])
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
module RANSACRegressor = struct
type tag = [`RANSACRegressor]
type t = [`BaseEstimator | `MetaEstimatorMixin | `MultiOutputMixin | `Object | `RANSACRegressor | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_meta_estimator x = (x :> [`MetaEstimatorMixin] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
                  let create ?base_estimator ?min_samples ?residual_threshold ?is_data_valid ?is_model_valid ?max_trials ?max_skips ?stop_n_inliers ?stop_score ?stop_probability ?loss ?random_state () =
                     Py.Module.get_function_with_keywords __wrap_namespace "RANSACRegressor"
                       [||]
                       (Wrap_utils.keyword_args [("base_estimator", Wrap_utils.Option.map base_estimator Np.Obj.to_pyobject); ("min_samples", Wrap_utils.Option.map min_samples (function
| `I x -> Py.Int.of_int x
| `Float_0_1_ x -> Wrap_utils.id x
)); ("residual_threshold", Wrap_utils.Option.map residual_threshold Py.Float.of_float); ("is_data_valid", is_data_valid); ("is_model_valid", is_model_valid); ("max_trials", Wrap_utils.Option.map max_trials Py.Int.of_int); ("max_skips", Wrap_utils.Option.map max_skips Py.Int.of_int); ("stop_n_inliers", Wrap_utils.Option.map stop_n_inliers Py.Int.of_int); ("stop_score", Wrap_utils.Option.map stop_score Py.Float.of_float); ("stop_probability", Wrap_utils.Option.map stop_probability Py.Float.of_float); ("loss", Wrap_utils.Option.map loss (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
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
let score ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject

let estimator_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "estimator_" with
  | None -> failwith "attribute estimator_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let estimator_ self = match estimator_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_trials_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_trials_" with
  | None -> failwith "attribute n_trials_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_trials_ self = match n_trials_opt self with
  | None -> raise Not_found
  | Some x -> x

let inlier_mask_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "inlier_mask_" with
  | None -> failwith "attribute inlier_mask_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let inlier_mask_ self = match inlier_mask_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_skips_no_inliers_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_skips_no_inliers_" with
  | None -> failwith "attribute n_skips_no_inliers_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_skips_no_inliers_ self = match n_skips_no_inliers_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_skips_invalid_data_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_skips_invalid_data_" with
  | None -> failwith "attribute n_skips_invalid_data_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_skips_invalid_data_ self = match n_skips_invalid_data_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_skips_invalid_model_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_skips_invalid_model_" with
  | None -> failwith "attribute n_skips_invalid_model_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_skips_invalid_model_ self = match n_skips_invalid_model_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Ridge = struct
type tag = [`Ridge]
type t = [`BaseEstimator | `MultiOutputMixin | `Object | `RegressorMixin | `Ridge] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
                  let create ?alpha ?fit_intercept ?normalize ?copy_X ?max_iter ?tol ?solver ?random_state () =
                     Py.Module.get_function_with_keywords __wrap_namespace "Ridge"
                       [||]
                       (Wrap_utils.keyword_args [("alpha", Wrap_utils.Option.map alpha Np.Obj.to_pyobject); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("copy_X", Wrap_utils.Option.map copy_X Py.Bool.of_bool); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("solver", Wrap_utils.Option.map solver (function
| `Auto -> Py.String.of_string "auto"
| `Svd -> Py.String.of_string "svd"
| `Cholesky -> Py.String.of_string "cholesky"
| `Lsqr -> Py.String.of_string "lsqr"
| `Sparse_cg -> Py.String.of_string "sparse_cg"
| `Sag -> Py.String.of_string "sag"
| `Saga -> Py.String.of_string "saga"
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
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
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RidgeCV = struct
type tag = [`RidgeCV]
type t = [`BaseEstimator | `MultiOutputMixin | `Object | `RegressorMixin | `RidgeCV] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
                  let create ?alphas ?fit_intercept ?normalize ?scoring ?cv ?gcv_mode ?store_cv_values () =
                     Py.Module.get_function_with_keywords __wrap_namespace "RidgeCV"
                       [||]
                       (Wrap_utils.keyword_args [("alphas", Wrap_utils.Option.map alphas Np.Obj.to_pyobject); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("scoring", Wrap_utils.Option.map scoring (function
| `Neg_mean_absolute_error -> Py.String.of_string "neg_mean_absolute_error"
| `Completeness_score -> Py.String.of_string "completeness_score"
| `Roc_auc_ovr -> Py.String.of_string "roc_auc_ovr"
| `Neg_mean_squared_log_error -> Py.String.of_string "neg_mean_squared_log_error"
| `Neg_mean_gamma_deviance -> Py.String.of_string "neg_mean_gamma_deviance"
| `Precision_macro -> Py.String.of_string "precision_macro"
| `R2 -> Py.String.of_string "r2"
| `Precision_micro -> Py.String.of_string "precision_micro"
| `F1_weighted -> Py.String.of_string "f1_weighted"
| `Balanced_accuracy -> Py.String.of_string "balanced_accuracy"
| `Neg_mean_squared_error -> Py.String.of_string "neg_mean_squared_error"
| `F1_samples -> Py.String.of_string "f1_samples"
| `Jaccard_micro -> Py.String.of_string "jaccard_micro"
| `Normalized_mutual_info_score -> Py.String.of_string "normalized_mutual_info_score"
| `F1_micro -> Py.String.of_string "f1_micro"
| `Roc_auc -> Py.String.of_string "roc_auc"
| `Mutual_info_score -> Py.String.of_string "mutual_info_score"
| `Adjusted_rand_score -> Py.String.of_string "adjusted_rand_score"
| `Average_precision -> Py.String.of_string "average_precision"
| `Jaccard -> Py.String.of_string "jaccard"
| `Homogeneity_score -> Py.String.of_string "homogeneity_score"
| `Accuracy -> Py.String.of_string "accuracy"
| `Jaccard_macro -> Py.String.of_string "jaccard_macro"
| `Jaccard_weighted -> Py.String.of_string "jaccard_weighted"
| `Recall_micro -> Py.String.of_string "recall_micro"
| `Explained_variance -> Py.String.of_string "explained_variance"
| `Precision -> Py.String.of_string "precision"
| `Callable x -> Wrap_utils.id x
| `V_measure_score -> Py.String.of_string "v_measure_score"
| `F1 -> Py.String.of_string "f1"
| `Roc_auc_ovo -> Py.String.of_string "roc_auc_ovo"
| `Neg_mean_poisson_deviance -> Py.String.of_string "neg_mean_poisson_deviance"
| `Recall_samples -> Py.String.of_string "recall_samples"
| `Adjusted_mutual_info_score -> Py.String.of_string "adjusted_mutual_info_score"
| `Neg_brier_score -> Py.String.of_string "neg_brier_score"
| `Roc_auc_ovo_weighted -> Py.String.of_string "roc_auc_ovo_weighted"
| `Recall -> Py.String.of_string "recall"
| `Fowlkes_mallows_score -> Py.String.of_string "fowlkes_mallows_score"
| `Neg_log_loss -> Py.String.of_string "neg_log_loss"
| `Neg_root_mean_squared_error -> Py.String.of_string "neg_root_mean_squared_error"
| `Precision_samples -> Py.String.of_string "precision_samples"
| `F1_macro -> Py.String.of_string "f1_macro"
| `Roc_auc_ovr_weighted -> Py.String.of_string "roc_auc_ovr_weighted"
| `Recall_weighted -> Py.String.of_string "recall_weighted"
| `Neg_median_absolute_error -> Py.String.of_string "neg_median_absolute_error"
| `Jaccard_samples -> Py.String.of_string "jaccard_samples"
| `Precision_weighted -> Py.String.of_string "precision_weighted"
| `Max_error -> Py.String.of_string "max_error"
| `Recall_macro -> Py.String.of_string "recall_macro"
)); ("cv", Wrap_utils.Option.map cv (function
| `BaseCrossValidator x -> Np.Obj.to_pyobject x
| `Arr x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("gcv_mode", Wrap_utils.Option.map gcv_mode (function
| `Svd -> Py.String.of_string "svd"
| `Auto -> Py.String.of_string "auto"
| `Eigen -> Py.String.of_string "eigen"
)); ("store_cv_values", Wrap_utils.Option.map store_cv_values Py.Bool.of_bool)])
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

let cv_values_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "cv_values_" with
  | None -> failwith "attribute cv_values_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let cv_values_ self = match cv_values_opt self with
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

let alpha_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "alpha_" with
  | None -> failwith "attribute alpha_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let alpha_ self = match alpha_opt self with
  | None -> raise Not_found
  | Some x -> x

let best_score_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "best_score_" with
  | None -> failwith "attribute best_score_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let best_score_ self = match best_score_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RidgeClassifier = struct
type tag = [`RidgeClassifier]
type t = [`BaseEstimator | `ClassifierMixin | `LinearClassifierMixin | `Object | `RidgeClassifier] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_linear_classifier x = (x :> [`LinearClassifierMixin] Obj.t)
                  let create ?alpha ?fit_intercept ?normalize ?copy_X ?max_iter ?tol ?class_weight ?solver ?random_state () =
                     Py.Module.get_function_with_keywords __wrap_namespace "RidgeClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("copy_X", Wrap_utils.Option.map copy_X Py.Bool.of_bool); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("class_weight", Wrap_utils.Option.map class_weight (function
| `Balanced -> Py.String.of_string "balanced"
| `DictIntToFloat x -> (Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float) x
)); ("solver", Wrap_utils.Option.map solver (function
| `Auto -> Py.String.of_string "auto"
| `Svd -> Py.String.of_string "svd"
| `Cholesky -> Py.String.of_string "cholesky"
| `Lsqr -> Py.String.of_string "lsqr"
| `Sparse_cg -> Py.String.of_string "sparse_cg"
| `Sag -> Py.String.of_string "sag"
| `Saga -> Py.String.of_string "saga"
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
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
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let n_iter_ self = match n_iter_opt self with
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
module RidgeClassifierCV = struct
type tag = [`RidgeClassifierCV]
type t = [`BaseEstimator | `ClassifierMixin | `LinearClassifierMixin | `Object | `RidgeClassifierCV] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_linear_classifier x = (x :> [`LinearClassifierMixin] Obj.t)
                  let create ?alphas ?fit_intercept ?normalize ?scoring ?cv ?class_weight ?store_cv_values () =
                     Py.Module.get_function_with_keywords __wrap_namespace "RidgeClassifierCV"
                       [||]
                       (Wrap_utils.keyword_args [("alphas", Wrap_utils.Option.map alphas Np.Obj.to_pyobject); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("scoring", Wrap_utils.Option.map scoring (function
| `Neg_mean_absolute_error -> Py.String.of_string "neg_mean_absolute_error"
| `Completeness_score -> Py.String.of_string "completeness_score"
| `Roc_auc_ovr -> Py.String.of_string "roc_auc_ovr"
| `Neg_mean_squared_log_error -> Py.String.of_string "neg_mean_squared_log_error"
| `Neg_mean_gamma_deviance -> Py.String.of_string "neg_mean_gamma_deviance"
| `Precision_macro -> Py.String.of_string "precision_macro"
| `R2 -> Py.String.of_string "r2"
| `Precision_micro -> Py.String.of_string "precision_micro"
| `F1_weighted -> Py.String.of_string "f1_weighted"
| `Balanced_accuracy -> Py.String.of_string "balanced_accuracy"
| `Neg_mean_squared_error -> Py.String.of_string "neg_mean_squared_error"
| `F1_samples -> Py.String.of_string "f1_samples"
| `Jaccard_micro -> Py.String.of_string "jaccard_micro"
| `Normalized_mutual_info_score -> Py.String.of_string "normalized_mutual_info_score"
| `F1_micro -> Py.String.of_string "f1_micro"
| `Roc_auc -> Py.String.of_string "roc_auc"
| `Mutual_info_score -> Py.String.of_string "mutual_info_score"
| `Adjusted_rand_score -> Py.String.of_string "adjusted_rand_score"
| `Average_precision -> Py.String.of_string "average_precision"
| `Jaccard -> Py.String.of_string "jaccard"
| `Homogeneity_score -> Py.String.of_string "homogeneity_score"
| `Accuracy -> Py.String.of_string "accuracy"
| `Jaccard_macro -> Py.String.of_string "jaccard_macro"
| `Jaccard_weighted -> Py.String.of_string "jaccard_weighted"
| `Recall_micro -> Py.String.of_string "recall_micro"
| `Explained_variance -> Py.String.of_string "explained_variance"
| `Precision -> Py.String.of_string "precision"
| `Callable x -> Wrap_utils.id x
| `V_measure_score -> Py.String.of_string "v_measure_score"
| `F1 -> Py.String.of_string "f1"
| `Roc_auc_ovo -> Py.String.of_string "roc_auc_ovo"
| `Neg_mean_poisson_deviance -> Py.String.of_string "neg_mean_poisson_deviance"
| `Recall_samples -> Py.String.of_string "recall_samples"
| `Adjusted_mutual_info_score -> Py.String.of_string "adjusted_mutual_info_score"
| `Neg_brier_score -> Py.String.of_string "neg_brier_score"
| `Roc_auc_ovo_weighted -> Py.String.of_string "roc_auc_ovo_weighted"
| `Recall -> Py.String.of_string "recall"
| `Fowlkes_mallows_score -> Py.String.of_string "fowlkes_mallows_score"
| `Neg_log_loss -> Py.String.of_string "neg_log_loss"
| `Neg_root_mean_squared_error -> Py.String.of_string "neg_root_mean_squared_error"
| `Precision_samples -> Py.String.of_string "precision_samples"
| `F1_macro -> Py.String.of_string "f1_macro"
| `Roc_auc_ovr_weighted -> Py.String.of_string "roc_auc_ovr_weighted"
| `Recall_weighted -> Py.String.of_string "recall_weighted"
| `Neg_median_absolute_error -> Py.String.of_string "neg_median_absolute_error"
| `Jaccard_samples -> Py.String.of_string "jaccard_samples"
| `Precision_weighted -> Py.String.of_string "precision_weighted"
| `Max_error -> Py.String.of_string "max_error"
| `Recall_macro -> Py.String.of_string "recall_macro"
)); ("cv", Wrap_utils.Option.map cv (function
| `BaseCrossValidator x -> Np.Obj.to_pyobject x
| `Arr x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("class_weight", Wrap_utils.Option.map class_weight (function
| `Balanced -> Py.String.of_string "balanced"
| `DictIntToFloat x -> (Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float) x
)); ("store_cv_values", Wrap_utils.Option.map store_cv_values Py.Bool.of_bool)])
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

let cv_values_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "cv_values_" with
  | None -> failwith "attribute cv_values_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let cv_values_ self = match cv_values_opt self with
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

let alpha_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "alpha_" with
  | None -> failwith "attribute alpha_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let alpha_ self = match alpha_opt self with
  | None -> raise Not_found
  | Some x -> x

let best_score_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "best_score_" with
  | None -> failwith "attribute best_score_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let best_score_ self = match best_score_opt self with
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
module SGDClassifier = struct
type tag = [`SGDClassifier]
type t = [`BaseEstimator | `BaseSGD | `BaseSGDClassifier | `ClassifierMixin | `LinearClassifierMixin | `Object | `SGDClassifier | `SparseCoefMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_sgd x = (x :> [`BaseSGD] Obj.t)
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_sparse_coef x = (x :> [`SparseCoefMixin] Obj.t)
let as_linear_classifier x = (x :> [`LinearClassifierMixin] Obj.t)
let as_sgd_classifier x = (x :> [`BaseSGDClassifier] Obj.t)
                  let create ?loss ?penalty ?alpha ?l1_ratio ?fit_intercept ?max_iter ?tol ?shuffle ?verbose ?epsilon ?n_jobs ?random_state ?learning_rate ?eta0 ?power_t ?early_stopping ?validation_fraction ?n_iter_no_change ?class_weight ?warm_start ?average () =
                     Py.Module.get_function_with_keywords __wrap_namespace "SGDClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("loss", Wrap_utils.Option.map loss Py.String.of_string); ("penalty", Wrap_utils.Option.map penalty (function
| `L2 -> Py.String.of_string "l2"
| `L1 -> Py.String.of_string "l1"
| `Elasticnet -> Py.String.of_string "elasticnet"
)); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("l1_ratio", Wrap_utils.Option.map l1_ratio Py.Float.of_float); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("epsilon", Wrap_utils.Option.map epsilon Py.Float.of_float); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("learning_rate", Wrap_utils.Option.map learning_rate Py.String.of_string); ("eta0", Wrap_utils.Option.map eta0 Py.Float.of_float); ("power_t", Wrap_utils.Option.map power_t Py.Float.of_float); ("early_stopping", Wrap_utils.Option.map early_stopping Py.Bool.of_bool); ("validation_fraction", Wrap_utils.Option.map validation_fraction Py.Float.of_float); ("n_iter_no_change", Wrap_utils.Option.map n_iter_no_change Py.Int.of_int); ("class_weight", Wrap_utils.Option.map class_weight (function
| `T_class_label_weight_ x -> Wrap_utils.id x
| `Balanced -> Py.String.of_string "balanced"
| `DictIntToFloat x -> (Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float) x
)); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("average", Wrap_utils.Option.map average (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
))])
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
let fit ?coef_init ?intercept_init ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("coef_init", Wrap_utils.Option.map coef_init Np.Obj.to_pyobject); ("intercept_init", Wrap_utils.Option.map intercept_init Np.Obj.to_pyobject); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let partial_fit ?classes ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("classes", Wrap_utils.Option.map classes Np.Obj.to_pyobject); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
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
let set_params ?kwargs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match kwargs with None -> [] | Some x -> x)
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

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x

let loss_function_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "loss_function_" with
  | None -> failwith "attribute loss_function_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> fun x y -> Py.Callable.to_function py
       [|Np.NumpyRaw.Ndarray.to_pyobject x; Np.NumpyRaw.Ndarray.to_pyobject y|] |> Py.Float.to_float) x)

let loss_function_ self = match loss_function_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let t_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t_" with
  | None -> failwith "attribute t_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let t_ self = match t_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SGDRegressor = struct
type tag = [`SGDRegressor]
type t = [`BaseEstimator | `BaseSGD | `BaseSGDRegressor | `Object | `RegressorMixin | `SGDRegressor | `SparseCoefMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_sgd x = (x :> [`BaseSGD] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_sparse_coef x = (x :> [`SparseCoefMixin] Obj.t)
let as_sgd_regressor x = (x :> [`BaseSGDRegressor] Obj.t)
                  let create ?loss ?penalty ?alpha ?l1_ratio ?fit_intercept ?max_iter ?tol ?shuffle ?verbose ?epsilon ?random_state ?learning_rate ?eta0 ?power_t ?early_stopping ?validation_fraction ?n_iter_no_change ?warm_start ?average () =
                     Py.Module.get_function_with_keywords __wrap_namespace "SGDRegressor"
                       [||]
                       (Wrap_utils.keyword_args [("loss", Wrap_utils.Option.map loss Py.String.of_string); ("penalty", Wrap_utils.Option.map penalty (function
| `L2 -> Py.String.of_string "l2"
| `L1 -> Py.String.of_string "l1"
| `Elasticnet -> Py.String.of_string "elasticnet"
)); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("l1_ratio", Wrap_utils.Option.map l1_ratio Py.Float.of_float); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("epsilon", Wrap_utils.Option.map epsilon Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("learning_rate", Wrap_utils.Option.map learning_rate Py.String.of_string); ("eta0", Wrap_utils.Option.map eta0 Py.Float.of_float); ("power_t", Wrap_utils.Option.map power_t Py.Float.of_float); ("early_stopping", Wrap_utils.Option.map early_stopping Py.Bool.of_bool); ("validation_fraction", Wrap_utils.Option.map validation_fraction Py.Float.of_float); ("n_iter_no_change", Wrap_utils.Option.map n_iter_no_change Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("average", Wrap_utils.Option.map average (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
))])
                       |> of_pyobject
let densify self =
   Py.Module.get_function_with_keywords (to_pyobject self) "densify"
     [||]
     []
     |> of_pyobject
let fit ?coef_init ?intercept_init ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("coef_init", Wrap_utils.Option.map coef_init Np.Obj.to_pyobject); ("intercept_init", Wrap_utils.Option.map intercept_init Np.Obj.to_pyobject); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let partial_fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
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
let set_params ?kwargs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match kwargs with None -> [] | Some x -> x)
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

let average_coef_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "average_coef_" with
  | None -> failwith "attribute average_coef_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let average_coef_ self = match average_coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let average_intercept_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "average_intercept_" with
  | None -> failwith "attribute average_intercept_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let average_intercept_ self = match average_intercept_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x

let t_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t_" with
  | None -> failwith "attribute t_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let t_ self = match t_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SquaredLoss = struct
type tag = [`SquaredLoss]
type t = [`Object | `SquaredLoss] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module TheilSenRegressor = struct
type tag = [`TheilSenRegressor]
type t = [`BaseEstimator | `Object | `RegressorMixin | `TheilSenRegressor] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let create ?fit_intercept ?copy_X ?max_subpopulation ?n_subsamples ?max_iter ?tol ?random_state ?n_jobs ?verbose () =
   Py.Module.get_function_with_keywords __wrap_namespace "TheilSenRegressor"
     [||]
     (Wrap_utils.keyword_args [("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("copy_X", Wrap_utils.Option.map copy_X Py.Bool.of_bool); ("max_subpopulation", Wrap_utils.Option.map max_subpopulation Py.Int.of_int); ("n_subsamples", Wrap_utils.Option.map n_subsamples Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int)])
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

let breakdown_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "breakdown_" with
  | None -> failwith "attribute breakdown_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let breakdown_ self = match breakdown_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_subpopulation_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_subpopulation_" with
  | None -> failwith "attribute n_subpopulation_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_subpopulation_ self = match n_subpopulation_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module TweedieRegressor = struct
type tag = [`TweedieRegressor]
type t = [`BaseEstimator | `Object | `RegressorMixin | `TweedieRegressor] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
                  let create ?power ?alpha ?fit_intercept ?link ?max_iter ?tol ?warm_start ?verbose () =
                     Py.Module.get_function_with_keywords __wrap_namespace "TweedieRegressor"
                       [||]
                       (Wrap_utils.keyword_args [("power", Wrap_utils.Option.map power Py.Float.of_float); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("link", Wrap_utils.Option.map link (function
| `Auto -> Py.String.of_string "auto"
| `Identity -> Py.String.of_string "identity"
| `Log -> Py.String.of_string "log"
)); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int)])
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
                  let enet_path ?l1_ratio ?eps ?n_alphas ?alphas ?precompute ?xy ?copy_X ?coef_init ?verbose ?return_n_iter ?positive ?check_input ?params ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "enet_path"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("l1_ratio", Wrap_utils.Option.map l1_ratio Py.Float.of_float); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("n_alphas", Wrap_utils.Option.map n_alphas Py.Int.of_int); ("alphas", Wrap_utils.Option.map alphas Np.Obj.to_pyobject); ("precompute", Wrap_utils.Option.map precompute (function
| `Arr x -> Np.Obj.to_pyobject x
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("Xy", Wrap_utils.Option.map xy Np.Obj.to_pyobject); ("copy_X", Wrap_utils.Option.map copy_X Py.Bool.of_bool); ("coef_init", Wrap_utils.Option.map coef_init Np.Obj.to_pyobject); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("return_n_iter", Wrap_utils.Option.map return_n_iter Py.Bool.of_bool); ("positive", Wrap_utils.Option.map positive Py.Bool.of_bool); ("check_input", Wrap_utils.Option.map check_input Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))]) (match params with None -> [] | Some x -> x))
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3))))
                  let lars_path ?xy ?gram ?max_iter ?alpha_min ?method_ ?copy_X ?eps ?copy_Gram ?verbose ?return_path ?return_n_iter ?positive ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "lars_path"
                       [||]
                       (Wrap_utils.keyword_args [("Xy", Wrap_utils.Option.map xy Np.Obj.to_pyobject); ("Gram", Wrap_utils.Option.map gram (function
| `Auto -> Py.String.of_string "auto"
| `Arr x -> Np.Obj.to_pyobject x
)); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("alpha_min", Wrap_utils.Option.map alpha_min Py.Float.of_float); ("method", Wrap_utils.Option.map method_ (function
| `Lar -> Py.String.of_string "lar"
| `Lasso -> Py.String.of_string "lasso"
)); ("copy_X", Wrap_utils.Option.map copy_X Py.Bool.of_bool); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("copy_Gram", Wrap_utils.Option.map copy_Gram Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("return_path", Wrap_utils.Option.map return_path Py.Bool.of_bool); ("return_n_iter", Wrap_utils.Option.map return_n_iter Py.Bool.of_bool); ("positive", Wrap_utils.Option.map positive Py.Bool.of_bool); ("X", Some(x |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `None -> Py.none
))); ("y", Some(y |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `None -> Py.none
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 2)), (Py.Int.to_int (Py.Tuple.get x 3))))
                  let lars_path_gram ?max_iter ?alpha_min ?method_ ?copy_X ?eps ?copy_Gram ?verbose ?return_path ?return_n_iter ?positive ~xy ~gram ~n_samples () =
                     Py.Module.get_function_with_keywords __wrap_namespace "lars_path_gram"
                       [||]
                       (Wrap_utils.keyword_args [("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("alpha_min", Wrap_utils.Option.map alpha_min Py.Float.of_float); ("method", Wrap_utils.Option.map method_ (function
| `Lar -> Py.String.of_string "lar"
| `Lasso -> Py.String.of_string "lasso"
)); ("copy_X", Wrap_utils.Option.map copy_X Py.Bool.of_bool); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("copy_Gram", Wrap_utils.Option.map copy_Gram Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("return_path", Wrap_utils.Option.map return_path Py.Bool.of_bool); ("return_n_iter", Wrap_utils.Option.map return_n_iter Py.Bool.of_bool); ("positive", Wrap_utils.Option.map positive Py.Bool.of_bool); ("Xy", Some(xy |> Np.Obj.to_pyobject)); ("Gram", Some(gram |> Np.Obj.to_pyobject)); ("n_samples", Some(n_samples |> (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 2)), (Py.Int.to_int (Py.Tuple.get x 3))))
                  let lasso_path ?eps ?n_alphas ?alphas ?precompute ?xy ?copy_X ?coef_init ?verbose ?return_n_iter ?positive ?params ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "lasso_path"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("n_alphas", Wrap_utils.Option.map n_alphas Py.Int.of_int); ("alphas", Wrap_utils.Option.map alphas Np.Obj.to_pyobject); ("precompute", Wrap_utils.Option.map precompute (function
| `Arr x -> Np.Obj.to_pyobject x
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("Xy", Wrap_utils.Option.map xy Np.Obj.to_pyobject); ("copy_X", Wrap_utils.Option.map copy_X Py.Bool.of_bool); ("coef_init", Wrap_utils.Option.map coef_init Np.Obj.to_pyobject); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("return_n_iter", Wrap_utils.Option.map return_n_iter Py.Bool.of_bool); ("positive", Wrap_utils.Option.map positive Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))]) (match params with None -> [] | Some x -> x))
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3))))
                  let orthogonal_mp ?n_nonzero_coefs ?tol ?precompute ?copy_X ?return_path ?return_n_iter ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "orthogonal_mp"
                       [||]
                       (Wrap_utils.keyword_args [("n_nonzero_coefs", Wrap_utils.Option.map n_nonzero_coefs Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("precompute", Wrap_utils.Option.map precompute (function
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("copy_X", Wrap_utils.Option.map copy_X Py.Bool.of_bool); ("return_path", Wrap_utils.Option.map return_path Py.Bool.of_bool); ("return_n_iter", Wrap_utils.Option.map return_n_iter Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let orthogonal_mp_gram ?n_nonzero_coefs ?tol ?norms_squared ?copy_Gram ?copy_Xy ?return_path ?return_n_iter ~gram ~xy () =
   Py.Module.get_function_with_keywords __wrap_namespace "orthogonal_mp_gram"
     [||]
     (Wrap_utils.keyword_args [("n_nonzero_coefs", Wrap_utils.Option.map n_nonzero_coefs Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("norms_squared", Wrap_utils.Option.map norms_squared Np.Obj.to_pyobject); ("copy_Gram", Wrap_utils.Option.map copy_Gram Py.Bool.of_bool); ("copy_Xy", Wrap_utils.Option.map copy_Xy Py.Bool.of_bool); ("return_path", Wrap_utils.Option.map return_path Py.Bool.of_bool); ("return_n_iter", Wrap_utils.Option.map return_n_iter Py.Bool.of_bool); ("Gram", Some(gram |> Np.Obj.to_pyobject)); ("Xy", Some(xy |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let ridge_regression ?sample_weight ?solver ?max_iter ?tol ?verbose ?random_state ?return_n_iter ?return_intercept ?check_input ~x ~y ~alpha () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ridge_regression"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("solver", Wrap_utils.Option.map solver (function
| `Auto -> Py.String.of_string "auto"
| `Svd -> Py.String.of_string "svd"
| `Cholesky -> Py.String.of_string "cholesky"
| `Lsqr -> Py.String.of_string "lsqr"
| `Sparse_cg -> Py.String.of_string "sparse_cg"
| `Sag -> Py.String.of_string "sag"
| `Saga -> Py.String.of_string "saga"
)); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("return_n_iter", Wrap_utils.Option.map return_n_iter Py.Bool.of_bool); ("return_intercept", Wrap_utils.Option.map return_intercept Py.Bool.of_bool); ("check_input", Wrap_utils.Option.map check_input Py.Bool.of_bool); ("X", Some(x |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
))); ("y", Some(y |> Np.Obj.to_pyobject)); ("alpha", Some(alpha |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 2))))
