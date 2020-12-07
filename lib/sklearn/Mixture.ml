let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.mixture"

let get_py name = Py.Module.get __wrap_namespace name
module BayesianGaussianMixture = struct
type tag = [`BayesianGaussianMixture]
type t = [`BaseEstimator | `BaseMixture | `BayesianGaussianMixture | `DensityMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_mixture x = (x :> [`BaseMixture] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_density x = (x :> [`DensityMixin] Obj.t)
                  let create ?n_components ?covariance_type ?tol ?reg_covar ?max_iter ?n_init ?init_params ?weight_concentration_prior_type ?weight_concentration_prior ?mean_precision_prior ?mean_prior ?degrees_of_freedom_prior ?covariance_prior ?random_state ?warm_start ?verbose ?verbose_interval () =
                     Py.Module.get_function_with_keywords __wrap_namespace "BayesianGaussianMixture"
                       [||]
                       (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("covariance_type", Wrap_utils.Option.map covariance_type (function
| `Full -> Py.String.of_string "full"
| `Tied -> Py.String.of_string "tied"
| `Diag -> Py.String.of_string "diag"
| `Spherical -> Py.String.of_string "spherical"
)); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("reg_covar", Wrap_utils.Option.map reg_covar Py.Float.of_float); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("n_init", Wrap_utils.Option.map n_init Py.Int.of_int); ("init_params", Wrap_utils.Option.map init_params (function
| `Kmeans -> Py.String.of_string "kmeans"
| `Random -> Py.String.of_string "random"
)); ("weight_concentration_prior_type", Wrap_utils.Option.map weight_concentration_prior_type Py.String.of_string); ("weight_concentration_prior", Wrap_utils.Option.map weight_concentration_prior Py.Float.of_float); ("mean_precision_prior", Wrap_utils.Option.map mean_precision_prior Py.Float.of_float); ("mean_prior", Wrap_utils.Option.map mean_prior Np.Obj.to_pyobject); ("degrees_of_freedom_prior", Wrap_utils.Option.map degrees_of_freedom_prior Py.Float.of_float); ("covariance_prior", Wrap_utils.Option.map covariance_prior Np.Obj.to_pyobject); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("verbose_interval", Wrap_utils.Option.map verbose_interval Py.Int.of_int)])
                       |> of_pyobject
let fit ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
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
let predict_proba ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let sample ?n_samples self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sample"
     [||]
     (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int)])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
let score ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
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

let weights_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "weights_" with
  | None -> failwith "attribute weights_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let weights_ self = match weights_opt self with
  | None -> raise Not_found
  | Some x -> x

let means_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "means_" with
  | None -> failwith "attribute means_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let means_ self = match means_opt self with
  | None -> raise Not_found
  | Some x -> x

let covariances_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "covariances_" with
  | None -> failwith "attribute covariances_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let covariances_ self = match covariances_opt self with
  | None -> raise Not_found
  | Some x -> x

let precisions_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "precisions_" with
  | None -> failwith "attribute precisions_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let precisions_ self = match precisions_opt self with
  | None -> raise Not_found
  | Some x -> x

let precisions_cholesky_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "precisions_cholesky_" with
  | None -> failwith "attribute precisions_cholesky_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let precisions_cholesky_ self = match precisions_cholesky_opt self with
  | None -> raise Not_found
  | Some x -> x

let converged_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "converged_" with
  | None -> failwith "attribute converged_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Bool.to_bool x)

let converged_ self = match converged_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x

let lower_bound_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "lower_bound_" with
  | None -> failwith "attribute lower_bound_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let lower_bound_ self = match lower_bound_opt self with
  | None -> raise Not_found
  | Some x -> x

let weight_concentration_prior_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "weight_concentration_prior_" with
  | None -> failwith "attribute weight_concentration_prior_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let weight_concentration_prior_ self = match weight_concentration_prior_opt self with
  | None -> raise Not_found
  | Some x -> x

let weight_concentration_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "weight_concentration_" with
  | None -> failwith "attribute weight_concentration_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let weight_concentration_ self = match weight_concentration_opt self with
  | None -> raise Not_found
  | Some x -> x

let mean_precision_prior_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "mean_precision_prior_" with
  | None -> failwith "attribute mean_precision_prior_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let mean_precision_prior_ self = match mean_precision_prior_opt self with
  | None -> raise Not_found
  | Some x -> x

let mean_precision_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "mean_precision_" with
  | None -> failwith "attribute mean_precision_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let mean_precision_ self = match mean_precision_opt self with
  | None -> raise Not_found
  | Some x -> x

let mean_prior_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "mean_prior_" with
  | None -> failwith "attribute mean_prior_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let mean_prior_ self = match mean_prior_opt self with
  | None -> raise Not_found
  | Some x -> x

let degrees_of_freedom_prior_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "degrees_of_freedom_prior_" with
  | None -> failwith "attribute degrees_of_freedom_prior_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let degrees_of_freedom_prior_ self = match degrees_of_freedom_prior_opt self with
  | None -> raise Not_found
  | Some x -> x

let degrees_of_freedom_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "degrees_of_freedom_" with
  | None -> failwith "attribute degrees_of_freedom_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let degrees_of_freedom_ self = match degrees_of_freedom_opt self with
  | None -> raise Not_found
  | Some x -> x

let covariance_prior_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "covariance_prior_" with
  | None -> failwith "attribute covariance_prior_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let covariance_prior_ self = match covariance_prior_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module GaussianMixture = struct
type tag = [`GaussianMixture]
type t = [`BaseEstimator | `BaseMixture | `DensityMixin | `GaussianMixture | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_mixture x = (x :> [`BaseMixture] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_density x = (x :> [`DensityMixin] Obj.t)
                  let create ?n_components ?covariance_type ?tol ?reg_covar ?max_iter ?n_init ?init_params ?weights_init ?means_init ?precisions_init ?random_state ?warm_start ?verbose ?verbose_interval () =
                     Py.Module.get_function_with_keywords __wrap_namespace "GaussianMixture"
                       [||]
                       (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("covariance_type", Wrap_utils.Option.map covariance_type (function
| `Full -> Py.String.of_string "full"
| `Tied -> Py.String.of_string "tied"
| `Diag -> Py.String.of_string "diag"
| `Spherical -> Py.String.of_string "spherical"
)); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("reg_covar", Wrap_utils.Option.map reg_covar Py.Float.of_float); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("n_init", Wrap_utils.Option.map n_init Py.Int.of_int); ("init_params", Wrap_utils.Option.map init_params (function
| `Kmeans -> Py.String.of_string "kmeans"
| `Random -> Py.String.of_string "random"
)); ("weights_init", Wrap_utils.Option.map weights_init Np.Obj.to_pyobject); ("means_init", Wrap_utils.Option.map means_init Np.Obj.to_pyobject); ("precisions_init", Wrap_utils.Option.map precisions_init Np.Obj.to_pyobject); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("verbose_interval", Wrap_utils.Option.map verbose_interval Py.Int.of_int)])
                       |> of_pyobject
let aic ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "aic"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let bic ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "bic"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let fit ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
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
let predict_proba ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let sample ?n_samples self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sample"
     [||]
     (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int)])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
let score ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
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

let weights_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "weights_" with
  | None -> failwith "attribute weights_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let weights_ self = match weights_opt self with
  | None -> raise Not_found
  | Some x -> x

let means_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "means_" with
  | None -> failwith "attribute means_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let means_ self = match means_opt self with
  | None -> raise Not_found
  | Some x -> x

let covariances_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "covariances_" with
  | None -> failwith "attribute covariances_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let covariances_ self = match covariances_opt self with
  | None -> raise Not_found
  | Some x -> x

let precisions_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "precisions_" with
  | None -> failwith "attribute precisions_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let precisions_ self = match precisions_opt self with
  | None -> raise Not_found
  | Some x -> x

let precisions_cholesky_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "precisions_cholesky_" with
  | None -> failwith "attribute precisions_cholesky_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let precisions_cholesky_ self = match precisions_cholesky_opt self with
  | None -> raise Not_found
  | Some x -> x

let converged_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "converged_" with
  | None -> failwith "attribute converged_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Bool.to_bool x)

let converged_ self = match converged_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x

let lower_bound_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "lower_bound_" with
  | None -> failwith "attribute lower_bound_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let lower_bound_ self = match lower_bound_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
