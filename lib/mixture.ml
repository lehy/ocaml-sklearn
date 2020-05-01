let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.mixture"

let get_py name = Py.Module.get ns name
module BayesianGaussianMixture = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_components ?covariance_type ?tol ?reg_covar ?max_iter ?n_init ?init_params ?weight_concentration_prior_type ?weight_concentration_prior ?mean_precision_prior ?mean_prior ?degrees_of_freedom_prior ?covariance_prior ?random_state ?warm_start ?verbose ?verbose_interval () =
                     Py.Module.get_function_with_keywords ns "BayesianGaussianMixture"
                       [||]
                       (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("covariance_type", Wrap_utils.Option.map covariance_type (function
| `Full -> Py.String.of_string "full"
| `Tied -> Py.String.of_string "tied"
| `Diag -> Py.String.of_string "diag"
| `Spherical -> Py.String.of_string "spherical"
)); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("reg_covar", Wrap_utils.Option.map reg_covar Py.Float.of_float); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("n_init", Wrap_utils.Option.map n_init Py.Int.of_int); ("init_params", Wrap_utils.Option.map init_params (function
| `Kmeans -> Py.String.of_string "kmeans"
| `Random -> Py.String.of_string "random"
)); ("weight_concentration_prior_type", Wrap_utils.Option.map weight_concentration_prior_type Py.String.of_string); ("weight_concentration_prior", Wrap_utils.Option.map weight_concentration_prior Py.Float.of_float); ("mean_precision_prior", Wrap_utils.Option.map mean_precision_prior Py.Float.of_float); ("mean_prior", Wrap_utils.Option.map mean_prior Arr.to_pyobject); ("degrees_of_freedom_prior", Wrap_utils.Option.map degrees_of_freedom_prior Py.Float.of_float); ("covariance_prior", Wrap_utils.Option.map covariance_prior (function
| `F x -> Py.Float.of_float x
| `Arr x -> Arr.to_pyobject x
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("verbose_interval", Wrap_utils.Option.map verbose_interval Py.Int.of_int)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])

let fit_predict ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_predict"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])
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
let predict_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let sample ?n_samples self =
   Py.Module.get_function_with_keywords self "sample"
     [||]
     (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int)])
     |> (fun x -> ((Arr.of_pyobject (Py.Tuple.get x 0)), (Arr.of_pyobject (Py.Tuple.get x 1))))
let score ?y ~x self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])
     |> Py.Float.to_float
let score_samples ~x self =
   Py.Module.get_function_with_keywords self "score_samples"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)


let weights_opt self =
  match Py.Object.get_attr_string self "weights_" with
  | None -> failwith "attribute weights_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let weights_ self = match weights_opt self with
  | None -> raise Not_found
  | Some x -> x

let means_opt self =
  match Py.Object.get_attr_string self "means_" with
  | None -> failwith "attribute means_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let means_ self = match means_opt self with
  | None -> raise Not_found
  | Some x -> x

let covariances_opt self =
  match Py.Object.get_attr_string self "covariances_" with
  | None -> failwith "attribute covariances_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let covariances_ self = match covariances_opt self with
  | None -> raise Not_found
  | Some x -> x

let precisions_opt self =
  match Py.Object.get_attr_string self "precisions_" with
  | None -> failwith "attribute precisions_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let precisions_ self = match precisions_opt self with
  | None -> raise Not_found
  | Some x -> x

let precisions_cholesky_opt self =
  match Py.Object.get_attr_string self "precisions_cholesky_" with
  | None -> failwith "attribute precisions_cholesky_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let precisions_cholesky_ self = match precisions_cholesky_opt self with
  | None -> raise Not_found
  | Some x -> x

let converged_opt self =
  match Py.Object.get_attr_string self "converged_" with
  | None -> failwith "attribute converged_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Bool.to_bool x)

let converged_ self = match converged_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string self "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x

let lower_bound_opt self =
  match Py.Object.get_attr_string self "lower_bound_" with
  | None -> failwith "attribute lower_bound_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let lower_bound_ self = match lower_bound_opt self with
  | None -> raise Not_found
  | Some x -> x

let weight_concentration_prior_opt self =
  match Py.Object.get_attr_string self "weight_concentration_prior_" with
  | None -> failwith "attribute weight_concentration_prior_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let weight_concentration_prior_ self = match weight_concentration_prior_opt self with
  | None -> raise Not_found
  | Some x -> x

let weight_concentration_opt self =
  match Py.Object.get_attr_string self "weight_concentration_" with
  | None -> failwith "attribute weight_concentration_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let weight_concentration_ self = match weight_concentration_opt self with
  | None -> raise Not_found
  | Some x -> x

let mean_precision_prior_opt self =
  match Py.Object.get_attr_string self "mean_precision_prior_" with
  | None -> failwith "attribute mean_precision_prior_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let mean_precision_prior_ self = match mean_precision_prior_opt self with
  | None -> raise Not_found
  | Some x -> x

let mean_precision_opt self =
  match Py.Object.get_attr_string self "mean_precision_" with
  | None -> failwith "attribute mean_precision_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let mean_precision_ self = match mean_precision_opt self with
  | None -> raise Not_found
  | Some x -> x

let mean_prior_opt self =
  match Py.Object.get_attr_string self "mean_prior_" with
  | None -> failwith "attribute mean_prior_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let mean_prior_ self = match mean_prior_opt self with
  | None -> raise Not_found
  | Some x -> x

let degrees_of_freedom_prior_opt self =
  match Py.Object.get_attr_string self "degrees_of_freedom_prior_" with
  | None -> failwith "attribute degrees_of_freedom_prior_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let degrees_of_freedom_prior_ self = match degrees_of_freedom_prior_opt self with
  | None -> raise Not_found
  | Some x -> x

let degrees_of_freedom_opt self =
  match Py.Object.get_attr_string self "degrees_of_freedom_" with
  | None -> failwith "attribute degrees_of_freedom_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let degrees_of_freedom_ self = match degrees_of_freedom_opt self with
  | None -> raise Not_found
  | Some x -> x

let covariance_prior_opt self =
  match Py.Object.get_attr_string self "covariance_prior_" with
  | None -> failwith "attribute covariance_prior_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun x -> if Py.Float.check x then `F (Py.Float.to_float x) else if (fun x -> (Wrap_utils.isinstance Wrap_utils.ndarray x) || (Wrap_utils.isinstance Wrap_utils.csr_matrix x)) x then `Arr (Arr.of_pyobject x) else failwith "could not identify type from Python value") x)

let covariance_prior_ self = match covariance_prior_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module GaussianMixture = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_components ?covariance_type ?tol ?reg_covar ?max_iter ?n_init ?init_params ?weights_init ?means_init ?precisions_init ?random_state ?warm_start ?verbose ?verbose_interval () =
                     Py.Module.get_function_with_keywords ns "GaussianMixture"
                       [||]
                       (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("covariance_type", Wrap_utils.Option.map covariance_type (function
| `Full -> Py.String.of_string "full"
| `Tied -> Py.String.of_string "tied"
| `Diag -> Py.String.of_string "diag"
| `Spherical -> Py.String.of_string "spherical"
)); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("reg_covar", Wrap_utils.Option.map reg_covar Py.Float.of_float); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("n_init", Wrap_utils.Option.map n_init Py.Int.of_int); ("init_params", Wrap_utils.Option.map init_params (function
| `Kmeans -> Py.String.of_string "kmeans"
| `Random -> Py.String.of_string "random"
)); ("weights_init", Wrap_utils.Option.map weights_init Arr.to_pyobject); ("means_init", Wrap_utils.Option.map means_init Arr.to_pyobject); ("precisions_init", Wrap_utils.Option.map precisions_init Arr.to_pyobject); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("verbose_interval", Wrap_utils.Option.map verbose_interval Py.Int.of_int)])

let aic ~x self =
   Py.Module.get_function_with_keywords self "aic"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Py.Float.to_float
let bic ~x self =
   Py.Module.get_function_with_keywords self "bic"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Py.Float.to_float
let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])

let fit_predict ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_predict"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])
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
let predict_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let sample ?n_samples self =
   Py.Module.get_function_with_keywords self "sample"
     [||]
     (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int)])
     |> (fun x -> ((Arr.of_pyobject (Py.Tuple.get x 0)), (Arr.of_pyobject (Py.Tuple.get x 1))))
let score ?y ~x self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])
     |> Py.Float.to_float
let score_samples ~x self =
   Py.Module.get_function_with_keywords self "score_samples"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)


let weights_opt self =
  match Py.Object.get_attr_string self "weights_" with
  | None -> failwith "attribute weights_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let weights_ self = match weights_opt self with
  | None -> raise Not_found
  | Some x -> x

let means_opt self =
  match Py.Object.get_attr_string self "means_" with
  | None -> failwith "attribute means_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let means_ self = match means_opt self with
  | None -> raise Not_found
  | Some x -> x

let covariances_opt self =
  match Py.Object.get_attr_string self "covariances_" with
  | None -> failwith "attribute covariances_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let covariances_ self = match covariances_opt self with
  | None -> raise Not_found
  | Some x -> x

let precisions_opt self =
  match Py.Object.get_attr_string self "precisions_" with
  | None -> failwith "attribute precisions_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let precisions_ self = match precisions_opt self with
  | None -> raise Not_found
  | Some x -> x

let precisions_cholesky_opt self =
  match Py.Object.get_attr_string self "precisions_cholesky_" with
  | None -> failwith "attribute precisions_cholesky_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let precisions_cholesky_ self = match precisions_cholesky_opt self with
  | None -> raise Not_found
  | Some x -> x

let converged_opt self =
  match Py.Object.get_attr_string self "converged_" with
  | None -> failwith "attribute converged_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Bool.to_bool x)

let converged_ self = match converged_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string self "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x

let lower_bound_opt self =
  match Py.Object.get_attr_string self "lower_bound_" with
  | None -> failwith "attribute lower_bound_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let lower_bound_ self = match lower_bound_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
