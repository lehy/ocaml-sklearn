let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.mixture"

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
)); ("weight_concentration_prior_type", Wrap_utils.Option.map weight_concentration_prior_type Py.String.of_string); ("weight_concentration_prior", Wrap_utils.Option.map weight_concentration_prior (function
| `Float x -> Py.Float.of_float x
| `None -> Py.String.of_string "None"
)); ("mean_precision_prior", Wrap_utils.Option.map mean_precision_prior (function
| `Float x -> Py.Float.of_float x
| `None -> Py.String.of_string "None"
)); ("mean_prior", Wrap_utils.Option.map mean_prior Ndarray.to_pyobject); ("degrees_of_freedom_prior", Wrap_utils.Option.map degrees_of_freedom_prior (function
| `Float x -> Py.Float.of_float x
| `None -> Py.String.of_string "None"
)); ("covariance_prior", Wrap_utils.Option.map covariance_prior (function
| `Float x -> Py.Float.of_float x
| `Ndarray x -> Ndarray.to_pyobject x
)); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("verbose_interval", Wrap_utils.Option.map verbose_interval Py.Int.of_int)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let fit_predict ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_predict"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let predict_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let sample ?n_samples self =
   Py.Module.get_function_with_keywords self "sample"
     [||]
     (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int)])
     |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
let score ?y ~x self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let score_samples ~x self =
   Py.Module.get_function_with_keywords self "score_samples"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let weights_ self =
  match Py.Object.get_attr_string self "weights_" with
| None -> raise (Wrap_utils.Attribute_not_found "weights_")
| Some x -> Ndarray.of_pyobject x
let means_ self =
  match Py.Object.get_attr_string self "means_" with
| None -> raise (Wrap_utils.Attribute_not_found "means_")
| Some x -> Ndarray.of_pyobject x
let covariances_ self =
  match Py.Object.get_attr_string self "covariances_" with
| None -> raise (Wrap_utils.Attribute_not_found "covariances_")
| Some x -> Ndarray.of_pyobject x
let precisions_ self =
  match Py.Object.get_attr_string self "precisions_" with
| None -> raise (Wrap_utils.Attribute_not_found "precisions_")
| Some x -> Ndarray.of_pyobject x
let precisions_cholesky_ self =
  match Py.Object.get_attr_string self "precisions_cholesky_" with
| None -> raise (Wrap_utils.Attribute_not_found "precisions_cholesky_")
| Some x -> Ndarray.of_pyobject x
let converged_ self =
  match Py.Object.get_attr_string self "converged_" with
| None -> raise (Wrap_utils.Attribute_not_found "converged_")
| Some x -> Py.Bool.to_bool x
let n_iter_ self =
  match Py.Object.get_attr_string self "n_iter_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_iter_")
| Some x -> Py.Int.to_int x
let lower_bound_ self =
  match Py.Object.get_attr_string self "lower_bound_" with
| None -> raise (Wrap_utils.Attribute_not_found "lower_bound_")
| Some x -> Py.Float.to_float x
let weight_concentration_prior_ self =
  match Py.Object.get_attr_string self "weight_concentration_prior_" with
| None -> raise (Wrap_utils.Attribute_not_found "weight_concentration_prior_")
| Some x -> Wrap_utils.id x
let weight_concentration_ self =
  match Py.Object.get_attr_string self "weight_concentration_" with
| None -> raise (Wrap_utils.Attribute_not_found "weight_concentration_")
| Some x -> Ndarray.of_pyobject x
let mean_precision_prior_ self =
  match Py.Object.get_attr_string self "mean_precision_prior_" with
| None -> raise (Wrap_utils.Attribute_not_found "mean_precision_prior_")
| Some x -> Py.Float.to_float x
let mean_precision_ self =
  match Py.Object.get_attr_string self "mean_precision_" with
| None -> raise (Wrap_utils.Attribute_not_found "mean_precision_")
| Some x -> Ndarray.of_pyobject x
let mean_prior_ self =
  match Py.Object.get_attr_string self "mean_prior_" with
| None -> raise (Wrap_utils.Attribute_not_found "mean_prior_")
| Some x -> Ndarray.of_pyobject x
let degrees_of_freedom_prior_ self =
  match Py.Object.get_attr_string self "degrees_of_freedom_prior_" with
| None -> raise (Wrap_utils.Attribute_not_found "degrees_of_freedom_prior_")
| Some x -> Py.Float.to_float x
let degrees_of_freedom_ self =
  match Py.Object.get_attr_string self "degrees_of_freedom_" with
| None -> raise (Wrap_utils.Attribute_not_found "degrees_of_freedom_")
| Some x -> Ndarray.of_pyobject x
let covariance_prior_ self =
  match Py.Object.get_attr_string self "covariance_prior_" with
| None -> raise (Wrap_utils.Attribute_not_found "covariance_prior_")
| Some x -> Wrap_utils.id x
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
                       (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("covariance_type", covariance_type); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("reg_covar", Wrap_utils.Option.map reg_covar Py.Float.of_float); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("n_init", Wrap_utils.Option.map n_init Py.Int.of_int); ("init_params", Wrap_utils.Option.map init_params (function
| `Kmeans -> Py.String.of_string "kmeans"
| `Random -> Py.String.of_string "random"
)); ("weights_init", Wrap_utils.Option.map weights_init Ndarray.to_pyobject); ("means_init", Wrap_utils.Option.map means_init Ndarray.to_pyobject); ("precisions_init", Wrap_utils.Option.map precisions_init Ndarray.to_pyobject); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("verbose_interval", Wrap_utils.Option.map verbose_interval Py.Int.of_int)])

let aic ~x self =
   Py.Module.get_function_with_keywords self "aic"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let bic ~x self =
   Py.Module.get_function_with_keywords self "bic"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let fit_predict ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_predict"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let predict_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let sample ?n_samples self =
   Py.Module.get_function_with_keywords self "sample"
     [||]
     (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int)])
     |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
let score ?y ~x self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let score_samples ~x self =
   Py.Module.get_function_with_keywords self "score_samples"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let weights_ self =
  match Py.Object.get_attr_string self "weights_" with
| None -> raise (Wrap_utils.Attribute_not_found "weights_")
| Some x -> Ndarray.of_pyobject x
let means_ self =
  match Py.Object.get_attr_string self "means_" with
| None -> raise (Wrap_utils.Attribute_not_found "means_")
| Some x -> Ndarray.of_pyobject x
let covariances_ self =
  match Py.Object.get_attr_string self "covariances_" with
| None -> raise (Wrap_utils.Attribute_not_found "covariances_")
| Some x -> Ndarray.of_pyobject x
let precisions_ self =
  match Py.Object.get_attr_string self "precisions_" with
| None -> raise (Wrap_utils.Attribute_not_found "precisions_")
| Some x -> Ndarray.of_pyobject x
let precisions_cholesky_ self =
  match Py.Object.get_attr_string self "precisions_cholesky_" with
| None -> raise (Wrap_utils.Attribute_not_found "precisions_cholesky_")
| Some x -> Ndarray.of_pyobject x
let converged_ self =
  match Py.Object.get_attr_string self "converged_" with
| None -> raise (Wrap_utils.Attribute_not_found "converged_")
| Some x -> Py.Bool.to_bool x
let n_iter_ self =
  match Py.Object.get_attr_string self "n_iter_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_iter_")
| Some x -> Py.Int.to_int x
let lower_bound_ self =
  match Py.Object.get_attr_string self "lower_bound_" with
| None -> raise (Wrap_utils.Attribute_not_found "lower_bound_")
| Some x -> Py.Float.to_float x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
