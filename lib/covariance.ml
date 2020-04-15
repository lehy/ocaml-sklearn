let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.covariance"

module EllipticEnvelope = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?store_precision ?assume_centered ?support_fraction ?contamination ?random_state () =
                     Py.Module.get_function_with_keywords ns "EllipticEnvelope"
                       [||]
                       (Wrap_utils.keyword_args [("store_precision", Wrap_utils.Option.map store_precision Py.Bool.of_bool); ("assume_centered", Wrap_utils.Option.map assume_centered Py.Bool.of_bool); ("support_fraction", Wrap_utils.Option.map support_fraction Py.Float.of_float); ("contamination", Wrap_utils.Option.map contamination Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])

let correct_covariance ~data self =
   Py.Module.get_function_with_keywords self "correct_covariance"
     [||]
     (Wrap_utils.keyword_args [("data", Some(data |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let decision_function ~x self =
   Py.Module.get_function_with_keywords self "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let error_norm ?norm ?scaling ?squared ~comp_cov self =
   Py.Module.get_function_with_keywords self "error_norm"
     [||]
     (Wrap_utils.keyword_args [("norm", Wrap_utils.Option.map norm Py.String.of_string); ("scaling", Wrap_utils.Option.map scaling Py.Bool.of_bool); ("squared", Wrap_utils.Option.map squared Py.Bool.of_bool); ("comp_cov", Some(comp_cov |> Ndarray.to_pyobject))])

                  let fit ?y ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

let fit_predict ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_predict"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let get_precision self =
   Py.Module.get_function_with_keywords self "get_precision"
     [||]
     []
     |> Ndarray.of_pyobject
let mahalanobis ~x self =
   Py.Module.get_function_with_keywords self "mahalanobis"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let reweight_covariance ~data self =
   Py.Module.get_function_with_keywords self "reweight_covariance"
     [||]
     (Wrap_utils.keyword_args [("data", Some(data |> Ndarray.to_pyobject))])
     |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])
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

let location_ self =
  match Py.Object.get_attr_string self "location_" with
| None -> raise (Wrap_utils.Attribute_not_found "location_")
| Some x -> Ndarray.of_pyobject x
let covariance_ self =
  match Py.Object.get_attr_string self "covariance_" with
| None -> raise (Wrap_utils.Attribute_not_found "covariance_")
| Some x -> Ndarray.of_pyobject x
let precision_ self =
  match Py.Object.get_attr_string self "precision_" with
| None -> raise (Wrap_utils.Attribute_not_found "precision_")
| Some x -> Ndarray.of_pyobject x
let support_ self =
  match Py.Object.get_attr_string self "support_" with
| None -> raise (Wrap_utils.Attribute_not_found "support_")
| Some x -> Ndarray.of_pyobject x
let offset_ self =
  match Py.Object.get_attr_string self "offset_" with
| None -> raise (Wrap_utils.Attribute_not_found "offset_")
| Some x -> Py.Float.to_float x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module EmpiricalCovariance = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?store_precision ?assume_centered () =
   Py.Module.get_function_with_keywords ns "EmpiricalCovariance"
     [||]
     (Wrap_utils.keyword_args [("store_precision", Wrap_utils.Option.map store_precision Py.Bool.of_bool); ("assume_centered", Wrap_utils.Option.map assume_centered Py.Bool.of_bool)])

let error_norm ?norm ?scaling ?squared ~comp_cov self =
   Py.Module.get_function_with_keywords self "error_norm"
     [||]
     (Wrap_utils.keyword_args [("norm", Wrap_utils.Option.map norm Py.String.of_string); ("scaling", Wrap_utils.Option.map scaling Py.Bool.of_bool); ("squared", Wrap_utils.Option.map squared Py.Bool.of_bool); ("comp_cov", Some(comp_cov |> Ndarray.to_pyobject))])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let get_precision self =
   Py.Module.get_function_with_keywords self "get_precision"
     [||]
     []
     |> Ndarray.of_pyobject
let mahalanobis ~x self =
   Py.Module.get_function_with_keywords self "mahalanobis"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let score ?y ~x_test self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X_test", Some(x_test |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let location_ self =
  match Py.Object.get_attr_string self "location_" with
| None -> raise (Wrap_utils.Attribute_not_found "location_")
| Some x -> Ndarray.of_pyobject x
let covariance_ self =
  match Py.Object.get_attr_string self "covariance_" with
| None -> raise (Wrap_utils.Attribute_not_found "covariance_")
| Some x -> Wrap_utils.id x
let precision_ self =
  match Py.Object.get_attr_string self "precision_" with
| None -> raise (Wrap_utils.Attribute_not_found "precision_")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module GraphicalLasso = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?alpha ?mode ?tol ?enet_tol ?max_iter ?verbose ?assume_centered () =
                     Py.Module.get_function_with_keywords ns "GraphicalLasso"
                       [||]
                       (Wrap_utils.keyword_args [("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("mode", Wrap_utils.Option.map mode (function
| `Cd -> Py.String.of_string "cd"
| `Lars -> Py.String.of_string "lars"
)); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("enet_tol", Wrap_utils.Option.map enet_tol Py.Float.of_float); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("assume_centered", Wrap_utils.Option.map assume_centered Py.Bool.of_bool)])

let error_norm ?norm ?scaling ?squared ~comp_cov self =
   Py.Module.get_function_with_keywords self "error_norm"
     [||]
     (Wrap_utils.keyword_args [("norm", Wrap_utils.Option.map norm Py.String.of_string); ("scaling", Wrap_utils.Option.map scaling Py.Bool.of_bool); ("squared", Wrap_utils.Option.map squared Py.Bool.of_bool); ("comp_cov", Some(comp_cov |> Ndarray.to_pyobject))])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let get_precision self =
   Py.Module.get_function_with_keywords self "get_precision"
     [||]
     []
     |> Ndarray.of_pyobject
let mahalanobis ~x self =
   Py.Module.get_function_with_keywords self "mahalanobis"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let score ?y ~x_test self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X_test", Some(x_test |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let location_ self =
  match Py.Object.get_attr_string self "location_" with
| None -> raise (Wrap_utils.Attribute_not_found "location_")
| Some x -> Ndarray.of_pyobject x
let covariance_ self =
  match Py.Object.get_attr_string self "covariance_" with
| None -> raise (Wrap_utils.Attribute_not_found "covariance_")
| Some x -> Ndarray.of_pyobject x
let precision_ self =
  match Py.Object.get_attr_string self "precision_" with
| None -> raise (Wrap_utils.Attribute_not_found "precision_")
| Some x -> Ndarray.of_pyobject x
let n_iter_ self =
  match Py.Object.get_attr_string self "n_iter_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_iter_")
| Some x -> Py.Int.to_int x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module GraphicalLassoCV = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?alphas ?n_refinements ?cv ?tol ?enet_tol ?max_iter ?mode ?n_jobs ?verbose ?assume_centered () =
                     Py.Module.get_function_with_keywords ns "GraphicalLassoCV"
                       [||]
                       (Wrap_utils.keyword_args [("alphas", Wrap_utils.Option.map alphas (function
| `Int x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)); ("n_refinements", n_refinements); ("cv", Wrap_utils.Option.map cv (function
| `Int x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Ndarray x -> Ndarray.to_pyobject x
)); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("enet_tol", Wrap_utils.Option.map enet_tol Py.Float.of_float); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Cd -> Py.String.of_string "cd"
| `Lars -> Py.String.of_string "lars"
)); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("assume_centered", Wrap_utils.Option.map assume_centered Py.Bool.of_bool)])

let error_norm ?norm ?scaling ?squared ~comp_cov self =
   Py.Module.get_function_with_keywords self "error_norm"
     [||]
     (Wrap_utils.keyword_args [("norm", Wrap_utils.Option.map norm Py.String.of_string); ("scaling", Wrap_utils.Option.map scaling Py.Bool.of_bool); ("squared", Wrap_utils.Option.map squared Py.Bool.of_bool); ("comp_cov", Some(comp_cov |> Ndarray.to_pyobject))])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let get_precision self =
   Py.Module.get_function_with_keywords self "get_precision"
     [||]
     []
     |> Ndarray.of_pyobject
let mahalanobis ~x self =
   Py.Module.get_function_with_keywords self "mahalanobis"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let score ?y ~x_test self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X_test", Some(x_test |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let location_ self =
  match Py.Object.get_attr_string self "location_" with
| None -> raise (Wrap_utils.Attribute_not_found "location_")
| Some x -> Ndarray.of_pyobject x
let covariance_ self =
  match Py.Object.get_attr_string self "covariance_" with
| None -> raise (Wrap_utils.Attribute_not_found "covariance_")
| Some x -> Wrap_utils.id x
let precision_ self =
  match Py.Object.get_attr_string self "precision_" with
| None -> raise (Wrap_utils.Attribute_not_found "precision_")
| Some x -> Wrap_utils.id x
let alpha_ self =
  match Py.Object.get_attr_string self "alpha_" with
| None -> raise (Wrap_utils.Attribute_not_found "alpha_")
| Some x -> Py.Float.to_float x
let cv_alphas_ self =
  match Py.Object.get_attr_string self "cv_alphas_" with
| None -> raise (Wrap_utils.Attribute_not_found "cv_alphas_")
| Some x -> Wrap_utils.id x
let grid_scores_ self =
  match Py.Object.get_attr_string self "grid_scores_" with
| None -> raise (Wrap_utils.Attribute_not_found "grid_scores_")
| Some x -> Wrap_utils.id x
let n_iter_ self =
  match Py.Object.get_attr_string self "n_iter_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_iter_")
| Some x -> Py.Int.to_int x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LedoitWolf = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?store_precision ?assume_centered ?block_size () =
   Py.Module.get_function_with_keywords ns "LedoitWolf"
     [||]
     (Wrap_utils.keyword_args [("store_precision", Wrap_utils.Option.map store_precision Py.Bool.of_bool); ("assume_centered", Wrap_utils.Option.map assume_centered Py.Bool.of_bool); ("block_size", Wrap_utils.Option.map block_size Py.Int.of_int)])

let error_norm ?norm ?scaling ?squared ~comp_cov self =
   Py.Module.get_function_with_keywords self "error_norm"
     [||]
     (Wrap_utils.keyword_args [("norm", Wrap_utils.Option.map norm Py.String.of_string); ("scaling", Wrap_utils.Option.map scaling Py.Bool.of_bool); ("squared", Wrap_utils.Option.map squared Py.Bool.of_bool); ("comp_cov", Some(comp_cov |> Ndarray.to_pyobject))])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let get_precision self =
   Py.Module.get_function_with_keywords self "get_precision"
     [||]
     []
     |> Ndarray.of_pyobject
let mahalanobis ~x self =
   Py.Module.get_function_with_keywords self "mahalanobis"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let score ?y ~x_test self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X_test", Some(x_test |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let location_ self =
  match Py.Object.get_attr_string self "location_" with
| None -> raise (Wrap_utils.Attribute_not_found "location_")
| Some x -> Ndarray.of_pyobject x
let covariance_ self =
  match Py.Object.get_attr_string self "covariance_" with
| None -> raise (Wrap_utils.Attribute_not_found "covariance_")
| Some x -> Ndarray.of_pyobject x
let precision_ self =
  match Py.Object.get_attr_string self "precision_" with
| None -> raise (Wrap_utils.Attribute_not_found "precision_")
| Some x -> Ndarray.of_pyobject x
let shrinkage_ self =
  match Py.Object.get_attr_string self "shrinkage_" with
| None -> raise (Wrap_utils.Attribute_not_found "shrinkage_")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MinCovDet = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?store_precision ?assume_centered ?support_fraction ?random_state () =
                     Py.Module.get_function_with_keywords ns "MinCovDet"
                       [||]
                       (Wrap_utils.keyword_args [("store_precision", Wrap_utils.Option.map store_precision Py.Bool.of_bool); ("assume_centered", Wrap_utils.Option.map assume_centered Py.Bool.of_bool); ("support_fraction", Wrap_utils.Option.map support_fraction (function
| `Float x -> Py.Float.of_float x
| `PyObject x -> Wrap_utils.id x
)); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])

let correct_covariance ~data self =
   Py.Module.get_function_with_keywords self "correct_covariance"
     [||]
     (Wrap_utils.keyword_args [("data", Some(data |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let error_norm ?norm ?scaling ?squared ~comp_cov self =
   Py.Module.get_function_with_keywords self "error_norm"
     [||]
     (Wrap_utils.keyword_args [("norm", Wrap_utils.Option.map norm Py.String.of_string); ("scaling", Wrap_utils.Option.map scaling Py.Bool.of_bool); ("squared", Wrap_utils.Option.map squared Py.Bool.of_bool); ("comp_cov", Some(comp_cov |> Ndarray.to_pyobject))])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let get_precision self =
   Py.Module.get_function_with_keywords self "get_precision"
     [||]
     []
     |> Ndarray.of_pyobject
let mahalanobis ~x self =
   Py.Module.get_function_with_keywords self "mahalanobis"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let reweight_covariance ~data self =
   Py.Module.get_function_with_keywords self "reweight_covariance"
     [||]
     (Wrap_utils.keyword_args [("data", Some(data |> Ndarray.to_pyobject))])
     |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
let score ?y ~x_test self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X_test", Some(x_test |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let raw_location_ self =
  match Py.Object.get_attr_string self "raw_location_" with
| None -> raise (Wrap_utils.Attribute_not_found "raw_location_")
| Some x -> Ndarray.of_pyobject x
let raw_covariance_ self =
  match Py.Object.get_attr_string self "raw_covariance_" with
| None -> raise (Wrap_utils.Attribute_not_found "raw_covariance_")
| Some x -> Ndarray.of_pyobject x
let raw_support_ self =
  match Py.Object.get_attr_string self "raw_support_" with
| None -> raise (Wrap_utils.Attribute_not_found "raw_support_")
| Some x -> Ndarray.of_pyobject x
let location_ self =
  match Py.Object.get_attr_string self "location_" with
| None -> raise (Wrap_utils.Attribute_not_found "location_")
| Some x -> Ndarray.of_pyobject x
let covariance_ self =
  match Py.Object.get_attr_string self "covariance_" with
| None -> raise (Wrap_utils.Attribute_not_found "covariance_")
| Some x -> Ndarray.of_pyobject x
let precision_ self =
  match Py.Object.get_attr_string self "precision_" with
| None -> raise (Wrap_utils.Attribute_not_found "precision_")
| Some x -> Ndarray.of_pyobject x
let support_ self =
  match Py.Object.get_attr_string self "support_" with
| None -> raise (Wrap_utils.Attribute_not_found "support_")
| Some x -> Ndarray.of_pyobject x
let dist_ self =
  match Py.Object.get_attr_string self "dist_" with
| None -> raise (Wrap_utils.Attribute_not_found "dist_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module OAS = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?store_precision ?assume_centered () =
   Py.Module.get_function_with_keywords ns "OAS"
     [||]
     (Wrap_utils.keyword_args [("store_precision", Wrap_utils.Option.map store_precision Py.Bool.of_bool); ("assume_centered", Wrap_utils.Option.map assume_centered Py.Bool.of_bool)])

let error_norm ?norm ?scaling ?squared ~comp_cov self =
   Py.Module.get_function_with_keywords self "error_norm"
     [||]
     (Wrap_utils.keyword_args [("norm", Wrap_utils.Option.map norm Py.String.of_string); ("scaling", Wrap_utils.Option.map scaling Py.Bool.of_bool); ("squared", Wrap_utils.Option.map squared Py.Bool.of_bool); ("comp_cov", Some(comp_cov |> Ndarray.to_pyobject))])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let get_precision self =
   Py.Module.get_function_with_keywords self "get_precision"
     [||]
     []
     |> Ndarray.of_pyobject
let mahalanobis ~x self =
   Py.Module.get_function_with_keywords self "mahalanobis"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let score ?y ~x_test self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X_test", Some(x_test |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let covariance_ self =
  match Py.Object.get_attr_string self "covariance_" with
| None -> raise (Wrap_utils.Attribute_not_found "covariance_")
| Some x -> Ndarray.of_pyobject x
let precision_ self =
  match Py.Object.get_attr_string self "precision_" with
| None -> raise (Wrap_utils.Attribute_not_found "precision_")
| Some x -> Ndarray.of_pyobject x
let shrinkage_ self =
  match Py.Object.get_attr_string self "shrinkage_" with
| None -> raise (Wrap_utils.Attribute_not_found "shrinkage_")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ShrunkCovariance = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?store_precision ?assume_centered ?shrinkage () =
                     Py.Module.get_function_with_keywords ns "ShrunkCovariance"
                       [||]
                       (Wrap_utils.keyword_args [("store_precision", Wrap_utils.Option.map store_precision Py.Bool.of_bool); ("assume_centered", Wrap_utils.Option.map assume_centered Py.Bool.of_bool); ("shrinkage", Wrap_utils.Option.map shrinkage (function
| `Float x -> Py.Float.of_float x
| `PyObject x -> Wrap_utils.id x
))])

let error_norm ?norm ?scaling ?squared ~comp_cov self =
   Py.Module.get_function_with_keywords self "error_norm"
     [||]
     (Wrap_utils.keyword_args [("norm", Wrap_utils.Option.map norm Py.String.of_string); ("scaling", Wrap_utils.Option.map scaling Py.Bool.of_bool); ("squared", Wrap_utils.Option.map squared Py.Bool.of_bool); ("comp_cov", Some(comp_cov |> Ndarray.to_pyobject))])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let get_precision self =
   Py.Module.get_function_with_keywords self "get_precision"
     [||]
     []
     |> Ndarray.of_pyobject
let mahalanobis ~x self =
   Py.Module.get_function_with_keywords self "mahalanobis"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let score ?y ~x_test self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X_test", Some(x_test |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let location_ self =
  match Py.Object.get_attr_string self "location_" with
| None -> raise (Wrap_utils.Attribute_not_found "location_")
| Some x -> Ndarray.of_pyobject x
let covariance_ self =
  match Py.Object.get_attr_string self "covariance_" with
| None -> raise (Wrap_utils.Attribute_not_found "covariance_")
| Some x -> Ndarray.of_pyobject x
let precision_ self =
  match Py.Object.get_attr_string self "precision_" with
| None -> raise (Wrap_utils.Attribute_not_found "precision_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let empirical_covariance ?assume_centered ~x () =
   Py.Module.get_function_with_keywords ns "empirical_covariance"
     [||]
     (Wrap_utils.keyword_args [("assume_centered", Wrap_utils.Option.map assume_centered Py.Bool.of_bool); ("X", Some(x |> Ndarray.to_pyobject))])

                  let fast_mcd ?support_fraction ?cov_computation_method ?random_state ~x () =
                     Py.Module.get_function_with_keywords ns "fast_mcd"
                       [||]
                       (Wrap_utils.keyword_args [("support_fraction", Wrap_utils.Option.map support_fraction (function
| `Float x -> Py.Float.of_float x
| `PyObject x -> Wrap_utils.id x
)); ("cov_computation_method", cov_computation_method); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("X", Some(x |> Ndarray.to_pyobject))])
                       |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
                  let graphical_lasso ?cov_init ?mode ?tol ?enet_tol ?max_iter ?verbose ?return_costs ?eps ?return_n_iter ~emp_cov ~alpha () =
                     Py.Module.get_function_with_keywords ns "graphical_lasso"
                       [||]
                       (Wrap_utils.keyword_args [("cov_init", cov_init); ("mode", Wrap_utils.Option.map mode (function
| `Cd -> Py.String.of_string "cd"
| `Lars -> Py.String.of_string "lars"
)); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("enet_tol", Wrap_utils.Option.map enet_tol Py.Float.of_float); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("return_costs", Wrap_utils.Option.map return_costs Py.Bool.of_bool); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("return_n_iter", Wrap_utils.Option.map return_n_iter Py.Bool.of_bool); ("emp_cov", Some(emp_cov )); ("alpha", Some(alpha |> Py.Float.of_float))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), (Py.Int.to_int (Py.Tuple.get x 3))))
let ledoit_wolf ?assume_centered ?block_size ~x () =
   Py.Module.get_function_with_keywords ns "ledoit_wolf"
     [||]
     (Wrap_utils.keyword_args [("assume_centered", Wrap_utils.Option.map assume_centered Py.Bool.of_bool); ("block_size", Wrap_utils.Option.map block_size Py.Int.of_int); ("X", Some(x |> Ndarray.to_pyobject))])
     |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))
let ledoit_wolf_shrinkage ?assume_centered ?block_size ~x () =
   Py.Module.get_function_with_keywords ns "ledoit_wolf_shrinkage"
     [||]
     (Wrap_utils.keyword_args [("assume_centered", Wrap_utils.Option.map assume_centered Py.Bool.of_bool); ("block_size", Wrap_utils.Option.map block_size Py.Int.of_int); ("X", Some(x |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let log_likelihood ~emp_cov ~precision () =
   Py.Module.get_function_with_keywords ns "log_likelihood"
     [||]
     (Wrap_utils.keyword_args [("emp_cov", Some(emp_cov )); ("precision", Some(precision ))])

let oas ?assume_centered ~x () =
   Py.Module.get_function_with_keywords ns "oas"
     [||]
     (Wrap_utils.keyword_args [("assume_centered", Wrap_utils.Option.map assume_centered Py.Bool.of_bool); ("X", Some(x |> Ndarray.to_pyobject))])
     |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))
                  let shrunk_covariance ?shrinkage ~emp_cov () =
                     Py.Module.get_function_with_keywords ns "shrunk_covariance"
                       [||]
                       (Wrap_utils.keyword_args [("shrinkage", Wrap_utils.Option.map shrinkage (function
| `Float x -> Py.Float.of_float x
| `PyObject x -> Wrap_utils.id x
)); ("emp_cov", Some(emp_cov |> Ndarray.to_pyobject))])
                       |> Ndarray.of_pyobject
