let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.decomposition"

let get_py name = Py.Module.get ns name
module DictionaryLearning = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_components ?alpha ?max_iter ?tol ?fit_algorithm ?transform_algorithm ?transform_n_nonzero_coefs ?transform_alpha ?n_jobs ?code_init ?dict_init ?verbose ?split_sign ?random_state ?positive_code ?positive_dict ?transform_max_iter () =
                     Py.Module.get_function_with_keywords ns "DictionaryLearning"
                       [||]
                       (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("fit_algorithm", Wrap_utils.Option.map fit_algorithm (function
| `Lars -> Py.String.of_string "lars"
| `Cd -> Py.String.of_string "cd"
)); ("transform_algorithm", Wrap_utils.Option.map transform_algorithm (function
| `Lasso_lars -> Py.String.of_string "lasso_lars"
| `Lasso_cd -> Py.String.of_string "lasso_cd"
| `Lars -> Py.String.of_string "lars"
| `Omp -> Py.String.of_string "omp"
| `Threshold -> Py.String.of_string "threshold"
)); ("transform_n_nonzero_coefs", Wrap_utils.Option.map transform_n_nonzero_coefs Py.Int.of_int); ("transform_alpha", Wrap_utils.Option.map transform_alpha Py.Float.of_float); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("code_init", Wrap_utils.Option.map code_init Arr.to_pyobject); ("dict_init", Wrap_utils.Option.map dict_init Arr.to_pyobject); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("split_sign", Wrap_utils.Option.map split_sign Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("positive_code", Wrap_utils.Option.map positive_code Py.Bool.of_bool); ("positive_dict", Wrap_utils.Option.map positive_dict Py.Bool.of_bool); ("transform_max_iter", Wrap_utils.Option.map transform_max_iter Py.Int.of_int)])

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
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let components_opt self =
  match Py.Object.get_attr_string self "components_" with
  | None -> failwith "attribute components_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let components_ self = match components_opt self with
  | None -> raise Not_found
  | Some x -> x

let error_opt self =
  match Py.Object.get_attr_string self "error_" with
  | None -> failwith "attribute error_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let error_ self = match error_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string self "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module FactorAnalysis = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_components ?tol ?copy ?max_iter ?noise_variance_init ?svd_method ?iterated_power ?random_state () =
                     Py.Module.get_function_with_keywords ns "FactorAnalysis"
                       [||]
                       (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("noise_variance_init", Wrap_utils.Option.map noise_variance_init Arr.to_pyobject); ("svd_method", Wrap_utils.Option.map svd_method (function
| `Lapack -> Py.String.of_string "lapack"
| `Randomized -> Py.String.of_string "randomized"
)); ("iterated_power", Wrap_utils.Option.map iterated_power Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Arr.of_pyobject
let get_covariance self =
   Py.Module.get_function_with_keywords self "get_covariance"
     [||]
     []
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let get_precision self =
   Py.Module.get_function_with_keywords self "get_precision"
     [||]
     []
     |> Arr.of_pyobject
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

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let components_opt self =
  match Py.Object.get_attr_string self "components_" with
  | None -> failwith "attribute components_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let components_ self = match components_opt self with
  | None -> raise Not_found
  | Some x -> x

let loglike_opt self =
  match Py.Object.get_attr_string self "loglike_" with
  | None -> failwith "attribute loglike_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let loglike_ self = match loglike_opt self with
  | None -> raise Not_found
  | Some x -> x

let noise_variance_opt self =
  match Py.Object.get_attr_string self "noise_variance_" with
  | None -> failwith "attribute noise_variance_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let noise_variance_ self = match noise_variance_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string self "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x

let mean_opt self =
  match Py.Object.get_attr_string self "mean_" with
  | None -> failwith "attribute mean_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let mean_ self = match mean_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module FastICA = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_components ?algorithm ?whiten ?fun_ ?fun_args ?max_iter ?tol ?w_init ?random_state () =
                     Py.Module.get_function_with_keywords ns "FastICA"
                       [||]
                       (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Parallel -> Py.String.of_string "parallel"
| `Deflation -> Py.String.of_string "deflation"
)); ("whiten", Wrap_utils.Option.map whiten Py.Bool.of_bool); ("fun", Wrap_utils.Option.map fun_ (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("fun_args", Wrap_utils.Option.map fun_args Dict.to_pyobject); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("w_init", w_init); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])

let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])
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
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ?copy ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let components_opt self =
  match Py.Object.get_attr_string self "components_" with
  | None -> failwith "attribute components_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let components_ self = match components_opt self with
  | None -> raise Not_found
  | Some x -> x

let mixing_opt self =
  match Py.Object.get_attr_string self "mixing_" with
  | None -> failwith "attribute mixing_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let mixing_ self = match mixing_opt self with
  | None -> raise Not_found
  | Some x -> x

let mean_opt self =
  match Py.Object.get_attr_string self "mean_" with
  | None -> failwith "attribute mean_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let mean_ self = match mean_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string self "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x

let whitening_opt self =
  match Py.Object.get_attr_string self "whitening_" with
  | None -> failwith "attribute whitening_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let whitening_ self = match whitening_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module IncrementalPCA = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?n_components ?whiten ?copy ?batch_size () =
   Py.Module.get_function_with_keywords ns "IncrementalPCA"
     [||]
     (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("whiten", Wrap_utils.Option.map whiten Py.Bool.of_bool); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("batch_size", Wrap_utils.Option.map batch_size Py.Int.of_int)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Arr.of_pyobject
let get_covariance self =
   Py.Module.get_function_with_keywords self "get_covariance"
     [||]
     []
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let get_precision self =
   Py.Module.get_function_with_keywords self "get_precision"
     [||]
     []
     |> Arr.of_pyobject
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])

let partial_fit ?y ?check_input ~x self =
   Py.Module.get_function_with_keywords self "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("check_input", check_input); ("X", Some(x |> Arr.to_pyobject))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let components_opt self =
  match Py.Object.get_attr_string self "components_" with
  | None -> failwith "attribute components_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let components_ self = match components_opt self with
  | None -> raise Not_found
  | Some x -> x

let explained_variance_opt self =
  match Py.Object.get_attr_string self "explained_variance_" with
  | None -> failwith "attribute explained_variance_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let explained_variance_ self = match explained_variance_opt self with
  | None -> raise Not_found
  | Some x -> x

let explained_variance_ratio_opt self =
  match Py.Object.get_attr_string self "explained_variance_ratio_" with
  | None -> failwith "attribute explained_variance_ratio_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let explained_variance_ratio_ self = match explained_variance_ratio_opt self with
  | None -> raise Not_found
  | Some x -> x

let singular_values_opt self =
  match Py.Object.get_attr_string self "singular_values_" with
  | None -> failwith "attribute singular_values_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let singular_values_ self = match singular_values_opt self with
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

let noise_variance_opt self =
  match Py.Object.get_attr_string self "noise_variance_" with
  | None -> failwith "attribute noise_variance_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let noise_variance_ self = match noise_variance_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_components_opt self =
  match Py.Object.get_attr_string self "n_components_" with
  | None -> failwith "attribute n_components_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_components_ self = match n_components_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_samples_seen_opt self =
  match Py.Object.get_attr_string self "n_samples_seen_" with
  | None -> failwith "attribute n_samples_seen_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_samples_seen_ self = match n_samples_seen_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module KernelPCA = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_components ?kernel ?gamma ?degree ?coef0 ?kernel_params ?alpha ?fit_inverse_transform ?eigen_solver ?tol ?max_iter ?remove_zero_eig ?random_state ?copy_X ?n_jobs () =
                     Py.Module.get_function_with_keywords ns "KernelPCA"
                       [||]
                       (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("kernel", Wrap_utils.Option.map kernel (function
| `Linear -> Py.String.of_string "linear"
| `Poly -> Py.String.of_string "poly"
| `Rbf -> Py.String.of_string "rbf"
| `Sigmoid -> Py.String.of_string "sigmoid"
| `Cosine -> Py.String.of_string "cosine"
| `Precomputed -> Py.String.of_string "precomputed"
)); ("gamma", Wrap_utils.Option.map gamma Py.Float.of_float); ("degree", Wrap_utils.Option.map degree Py.Int.of_int); ("coef0", Wrap_utils.Option.map coef0 Py.Float.of_float); ("kernel_params", Wrap_utils.Option.map kernel_params Dict.to_pyobject); ("alpha", Wrap_utils.Option.map alpha Py.Int.of_int); ("fit_inverse_transform", Wrap_utils.Option.map fit_inverse_transform Py.Bool.of_bool); ("eigen_solver", Wrap_utils.Option.map eigen_solver (function
| `Auto -> Py.String.of_string "auto"
| `Dense -> Py.String.of_string "dense"
| `Arpack -> Py.String.of_string "arpack"
)); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("remove_zero_eig", Wrap_utils.Option.map remove_zero_eig Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("copy_X", Wrap_utils.Option.map copy_X Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])

let fit_transform ?y ?params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))]) (match params with None -> [] | Some x -> x))
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let lambdas_opt self =
  match Py.Object.get_attr_string self "lambdas_" with
  | None -> failwith "attribute lambdas_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let lambdas_ self = match lambdas_opt self with
  | None -> raise Not_found
  | Some x -> x

let alphas_opt self =
  match Py.Object.get_attr_string self "alphas_" with
  | None -> failwith "attribute alphas_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let alphas_ self = match alphas_opt self with
  | None -> raise Not_found
  | Some x -> x

let dual_coef_opt self =
  match Py.Object.get_attr_string self "dual_coef_" with
  | None -> failwith "attribute dual_coef_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let dual_coef_ self = match dual_coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_transformed_fit_opt self =
  match Py.Object.get_attr_string self "X_transformed_fit_" with
  | None -> failwith "attribute X_transformed_fit_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let x_transformed_fit_ self = match x_transformed_fit_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_fit_opt self =
  match Py.Object.get_attr_string self "X_fit_" with
  | None -> failwith "attribute X_fit_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let x_fit_ self = match x_fit_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LatentDirichletAllocation = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_components ?doc_topic_prior ?topic_word_prior ?learning_method ?learning_decay ?learning_offset ?max_iter ?batch_size ?evaluate_every ?total_samples ?perp_tol ?mean_change_tol ?max_doc_update_iter ?n_jobs ?verbose ?random_state () =
                     Py.Module.get_function_with_keywords ns "LatentDirichletAllocation"
                       [||]
                       (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("doc_topic_prior", Wrap_utils.Option.map doc_topic_prior Py.Float.of_float); ("topic_word_prior", Wrap_utils.Option.map topic_word_prior Py.Float.of_float); ("learning_method", Wrap_utils.Option.map learning_method (function
| `Batch -> Py.String.of_string "batch"
| `Online -> Py.String.of_string "online"
)); ("learning_decay", Wrap_utils.Option.map learning_decay Py.Float.of_float); ("learning_offset", Wrap_utils.Option.map learning_offset Py.Float.of_float); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("batch_size", Wrap_utils.Option.map batch_size Py.Int.of_int); ("evaluate_every", Wrap_utils.Option.map evaluate_every Py.Int.of_int); ("total_samples", Wrap_utils.Option.map total_samples Py.Int.of_int); ("perp_tol", Wrap_utils.Option.map perp_tol Py.Float.of_float); ("mean_change_tol", Wrap_utils.Option.map mean_change_tol Py.Float.of_float); ("max_doc_update_iter", Wrap_utils.Option.map max_doc_update_iter Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])

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
let partial_fit ?y ~x self =
   Py.Module.get_function_with_keywords self "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])

let perplexity ?sub_sampling ~x self =
   Py.Module.get_function_with_keywords self "perplexity"
     [||]
     (Wrap_utils.keyword_args [("sub_sampling", Wrap_utils.Option.map sub_sampling Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])
     |> Py.Float.to_float
let score ?y ~x self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])
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

let components_opt self =
  match Py.Object.get_attr_string self "components_" with
  | None -> failwith "attribute components_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let components_ self = match components_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_batch_iter_opt self =
  match Py.Object.get_attr_string self "n_batch_iter_" with
  | None -> failwith "attribute n_batch_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_batch_iter_ self = match n_batch_iter_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string self "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x

let bound_opt self =
  match Py.Object.get_attr_string self "bound_" with
  | None -> failwith "attribute bound_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let bound_ self = match bound_opt self with
  | None -> raise Not_found
  | Some x -> x

let doc_topic_prior_opt self =
  match Py.Object.get_attr_string self "doc_topic_prior_" with
  | None -> failwith "attribute doc_topic_prior_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let doc_topic_prior_ self = match doc_topic_prior_opt self with
  | None -> raise Not_found
  | Some x -> x

let topic_word_prior_opt self =
  match Py.Object.get_attr_string self "topic_word_prior_" with
  | None -> failwith "attribute topic_word_prior_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let topic_word_prior_ self = match topic_word_prior_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module PCA = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_components ?copy ?whiten ?svd_solver ?tol ?iterated_power ?random_state () =
                     Py.Module.get_function_with_keywords ns "PCA"
                       [||]
                       (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
| `S x -> Py.String.of_string x
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("whiten", Wrap_utils.Option.map whiten Py.Bool.of_bool); ("svd_solver", Wrap_utils.Option.map svd_solver (function
| `Auto -> Py.String.of_string "auto"
| `Full -> Py.String.of_string "full"
| `Arpack -> Py.String.of_string "arpack"
| `Randomized -> Py.String.of_string "randomized"
)); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("iterated_power", Wrap_utils.Option.map iterated_power (function
| `I x -> Py.Int.of_int x
| `Auto -> Py.String.of_string "auto"
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])

let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let get_covariance self =
   Py.Module.get_function_with_keywords self "get_covariance"
     [||]
     []
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let get_precision self =
   Py.Module.get_function_with_keywords self "get_precision"
     [||]
     []
     |> Arr.of_pyobject
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])

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

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let components_opt self =
  match Py.Object.get_attr_string self "components_" with
  | None -> failwith "attribute components_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let components_ self = match components_opt self with
  | None -> raise Not_found
  | Some x -> x

let explained_variance_opt self =
  match Py.Object.get_attr_string self "explained_variance_" with
  | None -> failwith "attribute explained_variance_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let explained_variance_ self = match explained_variance_opt self with
  | None -> raise Not_found
  | Some x -> x

let explained_variance_ratio_opt self =
  match Py.Object.get_attr_string self "explained_variance_ratio_" with
  | None -> failwith "attribute explained_variance_ratio_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let explained_variance_ratio_ self = match explained_variance_ratio_opt self with
  | None -> raise Not_found
  | Some x -> x

let singular_values_opt self =
  match Py.Object.get_attr_string self "singular_values_" with
  | None -> failwith "attribute singular_values_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let singular_values_ self = match singular_values_opt self with
  | None -> raise Not_found
  | Some x -> x

let mean_opt self =
  match Py.Object.get_attr_string self "mean_" with
  | None -> failwith "attribute mean_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let mean_ self = match mean_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_components_opt self =
  match Py.Object.get_attr_string self "n_components_" with
  | None -> failwith "attribute n_components_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_components_ self = match n_components_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_features_opt self =
  match Py.Object.get_attr_string self "n_features_" with
  | None -> failwith "attribute n_features_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_features_ self = match n_features_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_samples_opt self =
  match Py.Object.get_attr_string self "n_samples_" with
  | None -> failwith "attribute n_samples_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_samples_ self = match n_samples_opt self with
  | None -> raise Not_found
  | Some x -> x

let noise_variance_opt self =
  match Py.Object.get_attr_string self "noise_variance_" with
  | None -> failwith "attribute noise_variance_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let noise_variance_ self = match noise_variance_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SparseCoder = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?transform_algorithm ?transform_n_nonzero_coefs ?transform_alpha ?split_sign ?n_jobs ?positive_code ?transform_max_iter ~dictionary () =
                     Py.Module.get_function_with_keywords ns "SparseCoder"
                       [||]
                       (Wrap_utils.keyword_args [("transform_algorithm", Wrap_utils.Option.map transform_algorithm (function
| `Lasso_lars -> Py.String.of_string "lasso_lars"
| `Lasso_cd -> Py.String.of_string "lasso_cd"
| `Lars -> Py.String.of_string "lars"
| `Omp -> Py.String.of_string "omp"
| `Threshold -> Py.String.of_string "threshold"
)); ("transform_n_nonzero_coefs", Wrap_utils.Option.map transform_n_nonzero_coefs Py.Int.of_int); ("transform_alpha", Wrap_utils.Option.map transform_alpha Py.Float.of_float); ("split_sign", Wrap_utils.Option.map split_sign Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("positive_code", Wrap_utils.Option.map positive_code Py.Bool.of_bool); ("transform_max_iter", Wrap_utils.Option.map transform_max_iter Py.Int.of_int); ("dictionary", Some(dictionary |> Arr.to_pyobject))])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x ))])

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
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let components_opt self =
  match Py.Object.get_attr_string self "components_" with
  | None -> failwith "attribute components_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let components_ self = match components_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SparsePCA = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_components ?alpha ?ridge_alpha ?max_iter ?tol ?method_ ?n_jobs ?u_init ?v_init ?verbose ?random_state ?normalize_components () =
                     Py.Module.get_function_with_keywords ns "SparsePCA"
                       [||]
                       (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("ridge_alpha", Wrap_utils.Option.map ridge_alpha Py.Float.of_float); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("method", Wrap_utils.Option.map method_ (function
| `Lars -> Py.String.of_string "lars"
| `Cd -> Py.String.of_string "cd"
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("U_init", Wrap_utils.Option.map u_init Arr.to_pyobject); ("V_init", Wrap_utils.Option.map v_init Arr.to_pyobject); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("normalize_components", Wrap_utils.Option.map normalize_components Py.String.of_string)])

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
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let components_opt self =
  match Py.Object.get_attr_string self "components_" with
  | None -> failwith "attribute components_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let components_ self = match components_opt self with
  | None -> raise Not_found
  | Some x -> x

let error_opt self =
  match Py.Object.get_attr_string self "error_" with
  | None -> failwith "attribute error_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let error_ self = match error_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string self "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x

let mean_opt self =
  match Py.Object.get_attr_string self "mean_" with
  | None -> failwith "attribute mean_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let mean_ self = match mean_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module TruncatedSVD = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?n_components ?algorithm ?n_iter ?random_state ?tol () =
   Py.Module.get_function_with_keywords ns "TruncatedSVD"
     [||]
     (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("algorithm", Wrap_utils.Option.map algorithm Py.String.of_string); ("n_iter", Wrap_utils.Option.map n_iter Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])

let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let components_opt self =
  match Py.Object.get_attr_string self "components_" with
  | None -> failwith "attribute components_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let components_ self = match components_opt self with
  | None -> raise Not_found
  | Some x -> x

let explained_variance_opt self =
  match Py.Object.get_attr_string self "explained_variance_" with
  | None -> failwith "attribute explained_variance_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let explained_variance_ self = match explained_variance_opt self with
  | None -> raise Not_found
  | Some x -> x

let explained_variance_ratio_opt self =
  match Py.Object.get_attr_string self "explained_variance_ratio_" with
  | None -> failwith "attribute explained_variance_ratio_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let explained_variance_ratio_ self = match explained_variance_ratio_opt self with
  | None -> raise Not_found
  | Some x -> x

let singular_values_opt self =
  match Py.Object.get_attr_string self "singular_values_" with
  | None -> failwith "attribute singular_values_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let singular_values_ self = match singular_values_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let dict_learning ?max_iter ?tol ?method_ ?n_jobs ?dict_init ?code_init ?callback ?verbose ?random_state ?return_n_iter ?positive_dict ?positive_code ?method_max_iter ~x ~n_components ~alpha () =
                     Py.Module.get_function_with_keywords ns "dict_learning"
                       [||]
                       (Wrap_utils.keyword_args [("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("method", Wrap_utils.Option.map method_ (function
| `Lars -> Py.String.of_string "lars"
| `Cd -> Py.String.of_string "cd"
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("dict_init", Wrap_utils.Option.map dict_init Arr.to_pyobject); ("code_init", Wrap_utils.Option.map code_init Arr.to_pyobject); ("callback", callback); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("return_n_iter", Wrap_utils.Option.map return_n_iter Py.Bool.of_bool); ("positive_dict", Wrap_utils.Option.map positive_dict Py.Bool.of_bool); ("positive_code", Wrap_utils.Option.map positive_code Py.Bool.of_bool); ("method_max_iter", Wrap_utils.Option.map method_max_iter Py.Int.of_int); ("X", Some(x |> Arr.to_pyobject)); ("n_components", Some(n_components |> Py.Int.of_int)); ("alpha", Some(alpha |> Py.Int.of_int))])
                       |> (fun x -> ((Arr.of_pyobject (Py.Tuple.get x 0)), (Arr.of_pyobject (Py.Tuple.get x 1)), (Arr.of_pyobject (Py.Tuple.get x 2)), (Py.Int.to_int (Py.Tuple.get x 3))))
                  let dict_learning_online ?n_components ?alpha ?n_iter ?return_code ?dict_init ?callback ?batch_size ?verbose ?shuffle ?n_jobs ?method_ ?iter_offset ?random_state ?return_inner_stats ?inner_stats ?return_n_iter ?positive_dict ?positive_code ?method_max_iter ~x () =
                     Py.Module.get_function_with_keywords ns "dict_learning_online"
                       [||]
                       (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("n_iter", Wrap_utils.Option.map n_iter Py.Int.of_int); ("return_code", Wrap_utils.Option.map return_code Py.Bool.of_bool); ("dict_init", Wrap_utils.Option.map dict_init Arr.to_pyobject); ("callback", callback); ("batch_size", Wrap_utils.Option.map batch_size Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("method", Wrap_utils.Option.map method_ (function
| `Lars -> Py.String.of_string "lars"
| `Cd -> Py.String.of_string "cd"
)); ("iter_offset", Wrap_utils.Option.map iter_offset Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("return_inner_stats", Wrap_utils.Option.map return_inner_stats Py.Bool.of_bool); ("inner_stats", inner_stats); ("return_n_iter", Wrap_utils.Option.map return_n_iter Py.Bool.of_bool); ("positive_dict", Wrap_utils.Option.map positive_dict Py.Bool.of_bool); ("positive_code", Wrap_utils.Option.map positive_code Py.Bool.of_bool); ("method_max_iter", Wrap_utils.Option.map method_max_iter Py.Int.of_int); ("X", Some(x |> Arr.to_pyobject))])
                       |> (fun x -> ((Arr.of_pyobject (Py.Tuple.get x 0)), (Arr.of_pyobject (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2))))
                  let fastica ?n_components ?algorithm ?whiten ?fun_ ?fun_args ?max_iter ?tol ?w_init ?random_state ?return_X_mean ?compute_sources ?return_n_iter ~x () =
                     Py.Module.get_function_with_keywords ns "fastica"
                       [||]
                       (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Parallel -> Py.String.of_string "parallel"
| `Deflation -> Py.String.of_string "deflation"
)); ("whiten", Wrap_utils.Option.map whiten Py.Bool.of_bool); ("fun", Wrap_utils.Option.map fun_ (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("fun_args", Wrap_utils.Option.map fun_args Dict.to_pyobject); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("w_init", w_init); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("return_X_mean", Wrap_utils.Option.map return_X_mean Py.Bool.of_bool); ("compute_sources", Wrap_utils.Option.map compute_sources Py.Bool.of_bool); ("return_n_iter", Wrap_utils.Option.map return_n_iter Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Arr.of_pyobject (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), (Arr.of_pyobject (Py.Tuple.get x 3)), (Py.Int.to_int (Py.Tuple.get x 4))))
                  let non_negative_factorization ?w ?h ?n_components ?init ?update_H ?solver ?beta_loss ?tol ?max_iter ?alpha ?l1_ratio ?regularization ?random_state ?verbose ?shuffle ~x () =
                     Py.Module.get_function_with_keywords ns "non_negative_factorization"
                       [||]
                       (Wrap_utils.keyword_args [("W", Wrap_utils.Option.map w Arr.to_pyobject); ("H", Wrap_utils.Option.map h Arr.to_pyobject); ("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("init", Wrap_utils.Option.map init (function
| `Random -> Py.String.of_string "random"
| `Nndsvd -> Py.String.of_string "nndsvd"
| `Nndsvda -> Py.String.of_string "nndsvda"
| `Nndsvdar -> Py.String.of_string "nndsvdar"
| `Custom -> Py.String.of_string "custom"
| `None -> Py.none
)); ("update_H", Wrap_utils.Option.map update_H Py.Bool.of_bool); ("solver", Wrap_utils.Option.map solver (function
| `Cd -> Py.String.of_string "cd"
| `Mu -> Py.String.of_string "mu"
)); ("beta_loss", Wrap_utils.Option.map beta_loss (function
| `F x -> Py.Float.of_float x
| `S x -> Py.String.of_string x
)); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("l1_ratio", Wrap_utils.Option.map l1_ratio Py.Float.of_float); ("regularization", Wrap_utils.Option.map regularization (function
| `Both -> Py.String.of_string "both"
| `Components -> Py.String.of_string "components"
| `Transformation -> Py.String.of_string "transformation"
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])
                       |> (fun x -> ((Arr.of_pyobject (Py.Tuple.get x 0)), (Arr.of_pyobject (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2))))
                  let randomized_svd ?n_oversamples ?n_iter ?power_iteration_normalizer ?transpose ?flip_sign ?random_state ~m ~n_components () =
                     Py.Module.get_function_with_keywords ns "randomized_svd"
                       [||]
                       (Wrap_utils.keyword_args [("n_oversamples", n_oversamples); ("n_iter", Wrap_utils.Option.map n_iter (function
| `I x -> Py.Int.of_int x
| `T_auto_ x -> Wrap_utils.id x
)); ("power_iteration_normalizer", Wrap_utils.Option.map power_iteration_normalizer (function
| `Auto -> Py.String.of_string "auto"
| `QR -> Py.String.of_string "QR"
| `LU -> Py.String.of_string "LU"
| `None -> Py.String.of_string "none"
)); ("transpose", Wrap_utils.Option.map transpose (function
| `Bool x -> Py.Bool.of_bool x
| `Auto -> Py.String.of_string "auto"
)); ("flip_sign", Wrap_utils.Option.map flip_sign (function
| `Bool x -> Py.Bool.of_bool x
| `T_True_by x -> Wrap_utils.id x
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("M", Some(m |> Arr.to_pyobject)); ("n_components", Some(n_components |> Py.Int.of_int))])

                  let sparse_encode ?gram ?cov ?algorithm ?n_nonzero_coefs ?alpha ?copy_cov ?init ?max_iter ?n_jobs ?check_input ?verbose ?positive ~x ~dictionary () =
                     Py.Module.get_function_with_keywords ns "sparse_encode"
                       [||]
                       (Wrap_utils.keyword_args [("gram", Wrap_utils.Option.map gram Arr.to_pyobject); ("cov", Wrap_utils.Option.map cov Arr.to_pyobject); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Lasso_lars -> Py.String.of_string "lasso_lars"
| `Lasso_cd -> Py.String.of_string "lasso_cd"
| `Lars -> Py.String.of_string "lars"
| `Omp -> Py.String.of_string "omp"
| `Threshold -> Py.String.of_string "threshold"
)); ("n_nonzero_coefs", Wrap_utils.Option.map n_nonzero_coefs (function
| `I x -> Py.Int.of_int x
| `T0_1_ x -> Wrap_utils.id x
)); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("copy_cov", Wrap_utils.Option.map copy_cov Py.Bool.of_bool); ("init", Wrap_utils.Option.map init Arr.to_pyobject); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("check_input", Wrap_utils.Option.map check_input Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("positive", Wrap_utils.Option.map positive Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject)); ("dictionary", Some(dictionary |> Arr.to_pyobject))])
                       |> Arr.of_pyobject
