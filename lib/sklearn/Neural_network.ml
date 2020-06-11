let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.neural_network"

let get_py name = Py.Module.get __wrap_namespace name
module BernoulliRBM = struct
type tag = [`BernoulliRBM]
type t = [`BaseEstimator | `BernoulliRBM | `Object | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let create ?n_components ?learning_rate ?batch_size ?n_iter ?verbose ?random_state () =
   Py.Module.get_function_with_keywords __wrap_namespace "BernoulliRBM"
     [||]
     (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("learning_rate", Wrap_utils.Option.map learning_rate Py.Float.of_float); ("batch_size", Wrap_utils.Option.map batch_size Py.Int.of_int); ("n_iter", Wrap_utils.Option.map n_iter Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
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
let gibbs ~v self =
   Py.Module.get_function_with_keywords (to_pyobject self) "gibbs"
     [||]
     (Wrap_utils.keyword_args [("v", Some(v |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let partial_fit ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> of_pyobject
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
let transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

let intercept_hidden_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "intercept_hidden_" with
  | None -> failwith "attribute intercept_hidden_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let intercept_hidden_ self = match intercept_hidden_opt self with
  | None -> raise Not_found
  | Some x -> x

let intercept_visible_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "intercept_visible_" with
  | None -> failwith "attribute intercept_visible_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let intercept_visible_ self = match intercept_visible_opt self with
  | None -> raise Not_found
  | Some x -> x

let components_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "components_" with
  | None -> failwith "attribute components_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let components_ self = match components_opt self with
  | None -> raise Not_found
  | Some x -> x

let h_samples_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "h_samples_" with
  | None -> failwith "attribute h_samples_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let h_samples_ self = match h_samples_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MLPClassifier = struct
type tag = [`MLPClassifier]
type t = [`BaseEstimator | `BaseMultilayerPerceptron | `ClassifierMixin | `MLPClassifier | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_multilayer_perceptron x = (x :> [`BaseMultilayerPerceptron] Obj.t)
                  let create ?hidden_layer_sizes ?activation ?solver ?alpha ?batch_size ?learning_rate ?learning_rate_init ?power_t ?max_iter ?shuffle ?random_state ?tol ?verbose ?warm_start ?momentum ?nesterovs_momentum ?early_stopping ?validation_fraction ?beta_1 ?beta_2 ?epsilon ?n_iter_no_change ?max_fun () =
                     Py.Module.get_function_with_keywords __wrap_namespace "MLPClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("hidden_layer_sizes", hidden_layer_sizes); ("activation", Wrap_utils.Option.map activation (function
| `Identity -> Py.String.of_string "identity"
| `Logistic -> Py.String.of_string "logistic"
| `Tanh -> Py.String.of_string "tanh"
| `Relu -> Py.String.of_string "relu"
)); ("solver", Wrap_utils.Option.map solver (function
| `Lbfgs -> Py.String.of_string "lbfgs"
| `Sgd -> Py.String.of_string "sgd"
| `Adam -> Py.String.of_string "adam"
)); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("batch_size", Wrap_utils.Option.map batch_size Py.Int.of_int); ("learning_rate", Wrap_utils.Option.map learning_rate (function
| `Constant -> Py.String.of_string "constant"
| `Invscaling -> Py.String.of_string "invscaling"
| `Adaptive -> Py.String.of_string "adaptive"
)); ("learning_rate_init", Wrap_utils.Option.map learning_rate_init Py.Float.of_float); ("power_t", Wrap_utils.Option.map power_t Py.Float.of_float); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("momentum", Wrap_utils.Option.map momentum Py.Float.of_float); ("nesterovs_momentum", Wrap_utils.Option.map nesterovs_momentum Py.Bool.of_bool); ("early_stopping", Wrap_utils.Option.map early_stopping Py.Bool.of_bool); ("validation_fraction", Wrap_utils.Option.map validation_fraction Py.Float.of_float); ("beta_1", Wrap_utils.Option.map beta_1 Py.Float.of_float); ("beta_2", Wrap_utils.Option.map beta_2 Py.Float.of_float); ("epsilon", Wrap_utils.Option.map epsilon Py.Float.of_float); ("n_iter_no_change", Wrap_utils.Option.map n_iter_no_change Py.Int.of_int); ("max_fun", Wrap_utils.Option.map max_fun Py.Int.of_int)])
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
let partial_fit ?classes ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("classes", classes); ("X", Some(x )); ("y", Some(y ))])
     |> of_pyobject
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

let classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let loss_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "loss_" with
  | None -> failwith "attribute loss_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let loss_ self = match loss_opt self with
  | None -> raise Not_found
  | Some x -> x

let coefs_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "coefs_" with
  | None -> failwith "attribute coefs_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let coefs_ self = match coefs_opt self with
  | None -> raise Not_found
  | Some x -> x

let intercepts_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "intercepts_" with
  | None -> failwith "attribute intercepts_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let intercepts_ self = match intercepts_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_layers_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_layers_" with
  | None -> failwith "attribute n_layers_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_layers_ self = match n_layers_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_outputs_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_outputs_" with
  | None -> failwith "attribute n_outputs_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_outputs_ self = match n_outputs_opt self with
  | None -> raise Not_found
  | Some x -> x

let out_activation_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "out_activation_" with
  | None -> failwith "attribute out_activation_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.String.to_string x)

let out_activation_ self = match out_activation_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MLPRegressor = struct
type tag = [`MLPRegressor]
type t = [`BaseEstimator | `BaseMultilayerPerceptron | `MLPRegressor | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_multilayer_perceptron x = (x :> [`BaseMultilayerPerceptron] Obj.t)
                  let create ?hidden_layer_sizes ?activation ?solver ?alpha ?batch_size ?learning_rate ?learning_rate_init ?power_t ?max_iter ?shuffle ?random_state ?tol ?verbose ?warm_start ?momentum ?nesterovs_momentum ?early_stopping ?validation_fraction ?beta_1 ?beta_2 ?epsilon ?n_iter_no_change ?max_fun () =
                     Py.Module.get_function_with_keywords __wrap_namespace "MLPRegressor"
                       [||]
                       (Wrap_utils.keyword_args [("hidden_layer_sizes", hidden_layer_sizes); ("activation", Wrap_utils.Option.map activation (function
| `Identity -> Py.String.of_string "identity"
| `Logistic -> Py.String.of_string "logistic"
| `Tanh -> Py.String.of_string "tanh"
| `Relu -> Py.String.of_string "relu"
)); ("solver", Wrap_utils.Option.map solver (function
| `Lbfgs -> Py.String.of_string "lbfgs"
| `Sgd -> Py.String.of_string "sgd"
| `Adam -> Py.String.of_string "adam"
)); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("batch_size", Wrap_utils.Option.map batch_size Py.Int.of_int); ("learning_rate", Wrap_utils.Option.map learning_rate (function
| `Constant -> Py.String.of_string "constant"
| `Invscaling -> Py.String.of_string "invscaling"
| `Adaptive -> Py.String.of_string "adaptive"
)); ("learning_rate_init", Wrap_utils.Option.map learning_rate_init Py.Float.of_float); ("power_t", Wrap_utils.Option.map power_t Py.Float.of_float); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("momentum", Wrap_utils.Option.map momentum Py.Float.of_float); ("nesterovs_momentum", Wrap_utils.Option.map nesterovs_momentum Py.Bool.of_bool); ("early_stopping", Wrap_utils.Option.map early_stopping Py.Bool.of_bool); ("validation_fraction", Wrap_utils.Option.map validation_fraction Py.Float.of_float); ("beta_1", Wrap_utils.Option.map beta_1 Py.Float.of_float); ("beta_2", Wrap_utils.Option.map beta_2 Py.Float.of_float); ("epsilon", Wrap_utils.Option.map epsilon Py.Float.of_float); ("n_iter_no_change", Wrap_utils.Option.map n_iter_no_change Py.Int.of_int); ("max_fun", Wrap_utils.Option.map max_fun Py.Int.of_int)])
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
let partial_fit ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("y", Some(y ))])
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
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject

let loss_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "loss_" with
  | None -> failwith "attribute loss_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let loss_ self = match loss_opt self with
  | None -> raise Not_found
  | Some x -> x

let coefs_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "coefs_" with
  | None -> failwith "attribute coefs_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let coefs_ self = match coefs_opt self with
  | None -> raise Not_found
  | Some x -> x

let intercepts_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "intercepts_" with
  | None -> failwith "attribute intercepts_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let intercepts_ self = match intercepts_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_layers_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_layers_" with
  | None -> failwith "attribute n_layers_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_layers_ self = match n_layers_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_outputs_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_outputs_" with
  | None -> failwith "attribute n_outputs_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_outputs_ self = match n_outputs_opt self with
  | None -> raise Not_found
  | Some x -> x

let out_activation_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "out_activation_" with
  | None -> failwith "attribute out_activation_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.String.to_string x)

let out_activation_ self = match out_activation_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
