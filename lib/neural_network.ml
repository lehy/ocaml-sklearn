let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.neural_network"

module BernoulliRBM = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_components ?learning_rate ?batch_size ?n_iter ?verbose ?random_state () =
                     Py.Module.get_function_with_keywords ns "BernoulliRBM"
                       [||]
                       (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("learning_rate", Wrap_utils.Option.map learning_rate Py.Float.of_float); ("batch_size", Wrap_utils.Option.map batch_size Py.Int.of_int); ("n_iter", Wrap_utils.Option.map n_iter Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
))])

                  let fit ?y ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let gibbs ~v self =
   Py.Module.get_function_with_keywords self "gibbs"
     [||]
     (Wrap_utils.keyword_args [("v", Some(v |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let partial_fit ?y ~x self =
   Py.Module.get_function_with_keywords self "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

                  let score_samples ~x self =
                     Py.Module.get_function_with_keywords self "score_samples"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

                  let transform ~x self =
                     Py.Module.get_function_with_keywords self "transform"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let intercept_hidden_ self =
  match Py.Object.get_attr_string self "intercept_hidden_" with
| None -> raise (Wrap_utils.Attribute_not_found "intercept_hidden_")
| Some x -> Ndarray.of_pyobject x
let intercept_visible_ self =
  match Py.Object.get_attr_string self "intercept_visible_" with
| None -> raise (Wrap_utils.Attribute_not_found "intercept_visible_")
| Some x -> Ndarray.of_pyobject x
let components_ self =
  match Py.Object.get_attr_string self "components_" with
| None -> raise (Wrap_utils.Attribute_not_found "components_")
| Some x -> Ndarray.of_pyobject x
let h_samples_ self =
  match Py.Object.get_attr_string self "h_samples_" with
| None -> raise (Wrap_utils.Attribute_not_found "h_samples_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MLPClassifier = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?hidden_layer_sizes ?activation ?solver ?alpha ?batch_size ?learning_rate ?learning_rate_init ?power_t ?max_iter ?shuffle ?random_state ?tol ?verbose ?warm_start ?momentum ?nesterovs_momentum ?early_stopping ?validation_fraction ?beta_1 ?beta_2 ?epsilon ?n_iter_no_change ?max_fun () =
                     Py.Module.get_function_with_keywords ns "MLPClassifier"
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
)); ("learning_rate_init", Wrap_utils.Option.map learning_rate_init Py.Float.of_float); ("power_t", Wrap_utils.Option.map power_t Py.Float.of_float); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("momentum", Wrap_utils.Option.map momentum Py.Float.of_float); ("nesterovs_momentum", Wrap_utils.Option.map nesterovs_momentum Py.Bool.of_bool); ("early_stopping", Wrap_utils.Option.map early_stopping Py.Bool.of_bool); ("validation_fraction", Wrap_utils.Option.map validation_fraction Py.Float.of_float); ("beta_1", Wrap_utils.Option.map beta_1 Py.Float.of_float); ("beta_2", Wrap_utils.Option.map beta_2 Py.Float.of_float); ("epsilon", Wrap_utils.Option.map epsilon Py.Float.of_float); ("n_iter_no_change", Wrap_utils.Option.map n_iter_no_change Py.Int.of_int); ("max_fun", Wrap_utils.Option.map max_fun Py.Int.of_int)])

                  let fit ~x ~y self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
))); ("y", Some(y |> Ndarray.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let partial_fit ?classes ~x ~y self =
   Py.Module.get_function_with_keywords self "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("classes", classes); ("X", Some(x )); ("y", Some(y ))])

                  let predict ~x self =
                     Py.Module.get_function_with_keywords self "predict"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let predict_log_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_log_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
                  let predict_proba ~x self =
                     Py.Module.get_function_with_keywords self "predict_proba"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let classes_ self =
  match Py.Object.get_attr_string self "classes_" with
| None -> raise (Wrap_utils.Attribute_not_found "classes_")
| Some x -> Ndarray.of_pyobject x
let loss_ self =
  match Py.Object.get_attr_string self "loss_" with
| None -> raise (Wrap_utils.Attribute_not_found "loss_")
| Some x -> Py.Float.to_float x
let coefs_ self =
  match Py.Object.get_attr_string self "coefs_" with
| None -> raise (Wrap_utils.Attribute_not_found "coefs_")
| Some x -> Wrap_utils.id x
let intercepts_ self =
  match Py.Object.get_attr_string self "intercepts_" with
| None -> raise (Wrap_utils.Attribute_not_found "intercepts_")
| Some x -> Wrap_utils.id x
let n_iter_ self =
  match Py.Object.get_attr_string self "n_iter_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_iter_")
| Some x -> Py.Int.to_int x
let n_layers_ self =
  match Py.Object.get_attr_string self "n_layers_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_layers_")
| Some x -> Py.Int.to_int x
let n_outputs_ self =
  match Py.Object.get_attr_string self "n_outputs_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_outputs_")
| Some x -> Py.Int.to_int x
let out_activation_ self =
  match Py.Object.get_attr_string self "out_activation_" with
| None -> raise (Wrap_utils.Attribute_not_found "out_activation_")
| Some x -> Py.String.to_string x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MLPRegressor = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?hidden_layer_sizes ?activation ?solver ?alpha ?batch_size ?learning_rate ?learning_rate_init ?power_t ?max_iter ?shuffle ?random_state ?tol ?verbose ?warm_start ?momentum ?nesterovs_momentum ?early_stopping ?validation_fraction ?beta_1 ?beta_2 ?epsilon ?n_iter_no_change ?max_fun () =
                     Py.Module.get_function_with_keywords ns "MLPRegressor"
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
)); ("learning_rate_init", Wrap_utils.Option.map learning_rate_init Py.Float.of_float); ("power_t", Wrap_utils.Option.map power_t Py.Float.of_float); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("momentum", Wrap_utils.Option.map momentum Py.Float.of_float); ("nesterovs_momentum", Wrap_utils.Option.map nesterovs_momentum Py.Bool.of_bool); ("early_stopping", Wrap_utils.Option.map early_stopping Py.Bool.of_bool); ("validation_fraction", Wrap_utils.Option.map validation_fraction Py.Float.of_float); ("beta_1", Wrap_utils.Option.map beta_1 Py.Float.of_float); ("beta_2", Wrap_utils.Option.map beta_2 Py.Float.of_float); ("epsilon", Wrap_utils.Option.map epsilon Py.Float.of_float); ("n_iter_no_change", Wrap_utils.Option.map n_iter_no_change Py.Int.of_int); ("max_fun", Wrap_utils.Option.map max_fun Py.Int.of_int)])

                  let fit ~x ~y self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
))); ("y", Some(y |> Ndarray.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let partial_fit ~x ~y self =
   Py.Module.get_function_with_keywords self "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("y", Some(y ))])

                  let predict ~x self =
                     Py.Module.get_function_with_keywords self "predict"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let loss_ self =
  match Py.Object.get_attr_string self "loss_" with
| None -> raise (Wrap_utils.Attribute_not_found "loss_")
| Some x -> Py.Float.to_float x
let coefs_ self =
  match Py.Object.get_attr_string self "coefs_" with
| None -> raise (Wrap_utils.Attribute_not_found "coefs_")
| Some x -> Wrap_utils.id x
let intercepts_ self =
  match Py.Object.get_attr_string self "intercepts_" with
| None -> raise (Wrap_utils.Attribute_not_found "intercepts_")
| Some x -> Wrap_utils.id x
let n_iter_ self =
  match Py.Object.get_attr_string self "n_iter_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_iter_")
| Some x -> Py.Int.to_int x
let n_layers_ self =
  match Py.Object.get_attr_string self "n_layers_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_layers_")
| Some x -> Py.Int.to_int x
let n_outputs_ self =
  match Py.Object.get_attr_string self "n_outputs_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_outputs_")
| Some x -> Py.Int.to_int x
let out_activation_ self =
  match Py.Object.get_attr_string self "out_activation_" with
| None -> raise (Wrap_utils.Attribute_not_found "out_activation_")
| Some x -> Py.String.to_string x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
