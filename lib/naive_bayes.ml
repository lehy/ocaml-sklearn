let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.naive_bayes"

let get_py name = Py.Module.get ns name
module ABCMeta = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?kwargs ~name ~bases ~namespace () =
   Py.Module.get_function_with_keywords ns "ABCMeta"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("name", Some(name )); ("bases", Some(bases )); ("namespace", Some(namespace ))]) (match kwargs with None -> [] | Some x -> x))

let mro self =
   Py.Module.get_function_with_keywords self "mro"
     [||]
     []

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
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
module BernoulliNB = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?alpha ?binarize ?fit_prior ?class_prior () =
                     Py.Module.get_function_with_keywords ns "BernoulliNB"
                       [||]
                       (Wrap_utils.keyword_args [("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("binarize", Wrap_utils.Option.map binarize (function
| `F x -> Py.Float.of_float x
| `None -> Py.none
)); ("fit_prior", Wrap_utils.Option.map fit_prior Py.Bool.of_bool); ("class_prior", Wrap_utils.Option.map class_prior (function
| `Arr x -> Arr.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))])

let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let partial_fit ?classes ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("classes", Wrap_utils.Option.map classes Arr.to_pyobject); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

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


let class_count_opt self =
  match Py.Object.get_attr_string self "class_count_" with
  | None -> failwith "attribute class_count_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let class_count_ self = match class_count_opt self with
  | None -> raise Not_found
  | Some x -> x

let class_log_prior_opt self =
  match Py.Object.get_attr_string self "class_log_prior_" with
  | None -> failwith "attribute class_log_prior_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let class_log_prior_ self = match class_log_prior_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string self "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let feature_count_opt self =
  match Py.Object.get_attr_string self "feature_count_" with
  | None -> failwith "attribute feature_count_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let feature_count_ self = match feature_count_opt self with
  | None -> raise Not_found
  | Some x -> x

let feature_log_prob_opt self =
  match Py.Object.get_attr_string self "feature_log_prob_" with
  | None -> failwith "attribute feature_log_prob_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let feature_log_prob_ self = match feature_log_prob_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_features_opt self =
  match Py.Object.get_attr_string self "n_features_" with
  | None -> failwith "attribute n_features_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_features_ self = match n_features_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module CategoricalNB = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?alpha ?fit_prior ?class_prior () =
                     Py.Module.get_function_with_keywords ns "CategoricalNB"
                       [||]
                       (Wrap_utils.keyword_args [("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("fit_prior", Wrap_utils.Option.map fit_prior Py.Bool.of_bool); ("class_prior", Wrap_utils.Option.map class_prior (function
| `Arr x -> Arr.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))])

let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let partial_fit ?classes ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("classes", Wrap_utils.Option.map classes Arr.to_pyobject); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

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


let category_count_opt self =
  match Py.Object.get_attr_string self "category_count_" with
  | None -> failwith "attribute category_count_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let category_count_ self = match category_count_opt self with
  | None -> raise Not_found
  | Some x -> x

let class_count_opt self =
  match Py.Object.get_attr_string self "class_count_" with
  | None -> failwith "attribute class_count_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let class_count_ self = match class_count_opt self with
  | None -> raise Not_found
  | Some x -> x

let class_log_prior_opt self =
  match Py.Object.get_attr_string self "class_log_prior_" with
  | None -> failwith "attribute class_log_prior_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let class_log_prior_ self = match class_log_prior_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string self "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let feature_log_prob_opt self =
  match Py.Object.get_attr_string self "feature_log_prob_" with
  | None -> failwith "attribute feature_log_prob_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let feature_log_prob_ self = match feature_log_prob_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_features_opt self =
  match Py.Object.get_attr_string self "n_features_" with
  | None -> failwith "attribute n_features_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_features_ self = match n_features_opt self with
  | None -> raise Not_found
  | Some x -> x
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
module ComplementNB = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?alpha ?fit_prior ?class_prior ?norm () =
                     Py.Module.get_function_with_keywords ns "ComplementNB"
                       [||]
                       (Wrap_utils.keyword_args [("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("fit_prior", Wrap_utils.Option.map fit_prior Py.Bool.of_bool); ("class_prior", Wrap_utils.Option.map class_prior (function
| `Arr x -> Arr.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("norm", Wrap_utils.Option.map norm Py.Bool.of_bool)])

let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let partial_fit ?classes ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("classes", Wrap_utils.Option.map classes Arr.to_pyobject); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

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


let class_count_opt self =
  match Py.Object.get_attr_string self "class_count_" with
  | None -> failwith "attribute class_count_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let class_count_ self = match class_count_opt self with
  | None -> raise Not_found
  | Some x -> x

let class_log_prior_opt self =
  match Py.Object.get_attr_string self "class_log_prior_" with
  | None -> failwith "attribute class_log_prior_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let class_log_prior_ self = match class_log_prior_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string self "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let feature_all_opt self =
  match Py.Object.get_attr_string self "feature_all_" with
  | None -> failwith "attribute feature_all_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let feature_all_ self = match feature_all_opt self with
  | None -> raise Not_found
  | Some x -> x

let feature_count_opt self =
  match Py.Object.get_attr_string self "feature_count_" with
  | None -> failwith "attribute feature_count_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let feature_count_ self = match feature_count_opt self with
  | None -> raise Not_found
  | Some x -> x

let feature_log_prob_opt self =
  match Py.Object.get_attr_string self "feature_log_prob_" with
  | None -> failwith "attribute feature_log_prob_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let feature_log_prob_ self = match feature_log_prob_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_features_opt self =
  match Py.Object.get_attr_string self "n_features_" with
  | None -> failwith "attribute n_features_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_features_ self = match n_features_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module GaussianNB = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?priors ?var_smoothing () =
   Py.Module.get_function_with_keywords ns "GaussianNB"
     [||]
     (Wrap_utils.keyword_args [("priors", Wrap_utils.Option.map priors Arr.to_pyobject); ("var_smoothing", Wrap_utils.Option.map var_smoothing Py.Float.of_float)])

let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let partial_fit ?classes ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("classes", Wrap_utils.Option.map classes Arr.to_pyobject); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

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


let class_count_opt self =
  match Py.Object.get_attr_string self "class_count_" with
  | None -> failwith "attribute class_count_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let class_count_ self = match class_count_opt self with
  | None -> raise Not_found
  | Some x -> x

let class_prior_opt self =
  match Py.Object.get_attr_string self "class_prior_" with
  | None -> failwith "attribute class_prior_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let class_prior_ self = match class_prior_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string self "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let epsilon_opt self =
  match Py.Object.get_attr_string self "epsilon_" with
  | None -> failwith "attribute epsilon_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let epsilon_ self = match epsilon_opt self with
  | None -> raise Not_found
  | Some x -> x

let sigma_opt self =
  match Py.Object.get_attr_string self "sigma_" with
  | None -> failwith "attribute sigma_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let sigma_ self = match sigma_opt self with
  | None -> raise Not_found
  | Some x -> x

let theta_opt self =
  match Py.Object.get_attr_string self "theta_" with
  | None -> failwith "attribute theta_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let theta_ self = match theta_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LabelBinarizer = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?neg_label ?pos_label ?sparse_output () =
   Py.Module.get_function_with_keywords ns "LabelBinarizer"
     [||]
     (Wrap_utils.keyword_args [("neg_label", Wrap_utils.Option.map neg_label Py.Int.of_int); ("pos_label", Wrap_utils.Option.map pos_label Py.Int.of_int); ("sparse_output", Wrap_utils.Option.map sparse_output Py.Bool.of_bool)])

let fit ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Arr.to_pyobject))])

let fit_transform ~y self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let inverse_transform ?threshold ~y self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("threshold", Wrap_utils.Option.map threshold Py.Float.of_float); ("Y", Some(y |> Arr.to_pyobject))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~y self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let classes_opt self =
  match Py.Object.get_attr_string self "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_type_opt self =
  match Py.Object.get_attr_string self "y_type_" with
  | None -> failwith "attribute y_type_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.String.to_string x)

let y_type_ self = match y_type_opt self with
  | None -> raise Not_found
  | Some x -> x

let sparse_input_opt self =
  match Py.Object.get_attr_string self "sparse_input_" with
  | None -> failwith "attribute sparse_input_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Bool.to_bool x)

let sparse_input_ self = match sparse_input_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MultinomialNB = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?alpha ?fit_prior ?class_prior () =
                     Py.Module.get_function_with_keywords ns "MultinomialNB"
                       [||]
                       (Wrap_utils.keyword_args [("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("fit_prior", Wrap_utils.Option.map fit_prior Py.Bool.of_bool); ("class_prior", Wrap_utils.Option.map class_prior (function
| `Arr x -> Arr.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))])

let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let partial_fit ?classes ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("classes", Wrap_utils.Option.map classes Arr.to_pyobject); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

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


let class_count_opt self =
  match Py.Object.get_attr_string self "class_count_" with
  | None -> failwith "attribute class_count_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let class_count_ self = match class_count_opt self with
  | None -> raise Not_found
  | Some x -> x

let class_log_prior_opt self =
  match Py.Object.get_attr_string self "class_log_prior_" with
  | None -> failwith "attribute class_log_prior_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let class_log_prior_ self = match class_log_prior_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string self "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let coef_opt self =
  match Py.Object.get_attr_string self "coef_" with
  | None -> failwith "attribute coef_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let coef_ self = match coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let feature_count_opt self =
  match Py.Object.get_attr_string self "feature_count_" with
  | None -> failwith "attribute feature_count_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let feature_count_ self = match feature_count_opt self with
  | None -> raise Not_found
  | Some x -> x

let feature_log_prob_opt self =
  match Py.Object.get_attr_string self "feature_log_prob_" with
  | None -> failwith "attribute feature_log_prob_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let feature_log_prob_ self = match feature_log_prob_opt self with
  | None -> raise Not_found
  | Some x -> x

let intercept_opt self =
  match Py.Object.get_attr_string self "intercept_" with
  | None -> failwith "attribute intercept_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let intercept_ self = match intercept_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_features_opt self =
  match Py.Object.get_attr_string self "n_features_" with
  | None -> failwith "attribute n_features_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_features_ self = match n_features_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let abstractmethod ~funcobj () =
   Py.Module.get_function_with_keywords ns "abstractmethod"
     [||]
     (Wrap_utils.keyword_args [("funcobj", Some(funcobj ))])

                  let binarize ?threshold ?copy ~x () =
                     Py.Module.get_function_with_keywords ns "binarize"
                       [||]
                       (Wrap_utils.keyword_args [("threshold", Wrap_utils.Option.map threshold (function
| `F x -> Py.Float.of_float x
| `T_0_0_by x -> Wrap_utils.id x
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])

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

let check_non_negative ~x ~whom () =
   Py.Module.get_function_with_keywords ns "check_non_negative"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject)); ("whom", Some(whom |> Py.String.of_string))])

let column_or_1d ?warn ~y () =
   Py.Module.get_function_with_keywords ns "column_or_1d"
     [||]
     (Wrap_utils.keyword_args [("warn", Wrap_utils.Option.map warn Py.Bool.of_bool); ("y", Some(y |> Arr.to_pyobject))])
     |> Arr.of_pyobject
module Deprecated = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?extra () =
   Py.Module.get_function_with_keywords ns "deprecated"
     [||]
     (Wrap_utils.keyword_args [("extra", Wrap_utils.Option.map extra Py.String.of_string)])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let label_binarize ?neg_label ?pos_label ?sparse_output ~y ~classes () =
   Py.Module.get_function_with_keywords ns "label_binarize"
     [||]
     (Wrap_utils.keyword_args [("neg_label", Wrap_utils.Option.map neg_label Py.Int.of_int); ("pos_label", Wrap_utils.Option.map pos_label Py.Int.of_int); ("sparse_output", Wrap_utils.Option.map sparse_output Py.Bool.of_bool); ("y", Some(y |> Arr.to_pyobject)); ("classes", Some(classes |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let logsumexp ?axis ?b ?keepdims ?return_sign ~a () =
   Py.Module.get_function_with_keywords ns "logsumexp"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("b", Wrap_utils.Option.map b Arr.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("return_sign", Wrap_utils.Option.map return_sign Py.Bool.of_bool); ("a", Some(a |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let safe_sparse_dot ?dense_output ~a ~b () =
   Py.Module.get_function_with_keywords ns "safe_sparse_dot"
     [||]
     (Wrap_utils.keyword_args [("dense_output", dense_output); ("a", Some(a |> Arr.to_pyobject)); ("b", Some(b ))])
     |> Arr.of_pyobject
