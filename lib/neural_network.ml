let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.neural_network"

let get_py name = Py.Module.get ns name
module BernoulliRBM = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?n_components ?learning_rate ?batch_size ?n_iter ?verbose ?random_state () =
   Py.Module.get_function_with_keywords ns "BernoulliRBM"
     [||]
     (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("learning_rate", Wrap_utils.Option.map learning_rate Py.Float.of_float); ("batch_size", Wrap_utils.Option.map batch_size Py.Int.of_int); ("n_iter", Wrap_utils.Option.map n_iter Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])

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
let gibbs ~v self =
   Py.Module.get_function_with_keywords self "gibbs"
     [||]
     (Wrap_utils.keyword_args [("v", Some(v |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let partial_fit ?y ~x self =
   Py.Module.get_function_with_keywords self "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])

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

let intercept_hidden_opt self =
  match Py.Object.get_attr_string self "intercept_hidden_" with
  | None -> failwith "attribute intercept_hidden_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let intercept_hidden_ self = match intercept_hidden_opt self with
  | None -> raise Not_found
  | Some x -> x

let intercept_visible_opt self =
  match Py.Object.get_attr_string self "intercept_visible_" with
  | None -> failwith "attribute intercept_visible_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let intercept_visible_ self = match intercept_visible_opt self with
  | None -> raise Not_found
  | Some x -> x

let components_opt self =
  match Py.Object.get_attr_string self "components_" with
  | None -> failwith "attribute components_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let components_ self = match components_opt self with
  | None -> raise Not_found
  | Some x -> x

let h_samples_opt self =
  match Py.Object.get_attr_string self "h_samples_" with
  | None -> failwith "attribute h_samples_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let h_samples_ self = match h_samples_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
