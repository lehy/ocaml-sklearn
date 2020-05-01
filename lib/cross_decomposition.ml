let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.cross_decomposition"

let get_py name = Py.Module.get ns name
module CCA = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?n_components ?scale ?max_iter ?tol ?copy () =
   Py.Module.get_function_with_keywords ns "CCA"
     [||]
     (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("scale", Wrap_utils.Option.map scale Py.Bool.of_bool); ("max_iter", max_iter); ("tol", tol); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let fit ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject)); ("Y", Some(y |> Arr.to_pyobject))])

let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))])
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
let predict ?copy ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])
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

let transform ?y ?copy ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Arr.to_pyobject); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let x_weights_opt self =
  match Py.Object.get_attr_string self "x_weights_" with
  | None -> failwith "attribute x_weights_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let x_weights_ self = match x_weights_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_weights_opt self =
  match Py.Object.get_attr_string self "y_weights_" with
  | None -> failwith "attribute y_weights_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let y_weights_ self = match y_weights_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_loadings_opt self =
  match Py.Object.get_attr_string self "x_loadings_" with
  | None -> failwith "attribute x_loadings_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let x_loadings_ self = match x_loadings_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_loadings_opt self =
  match Py.Object.get_attr_string self "y_loadings_" with
  | None -> failwith "attribute y_loadings_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let y_loadings_ self = match y_loadings_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_scores_opt self =
  match Py.Object.get_attr_string self "x_scores_" with
  | None -> failwith "attribute x_scores_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let x_scores_ self = match x_scores_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_scores_opt self =
  match Py.Object.get_attr_string self "y_scores_" with
  | None -> failwith "attribute y_scores_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let y_scores_ self = match y_scores_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_rotations_opt self =
  match Py.Object.get_attr_string self "x_rotations_" with
  | None -> failwith "attribute x_rotations_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let x_rotations_ self = match x_rotations_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_rotations_opt self =
  match Py.Object.get_attr_string self "y_rotations_" with
  | None -> failwith "attribute y_rotations_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let y_rotations_ self = match y_rotations_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string self "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module PLSCanonical = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_components ?scale ?algorithm ?max_iter ?tol ?copy () =
                     Py.Module.get_function_with_keywords ns "PLSCanonical"
                       [||]
                       (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("scale", Wrap_utils.Option.map scale Py.Bool.of_bool); ("algorithm", Wrap_utils.Option.map algorithm (function
| `Nipals -> Py.String.of_string "nipals"
| `Svd -> Py.String.of_string "svd"
)); ("max_iter", max_iter); ("tol", tol); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let fit ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject)); ("Y", Some(y |> Arr.to_pyobject))])

let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))])
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
let predict ?copy ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])
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

let transform ?y ?copy ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Arr.to_pyobject); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let x_weights_opt self =
  match Py.Object.get_attr_string self "x_weights_" with
  | None -> failwith "attribute x_weights_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let x_weights_ self = match x_weights_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_weights_opt self =
  match Py.Object.get_attr_string self "y_weights_" with
  | None -> failwith "attribute y_weights_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let y_weights_ self = match y_weights_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_loadings_opt self =
  match Py.Object.get_attr_string self "x_loadings_" with
  | None -> failwith "attribute x_loadings_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let x_loadings_ self = match x_loadings_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_loadings_opt self =
  match Py.Object.get_attr_string self "y_loadings_" with
  | None -> failwith "attribute y_loadings_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let y_loadings_ self = match y_loadings_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_scores_opt self =
  match Py.Object.get_attr_string self "x_scores_" with
  | None -> failwith "attribute x_scores_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let x_scores_ self = match x_scores_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_scores_opt self =
  match Py.Object.get_attr_string self "y_scores_" with
  | None -> failwith "attribute y_scores_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let y_scores_ self = match y_scores_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_rotations_opt self =
  match Py.Object.get_attr_string self "x_rotations_" with
  | None -> failwith "attribute x_rotations_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let x_rotations_ self = match x_rotations_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_rotations_opt self =
  match Py.Object.get_attr_string self "y_rotations_" with
  | None -> failwith "attribute y_rotations_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let y_rotations_ self = match y_rotations_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string self "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module PLSRegression = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?n_components ?scale ?max_iter ?tol ?copy () =
   Py.Module.get_function_with_keywords ns "PLSRegression"
     [||]
     (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("scale", Wrap_utils.Option.map scale Py.Bool.of_bool); ("max_iter", max_iter); ("tol", tol); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let fit ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject)); ("Y", Some(y |> Arr.to_pyobject))])

let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))])
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
let predict ?copy ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])
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

let transform ?y ?copy ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Arr.to_pyobject); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let x_weights_opt self =
  match Py.Object.get_attr_string self "x_weights_" with
  | None -> failwith "attribute x_weights_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let x_weights_ self = match x_weights_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_weights_opt self =
  match Py.Object.get_attr_string self "y_weights_" with
  | None -> failwith "attribute y_weights_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let y_weights_ self = match y_weights_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_loadings_opt self =
  match Py.Object.get_attr_string self "x_loadings_" with
  | None -> failwith "attribute x_loadings_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let x_loadings_ self = match x_loadings_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_loadings_opt self =
  match Py.Object.get_attr_string self "y_loadings_" with
  | None -> failwith "attribute y_loadings_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let y_loadings_ self = match y_loadings_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_scores_opt self =
  match Py.Object.get_attr_string self "x_scores_" with
  | None -> failwith "attribute x_scores_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let x_scores_ self = match x_scores_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_scores_opt self =
  match Py.Object.get_attr_string self "y_scores_" with
  | None -> failwith "attribute y_scores_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let y_scores_ self = match y_scores_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_rotations_opt self =
  match Py.Object.get_attr_string self "x_rotations_" with
  | None -> failwith "attribute x_rotations_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let x_rotations_ self = match x_rotations_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_rotations_opt self =
  match Py.Object.get_attr_string self "y_rotations_" with
  | None -> failwith "attribute y_rotations_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let y_rotations_ self = match y_rotations_opt self with
  | None -> raise Not_found
  | Some x -> x

let coef_opt self =
  match Py.Object.get_attr_string self "coef_" with
  | None -> failwith "attribute coef_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let coef_ self = match coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string self "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module PLSSVD = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?n_components ?scale ?copy () =
   Py.Module.get_function_with_keywords ns "PLSSVD"
     [||]
     (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("scale", Wrap_utils.Option.map scale Py.Bool.of_bool); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let fit ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject)); ("Y", Some(y |> Arr.to_pyobject))])

let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))])
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

let transform ?y ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let x_weights_opt self =
  match Py.Object.get_attr_string self "x_weights_" with
  | None -> failwith "attribute x_weights_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let x_weights_ self = match x_weights_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_weights_opt self =
  match Py.Object.get_attr_string self "y_weights_" with
  | None -> failwith "attribute y_weights_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let y_weights_ self = match y_weights_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_scores_opt self =
  match Py.Object.get_attr_string self "x_scores_" with
  | None -> failwith "attribute x_scores_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let x_scores_ self = match x_scores_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_scores_opt self =
  match Py.Object.get_attr_string self "y_scores_" with
  | None -> failwith "attribute y_scores_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let y_scores_ self = match y_scores_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
