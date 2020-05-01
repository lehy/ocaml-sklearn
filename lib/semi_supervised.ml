let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.semi_supervised"

let get_py name = Py.Module.get ns name
module LabelPropagation = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?kernel ?gamma ?n_neighbors ?max_iter ?tol ?n_jobs () =
                     Py.Module.get_function_with_keywords ns "LabelPropagation"
                       [||]
                       (Wrap_utils.keyword_args [("kernel", Wrap_utils.Option.map kernel (function
| `Knn -> Py.String.of_string "knn"
| `Rbf -> Py.String.of_string "rbf"
| `Callable x -> Wrap_utils.id x
)); ("gamma", Wrap_utils.Option.map gamma Py.Float.of_float); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int)])

let fit ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("y", Some(y ))])

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
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)


let x_opt self =
  match Py.Object.get_attr_string self "X_" with
  | None -> failwith "attribute X_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let x_ self = match x_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string self "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let label_distributions_opt self =
  match Py.Object.get_attr_string self "label_distributions_" with
  | None -> failwith "attribute label_distributions_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let label_distributions_ self = match label_distributions_opt self with
  | None -> raise Not_found
  | Some x -> x

let transduction_opt self =
  match Py.Object.get_attr_string self "transduction_" with
  | None -> failwith "attribute transduction_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let transduction_ self = match transduction_opt self with
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
module LabelSpreading = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?kernel ?gamma ?n_neighbors ?alpha ?max_iter ?tol ?n_jobs () =
                     Py.Module.get_function_with_keywords ns "LabelSpreading"
                       [||]
                       (Wrap_utils.keyword_args [("kernel", Wrap_utils.Option.map kernel (function
| `Knn -> Py.String.of_string "knn"
| `Rbf -> Py.String.of_string "rbf"
| `Callable x -> Wrap_utils.id x
)); ("gamma", Wrap_utils.Option.map gamma Py.Float.of_float); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int)])

let fit ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

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
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)


let x_opt self =
  match Py.Object.get_attr_string self "X_" with
  | None -> failwith "attribute X_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let x_ self = match x_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string self "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let label_distributions_opt self =
  match Py.Object.get_attr_string self "label_distributions_" with
  | None -> failwith "attribute label_distributions_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let label_distributions_ self = match label_distributions_opt self with
  | None -> raise Not_found
  | Some x -> x

let transduction_opt self =
  match Py.Object.get_attr_string self "transduction_" with
  | None -> failwith "attribute transduction_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let transduction_ self = match transduction_opt self with
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
