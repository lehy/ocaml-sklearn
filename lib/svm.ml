let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.svm"

let get_py name = Py.Module.get ns name
module LinearSVC = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?penalty ?loss ?dual ?tol ?c ?multi_class ?fit_intercept ?intercept_scaling ?class_weight ?verbose ?random_state ?max_iter () =
                     Py.Module.get_function_with_keywords ns "LinearSVC"
                       [||]
                       (Wrap_utils.keyword_args [("penalty", Wrap_utils.Option.map penalty (function
| `L1 -> Py.String.of_string "l1"
| `L2 -> Py.String.of_string "l2"
)); ("loss", Wrap_utils.Option.map loss (function
| `Hinge -> Py.String.of_string "hinge"
| `Squared_hinge -> Py.String.of_string "squared_hinge"
)); ("dual", Wrap_utils.Option.map dual Py.Bool.of_bool); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("C", Wrap_utils.Option.map c Py.Float.of_float); ("multi_class", Wrap_utils.Option.map multi_class (function
| `Ovr -> Py.String.of_string "ovr"
| `Crammer_singer -> Py.String.of_string "crammer_singer"
)); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("intercept_scaling", Wrap_utils.Option.map intercept_scaling Py.Float.of_float); ("class_weight", Wrap_utils.Option.map class_weight (function
| `DictIntToFloat x -> (Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float) x
| `Balanced -> Py.String.of_string "balanced"
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int)])

let decision_function ~x self =
   Py.Module.get_function_with_keywords self "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let densify self =
   Py.Module.get_function_with_keywords self "densify"
     [||]
     []

let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

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
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let sparsify self =
   Py.Module.get_function_with_keywords self "sparsify"
     [||]
     []


let coef_opt self =
  match Py.Object.get_attr_string self "coef_" with
  | None -> failwith "attribute coef_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let coef_ self = match coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let intercept_opt self =
  match Py.Object.get_attr_string self "intercept_" with
  | None -> failwith "attribute intercept_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let intercept_ self = match intercept_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string self "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let classes_ self = match classes_opt self with
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
module LinearSVR = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?epsilon ?tol ?c ?loss ?fit_intercept ?intercept_scaling ?dual ?verbose ?random_state ?max_iter () =
   Py.Module.get_function_with_keywords ns "LinearSVR"
     [||]
     (Wrap_utils.keyword_args [("epsilon", Wrap_utils.Option.map epsilon Py.Float.of_float); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("C", Wrap_utils.Option.map c Py.Float.of_float); ("loss", Wrap_utils.Option.map loss Py.String.of_string); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("intercept_scaling", Wrap_utils.Option.map intercept_scaling Py.Float.of_float); ("dual", Wrap_utils.Option.map dual Py.Bool.of_bool); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int)])

let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

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
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)


let coef_opt self =
  match Py.Object.get_attr_string self "coef_" with
  | None -> failwith "attribute coef_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let coef_ self = match coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let intercept_opt self =
  match Py.Object.get_attr_string self "intercept_" with
  | None -> failwith "attribute intercept_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let intercept_ self = match intercept_opt self with
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
module NuSVC = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?nu ?kernel ?degree ?gamma ?coef0 ?shrinking ?probability ?tol ?cache_size ?class_weight ?verbose ?max_iter ?decision_function_shape ?break_ties ?random_state () =
                     Py.Module.get_function_with_keywords ns "NuSVC"
                       [||]
                       (Wrap_utils.keyword_args [("nu", Wrap_utils.Option.map nu Py.Float.of_float); ("kernel", Wrap_utils.Option.map kernel Py.String.of_string); ("degree", Wrap_utils.Option.map degree Py.Int.of_int); ("gamma", Wrap_utils.Option.map gamma (function
| `Scale -> Py.String.of_string "scale"
| `Auto -> Py.String.of_string "auto"
| `F x -> Py.Float.of_float x
)); ("coef0", Wrap_utils.Option.map coef0 Py.Float.of_float); ("shrinking", Wrap_utils.Option.map shrinking Py.Bool.of_bool); ("probability", Wrap_utils.Option.map probability Py.Bool.of_bool); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("cache_size", Wrap_utils.Option.map cache_size Py.Float.of_float); ("class_weight", Wrap_utils.Option.map class_weight (function
| `DictIntToFloat x -> (Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float) x
| `Balanced -> Py.String.of_string "balanced"
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("decision_function_shape", Wrap_utils.Option.map decision_function_shape (function
| `Ovo -> Py.String.of_string "ovo"
| `Ovr -> Py.String.of_string "ovr"
)); ("break_ties", Wrap_utils.Option.map break_ties Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])

let decision_function ~x self =
   Py.Module.get_function_with_keywords self "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

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
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)


let support_opt self =
  match Py.Object.get_attr_string self "support_" with
  | None -> failwith "attribute support_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let support_ self = match support_opt self with
  | None -> raise Not_found
  | Some x -> x

let support_vectors_opt self =
  match Py.Object.get_attr_string self "support_vectors_" with
  | None -> failwith "attribute support_vectors_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let support_vectors_ self = match support_vectors_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_support_opt self =
  match Py.Object.get_attr_string self "n_support_" with
  | None -> failwith "attribute n_support_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let n_support_ self = match n_support_opt self with
  | None -> raise Not_found
  | Some x -> x

let dual_coef_opt self =
  match Py.Object.get_attr_string self "dual_coef_" with
  | None -> failwith "attribute dual_coef_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let dual_coef_ self = match dual_coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let coef_opt self =
  match Py.Object.get_attr_string self "coef_" with
  | None -> failwith "attribute coef_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let coef_ self = match coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let intercept_opt self =
  match Py.Object.get_attr_string self "intercept_" with
  | None -> failwith "attribute intercept_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let intercept_ self = match intercept_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string self "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let fit_status_opt self =
  match Py.Object.get_attr_string self "fit_status_" with
  | None -> failwith "attribute fit_status_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let fit_status_ self = match fit_status_opt self with
  | None -> raise Not_found
  | Some x -> x

let probA_opt self =
  match Py.Object.get_attr_string self "probA_" with
  | None -> failwith "attribute probA_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let probA_ self = match probA_opt self with
  | None -> raise Not_found
  | Some x -> x

let class_weight_opt self =
  match Py.Object.get_attr_string self "class_weight_" with
  | None -> failwith "attribute class_weight_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let class_weight_ self = match class_weight_opt self with
  | None -> raise Not_found
  | Some x -> x

let shape_fit_opt self =
  match Py.Object.get_attr_string self "shape_fit_" with
  | None -> failwith "attribute shape_fit_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let shape_fit_ self = match shape_fit_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module NuSVR = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?nu ?c ?kernel ?degree ?gamma ?coef0 ?shrinking ?tol ?cache_size ?verbose ?max_iter () =
                     Py.Module.get_function_with_keywords ns "NuSVR"
                       [||]
                       (Wrap_utils.keyword_args [("nu", Wrap_utils.Option.map nu Py.Float.of_float); ("C", Wrap_utils.Option.map c Py.Float.of_float); ("kernel", Wrap_utils.Option.map kernel Py.String.of_string); ("degree", Wrap_utils.Option.map degree Py.Int.of_int); ("gamma", Wrap_utils.Option.map gamma (function
| `Scale -> Py.String.of_string "scale"
| `Auto -> Py.String.of_string "auto"
| `F x -> Py.Float.of_float x
)); ("coef0", Wrap_utils.Option.map coef0 Py.Float.of_float); ("shrinking", Wrap_utils.Option.map shrinking Py.Bool.of_bool); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("cache_size", Wrap_utils.Option.map cache_size Py.Float.of_float); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int)])

let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

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
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)


let support_opt self =
  match Py.Object.get_attr_string self "support_" with
  | None -> failwith "attribute support_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let support_ self = match support_opt self with
  | None -> raise Not_found
  | Some x -> x

let support_vectors_opt self =
  match Py.Object.get_attr_string self "support_vectors_" with
  | None -> failwith "attribute support_vectors_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let support_vectors_ self = match support_vectors_opt self with
  | None -> raise Not_found
  | Some x -> x

let dual_coef_opt self =
  match Py.Object.get_attr_string self "dual_coef_" with
  | None -> failwith "attribute dual_coef_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let dual_coef_ self = match dual_coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let coef_opt self =
  match Py.Object.get_attr_string self "coef_" with
  | None -> failwith "attribute coef_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let coef_ self = match coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let intercept_opt self =
  match Py.Object.get_attr_string self "intercept_" with
  | None -> failwith "attribute intercept_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let intercept_ self = match intercept_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module OneClassSVM = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?kernel ?degree ?gamma ?coef0 ?tol ?nu ?shrinking ?cache_size ?verbose ?max_iter () =
                     Py.Module.get_function_with_keywords ns "OneClassSVM"
                       [||]
                       (Wrap_utils.keyword_args [("kernel", Wrap_utils.Option.map kernel Py.String.of_string); ("degree", Wrap_utils.Option.map degree Py.Int.of_int); ("gamma", Wrap_utils.Option.map gamma (function
| `Scale -> Py.String.of_string "scale"
| `Auto -> Py.String.of_string "auto"
| `F x -> Py.Float.of_float x
)); ("coef0", Wrap_utils.Option.map coef0 Py.Float.of_float); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("nu", Wrap_utils.Option.map nu Py.Float.of_float); ("shrinking", Wrap_utils.Option.map shrinking Py.Bool.of_bool); ("cache_size", Wrap_utils.Option.map cache_size Py.Float.of_float); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int)])

let decision_function ~x self =
   Py.Module.get_function_with_keywords self "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let fit ?y ?sample_weight ?params ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", y); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))]) (match params with None -> [] | Some x -> x))

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
let score_samples ~x self =
   Py.Module.get_function_with_keywords self "score_samples"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)


let support_opt self =
  match Py.Object.get_attr_string self "support_" with
  | None -> failwith "attribute support_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let support_ self = match support_opt self with
  | None -> raise Not_found
  | Some x -> x

let support_vectors_opt self =
  match Py.Object.get_attr_string self "support_vectors_" with
  | None -> failwith "attribute support_vectors_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let support_vectors_ self = match support_vectors_opt self with
  | None -> raise Not_found
  | Some x -> x

let dual_coef_opt self =
  match Py.Object.get_attr_string self "dual_coef_" with
  | None -> failwith "attribute dual_coef_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let dual_coef_ self = match dual_coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let coef_opt self =
  match Py.Object.get_attr_string self "coef_" with
  | None -> failwith "attribute coef_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let coef_ self = match coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let intercept_opt self =
  match Py.Object.get_attr_string self "intercept_" with
  | None -> failwith "attribute intercept_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let intercept_ self = match intercept_opt self with
  | None -> raise Not_found
  | Some x -> x

let offset_opt self =
  match Py.Object.get_attr_string self "offset_" with
  | None -> failwith "attribute offset_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let offset_ self = match offset_opt self with
  | None -> raise Not_found
  | Some x -> x

let fit_status_opt self =
  match Py.Object.get_attr_string self "fit_status_" with
  | None -> failwith "attribute fit_status_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let fit_status_ self = match fit_status_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SVC = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?c ?kernel ?degree ?gamma ?coef0 ?shrinking ?probability ?tol ?cache_size ?class_weight ?verbose ?max_iter ?decision_function_shape ?break_ties ?random_state () =
                     Py.Module.get_function_with_keywords ns "SVC"
                       [||]
                       (Wrap_utils.keyword_args [("C", Wrap_utils.Option.map c Py.Float.of_float); ("kernel", Wrap_utils.Option.map kernel Py.String.of_string); ("degree", Wrap_utils.Option.map degree Py.Int.of_int); ("gamma", Wrap_utils.Option.map gamma (function
| `Scale -> Py.String.of_string "scale"
| `Auto -> Py.String.of_string "auto"
| `F x -> Py.Float.of_float x
)); ("coef0", Wrap_utils.Option.map coef0 Py.Float.of_float); ("shrinking", Wrap_utils.Option.map shrinking Py.Bool.of_bool); ("probability", Wrap_utils.Option.map probability Py.Bool.of_bool); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("cache_size", Wrap_utils.Option.map cache_size Py.Float.of_float); ("class_weight", Wrap_utils.Option.map class_weight (function
| `DictIntToFloat x -> (Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float) x
| `Balanced -> Py.String.of_string "balanced"
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("decision_function_shape", Wrap_utils.Option.map decision_function_shape (function
| `Ovo -> Py.String.of_string "ovo"
| `Ovr -> Py.String.of_string "ovr"
)); ("break_ties", Wrap_utils.Option.map break_ties Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])

let decision_function ~x self =
   Py.Module.get_function_with_keywords self "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

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
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)


let support_opt self =
  match Py.Object.get_attr_string self "support_" with
  | None -> failwith "attribute support_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let support_ self = match support_opt self with
  | None -> raise Not_found
  | Some x -> x

let support_vectors_opt self =
  match Py.Object.get_attr_string self "support_vectors_" with
  | None -> failwith "attribute support_vectors_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let support_vectors_ self = match support_vectors_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_support_opt self =
  match Py.Object.get_attr_string self "n_support_" with
  | None -> failwith "attribute n_support_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let n_support_ self = match n_support_opt self with
  | None -> raise Not_found
  | Some x -> x

let dual_coef_opt self =
  match Py.Object.get_attr_string self "dual_coef_" with
  | None -> failwith "attribute dual_coef_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let dual_coef_ self = match dual_coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let coef_opt self =
  match Py.Object.get_attr_string self "coef_" with
  | None -> failwith "attribute coef_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let coef_ self = match coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let intercept_opt self =
  match Py.Object.get_attr_string self "intercept_" with
  | None -> failwith "attribute intercept_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let intercept_ self = match intercept_opt self with
  | None -> raise Not_found
  | Some x -> x

let fit_status_opt self =
  match Py.Object.get_attr_string self "fit_status_" with
  | None -> failwith "attribute fit_status_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let fit_status_ self = match fit_status_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string self "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let probA_opt self =
  match Py.Object.get_attr_string self "probA_" with
  | None -> failwith "attribute probA_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let probA_ self = match probA_opt self with
  | None -> raise Not_found
  | Some x -> x

let class_weight_opt self =
  match Py.Object.get_attr_string self "class_weight_" with
  | None -> failwith "attribute class_weight_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let class_weight_ self = match class_weight_opt self with
  | None -> raise Not_found
  | Some x -> x

let shape_fit_opt self =
  match Py.Object.get_attr_string self "shape_fit_" with
  | None -> failwith "attribute shape_fit_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let shape_fit_ self = match shape_fit_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SVR = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?kernel ?degree ?gamma ?coef0 ?tol ?c ?epsilon ?shrinking ?cache_size ?verbose ?max_iter () =
                     Py.Module.get_function_with_keywords ns "SVR"
                       [||]
                       (Wrap_utils.keyword_args [("kernel", Wrap_utils.Option.map kernel Py.String.of_string); ("degree", Wrap_utils.Option.map degree Py.Int.of_int); ("gamma", Wrap_utils.Option.map gamma (function
| `Scale -> Py.String.of_string "scale"
| `Auto -> Py.String.of_string "auto"
| `F x -> Py.Float.of_float x
)); ("coef0", Wrap_utils.Option.map coef0 Py.Float.of_float); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("C", Wrap_utils.Option.map c Py.Float.of_float); ("epsilon", Wrap_utils.Option.map epsilon Py.Float.of_float); ("shrinking", Wrap_utils.Option.map shrinking Py.Bool.of_bool); ("cache_size", Wrap_utils.Option.map cache_size Py.Float.of_float); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int)])

let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

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
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)


let support_opt self =
  match Py.Object.get_attr_string self "support_" with
  | None -> failwith "attribute support_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let support_ self = match support_opt self with
  | None -> raise Not_found
  | Some x -> x

let support_vectors_opt self =
  match Py.Object.get_attr_string self "support_vectors_" with
  | None -> failwith "attribute support_vectors_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let support_vectors_ self = match support_vectors_opt self with
  | None -> raise Not_found
  | Some x -> x

let dual_coef_opt self =
  match Py.Object.get_attr_string self "dual_coef_" with
  | None -> failwith "attribute dual_coef_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let dual_coef_ self = match dual_coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let coef_opt self =
  match Py.Object.get_attr_string self "coef_" with
  | None -> failwith "attribute coef_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let coef_ self = match coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let fit_status_opt self =
  match Py.Object.get_attr_string self "fit_status_" with
  | None -> failwith "attribute fit_status_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let fit_status_ self = match fit_status_opt self with
  | None -> raise Not_found
  | Some x -> x

let intercept_opt self =
  match Py.Object.get_attr_string self "intercept_" with
  | None -> failwith "attribute intercept_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let intercept_ self = match intercept_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let l1_min_c ?loss ?fit_intercept ?intercept_scaling ~x ~y () =
                     Py.Module.get_function_with_keywords ns "l1_min_c"
                       [||]
                       (Wrap_utils.keyword_args [("loss", Wrap_utils.Option.map loss (function
| `Squared_hinge -> Py.String.of_string "squared_hinge"
| `Log -> Py.String.of_string "log"
)); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("intercept_scaling", Wrap_utils.Option.map intercept_scaling Py.Float.of_float); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
                       |> Py.Float.to_float
