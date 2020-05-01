let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.model_selection"

let get_py name = Py.Module.get ns name
module GridSearchCV = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?scoring ?n_jobs ?iid ?refit ?cv ?verbose ?pre_dispatch ?error_score ?return_train_score ~estimator ~param_grid () =
                     Py.Module.get_function_with_keywords ns "GridSearchCV"
                       [||]
                       (Wrap_utils.keyword_args [("scoring", Wrap_utils.Option.map scoring (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
| `List_tuple x -> Wrap_utils.id x
| `Dict x -> Dict.to_pyobject x
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("iid", Wrap_utils.Option.map iid Py.Bool.of_bool); ("refit", Wrap_utils.Option.map refit (function
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("cv", Wrap_utils.Option.map cv (function
| `I x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Arr x -> Arr.to_pyobject x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("pre_dispatch", Wrap_utils.Option.map pre_dispatch (function
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
)); ("error_score", Wrap_utils.Option.map error_score (function
| `Raise -> Py.String.of_string "raise"
| `F x -> Py.Float.of_float x
)); ("return_train_score", Wrap_utils.Option.map return_train_score Py.Bool.of_bool); ("estimator", Some(estimator )); ("param_grid", Some(param_grid |> (function
| `Grid x -> (fun x -> Dict.(of_param_grid_alist x |> to_pyobject)) x
| `List x -> (fun ml -> Py.List.of_list_map (fun x -> Dict.(of_param_grid_alist x |> to_pyobject)) ml) x
)))])

let decision_function ~x self =
   Py.Module.get_function_with_keywords self "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
                  let fit ?y ?groups ?fit_params ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Arr x -> Arr.to_pyobject x
| `With x -> Wrap_utils.id x
)); ("X", Some(x |> Arr.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let inverse_transform ~xt self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("Xt", Some(xt |> Arr.to_pyobject))])

let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let predict_log_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_log_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])

let predict_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let score ?y ~x self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))])
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

let cv_results_opt self =
  match Py.Object.get_attr_string self "cv_results_" with
  | None -> failwith "attribute cv_results_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let cv_results_ self = match cv_results_opt self with
  | None -> raise Not_found
  | Some x -> x

let best_estimator_opt self =
  match Py.Object.get_attr_string self "best_estimator_" with
  | None -> failwith "attribute best_estimator_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let best_estimator_ self = match best_estimator_opt self with
  | None -> raise Not_found
  | Some x -> x

let best_score_opt self =
  match Py.Object.get_attr_string self "best_score_" with
  | None -> failwith "attribute best_score_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let best_score_ self = match best_score_opt self with
  | None -> raise Not_found
  | Some x -> x

let best_params_opt self =
  match Py.Object.get_attr_string self "best_params_" with
  | None -> failwith "attribute best_params_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let best_params_ self = match best_params_opt self with
  | None -> raise Not_found
  | Some x -> x

let best_index_opt self =
  match Py.Object.get_attr_string self "best_index_" with
  | None -> failwith "attribute best_index_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let best_index_ self = match best_index_opt self with
  | None -> raise Not_found
  | Some x -> x

let scorer_opt self =
  match Py.Object.get_attr_string self "scorer_" with
  | None -> failwith "attribute scorer_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let scorer_ self = match scorer_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_splits_opt self =
  match Py.Object.get_attr_string self "n_splits_" with
  | None -> failwith "attribute n_splits_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_splits_ self = match n_splits_opt self with
  | None -> raise Not_found
  | Some x -> x

let refit_time_opt self =
  match Py.Object.get_attr_string self "refit_time_" with
  | None -> failwith "attribute refit_time_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let refit_time_ self = match refit_time_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module GroupShuffleSplit = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_splits ?test_size ?train_size ?random_state () =
                     Py.Module.get_function_with_keywords ns "GroupShuffleSplit"
                       [||]
                       (Wrap_utils.keyword_args [("n_splits", Wrap_utils.Option.map n_splits Py.Int.of_int); ("test_size", Wrap_utils.Option.map test_size (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("train_size", Wrap_utils.Option.map train_size (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])

let get_n_splits ?x ?y ?groups self =
   Py.Module.get_function_with_keywords self "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", groups)])
     |> Py.Int.to_int
                  let split ?y ?groups ~x self =
                     Py.Module.get_function_with_keywords self "split"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Arr x -> Arr.to_pyobject x
| `With x -> Wrap_utils.id x
)); ("X", Some(x |> Arr.to_pyobject))])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LeaveOneGroupOut = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create () =
   Py.Module.get_function_with_keywords ns "LeaveOneGroupOut"
     [||]
     []

                  let get_n_splits ?x ?y ?groups self =
                     Py.Module.get_function_with_keywords self "get_n_splits"
                       [||]
                       (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", Wrap_utils.Option.map groups (function
| `Arr x -> Arr.to_pyobject x
| `With x -> Wrap_utils.id x
))])
                       |> Py.Int.to_int
                  let split ?y ?groups ~x self =
                     Py.Module.get_function_with_keywords self "split"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Arr x -> Arr.to_pyobject x
| `With x -> Wrap_utils.id x
)); ("X", Some(x |> Arr.to_pyobject))])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LeaveOneOut = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create () =
   Py.Module.get_function_with_keywords ns "LeaveOneOut"
     [||]
     []

let get_n_splits ?y ?groups ~x self =
   Py.Module.get_function_with_keywords self "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("groups", groups); ("X", Some(x |> Arr.to_pyobject))])
     |> Py.Int.to_int
                  let split ?y ?groups ~x self =
                     Py.Module.get_function_with_keywords self "split"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Arr x -> Arr.to_pyobject x
| `With x -> Wrap_utils.id x
)); ("X", Some(x |> Arr.to_pyobject))])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LeavePGroupsOut = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ~n_groups () =
   Py.Module.get_function_with_keywords ns "LeavePGroupsOut"
     [||]
     (Wrap_utils.keyword_args [("n_groups", Some(n_groups |> Py.Int.of_int))])

                  let get_n_splits ?x ?y ?groups self =
                     Py.Module.get_function_with_keywords self "get_n_splits"
                       [||]
                       (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", Wrap_utils.Option.map groups (function
| `Arr x -> Arr.to_pyobject x
| `With x -> Wrap_utils.id x
))])
                       |> Py.Int.to_int
                  let split ?y ?groups ~x self =
                     Py.Module.get_function_with_keywords self "split"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Arr x -> Arr.to_pyobject x
| `With x -> Wrap_utils.id x
)); ("X", Some(x |> Arr.to_pyobject))])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LeavePOut = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ~p () =
   Py.Module.get_function_with_keywords ns "LeavePOut"
     [||]
     (Wrap_utils.keyword_args [("p", Some(p |> Py.Int.of_int))])

let get_n_splits ?y ?groups ~x self =
   Py.Module.get_function_with_keywords self "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("groups", groups); ("X", Some(x |> Arr.to_pyobject))])

                  let split ?y ?groups ~x self =
                     Py.Module.get_function_with_keywords self "split"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Arr x -> Arr.to_pyobject x
| `With x -> Wrap_utils.id x
)); ("X", Some(x |> Arr.to_pyobject))])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ParameterGrid = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ~param_grid () =
   Py.Module.get_function_with_keywords ns "ParameterGrid"
     [||]
     (Wrap_utils.keyword_args [("param_grid", Some(param_grid ))])

let get_item ~ind self =
   Py.Module.get_function_with_keywords self "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("ind", Some(ind |> Py.Int.of_int))])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ParameterSampler = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?random_state ~param_distributions ~n_iter () =
   Py.Module.get_function_with_keywords ns "ParameterSampler"
     [||]
     (Wrap_utils.keyword_args [("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("param_distributions", Some(param_distributions |> Dict.to_pyobject)); ("n_iter", Some(n_iter |> Py.Int.of_int))])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module PredefinedSplit = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ~test_fold () =
   Py.Module.get_function_with_keywords ns "PredefinedSplit"
     [||]
     (Wrap_utils.keyword_args [("test_fold", Some(test_fold |> Arr.to_pyobject))])

let get_n_splits ?x ?y ?groups self =
   Py.Module.get_function_with_keywords self "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", groups)])
     |> Py.Int.to_int
let split ?x ?y ?groups self =
   Py.Module.get_function_with_keywords self "split"
     [||]
     (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", groups)])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RandomizedSearchCV = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_iter ?scoring ?n_jobs ?iid ?refit ?cv ?verbose ?pre_dispatch ?random_state ?error_score ?return_train_score ~estimator ~param_distributions () =
                     Py.Module.get_function_with_keywords ns "RandomizedSearchCV"
                       [||]
                       (Wrap_utils.keyword_args [("n_iter", Wrap_utils.Option.map n_iter Py.Int.of_int); ("scoring", Wrap_utils.Option.map scoring (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
| `List_tuple x -> Wrap_utils.id x
| `Dict x -> Dict.to_pyobject x
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("iid", Wrap_utils.Option.map iid Py.Bool.of_bool); ("refit", Wrap_utils.Option.map refit (function
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("cv", Wrap_utils.Option.map cv (function
| `I x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Arr x -> Arr.to_pyobject x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("pre_dispatch", Wrap_utils.Option.map pre_dispatch (function
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("error_score", Wrap_utils.Option.map error_score (function
| `Raise -> Py.String.of_string "raise"
| `F x -> Py.Float.of_float x
)); ("return_train_score", Wrap_utils.Option.map return_train_score Py.Bool.of_bool); ("estimator", Some(estimator )); ("param_distributions", Some(param_distributions |> (function
| `Dict x -> Dict.to_pyobject x
| `List_of_dicts x -> Wrap_utils.id x
)))])

let decision_function ~x self =
   Py.Module.get_function_with_keywords self "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
                  let fit ?y ?groups ?fit_params ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Arr x -> Arr.to_pyobject x
| `With x -> Wrap_utils.id x
)); ("X", Some(x |> Arr.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let inverse_transform ~xt self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("Xt", Some(xt |> Arr.to_pyobject))])

let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let predict_log_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_log_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])

let predict_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let score ?y ~x self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))])
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

let cv_results_opt self =
  match Py.Object.get_attr_string self "cv_results_" with
  | None -> failwith "attribute cv_results_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let cv_results_ self = match cv_results_opt self with
  | None -> raise Not_found
  | Some x -> x

let best_estimator_opt self =
  match Py.Object.get_attr_string self "best_estimator_" with
  | None -> failwith "attribute best_estimator_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let best_estimator_ self = match best_estimator_opt self with
  | None -> raise Not_found
  | Some x -> x

let best_score_opt self =
  match Py.Object.get_attr_string self "best_score_" with
  | None -> failwith "attribute best_score_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let best_score_ self = match best_score_opt self with
  | None -> raise Not_found
  | Some x -> x

let best_params_opt self =
  match Py.Object.get_attr_string self "best_params_" with
  | None -> failwith "attribute best_params_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let best_params_ self = match best_params_opt self with
  | None -> raise Not_found
  | Some x -> x

let best_index_opt self =
  match Py.Object.get_attr_string self "best_index_" with
  | None -> failwith "attribute best_index_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let best_index_ self = match best_index_opt self with
  | None -> raise Not_found
  | Some x -> x

let scorer_opt self =
  match Py.Object.get_attr_string self "scorer_" with
  | None -> failwith "attribute scorer_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let scorer_ self = match scorer_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_splits_opt self =
  match Py.Object.get_attr_string self "n_splits_" with
  | None -> failwith "attribute n_splits_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_splits_ self = match n_splits_opt self with
  | None -> raise Not_found
  | Some x -> x

let refit_time_opt self =
  match Py.Object.get_attr_string self "refit_time_" with
  | None -> failwith "attribute refit_time_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let refit_time_ self = match refit_time_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RepeatedKFold = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?n_splits ?n_repeats ?random_state () =
   Py.Module.get_function_with_keywords ns "RepeatedKFold"
     [||]
     (Wrap_utils.keyword_args [("n_splits", Wrap_utils.Option.map n_splits Py.Int.of_int); ("n_repeats", Wrap_utils.Option.map n_repeats Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])

                  let get_n_splits ?x ?y ?groups self =
                     Py.Module.get_function_with_keywords self "get_n_splits"
                       [||]
                       (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", Wrap_utils.Option.map groups (function
| `Arr x -> Arr.to_pyobject x
| `With x -> Wrap_utils.id x
))])
                       |> Py.Int.to_int
                  let split ?y ?groups ~x self =
                     Py.Module.get_function_with_keywords self "split"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Arr x -> Arr.to_pyobject x
| `With x -> Wrap_utils.id x
)); ("X", Some(x |> Arr.to_pyobject))])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RepeatedStratifiedKFold = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?n_splits ?n_repeats ?random_state () =
   Py.Module.get_function_with_keywords ns "RepeatedStratifiedKFold"
     [||]
     (Wrap_utils.keyword_args [("n_splits", Wrap_utils.Option.map n_splits Py.Int.of_int); ("n_repeats", Wrap_utils.Option.map n_repeats Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])

                  let get_n_splits ?x ?y ?groups self =
                     Py.Module.get_function_with_keywords self "get_n_splits"
                       [||]
                       (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", Wrap_utils.Option.map groups (function
| `Arr x -> Arr.to_pyobject x
| `With x -> Wrap_utils.id x
))])
                       |> Py.Int.to_int
                  let split ?y ?groups ~x self =
                     Py.Module.get_function_with_keywords self "split"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Arr x -> Arr.to_pyobject x
| `With x -> Wrap_utils.id x
)); ("X", Some(x |> Arr.to_pyobject))])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ShuffleSplit = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_splits ?test_size ?train_size ?random_state () =
                     Py.Module.get_function_with_keywords ns "ShuffleSplit"
                       [||]
                       (Wrap_utils.keyword_args [("n_splits", Wrap_utils.Option.map n_splits Py.Int.of_int); ("test_size", Wrap_utils.Option.map test_size (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("train_size", Wrap_utils.Option.map train_size (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])

let get_n_splits ?x ?y ?groups self =
   Py.Module.get_function_with_keywords self "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", groups)])
     |> Py.Int.to_int
                  let split ?y ?groups ~x self =
                     Py.Module.get_function_with_keywords self "split"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Arr x -> Arr.to_pyobject x
| `With x -> Wrap_utils.id x
)); ("X", Some(x |> Arr.to_pyobject))])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module StratifiedShuffleSplit = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_splits ?test_size ?train_size ?random_state () =
                     Py.Module.get_function_with_keywords ns "StratifiedShuffleSplit"
                       [||]
                       (Wrap_utils.keyword_args [("n_splits", Wrap_utils.Option.map n_splits Py.Int.of_int); ("test_size", Wrap_utils.Option.map test_size (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("train_size", Wrap_utils.Option.map train_size (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])

let get_n_splits ?x ?y ?groups self =
   Py.Module.get_function_with_keywords self "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", groups)])
     |> Py.Int.to_int
let split ?groups ~x ~y self =
   Py.Module.get_function_with_keywords self "split"
     [||]
     (Wrap_utils.keyword_args [("groups", groups); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let check_cv ?cv ?y ?classifier () =
                     Py.Module.get_function_with_keywords ns "check_cv"
                       [||]
                       (Wrap_utils.keyword_args [("cv", Wrap_utils.Option.map cv (function
| `I x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Arr x -> Arr.to_pyobject x
)); ("y", Wrap_utils.Option.map y Arr.to_pyobject); ("classifier", Wrap_utils.Option.map classifier Py.Bool.of_bool)])

                  let cross_val_predict ?y ?groups ?cv ?n_jobs ?verbose ?fit_params ?pre_dispatch ?method_ ~estimator ~x () =
                     Py.Module.get_function_with_keywords ns "cross_val_predict"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Arr x -> Arr.to_pyobject x
| `With x -> Wrap_utils.id x
)); ("cv", Wrap_utils.Option.map cv (function
| `I x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Arr x -> Arr.to_pyobject x
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("fit_params", Wrap_utils.Option.map fit_params Dict.to_pyobject); ("pre_dispatch", Wrap_utils.Option.map pre_dispatch (function
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
)); ("method", Wrap_utils.Option.map method_ Py.String.of_string); ("estimator", Some(estimator )); ("X", Some(x |> Arr.to_pyobject))])
                       |> Arr.of_pyobject
                  let cross_val_score ?y ?groups ?scoring ?cv ?n_jobs ?verbose ?fit_params ?pre_dispatch ?error_score ~estimator ~x () =
                     Py.Module.get_function_with_keywords ns "cross_val_score"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Arr x -> Arr.to_pyobject x
| `With x -> Wrap_utils.id x
)); ("scoring", Wrap_utils.Option.map scoring (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("cv", Wrap_utils.Option.map cv (function
| `I x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Arr x -> Arr.to_pyobject x
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("fit_params", Wrap_utils.Option.map fit_params Dict.to_pyobject); ("pre_dispatch", Wrap_utils.Option.map pre_dispatch (function
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
)); ("error_score", Wrap_utils.Option.map error_score (function
| `Raise -> Py.String.of_string "raise"
| `F x -> Py.Float.of_float x
)); ("estimator", Some(estimator )); ("X", Some(x |> Arr.to_pyobject))])

                  let cross_validate ?y ?groups ?scoring ?cv ?n_jobs ?verbose ?fit_params ?pre_dispatch ?return_train_score ?return_estimator ?error_score ~estimator ~x () =
                     Py.Module.get_function_with_keywords ns "cross_validate"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Arr x -> Arr.to_pyobject x
| `With x -> Wrap_utils.id x
)); ("scoring", Wrap_utils.Option.map scoring (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
| `List_tuple x -> Wrap_utils.id x
| `Dict x -> Dict.to_pyobject x
)); ("cv", Wrap_utils.Option.map cv (function
| `I x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Arr x -> Arr.to_pyobject x
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("fit_params", Wrap_utils.Option.map fit_params Dict.to_pyobject); ("pre_dispatch", Wrap_utils.Option.map pre_dispatch (function
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
)); ("return_train_score", Wrap_utils.Option.map return_train_score Py.Bool.of_bool); ("return_estimator", Wrap_utils.Option.map return_estimator Py.Bool.of_bool); ("error_score", Wrap_utils.Option.map error_score (function
| `Raise -> Py.String.of_string "raise"
| `F x -> Py.Float.of_float x
)); ("estimator", Some(estimator )); ("X", Some(x |> Arr.to_pyobject))])

                  let fit_grid_point ?error_score ?fit_params ~x ~y ~estimator ~parameters ~train ~test ~scorer ~verbose () =
                     Py.Module.get_function_with_keywords ns "fit_grid_point"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("error_score", Wrap_utils.Option.map error_score (function
| `Raise -> Py.String.of_string "raise"
| `F x -> Py.Float.of_float x
)); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> (function
| `Arr x -> Arr.to_pyobject x
| `None -> Py.none
))); ("estimator", Some(estimator )); ("parameters", Some(parameters |> Dict.to_pyobject)); ("train", Some(train |> (function
| `Arr x -> Arr.to_pyobject x
| `Dtype_int x -> Wrap_utils.id x
| `Bool x -> Py.Bool.of_bool x
))); ("test", Some(test |> (function
| `Arr x -> Arr.to_pyobject x
| `Dtype_int x -> Wrap_utils.id x
| `Bool x -> Py.Bool.of_bool x
))); ("scorer", Some(scorer |> (function
| `Callable x -> Wrap_utils.id x
| `None -> Py.none
))); ("verbose", Some(verbose |> Py.Int.of_int))]) (match fit_params with None -> [] | Some x -> x))
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Dict.of_pyobject (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2))))
                  let learning_curve ?groups ?train_sizes ?cv ?scoring ?exploit_incremental_learning ?n_jobs ?pre_dispatch ?verbose ?shuffle ?random_state ?error_score ?return_times ~estimator ~x ~y () =
                     Py.Module.get_function_with_keywords ns "learning_curve"
                       [||]
                       (Wrap_utils.keyword_args [("groups", Wrap_utils.Option.map groups (function
| `Arr x -> Arr.to_pyobject x
| `With x -> Wrap_utils.id x
)); ("train_sizes", Wrap_utils.Option.map train_sizes (function
| `Arr x -> Arr.to_pyobject x
| `Dtype_float x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("cv", Wrap_utils.Option.map cv (function
| `I x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Arr x -> Arr.to_pyobject x
)); ("scoring", Wrap_utils.Option.map scoring (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("exploit_incremental_learning", Wrap_utils.Option.map exploit_incremental_learning Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("pre_dispatch", Wrap_utils.Option.map pre_dispatch (function
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("error_score", Wrap_utils.Option.map error_score (function
| `Raise -> Py.String.of_string "raise"
| `F x -> Py.Float.of_float x
)); ("return_times", Wrap_utils.Option.map return_times Py.Bool.of_bool); ("estimator", Some(estimator )); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Arr.of_pyobject (Py.Tuple.get x 1)), (Arr.of_pyobject (Py.Tuple.get x 2)), (Arr.of_pyobject (Py.Tuple.get x 3)), (Arr.of_pyobject (Py.Tuple.get x 4))))
                  let permutation_test_score ?groups ?cv ?n_permutations ?n_jobs ?random_state ?verbose ?scoring ~estimator ~x ~y () =
                     Py.Module.get_function_with_keywords ns "permutation_test_score"
                       [||]
                       (Wrap_utils.keyword_args [("groups", Wrap_utils.Option.map groups (function
| `Arr x -> Arr.to_pyobject x
| `With x -> Wrap_utils.id x
)); ("cv", Wrap_utils.Option.map cv (function
| `I x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Arr x -> Arr.to_pyobject x
)); ("n_permutations", Wrap_utils.Option.map n_permutations Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("scoring", Wrap_utils.Option.map scoring (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("estimator", Some(estimator )); ("X", Some(x )); ("y", Some(y |> Arr.to_pyobject))])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Arr.of_pyobject (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
                  let train_test_split ?test_size ?train_size ?random_state ?shuffle ?stratify arrays =
                     Py.Module.get_function_with_keywords ns "train_test_split"
                       (Wrap_utils.pos_arg Arr.to_pyobject arrays)
                       (Wrap_utils.keyword_args [("test_size", Wrap_utils.Option.map test_size (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("train_size", Wrap_utils.Option.map train_size (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("stratify", Wrap_utils.Option.map stratify Arr.to_pyobject)])
                       |> (fun py -> Py.List.to_list_map (Arr.of_pyobject) py)
                  let validation_curve ?groups ?cv ?scoring ?n_jobs ?pre_dispatch ?verbose ?error_score ~estimator ~x ~y ~param_name ~param_range () =
                     Py.Module.get_function_with_keywords ns "validation_curve"
                       [||]
                       (Wrap_utils.keyword_args [("groups", Wrap_utils.Option.map groups (function
| `Arr x -> Arr.to_pyobject x
| `With x -> Wrap_utils.id x
)); ("cv", Wrap_utils.Option.map cv (function
| `I x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Arr x -> Arr.to_pyobject x
)); ("scoring", Wrap_utils.Option.map scoring (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("pre_dispatch", Wrap_utils.Option.map pre_dispatch (function
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("error_score", Wrap_utils.Option.map error_score (function
| `Raise -> Py.String.of_string "raise"
| `F x -> Py.Float.of_float x
)); ("estimator", Some(estimator )); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject)); ("param_name", Some(param_name |> Py.String.of_string)); ("param_range", Some(param_range |> Arr.to_pyobject))])
                       |> (fun x -> ((Arr.of_pyobject (Py.Tuple.get x 0)), (Arr.of_pyobject (Py.Tuple.get x 1))))
