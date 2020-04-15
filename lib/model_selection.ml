let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.model_selection"

module GridSearchCV = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?scoring ?n_jobs ?iid ?refit ?cv ?verbose ?pre_dispatch ?error_score ?return_train_score ~estimator ~param_grid () =
                     Py.Module.get_function_with_keywords ns "GridSearchCV"
                       [||]
                       (Wrap_utils.keyword_args [("scoring", Wrap_utils.Option.map scoring (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
| `Dict x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
)); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("iid", Wrap_utils.Option.map iid Py.Bool.of_bool); ("refit", Wrap_utils.Option.map refit (function
| `Bool x -> Py.Bool.of_bool x
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("cv", Wrap_utils.Option.map cv (function
| `Int x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Ndarray x -> Ndarray.to_pyobject x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("pre_dispatch", Wrap_utils.Option.map pre_dispatch (function
| `Int x -> Py.Int.of_int x
| `String x -> Py.String.of_string x
)); ("error_score", Wrap_utils.Option.map error_score (function
| `Raise -> Py.String.of_string "raise"
| `PyObject x -> Wrap_utils.id x
)); ("return_train_score", Wrap_utils.Option.map return_train_score Py.Bool.of_bool); ("estimator", Some(estimator )); ("param_grid", Some(param_grid |> (function
| `Dict x -> Wrap_utils.id x
| `PyObject x -> Wrap_utils.id x
)))])

let decision_function ~x self =
   Py.Module.get_function_with_keywords self "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
                  let fit ?y ?groups ?fit_params ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let inverse_transform ~xt self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("Xt", Some(xt |> Ndarray.to_pyobject))])

let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let predict_log_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_log_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])

let predict_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let score ?y ~x self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let cv_results_ self =
  match Py.Object.get_attr_string self "cv_results_" with
| None -> raise (Wrap_utils.Attribute_not_found "cv_results_")
| Some x -> Wrap_utils.id x
let best_estimator_ self =
  match Py.Object.get_attr_string self "best_estimator_" with
| None -> raise (Wrap_utils.Attribute_not_found "best_estimator_")
| Some x -> Wrap_utils.id x
let best_score_ self =
  match Py.Object.get_attr_string self "best_score_" with
| None -> raise (Wrap_utils.Attribute_not_found "best_score_")
| Some x -> Py.Float.to_float x
let best_params_ self =
  match Py.Object.get_attr_string self "best_params_" with
| None -> raise (Wrap_utils.Attribute_not_found "best_params_")
| Some x -> Wrap_utils.id x
let best_index_ self =
  match Py.Object.get_attr_string self "best_index_" with
| None -> raise (Wrap_utils.Attribute_not_found "best_index_")
| Some x -> Py.Int.to_int x
let scorer_ self =
  match Py.Object.get_attr_string self "scorer_" with
| None -> raise (Wrap_utils.Attribute_not_found "scorer_")
| Some x -> Wrap_utils.id x
let n_splits_ self =
  match Py.Object.get_attr_string self "n_splits_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_splits_")
| Some x -> Py.Int.to_int x
let refit_time_ self =
  match Py.Object.get_attr_string self "refit_time_" with
| None -> raise (Wrap_utils.Attribute_not_found "refit_time_")
| Some x -> Py.Float.to_float x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module GroupKFold = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?n_splits () =
   Py.Module.get_function_with_keywords ns "GroupKFold"
     [||]
     (Wrap_utils.keyword_args [("n_splits", Wrap_utils.Option.map n_splits Py.Int.of_int)])

let get_n_splits ?x ?y ?groups self =
   Py.Module.get_function_with_keywords self "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", groups)])
     |> Py.Int.to_int
                  let split ?y ?groups ~x self =
                     Py.Module.get_function_with_keywords self "split"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("X", Some(x |> Ndarray.to_pyobject))])

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
| `Float x -> Py.Float.of_float x
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("train_size", Wrap_utils.Option.map train_size (function
| `Float x -> Py.Float.of_float x
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])

let get_n_splits ?x ?y ?groups self =
   Py.Module.get_function_with_keywords self "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", groups)])
     |> Py.Int.to_int
                  let split ?y ?groups ~x self =
                     Py.Module.get_function_with_keywords self "split"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("X", Some(x |> Ndarray.to_pyobject))])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module KFold = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_splits ?shuffle ?random_state () =
                     Py.Module.get_function_with_keywords ns "KFold"
                       [||]
                       (Wrap_utils.keyword_args [("n_splits", Wrap_utils.Option.map n_splits Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])

let get_n_splits ?x ?y ?groups self =
   Py.Module.get_function_with_keywords self "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", groups)])
     |> Py.Int.to_int
                  let split ?y ?groups ~x self =
                     Py.Module.get_function_with_keywords self "split"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("X", Some(x |> Ndarray.to_pyobject))])

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
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))])
                       |> Py.Int.to_int
                  let split ?y ?groups ~x self =
                     Py.Module.get_function_with_keywords self "split"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("X", Some(x |> Ndarray.to_pyobject))])

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
     (Wrap_utils.keyword_args [("y", y); ("groups", groups); ("X", Some(x |> Ndarray.to_pyobject))])
     |> Py.Int.to_int
                  let split ?y ?groups ~x self =
                     Py.Module.get_function_with_keywords self "split"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("X", Some(x |> Ndarray.to_pyobject))])

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
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))])
                       |> Py.Int.to_int
                  let split ?y ?groups ~x self =
                     Py.Module.get_function_with_keywords self "split"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("X", Some(x |> Ndarray.to_pyobject))])

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
     (Wrap_utils.keyword_args [("y", y); ("groups", groups); ("X", Some(x |> Ndarray.to_pyobject))])

                  let split ?y ?groups ~x self =
                     Py.Module.get_function_with_keywords self "split"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("X", Some(x |> Ndarray.to_pyobject))])

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
                       (Wrap_utils.keyword_args [("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("param_distributions", Some(param_distributions )); ("n_iter", Some(n_iter |> Py.Int.of_int))])

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
     (Wrap_utils.keyword_args [("test_fold", Some(test_fold |> Ndarray.to_pyobject))])

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
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
| `Dict x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
)); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("iid", Wrap_utils.Option.map iid Py.Bool.of_bool); ("refit", Wrap_utils.Option.map refit (function
| `Bool x -> Py.Bool.of_bool x
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("cv", Wrap_utils.Option.map cv (function
| `Int x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Ndarray x -> Ndarray.to_pyobject x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("pre_dispatch", Wrap_utils.Option.map pre_dispatch (function
| `Int x -> Py.Int.of_int x
| `String x -> Py.String.of_string x
)); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("error_score", Wrap_utils.Option.map error_score (function
| `Raise -> Py.String.of_string "raise"
| `PyObject x -> Wrap_utils.id x
)); ("return_train_score", Wrap_utils.Option.map return_train_score Py.Bool.of_bool); ("estimator", Some(estimator )); ("param_distributions", Some(param_distributions |> (function
| `Dict x -> Wrap_utils.id x
| `PyObject x -> Wrap_utils.id x
)))])

let decision_function ~x self =
   Py.Module.get_function_with_keywords self "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
                  let fit ?y ?groups ?fit_params ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let inverse_transform ~xt self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("Xt", Some(xt |> Ndarray.to_pyobject))])

let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let predict_log_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_log_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])

let predict_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let score ?y ~x self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let cv_results_ self =
  match Py.Object.get_attr_string self "cv_results_" with
| None -> raise (Wrap_utils.Attribute_not_found "cv_results_")
| Some x -> Wrap_utils.id x
let best_estimator_ self =
  match Py.Object.get_attr_string self "best_estimator_" with
| None -> raise (Wrap_utils.Attribute_not_found "best_estimator_")
| Some x -> Wrap_utils.id x
let best_score_ self =
  match Py.Object.get_attr_string self "best_score_" with
| None -> raise (Wrap_utils.Attribute_not_found "best_score_")
| Some x -> Py.Float.to_float x
let best_params_ self =
  match Py.Object.get_attr_string self "best_params_" with
| None -> raise (Wrap_utils.Attribute_not_found "best_params_")
| Some x -> Wrap_utils.id x
let best_index_ self =
  match Py.Object.get_attr_string self "best_index_" with
| None -> raise (Wrap_utils.Attribute_not_found "best_index_")
| Some x -> Py.Int.to_int x
let scorer_ self =
  match Py.Object.get_attr_string self "scorer_" with
| None -> raise (Wrap_utils.Attribute_not_found "scorer_")
| Some x -> Wrap_utils.id x
let n_splits_ self =
  match Py.Object.get_attr_string self "n_splits_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_splits_")
| Some x -> Py.Int.to_int x
let refit_time_ self =
  match Py.Object.get_attr_string self "refit_time_" with
| None -> raise (Wrap_utils.Attribute_not_found "refit_time_")
| Some x -> Py.Float.to_float x
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
                       (Wrap_utils.keyword_args [("n_splits", Wrap_utils.Option.map n_splits Py.Int.of_int); ("n_repeats", Wrap_utils.Option.map n_repeats Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])

                  let get_n_splits ?x ?y ?groups self =
                     Py.Module.get_function_with_keywords self "get_n_splits"
                       [||]
                       (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", Wrap_utils.Option.map groups (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))])
                       |> Py.Int.to_int
                  let split ?y ?groups ~x self =
                     Py.Module.get_function_with_keywords self "split"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("X", Some(x |> Ndarray.to_pyobject))])

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
                       (Wrap_utils.keyword_args [("n_splits", Wrap_utils.Option.map n_splits Py.Int.of_int); ("n_repeats", Wrap_utils.Option.map n_repeats Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])

                  let get_n_splits ?x ?y ?groups self =
                     Py.Module.get_function_with_keywords self "get_n_splits"
                       [||]
                       (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", Wrap_utils.Option.map groups (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))])
                       |> Py.Int.to_int
                  let split ?y ?groups ~x self =
                     Py.Module.get_function_with_keywords self "split"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("X", Some(x |> Ndarray.to_pyobject))])

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
| `Float x -> Py.Float.of_float x
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("train_size", Wrap_utils.Option.map train_size (function
| `Float x -> Py.Float.of_float x
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])

let get_n_splits ?x ?y ?groups self =
   Py.Module.get_function_with_keywords self "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", groups)])
     |> Py.Int.to_int
                  let split ?y ?groups ~x self =
                     Py.Module.get_function_with_keywords self "split"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("X", Some(x |> Ndarray.to_pyobject))])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module StratifiedKFold = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_splits ?shuffle ?random_state () =
                     Py.Module.get_function_with_keywords ns "StratifiedKFold"
                       [||]
                       (Wrap_utils.keyword_args [("n_splits", Wrap_utils.Option.map n_splits Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])

let get_n_splits ?x ?y ?groups self =
   Py.Module.get_function_with_keywords self "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", groups)])
     |> Py.Int.to_int
let split ?groups ~x ~y self =
   Py.Module.get_function_with_keywords self "split"
     [||]
     (Wrap_utils.keyword_args [("groups", groups); ("X", Some(x |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])

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
| `Float x -> Py.Float.of_float x
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("train_size", Wrap_utils.Option.map train_size (function
| `Float x -> Py.Float.of_float x
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])

let get_n_splits ?x ?y ?groups self =
   Py.Module.get_function_with_keywords self "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", groups)])
     |> Py.Int.to_int
let split ?groups ~x ~y self =
   Py.Module.get_function_with_keywords self "split"
     [||]
     (Wrap_utils.keyword_args [("groups", groups); ("X", Some(x |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module TimeSeriesSplit = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?n_splits ?max_train_size () =
   Py.Module.get_function_with_keywords ns "TimeSeriesSplit"
     [||]
     (Wrap_utils.keyword_args [("n_splits", Wrap_utils.Option.map n_splits Py.Int.of_int); ("max_train_size", Wrap_utils.Option.map max_train_size Py.Int.of_int)])

let get_n_splits ?x ?y ?groups self =
   Py.Module.get_function_with_keywords self "get_n_splits"
     [||]
     (Wrap_utils.keyword_args [("X", x); ("y", y); ("groups", groups)])
     |> Py.Int.to_int
                  let split ?y ?groups ~x self =
                     Py.Module.get_function_with_keywords self "split"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("X", Some(x |> Ndarray.to_pyobject))])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let check_cv ?cv ?y ?classifier () =
                     Py.Module.get_function_with_keywords ns "check_cv"
                       [||]
                       (Wrap_utils.keyword_args [("cv", Wrap_utils.Option.map cv (function
| `Int x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Ndarray x -> Ndarray.to_pyobject x
)); ("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("classifier", Wrap_utils.Option.map classifier Py.Bool.of_bool)])

                  let cross_val_predict ?y ?groups ?cv ?n_jobs ?verbose ?fit_params ?pre_dispatch ?method_ ~estimator ~x () =
                     Py.Module.get_function_with_keywords ns "cross_val_predict"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("cv", Wrap_utils.Option.map cv (function
| `Int x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Ndarray x -> Ndarray.to_pyobject x
)); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("fit_params", fit_params); ("pre_dispatch", Wrap_utils.Option.map pre_dispatch (function
| `Int x -> Py.Int.of_int x
| `String x -> Py.String.of_string x
)); ("method", Wrap_utils.Option.map method_ Py.String.of_string); ("estimator", Some(estimator )); ("X", Some(x |> Ndarray.to_pyobject))])
                       |> Ndarray.of_pyobject
                  let cross_val_score ?y ?groups ?scoring ?cv ?n_jobs ?verbose ?fit_params ?pre_dispatch ?error_score ~estimator ~x () =
                     Py.Module.get_function_with_keywords ns "cross_val_score"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("scoring", Wrap_utils.Option.map scoring (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("cv", Wrap_utils.Option.map cv (function
| `Int x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Ndarray x -> Ndarray.to_pyobject x
)); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("fit_params", fit_params); ("pre_dispatch", Wrap_utils.Option.map pre_dispatch (function
| `Int x -> Py.Int.of_int x
| `String x -> Py.String.of_string x
)); ("error_score", Wrap_utils.Option.map error_score (function
| `Raise -> Py.String.of_string "raise"
| `PyObject x -> Wrap_utils.id x
)); ("estimator", Some(estimator )); ("X", Some(x |> Ndarray.to_pyobject))])

                  let cross_validate ?y ?groups ?scoring ?cv ?n_jobs ?verbose ?fit_params ?pre_dispatch ?return_train_score ?return_estimator ?error_score ~estimator ~x () =
                     Py.Module.get_function_with_keywords ns "cross_validate"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("groups", Wrap_utils.Option.map groups (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("scoring", Wrap_utils.Option.map scoring (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
| `Dict x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
)); ("cv", Wrap_utils.Option.map cv (function
| `Int x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Ndarray x -> Ndarray.to_pyobject x
)); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("fit_params", fit_params); ("pre_dispatch", Wrap_utils.Option.map pre_dispatch (function
| `Int x -> Py.Int.of_int x
| `String x -> Py.String.of_string x
)); ("return_train_score", Wrap_utils.Option.map return_train_score Py.Bool.of_bool); ("return_estimator", Wrap_utils.Option.map return_estimator Py.Bool.of_bool); ("error_score", Wrap_utils.Option.map error_score (function
| `Raise -> Py.String.of_string "raise"
| `PyObject x -> Wrap_utils.id x
)); ("estimator", Some(estimator )); ("X", Some(x |> Ndarray.to_pyobject))])

                  let fit_grid_point ?error_score ?fit_params ~x ~y ~estimator ~parameters ~train ~test ~scorer ~verbose () =
                     Py.Module.get_function_with_keywords ns "fit_grid_point"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("error_score", Wrap_utils.Option.map error_score (function
| `Raise -> Py.String.of_string "raise"
| `PyObject x -> Wrap_utils.id x
)); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
| `ArrayLike x -> Wrap_utils.id x
))); ("y", Some(y |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `None -> Py.String.of_string "None"
))); ("estimator", Some(estimator )); ("parameters", Some(parameters )); ("train", Some(train |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `Bool x -> Py.Bool.of_bool x
| `PyObject x -> Wrap_utils.id x
))); ("test", Some(test |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `Bool x -> Py.Bool.of_bool x
| `PyObject x -> Wrap_utils.id x
))); ("scorer", Some(scorer |> (function
| `Callable x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))); ("verbose", Some(verbose |> Py.Int.of_int))]) (match fit_params with None -> [] | Some x -> x))
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2))))
                  let learning_curve ?groups ?train_sizes ?cv ?scoring ?exploit_incremental_learning ?n_jobs ?pre_dispatch ?verbose ?shuffle ?random_state ?error_score ?return_times ~estimator ~x ~y () =
                     Py.Module.get_function_with_keywords ns "learning_curve"
                       [||]
                       (Wrap_utils.keyword_args [("groups", Wrap_utils.Option.map groups (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("train_sizes", Wrap_utils.Option.map train_sizes (function
| `Ndarray x -> Ndarray.to_pyobject x
| `Int x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)); ("cv", Wrap_utils.Option.map cv (function
| `Int x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Ndarray x -> Ndarray.to_pyobject x
)); ("scoring", Wrap_utils.Option.map scoring (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("exploit_incremental_learning", Wrap_utils.Option.map exploit_incremental_learning Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("pre_dispatch", Wrap_utils.Option.map pre_dispatch (function
| `Int x -> Py.Int.of_int x
| `String x -> Py.String.of_string x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("error_score", Wrap_utils.Option.map error_score (function
| `Raise -> Py.String.of_string "raise"
| `PyObject x -> Wrap_utils.id x
)); ("return_times", Wrap_utils.Option.map return_times Py.Bool.of_bool); ("estimator", Some(estimator )); ("X", Some(x |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1)), (Ndarray.of_pyobject (Py.Tuple.get x 2)), (Ndarray.of_pyobject (Py.Tuple.get x 3)), (Ndarray.of_pyobject (Py.Tuple.get x 4))))
                  let permutation_test_score ?groups ?cv ?n_permutations ?n_jobs ?random_state ?verbose ?scoring ~estimator ~x ~y () =
                     Py.Module.get_function_with_keywords ns "permutation_test_score"
                       [||]
                       (Wrap_utils.keyword_args [("groups", Wrap_utils.Option.map groups (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("cv", Wrap_utils.Option.map cv (function
| `Int x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Ndarray x -> Ndarray.to_pyobject x
)); ("n_permutations", Wrap_utils.Option.map n_permutations Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("scoring", Wrap_utils.Option.map scoring (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("estimator", Some(estimator )); ("X", Some(x )); ("y", Some(y |> Ndarray.to_pyobject))])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
                  let train_test_split ?test_size ?train_size ?random_state ?shuffle ?stratify arrays =
                     Py.Module.get_function_with_keywords ns "train_test_split"
                       (Wrap_utils.pos_arg Ndarray.to_pyobject arrays)
                       (Wrap_utils.keyword_args [("test_size", Wrap_utils.Option.map test_size (function
| `Float x -> Py.Float.of_float x
| `Int x -> Py.Int.of_int x
| `None -> Py.none
)); ("train_size", Wrap_utils.Option.map train_size (function
| `Float x -> Py.Float.of_float x
| `Int x -> Py.Int.of_int x
| `None -> Py.none
)); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.none
)); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("stratify", Wrap_utils.Option.map stratify (function
| `Ndarray x -> Ndarray.to_pyobject x
| `None -> Py.none
))])
                       |> (fun py -> let len = Py.Sequence.length py in Array.init len (fun i -> Ndarray.of_pyobject (Py.Sequence.get_item py i)))
                  let validation_curve ?groups ?cv ?scoring ?n_jobs ?pre_dispatch ?verbose ?error_score ~estimator ~x ~y ~param_name ~param_range () =
                     Py.Module.get_function_with_keywords ns "validation_curve"
                       [||]
                       (Wrap_utils.keyword_args [("groups", Wrap_utils.Option.map groups (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("cv", Wrap_utils.Option.map cv (function
| `Int x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Ndarray x -> Ndarray.to_pyobject x
)); ("scoring", Wrap_utils.Option.map scoring (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("pre_dispatch", Wrap_utils.Option.map pre_dispatch (function
| `Int x -> Py.Int.of_int x
| `String x -> Py.String.of_string x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("error_score", Wrap_utils.Option.map error_score (function
| `Raise -> Py.String.of_string "raise"
| `PyObject x -> Wrap_utils.id x
)); ("estimator", Some(estimator )); ("X", Some(x |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject)); ("param_name", Some(param_name |> Py.String.of_string)); ("param_range", Some(param_range |> Ndarray.to_pyobject))])
                       |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
