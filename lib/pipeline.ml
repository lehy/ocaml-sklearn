let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.pipeline"

let get_py name = Py.Module.get ns name
module Bunch = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?kwargs () =
   Py.Module.get_function_with_keywords ns "Bunch"
     [||]
     (match kwargs with None -> [] | Some x -> x)

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module FeatureUnion = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?n_jobs ?transformer_weights ?verbose ~transformer_list () =
   Py.Module.get_function_with_keywords ns "FeatureUnion"
     [||]
     (Wrap_utils.keyword_args [("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("transformer_weights", Wrap_utils.Option.map transformer_weights Dict.to_pyobject); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("transformer_list", Some(transformer_list |> (fun ml -> Py.List.of_list_map (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.String.of_string ml_0); (Wrap_utils.id ml_1)]) ml)))])

                  let fit ?y ?fit_params ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> (function
| `Arr x -> Arr.to_pyobject x
| `Depending_on_transformers x -> Wrap_utils.id x
)))]) (match fit_params with None -> [] | Some x -> x))

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Arr.of_pyobject
let get_feature_names self =
   Py.Module.get_function_with_keywords self "get_feature_names"
     [||]
     []
     |> (Py.List.to_list_map Py.String.to_string)
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let set_params ?kwargs self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match kwargs with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Parallel = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_jobs ?backend ?verbose ?timeout ?pre_dispatch ?batch_size ?temp_folder ?max_nbytes ?mmap_mode ?prefer ?require () =
                     Py.Module.get_function_with_keywords ns "Parallel"
                       [||]
                       (Wrap_utils.keyword_args [("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("backend", backend); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("timeout", Wrap_utils.Option.map timeout Py.Float.of_float); ("pre_dispatch", Wrap_utils.Option.map pre_dispatch (function
| `All -> Py.String.of_string "all"
| `I x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)); ("batch_size", Wrap_utils.Option.map batch_size (function
| `I x -> Py.Int.of_int x
| `Auto -> Py.String.of_string "auto"
)); ("temp_folder", Wrap_utils.Option.map temp_folder Py.String.of_string); ("max_nbytes", max_nbytes); ("mmap_mode", Wrap_utils.Option.map mmap_mode (function
| `R_ -> Py.String.of_string "r+"
| `R -> Py.String.of_string "r"
| `W_ -> Py.String.of_string "w+"
| `C -> Py.String.of_string "c"
| `None -> Py.none
)); ("prefer", Wrap_utils.Option.map prefer (function
| `Processes -> Py.String.of_string "processes"
| `Threads -> Py.String.of_string "threads"
)); ("require", Wrap_utils.Option.map require Py.String.of_string)])

let debug ~msg self =
   Py.Module.get_function_with_keywords self "debug"
     [||]
     (Wrap_utils.keyword_args [("msg", Some(msg ))])

let dispatch_next self =
   Py.Module.get_function_with_keywords self "dispatch_next"
     [||]
     []

let dispatch_one_batch ~iterator self =
   Py.Module.get_function_with_keywords self "dispatch_one_batch"
     [||]
     (Wrap_utils.keyword_args [("iterator", Some(iterator ))])

let format ?indent ~obj self =
   Py.Module.get_function_with_keywords self "format"
     [||]
     (Wrap_utils.keyword_args [("indent", indent); ("obj", Some(obj ))])

let print_progress self =
   Py.Module.get_function_with_keywords self "print_progress"
     [||]
     []

let retrieve self =
   Py.Module.get_function_with_keywords self "retrieve"
     [||]
     []

let warn ~msg self =
   Py.Module.get_function_with_keywords self "warn"
     [||]
     (Wrap_utils.keyword_args [("msg", Some(msg ))])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Pipeline = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?memory ?verbose ~steps () =
                     Py.Module.get_function_with_keywords ns "Pipeline"
                       [||]
                       (Wrap_utils.keyword_args [("memory", Wrap_utils.Option.map memory (function
| `S x -> Py.String.of_string x
| `JoblibMemory x -> Wrap_utils.id x
)); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("steps", Some(steps |> (fun ml -> Py.List.of_list_map (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.String.of_string ml_0); (Wrap_utils.id ml_1)]) ml)))])

                  let get_item ~ind self =
                     Py.Module.get_function_with_keywords self "__getitem__"
                       [||]
                       (Wrap_utils.keyword_args [("ind", Some(ind |> (function
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| (`Slice _) as s -> Wrap_utils.Slice.of_variant s
)))])

let decision_function ~x self =
   Py.Module.get_function_with_keywords self "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let fit ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))

let fit_predict ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_predict"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Arr.of_pyobject
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
let inverse_transform ?x ?y self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Arr.to_pyobject); ("y", Wrap_utils.Option.map y Arr.to_pyobject)])
     |> Arr.of_pyobject
let predict ?predict_params ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))]) (match predict_params with None -> [] | Some x -> x))
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
let score ?y ?sample_weight ~x self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))])
     |> Py.Float.to_float
let score_samples ~x self =
   Py.Module.get_function_with_keywords self "score_samples"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let set_params ?kwargs self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match kwargs with None -> [] | Some x -> x)


let named_steps_opt self =
  match Py.Object.get_attr_string self "named_steps" with
  | None -> failwith "attribute named_steps not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let named_steps self = match named_steps_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module TransformerMixin = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create () =
   Py.Module.get_function_with_keywords ns "TransformerMixin"
     [||]
     []

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Arr.of_pyobject
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let check_memory ~memory () =
                     Py.Module.get_function_with_keywords ns "check_memory"
                       [||]
                       (Wrap_utils.keyword_args [("memory", Some(memory |> (function
| `S x -> Py.String.of_string x
| `JoblibMemory x -> Wrap_utils.id x
| `None -> Py.none
)))])

                  let clone ?safe ~estimator () =
                     Py.Module.get_function_with_keywords ns "clone"
                       [||]
                       (Wrap_utils.keyword_args [("safe", Wrap_utils.Option.map safe Py.Bool.of_bool); ("estimator", Some(estimator |> (function
| `Estimator x -> Wrap_utils.id x
| `Arr x -> Arr.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])

let delayed ?check_pickle ~function_ () =
   Py.Module.get_function_with_keywords ns "delayed"
     [||]
     (Wrap_utils.keyword_args [("check_pickle", check_pickle); ("function", Some(function_ ))])

                  let if_delegate_has_method ~delegate () =
                     Py.Module.get_function_with_keywords ns "if_delegate_has_method"
                       [||]
                       (Wrap_utils.keyword_args [("delegate", Some(delegate |> (function
| `S x -> Py.String.of_string x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `Tuple_of_strings x -> Wrap_utils.id x
)))])

let make_pipeline ?kwargs steps =
   Py.Module.get_function_with_keywords ns "make_pipeline"
     (Wrap_utils.pos_arg Wrap_utils.id steps)
     (match kwargs with None -> [] | Some x -> x)
     |> Pipeline.of_pyobject
let make_union ?kwargs transformers =
   Py.Module.get_function_with_keywords ns "make_union"
     (Wrap_utils.pos_arg Wrap_utils.id transformers)
     (match kwargs with None -> [] | Some x -> x)
     |> FeatureUnion.of_pyobject
