let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.pipeline"

let get_py name = Py.Module.get __wrap_namespace name
module Bunch = struct
type tag = [`Bunch]
type t = [`Bunch | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs () =
   Py.Module.get_function_with_keywords __wrap_namespace "Bunch"
     [||]
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module FeatureUnion = struct
type tag = [`FeatureUnion]
type t = [`BaseEstimator | `FeatureUnion | `Object | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let create ?n_jobs ?transformer_weights ?verbose ~transformer_list () =
   Py.Module.get_function_with_keywords __wrap_namespace "FeatureUnion"
     [||]
     (Wrap_utils.keyword_args [("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("transformer_weights", Wrap_utils.Option.map transformer_weights Dict.to_pyobject); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("transformer_list", Some(transformer_list |> (fun ml -> Py.List.of_list_map (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.String.of_string ml_0); (Np.Obj.to_pyobject ml_1)]) ml)))])
     |> of_pyobject
let fit ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("X", Some(x ))]) (match fit_params with None -> [] | Some x -> x))
     |> of_pyobject
let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_feature_names self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_feature_names"
     [||]
     []
     |> (Py.List.to_list_map Py.String.to_string)
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let set_params ?kwargs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Pipeline = struct
type tag = [`Pipeline]
type t = [`BaseEstimator | `Object | `Pipeline] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
                  let create ?memory ?verbose ~steps () =
                     Py.Module.get_function_with_keywords __wrap_namespace "Pipeline"
                       [||]
                       (Wrap_utils.keyword_args [("memory", Wrap_utils.Option.map memory (function
| `S x -> Py.String.of_string x
| `Joblib_Memory x -> Wrap_utils.id x
)); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("steps", Some(steps |> (fun ml -> Py.List.of_list_map (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.String.of_string ml_0); (Np.Obj.to_pyobject ml_1)]) ml)))])
                       |> of_pyobject
                  let get_item ~ind self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
                       [||]
                       (Wrap_utils.keyword_args [("ind", Some(ind |> (function
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `Slice x -> Np.Wrap_utils.Slice.to_pyobject x
)))])

let decision_function ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let fit ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> of_pyobject
let fit_predict ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_predict"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
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
let inverse_transform ?x ?y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("y", Wrap_utils.Option.map y Np.Obj.to_pyobject)])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let predict ?predict_params ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))]) (match predict_params with None -> [] | Some x -> x))
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
let score ?y ?sample_weight ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let score_samples ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score_samples"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let set_params ?kwargs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject

let named_steps_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "named_steps" with
  | None -> failwith "attribute named_steps not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let named_steps self = match named_steps_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Islice = struct
type tag = [`Islice]
type t = [`Islice | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ~iterable ~stop () =
   Py.Module.get_function_with_keywords __wrap_namespace "islice"
     [||]
     (Wrap_utils.keyword_args [("iterable", Some(iterable )); ("stop", Some(stop ))])
     |> of_pyobject
let iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []
     |> (fun py -> Py.Iter.to_seq py |> Seq.map Dict.of_pyobject)
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let check_memory memory =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_memory"
                       [||]
                       (Wrap_utils.keyword_args [("memory", Some(memory |> (function
| `S x -> Py.String.of_string x
| `Object_with_the_joblib_Memory_interface x -> Wrap_utils.id x
| `None -> Py.none
)))])

let clone ?safe ~estimator () =
   Py.Module.get_function_with_keywords __wrap_namespace "clone"
     [||]
     (Wrap_utils.keyword_args [("safe", Wrap_utils.Option.map safe Py.Bool.of_bool); ("estimator", Some(estimator |> Np.Obj.to_pyobject))])

let delayed ?check_pickle ~function_ () =
   Py.Module.get_function_with_keywords __wrap_namespace "delayed"
     [||]
     (Wrap_utils.keyword_args [("check_pickle", check_pickle); ("function", Some(function_ ))])

                  let if_delegate_has_method delegate =
                     Py.Module.get_function_with_keywords __wrap_namespace "if_delegate_has_method"
                       [||]
                       (Wrap_utils.keyword_args [("delegate", Some(delegate |> (function
| `S x -> Py.String.of_string x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
)))])

let make_pipeline ?kwargs steps =
   Py.Module.get_function_with_keywords __wrap_namespace "make_pipeline"
     (Array.of_list @@ List.concat [(List.map Np.Obj.to_pyobject steps)])
     (match kwargs with None -> [] | Some x -> x)
     |> Pipeline.of_pyobject
let make_union ?kwargs transformers =
   Py.Module.get_function_with_keywords __wrap_namespace "make_union"
     (Array.of_list @@ List.concat [(List.map Np.Obj.to_pyobject transformers)])
     (match kwargs with None -> [] | Some x -> x)
     |> FeatureUnion.of_pyobject
