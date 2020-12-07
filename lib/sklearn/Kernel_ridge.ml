let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.kernel_ridge"

let get_py name = Py.Module.get __wrap_namespace name
module KernelRidge = struct
type tag = [`KernelRidge]
type t = [`BaseEstimator | `KernelRidge | `MultiOutputMixin | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
                  let create ?alpha ?kernel ?gamma ?degree ?coef0 ?kernel_params () =
                     Py.Module.get_function_with_keywords __wrap_namespace "KernelRidge"
                       [||]
                       (Wrap_utils.keyword_args [("alpha", Wrap_utils.Option.map alpha Np.Obj.to_pyobject); ("kernel", Wrap_utils.Option.map kernel (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("gamma", Wrap_utils.Option.map gamma Py.Float.of_float); ("degree", Wrap_utils.Option.map degree Py.Float.of_float); ("coef0", Wrap_utils.Option.map coef0 Py.Float.of_float); ("kernel_params", Wrap_utils.Option.map kernel_params Dict.to_pyobject)])
                       |> of_pyobject
let fit ?y ?sample_weight ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let predict ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject

let dual_coef_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dual_coef_" with
  | None -> failwith "attribute dual_coef_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let dual_coef_ self = match dual_coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_fit_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "X_fit_" with
  | None -> failwith "attribute X_fit_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let x_fit_ self = match x_fit_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let check_is_fitted ?attributes ?msg ?all_or_any ~estimator () =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_is_fitted"
                       [||]
                       (Wrap_utils.keyword_args [("attributes", Wrap_utils.Option.map attributes (function
| `S x -> Py.String.of_string x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `Arr x -> Np.Obj.to_pyobject x
)); ("msg", Wrap_utils.Option.map msg Py.String.of_string); ("all_or_any", Wrap_utils.Option.map all_or_any (function
| `Callable x -> Wrap_utils.id x
| `PyObject x -> Wrap_utils.id x
)); ("estimator", Some(estimator |> Np.Obj.to_pyobject))])

                  let pairwise_kernels ?y ?metric ?filter_params ?n_jobs ?kwds ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "pairwise_kernels"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("metric", Wrap_utils.Option.map metric (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("filter_params", Wrap_utils.Option.map filter_params Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("X", Some(x |> (function
| `Otherwise x -> Wrap_utils.id x
| `Arr x -> Np.Obj.to_pyobject x
)))]) (match kwds with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
