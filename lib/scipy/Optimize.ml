let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.optimize"

let get_py name = Py.Module.get __wrap_namespace name
module BFGS = struct
type tag = [`BFGS]
type t = [`BFGS | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?exception_strategy ?min_curvature ?init_scale () =
                     Py.Module.get_function_with_keywords __wrap_namespace "BFGS"
                       [||]
                       (Wrap_utils.keyword_args [("exception_strategy", Wrap_utils.Option.map exception_strategy (function
| `Skip_update -> Py.String.of_string "skip_update"
| `Damp_update -> Py.String.of_string "damp_update"
)); ("min_curvature", Wrap_utils.Option.map min_curvature Py.Float.of_float); ("init_scale", Wrap_utils.Option.map init_scale (function
| `F x -> Py.Float.of_float x
| `Auto -> Py.String.of_string "auto"
))])
                       |> of_pyobject
let dot ~p self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("p", Some(p |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let get_matrix self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_matrix"
     [||]
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let initialize ~n ~approx_type self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "initialize"
                       [||]
                       (Wrap_utils.keyword_args [("n", Some(n |> Py.Int.of_int)); ("approx_type", Some(approx_type |> (function
| `Hess -> Py.String.of_string "hess"
| `Inv_hess -> Py.String.of_string "inv_hess"
)))])

let update ~delta_x ~delta_grad self =
   Py.Module.get_function_with_keywords (to_pyobject self) "update"
     [||]
     (Wrap_utils.keyword_args [("delta_x", Some(delta_x |> Np.Obj.to_pyobject)); ("delta_grad", Some(delta_grad |> Np.Obj.to_pyobject))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Bounds = struct
type tag = [`Bounds]
type t = [`Bounds | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?keep_feasible ~lb ~ub () =
   Py.Module.get_function_with_keywords __wrap_namespace "Bounds"
     [||]
     (Wrap_utils.keyword_args [("keep_feasible", keep_feasible); ("lb", Some(lb )); ("ub", Some(ub ))])
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module HessianUpdateStrategy = struct
type tag = [`HessianUpdateStrategy]
type t = [`HessianUpdateStrategy | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "HessianUpdateStrategy"
     [||]
     []
     |> of_pyobject
let dot ~p self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("p", Some(p |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let get_matrix self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_matrix"
     [||]
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let initialize ~n ~approx_type self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "initialize"
                       [||]
                       (Wrap_utils.keyword_args [("n", Some(n |> Py.Int.of_int)); ("approx_type", Some(approx_type |> (function
| `Hess -> Py.String.of_string "hess"
| `Inv_hess -> Py.String.of_string "inv_hess"
)))])

let update ~delta_x ~delta_grad self =
   Py.Module.get_function_with_keywords (to_pyobject self) "update"
     [||]
     (Wrap_utils.keyword_args [("delta_x", Some(delta_x |> Np.Obj.to_pyobject)); ("delta_grad", Some(delta_grad |> Np.Obj.to_pyobject))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LbfgsInvHessProduct = struct
type tag = [`LbfgsInvHessProduct]
type t = [`LbfgsInvHessProduct | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "LbfgsInvHessProduct"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let adjoint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "adjoint"
     [||]
     []

let dot ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let matmat ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matmat"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])

let matvec ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matvec"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let rmatmat ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatmat"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])

let rmatvec ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatvec"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let todense self =
   Py.Module.get_function_with_keywords (to_pyobject self) "todense"
     [||]
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let transpose self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LinearConstraint = struct
type tag = [`LinearConstraint]
type t = [`LinearConstraint | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?keep_feasible ~a ~lb ~ub () =
   Py.Module.get_function_with_keywords __wrap_namespace "LinearConstraint"
     [||]
     (Wrap_utils.keyword_args [("keep_feasible", keep_feasible); ("A", Some(a |> Np.Obj.to_pyobject)); ("lb", Some(lb )); ("ub", Some(ub ))])
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module NonlinearConstraint = struct
type tag = [`NonlinearConstraint]
type t = [`NonlinearConstraint | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?jac ?hess ?keep_feasible ?finite_diff_rel_step ?finite_diff_jac_sparsity ~fun_ ~lb ~ub () =
                     Py.Module.get_function_with_keywords __wrap_namespace "NonlinearConstraint"
                       [||]
                       (Wrap_utils.keyword_args [("jac", Wrap_utils.Option.map jac (function
| `T2_point -> Py.String.of_string "2-point"
| `Callable x -> Wrap_utils.id x
| `Cs -> Py.String.of_string "cs"
| `T3_point -> Py.String.of_string "3-point"
)); ("hess", Wrap_utils.Option.map hess (function
| `T2_point -> Py.String.of_string "2-point"
| `HessianUpdateStrategy x -> Wrap_utils.id x
| `Cs -> Py.String.of_string "cs"
| `Callable x -> Wrap_utils.id x
| `T3_point -> Py.String.of_string "3-point"
| `None -> Py.none
)); ("keep_feasible", keep_feasible); ("finite_diff_rel_step", Wrap_utils.Option.map finite_diff_rel_step Np.Obj.to_pyobject); ("finite_diff_jac_sparsity", Wrap_utils.Option.map finite_diff_jac_sparsity Np.Obj.to_pyobject); ("fun", Some(fun_ )); ("lb", Some(lb )); ("ub", Some(ub ))])
                       |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module OptimizeResult = struct
type tag = [`OptimizeResult]
type t = [`Object | `OptimizeResult] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x

let x_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "x" with
  | None -> failwith "attribute x not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let x self = match x_opt self with
  | None -> raise Not_found
  | Some x -> x

let success_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "success" with
  | None -> failwith "attribute success not found"
  | Some x -> if Py.is_none x then None else Some (Py.Bool.to_bool x)

let success self = match success_opt self with
  | None -> raise Not_found
  | Some x -> x

let status_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "status" with
  | None -> failwith "attribute status not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let status self = match status_opt self with
  | None -> raise Not_found
  | Some x -> x

let message_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "message" with
  | None -> failwith "attribute message not found"
  | Some x -> if Py.is_none x then None else Some (Py.String.to_string x)

let message self = match message_opt self with
  | None -> raise Not_found
  | Some x -> x

let hess_inv_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "hess_inv" with
  | None -> failwith "attribute hess_inv not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let hess_inv self = match hess_inv_opt self with
  | None -> raise Not_found
  | Some x -> x

let nit_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nit" with
  | None -> failwith "attribute nit not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let nit self = match nit_opt self with
  | None -> raise Not_found
  | Some x -> x

let maxcv_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "maxcv" with
  | None -> failwith "attribute maxcv not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let maxcv self = match maxcv_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module OptimizeWarning = struct
type tag = [`OptimizeWarning]
type t = [`BaseException | `Object | `OptimizeWarning] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_exception x = (x :> [`BaseException] Obj.t)
let with_traceback ~tb self =
   Py.Module.get_function_with_keywords (to_pyobject self) "with_traceback"
     [||]
     (Wrap_utils.keyword_args [("tb", Some(tb ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RootResults = struct
type tag = [`RootResults]
type t = [`Object | `RootResults] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ~root ~iterations ~function_calls ~flag () =
   Py.Module.get_function_with_keywords __wrap_namespace "RootResults"
     [||]
     (Wrap_utils.keyword_args [("root", Some(root )); ("iterations", Some(iterations )); ("function_calls", Some(function_calls )); ("flag", Some(flag ))])
     |> of_pyobject

let root_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "root" with
  | None -> failwith "attribute root not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let root self = match root_opt self with
  | None -> raise Not_found
  | Some x -> x

let iterations_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "iterations" with
  | None -> failwith "attribute iterations not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let iterations self = match iterations_opt self with
  | None -> raise Not_found
  | Some x -> x

let function_calls_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "function_calls" with
  | None -> failwith "attribute function_calls not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let function_calls self = match function_calls_opt self with
  | None -> raise Not_found
  | Some x -> x

let converged_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "converged" with
  | None -> failwith "attribute converged not found"
  | Some x -> if Py.is_none x then None else Some (Py.Bool.to_bool x)

let converged self = match converged_opt self with
  | None -> raise Not_found
  | Some x -> x

let flag_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "flag" with
  | None -> failwith "attribute flag not found"
  | Some x -> if Py.is_none x then None else Some (Py.String.to_string x)

let flag self = match flag_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SR1 = struct
type tag = [`SR1]
type t = [`Object | `SR1] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?min_denominator ?init_scale () =
                     Py.Module.get_function_with_keywords __wrap_namespace "SR1"
                       [||]
                       (Wrap_utils.keyword_args [("min_denominator", Wrap_utils.Option.map min_denominator Py.Float.of_float); ("init_scale", Wrap_utils.Option.map init_scale (function
| `F x -> Py.Float.of_float x
| `Auto -> Py.String.of_string "auto"
))])
                       |> of_pyobject
let dot ~p self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("p", Some(p |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let get_matrix self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_matrix"
     [||]
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let initialize ~n ~approx_type self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "initialize"
                       [||]
                       (Wrap_utils.keyword_args [("n", Some(n |> Py.Int.of_int)); ("approx_type", Some(approx_type |> (function
| `Hess -> Py.String.of_string "hess"
| `Inv_hess -> Py.String.of_string "inv_hess"
)))])

let update ~delta_x ~delta_grad self =
   Py.Module.get_function_with_keywords (to_pyobject self) "update"
     [||]
     (Wrap_utils.keyword_args [("delta_x", Some(delta_x |> Np.Obj.to_pyobject)); ("delta_grad", Some(delta_grad |> Np.Obj.to_pyobject))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Cobyla = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.optimize.cobyla"

let get_py name = Py.Module.get __wrap_namespace name
module Izip = struct
type tag = [`Zip]
type t = [`Object | `Zip] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create iterables =
   Py.Module.get_function_with_keywords __wrap_namespace "izip"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id iterables)])
     []
     |> of_pyobject
let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let rLock ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "RLock"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

                  let fmin_cobyla ?args ?consargs ?rhobeg ?rhoend ?maxfun ?disp ?catol ~func ~x0 ~cons () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fmin_cobyla"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("consargs", consargs); ("rhobeg", Wrap_utils.Option.map rhobeg Py.Float.of_float); ("rhoend", Wrap_utils.Option.map rhoend Py.Float.of_float); ("maxfun", Wrap_utils.Option.map maxfun Py.Int.of_int); ("disp", Wrap_utils.Option.map disp (function
| `Three -> Py.Int.of_int 3
| `Two -> Py.Int.of_int 2
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
)); ("catol", Wrap_utils.Option.map catol Py.Float.of_float); ("func", Some(func )); ("x0", Some(x0 |> Np.Obj.to_pyobject)); ("cons", Some(cons ))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let synchronized func =
   Py.Module.get_function_with_keywords __wrap_namespace "synchronized"
     [||]
     (Wrap_utils.keyword_args [("func", Some(func ))])


end
module Lbfgsb = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.optimize.lbfgsb"

let get_py name = Py.Module.get __wrap_namespace name
module LinearOperator = struct
type tag = [`LinearOperator]
type t = [`LinearOperator | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "LinearOperator"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let adjoint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "adjoint"
     [||]
     []

let dot ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let matmat ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matmat"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])

let matvec ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matvec"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let rmatmat ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatmat"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])

let rmatvec ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatvec"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let transpose self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     [||]
     []


let args_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "args" with
  | None -> failwith "attribute args not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let args self = match args_opt self with
  | None -> raise Not_found
  | Some x -> x

let ndim_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "ndim" with
  | None -> failwith "attribute ndim not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let ndim self = match ndim_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MemoizeJac = struct
type tag = [`MemoizeJac]
type t = [`MemoizeJac | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create fun_ =
   Py.Module.get_function_with_keywords __wrap_namespace "MemoizeJac"
     [||]
     (Wrap_utils.keyword_args [("fun", Some(fun_ ))])
     |> of_pyobject
let derivative ~x args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "derivative"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (Wrap_utils.keyword_args [("x", Some(x ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Float64 = struct
type tag = [`Float64]
type t = [`Float64 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?x () =
   Py.Module.get_function_with_keywords __wrap_namespace "float64"
     (Array.of_list @@ List.concat [(match x with None -> [] | Some x -> [x ])])
     []
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let fromhex ~string self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fromhex"
     (Array.of_list @@ List.concat [[string ]])
     []

let hex self =
   Py.Module.get_function_with_keywords (to_pyobject self) "hex"
     [||]
     []

let is_integer self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_integer"
     [||]
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Np.Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let array ?dtype ?copy ?order ?subok ?ndmin ~object_ () =
                     Py.Module.get_function_with_keywords __wrap_namespace "array"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("order", Wrap_utils.Option.map order (function
| `K -> Py.String.of_string "K"
| `A -> Py.String.of_string "A"
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("ndmin", Wrap_utils.Option.map ndmin Py.Int.of_int); ("object", Some(object_ |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let asarray ?dtype ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let fmin_l_bfgs_b ?fprime ?args ?approx_grad ?bounds ?m ?factr ?pgtol ?epsilon ?iprint ?maxfun ?maxiter ?disp ?callback ?maxls ~func ~x0 () =
   Py.Module.get_function_with_keywords __wrap_namespace "fmin_l_bfgs_b"
     [||]
     (Wrap_utils.keyword_args [("fprime", fprime); ("args", args); ("approx_grad", Wrap_utils.Option.map approx_grad Py.Bool.of_bool); ("bounds", Wrap_utils.Option.map bounds Np.Obj.to_pyobject); ("m", Wrap_utils.Option.map m Py.Int.of_int); ("factr", Wrap_utils.Option.map factr Py.Float.of_float); ("pgtol", Wrap_utils.Option.map pgtol Py.Float.of_float); ("epsilon", Wrap_utils.Option.map epsilon Py.Float.of_float); ("iprint", Wrap_utils.Option.map iprint Py.Int.of_int); ("maxfun", Wrap_utils.Option.map maxfun Py.Int.of_int); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("disp", Wrap_utils.Option.map disp Py.Int.of_int); ("callback", callback); ("maxls", Wrap_utils.Option.map maxls Py.Int.of_int); ("func", Some(func )); ("x0", Some(x0 |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
let old_bound_to_new bounds =
   Py.Module.get_function_with_keywords __wrap_namespace "old_bound_to_new"
     [||]
     (Wrap_utils.keyword_args [("bounds", Some(bounds ))])

                  let zeros ?dtype ?order ~shape () =
                     Py.Module.get_function_with_keywords __wrap_namespace "zeros"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
module Linesearch = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.optimize.linesearch"

let get_py name = Py.Module.get __wrap_namespace name
module LineSearchWarning = struct
type tag = [`LineSearchWarning]
type t = [`BaseException | `LineSearchWarning | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_exception x = (x :> [`BaseException] Obj.t)
let with_traceback ~tb self =
   Py.Module.get_function_with_keywords (to_pyobject self) "with_traceback"
     [||]
     (Wrap_utils.keyword_args [("tb", Some(tb ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let line_search ?gfk ?old_fval ?old_old_fval ?args ?c1 ?c2 ?amax ?amin ?xtol ~f ~fprime ~xk ~pk () =
   Py.Module.get_function_with_keywords __wrap_namespace "line_search"
     [||]
     (Wrap_utils.keyword_args [("gfk", Wrap_utils.Option.map gfk Np.Obj.to_pyobject); ("old_fval", Wrap_utils.Option.map old_fval Py.Float.of_float); ("old_old_fval", Wrap_utils.Option.map old_old_fval Py.Float.of_float); ("args", args); ("c1", c1); ("c2", c2); ("amax", amax); ("amin", amin); ("xtol", xtol); ("f", Some(f )); ("fprime", Some(fprime )); ("xk", Some(xk |> Np.Obj.to_pyobject)); ("pk", Some(pk |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let line_search_BFGS ?args ?c1 ?alpha0 ~f ~xk ~pk ~gfk ~old_fval () =
   Py.Module.get_function_with_keywords __wrap_namespace "line_search_BFGS"
     [||]
     (Wrap_utils.keyword_args [("args", args); ("c1", c1); ("alpha0", alpha0); ("f", Some(f )); ("xk", Some(xk )); ("pk", Some(pk )); ("gfk", Some(gfk )); ("old_fval", Some(old_fval ))])

                  let line_search_armijo ?args ?c1 ?alpha0 ~f ~xk ~pk ~gfk ~old_fval () =
                     Py.Module.get_function_with_keywords __wrap_namespace "line_search_armijo"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("c1", Wrap_utils.Option.map c1 Py.Float.of_float); ("alpha0", Wrap_utils.Option.map alpha0 (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("f", Some(f )); ("xk", Some(xk |> Np.Obj.to_pyobject)); ("pk", Some(pk |> Np.Obj.to_pyobject)); ("gfk", Some(gfk |> Np.Obj.to_pyobject)); ("old_fval", Some(old_fval |> Py.Float.of_float))])

let line_search_wolfe1 ?gfk ?old_fval ?old_old_fval ?args ?c1 ?c2 ?amax ?amin ?xtol ~f ~fprime ~xk ~pk () =
   Py.Module.get_function_with_keywords __wrap_namespace "line_search_wolfe1"
     [||]
     (Wrap_utils.keyword_args [("gfk", Wrap_utils.Option.map gfk Np.Obj.to_pyobject); ("old_fval", Wrap_utils.Option.map old_fval Py.Float.of_float); ("old_old_fval", Wrap_utils.Option.map old_old_fval Py.Float.of_float); ("args", args); ("c1", c1); ("c2", c2); ("amax", amax); ("amin", amin); ("xtol", xtol); ("f", Some(f )); ("fprime", Some(fprime )); ("xk", Some(xk |> Np.Obj.to_pyobject)); ("pk", Some(pk |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let line_search_wolfe2 ?gfk ?old_fval ?old_old_fval ?args ?c1 ?c2 ?amax ?extra_condition ?maxiter ~f ~myfprime ~xk ~pk () =
   Py.Module.get_function_with_keywords __wrap_namespace "line_search_wolfe2"
     [||]
     (Wrap_utils.keyword_args [("gfk", Wrap_utils.Option.map gfk Np.Obj.to_pyobject); ("old_fval", Wrap_utils.Option.map old_fval Py.Float.of_float); ("old_old_fval", Wrap_utils.Option.map old_old_fval Py.Float.of_float); ("args", args); ("c1", Wrap_utils.Option.map c1 Py.Float.of_float); ("c2", Wrap_utils.Option.map c2 Py.Float.of_float); ("amax", Wrap_utils.Option.map amax Py.Float.of_float); ("extra_condition", extra_condition); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("f", Some(f )); ("myfprime", Some(myfprime )); ("xk", Some(xk |> Np.Obj.to_pyobject)); ("pk", Some(pk |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> if Py.is_none py then None else Some (Py.Float.to_float py)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), ((fun py -> if Py.is_none py then None else Some (Py.Float.to_float py)) (Py.Tuple.get x 3)), (Py.Float.to_float (Py.Tuple.get x 4)), ((fun py -> if Py.is_none py then None else Some (Py.Float.to_float py)) (Py.Tuple.get x 5))))
let scalar_search_armijo ?c1 ?alpha0 ?amin ~phi ~phi0 ~derphi0 () =
   Py.Module.get_function_with_keywords __wrap_namespace "scalar_search_armijo"
     [||]
     (Wrap_utils.keyword_args [("c1", c1); ("alpha0", alpha0); ("amin", amin); ("phi", Some(phi )); ("phi0", Some(phi0 )); ("derphi0", Some(derphi0 ))])

let scalar_search_wolfe1 ?phi0 ?old_phi0 ?derphi0 ?c1 ?c2 ?amax ?amin ?xtol ~phi ~derphi () =
   Py.Module.get_function_with_keywords __wrap_namespace "scalar_search_wolfe1"
     [||]
     (Wrap_utils.keyword_args [("phi0", Wrap_utils.Option.map phi0 Py.Float.of_float); ("old_phi0", Wrap_utils.Option.map old_phi0 Py.Float.of_float); ("derphi0", Wrap_utils.Option.map derphi0 Py.Float.of_float); ("c1", Wrap_utils.Option.map c1 Py.Float.of_float); ("c2", Wrap_utils.Option.map c2 Py.Float.of_float); ("amax", amax); ("amin", amin); ("xtol", Wrap_utils.Option.map xtol Py.Float.of_float); ("phi", Some(phi )); ("derphi", Some(derphi ))])
     |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let scalar_search_wolfe2 ?phi0 ?old_phi0 ?derphi0 ?c1 ?c2 ?amax ?extra_condition ?maxiter ~phi ~derphi () =
   Py.Module.get_function_with_keywords __wrap_namespace "scalar_search_wolfe2"
     [||]
     (Wrap_utils.keyword_args [("phi0", Wrap_utils.Option.map phi0 Py.Float.of_float); ("old_phi0", Wrap_utils.Option.map old_phi0 Py.Float.of_float); ("derphi0", Wrap_utils.Option.map derphi0 Py.Float.of_float); ("c1", Wrap_utils.Option.map c1 Py.Float.of_float); ("c2", Wrap_utils.Option.map c2 Py.Float.of_float); ("amax", Wrap_utils.Option.map amax Py.Float.of_float); ("extra_condition", extra_condition); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("phi", Some(phi )); ("derphi", Some(derphi ))])
     |> (fun x -> (((fun py -> if Py.is_none py then None else Some (Py.Float.to_float py)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2)), ((fun py -> if Py.is_none py then None else Some (Py.Float.to_float py)) (Py.Tuple.get x 3))))
let warn ?category ?stacklevel ?source ~message () =
   Py.Module.get_function_with_keywords __wrap_namespace "warn"
     [||]
     (Wrap_utils.keyword_args [("category", category); ("stacklevel", stacklevel); ("source", source); ("message", Some(message ))])


end
module Minpack = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.optimize.minpack"

let get_py name = Py.Module.get __wrap_namespace name
module Error = struct
type tag = [`Error]
type t = [`BaseException | `Error | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_exception x = (x :> [`BaseException] Obj.t)
let with_traceback ~tb self =
   Py.Module.get_function_with_keywords (to_pyobject self) "with_traceback"
     [||]
     (Wrap_utils.keyword_args [("tb", Some(tb ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Finfo = struct
type tag = [`Finfo]
type t = [`Finfo | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create dtype =
                     Py.Module.get_function_with_keywords __wrap_namespace "finfo"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Some(dtype |> (function
| `F x -> Py.Float.of_float x
| `Instance x -> Wrap_utils.id x
| `Dtype x -> Np.Dtype.to_pyobject x
)))])
                       |> of_pyobject

let bits_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "bits" with
  | None -> failwith "attribute bits not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let bits self = match bits_opt self with
  | None -> raise Not_found
  | Some x -> x

let eps_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "eps" with
  | None -> failwith "attribute eps not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let eps self = match eps_opt self with
  | None -> raise Not_found
  | Some x -> x

let epsneg_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "epsneg" with
  | None -> failwith "attribute epsneg not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let epsneg self = match epsneg_opt self with
  | None -> raise Not_found
  | Some x -> x

let iexp_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "iexp" with
  | None -> failwith "attribute iexp not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let iexp self = match iexp_opt self with
  | None -> raise Not_found
  | Some x -> x

let machar_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "machar" with
  | None -> failwith "attribute machar not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let machar self = match machar_opt self with
  | None -> raise Not_found
  | Some x -> x

let machep_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "machep" with
  | None -> failwith "attribute machep not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let machep self = match machep_opt self with
  | None -> raise Not_found
  | Some x -> x

let max_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "max" with
  | None -> failwith "attribute max not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let max self = match max_opt self with
  | None -> raise Not_found
  | Some x -> x

let maxexp_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "maxexp" with
  | None -> failwith "attribute maxexp not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let maxexp self = match maxexp_opt self with
  | None -> raise Not_found
  | Some x -> x

let min_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "min" with
  | None -> failwith "attribute min not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let min self = match min_opt self with
  | None -> raise Not_found
  | Some x -> x

let minexp_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "minexp" with
  | None -> failwith "attribute minexp not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let minexp self = match minexp_opt self with
  | None -> raise Not_found
  | Some x -> x

let negep_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "negep" with
  | None -> failwith "attribute negep not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let negep self = match negep_opt self with
  | None -> raise Not_found
  | Some x -> x

let nexp_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nexp" with
  | None -> failwith "attribute nexp not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let nexp self = match nexp_opt self with
  | None -> raise Not_found
  | Some x -> x

let nmant_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nmant" with
  | None -> failwith "attribute nmant not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let nmant self = match nmant_opt self with
  | None -> raise Not_found
  | Some x -> x

let precision_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "precision" with
  | None -> failwith "attribute precision not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let precision self = match precision_opt self with
  | None -> raise Not_found
  | Some x -> x

let resolution_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "resolution" with
  | None -> failwith "attribute resolution not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let resolution self = match resolution_opt self with
  | None -> raise Not_found
  | Some x -> x

let tiny_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "tiny" with
  | None -> failwith "attribute tiny not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let tiny self = match tiny_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let asarray ?dtype ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let atleast_1d arys =
   Py.Module.get_function_with_keywords __wrap_namespace "atleast_1d"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arys)])
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let check_gradient ?args ?col_deriv ~fcn ~dfcn ~x0 () =
   Py.Module.get_function_with_keywords __wrap_namespace "check_gradient"
     [||]
     (Wrap_utils.keyword_args [("args", args); ("col_deriv", col_deriv); ("fcn", Some(fcn )); ("Dfcn", Some(dfcn )); ("x0", Some(x0 ))])

let cholesky ?lower ?overwrite_a ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "cholesky"
     [||]
     (Wrap_utils.keyword_args [("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let curve_fit ?p0 ?sigma ?absolute_sigma ?check_finite ?bounds ?method_ ?jac ?kwargs ~f ~xdata ~ydata () =
                     Py.Module.get_function_with_keywords __wrap_namespace "curve_fit"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("p0", Wrap_utils.Option.map p0 Np.Obj.to_pyobject); ("sigma", sigma); ("absolute_sigma", Wrap_utils.Option.map absolute_sigma Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("bounds", bounds); ("method", Wrap_utils.Option.map method_ (function
| `Dogbox -> Py.String.of_string "dogbox"
| `Trf -> Py.String.of_string "trf"
| `Lm -> Py.String.of_string "lm"
)); ("jac", Wrap_utils.Option.map jac (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("f", Some(f )); ("xdata", Some(xdata |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("ydata", Some(ydata |> Np.Obj.to_pyobject))]) (match kwargs with None -> [] | Some x -> x))
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let dot ?out ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "dot"
     [||]
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let eye ?m ?k ?dtype ?order ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "eye"
                       [||]
                       (Wrap_utils.keyword_args [("M", Wrap_utils.Option.map m Py.Int.of_int); ("k", Wrap_utils.Option.map k Py.Int.of_int); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("N", Some(n |> Py.Int.of_int))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let fixed_point ?args ?xtol ?maxiter ?method_ ~func ~x0 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fixed_point"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("xtol", Wrap_utils.Option.map xtol Py.Float.of_float); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("method", Wrap_utils.Option.map method_ (function
| `Del2 -> Py.String.of_string "del2"
| `Iteration -> Py.String.of_string "iteration"
)); ("func", Some(func )); ("x0", Some(x0 |> Np.Obj.to_pyobject))])

let fsolve ?args ?fprime ?full_output ?col_deriv ?xtol ?maxfev ?band ?epsfcn ?factor ?diag ~func ~x0 () =
   Py.Module.get_function_with_keywords __wrap_namespace "fsolve"
     [||]
     (Wrap_utils.keyword_args [("args", args); ("fprime", fprime); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("col_deriv", Wrap_utils.Option.map col_deriv Py.Bool.of_bool); ("xtol", Wrap_utils.Option.map xtol Py.Float.of_float); ("maxfev", Wrap_utils.Option.map maxfev Py.Int.of_int); ("band", band); ("epsfcn", Wrap_utils.Option.map epsfcn Py.Float.of_float); ("factor", Wrap_utils.Option.map factor Py.Float.of_float); ("diag", diag); ("func", Some(func )); ("x0", Some(x0 |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Py.String.to_string (Py.Tuple.get x 3))))
                  let greater ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "greater"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let inv ?overwrite_a ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "inv"
     [||]
     (Wrap_utils.keyword_args [("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let issubdtype ~arg1 ~arg2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "issubdtype"
     [||]
     (Wrap_utils.keyword_args [("arg1", Some(arg1 )); ("arg2", Some(arg2 ))])
     |> Py.Bool.to_bool
                  let least_squares ?jac ?bounds ?method_ ?ftol ?xtol ?gtol ?x_scale ?loss ?f_scale ?diff_step ?tr_solver ?tr_options ?jac_sparsity ?max_nfev ?verbose ?args ?kwargs ~fun_ ~x0 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "least_squares"
                       [||]
                       (Wrap_utils.keyword_args [("jac", Wrap_utils.Option.map jac (function
| `T2_point -> Py.String.of_string "2-point"
| `Callable x -> Wrap_utils.id x
| `Cs -> Py.String.of_string "cs"
| `T3_point -> Py.String.of_string "3-point"
)); ("bounds", bounds); ("method", Wrap_utils.Option.map method_ (function
| `Trf -> Py.String.of_string "trf"
| `Dogbox -> Py.String.of_string "dogbox"
| `Lm -> Py.String.of_string "lm"
)); ("ftol", Wrap_utils.Option.map ftol (function
| `F x -> Py.Float.of_float x
| `None -> Py.none
)); ("xtol", Wrap_utils.Option.map xtol (function
| `F x -> Py.Float.of_float x
| `None -> Py.none
)); ("gtol", Wrap_utils.Option.map gtol (function
| `F x -> Py.Float.of_float x
| `None -> Py.none
)); ("x_scale", Wrap_utils.Option.map x_scale (function
| `Jac -> Py.String.of_string "jac"
| `Ndarray x -> Np.Obj.to_pyobject x
)); ("loss", Wrap_utils.Option.map loss (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("f_scale", Wrap_utils.Option.map f_scale Py.Float.of_float); ("diff_step", Wrap_utils.Option.map diff_step Np.Obj.to_pyobject); ("tr_solver", Wrap_utils.Option.map tr_solver (function
| `Exact -> Py.String.of_string "exact"
| `Lsmr -> Py.String.of_string "lsmr"
)); ("tr_options", tr_options); ("jac_sparsity", Wrap_utils.Option.map jac_sparsity Np.Obj.to_pyobject); ("max_nfev", Wrap_utils.Option.map max_nfev Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose (function
| `Two -> Py.Int.of_int 2
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
)); ("args", args); ("kwargs", kwargs); ("fun", Some(fun_ )); ("x0", Some(x0 |> (function
| `F x -> Py.Float.of_float x
| `Ndarray x -> Np.Obj.to_pyobject x
)))])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 3)), (Py.Float.to_float (Py.Tuple.get x 4)), (Wrap_utils.id (Py.Tuple.get x 5)), (Py.Int.to_int (Py.Tuple.get x 6)), ((fun py -> if Py.is_none py then None else Some (Py.Int.to_int py)) (Py.Tuple.get x 7)), (Py.Int.to_int (Py.Tuple.get x 8)), (Py.String.to_string (Py.Tuple.get x 9)), (Py.Bool.to_bool (Py.Tuple.get x 10))))
let leastsq ?args ?dfun ?full_output ?col_deriv ?ftol ?xtol ?gtol ?maxfev ?epsfcn ?factor ?diag ~func ~x0 () =
   Py.Module.get_function_with_keywords __wrap_namespace "leastsq"
     [||]
     (Wrap_utils.keyword_args [("args", args); ("Dfun", dfun); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("col_deriv", Wrap_utils.Option.map col_deriv Py.Bool.of_bool); ("ftol", Wrap_utils.Option.map ftol Py.Float.of_float); ("xtol", Wrap_utils.Option.map xtol Py.Float.of_float); ("gtol", Wrap_utils.Option.map gtol Py.Float.of_float); ("maxfev", Wrap_utils.Option.map maxfev Py.Int.of_int); ("epsfcn", Wrap_utils.Option.map epsfcn Py.Float.of_float); ("factor", Wrap_utils.Option.map factor Py.Float.of_float); ("diag", diag); ("func", Some(func )); ("x0", Some(x0 |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), (Py.String.to_string (Py.Tuple.get x 3)), (Py.Int.to_int (Py.Tuple.get x 4))))
let prepare_bounds ~bounds ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "prepare_bounds"
     [||]
     (Wrap_utils.keyword_args [("bounds", Some(bounds )); ("n", Some(n ))])

                  let prod ?axis ?dtype ?out ?keepdims ?initial ?where ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "prod"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("initial", Wrap_utils.Option.map initial (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("where", where); ("a", Some(a |> Np.Obj.to_pyobject))])

        let shape a =
           Py.Module.get_function_with_keywords __wrap_namespace "shape"
             [||]
             (Wrap_utils.keyword_args [("a", Some(a |> Np.Obj.to_pyobject))])
             |> (fun py -> let len = Py.Sequence.length py in Array.init len
(fun i -> Py.Int.to_int (Py.Sequence.get_item py i)))
                  let solve_triangular ?trans ?lower ?unit_diagonal ?overwrite_b ?debug ?check_finite ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "solve_triangular"
                       [||]
                       (Wrap_utils.keyword_args [("trans", Wrap_utils.Option.map trans (function
| `C -> Py.String.of_string "C"
| `Two -> Py.Int.of_int 2
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `T -> Py.String.of_string "T"
| `N -> Py.String.of_string "N"
)); ("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("unit_diagonal", Wrap_utils.Option.map unit_diagonal Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("debug", debug); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b ))])

                  let svd ?full_matrices ?compute_uv ?overwrite_a ?check_finite ?lapack_driver ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "svd"
                       [||]
                       (Wrap_utils.keyword_args [("full_matrices", Wrap_utils.Option.map full_matrices Py.Bool.of_bool); ("compute_uv", Wrap_utils.Option.map compute_uv Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("lapack_driver", Wrap_utils.Option.map lapack_driver (function
| `Gesdd -> Py.String.of_string "gesdd"
| `Gesvd -> Py.String.of_string "gesvd"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
                  let take ?axis ?out ?mode ~a ~indices () =
                     Py.Module.get_function_with_keywords __wrap_namespace "take"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `T_Ni_Nj_Nk_ x -> Wrap_utils.id x
)); ("mode", Wrap_utils.Option.map mode (function
| `Raise -> Py.String.of_string "raise"
| `Wrap -> Py.String.of_string "wrap"
| `Clip -> Py.String.of_string "clip"
)); ("a", Some(a )); ("indices", Some(indices ))])

let transpose ?axes ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "transpose"
     [||]
     (Wrap_utils.keyword_args [("axes", axes); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let triu ?k ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "triu"
     [||]
     (Wrap_utils.keyword_args [("k", k); ("m", Some(m ))])

                  let zeros ?dtype ?order ~shape () =
                     Py.Module.get_function_with_keywords __wrap_namespace "zeros"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
module Minpack2 = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.optimize.minpack2"

let get_py name = Py.Module.get __wrap_namespace name

end
module ModuleTNC = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.optimize.moduleTNC"

let get_py name = Py.Module.get __wrap_namespace name

end
module Nonlin = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.optimize.nonlin"

let get_py name = Py.Module.get __wrap_namespace name
module Anderson = struct
type tag = [`Anderson]
type t = [`Anderson | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?alpha ?w0 ?m () =
   Py.Module.get_function_with_keywords __wrap_namespace "Anderson"
     [||]
     (Wrap_utils.keyword_args [("alpha", alpha); ("w0", Wrap_utils.Option.map w0 Py.Float.of_float); ("M", Wrap_utils.Option.map m Py.Float.of_float)])
     |> of_pyobject
let aspreconditioner self =
   Py.Module.get_function_with_keywords (to_pyobject self) "aspreconditioner"
     [||]
     []

let matvec ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matvec"
     [||]
     (Wrap_utils.keyword_args [("f", Some(f ))])

let setup ~x0 ~f0 ~func self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setup"
     [||]
     (Wrap_utils.keyword_args [("x0", Some(x0 )); ("f0", Some(f0 )); ("func", Some(func ))])

let solve ?tol ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "solve"
     [||]
     (Wrap_utils.keyword_args [("tol", tol); ("f", Some(f ))])

let update ~x ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "update"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x )); ("f", Some(f ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module BroydenFirst = struct
type tag = [`BroydenFirst]
type t = [`BroydenFirst | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?alpha ?reduction_method ?max_rank () =
   Py.Module.get_function_with_keywords __wrap_namespace "BroydenFirst"
     [||]
     (Wrap_utils.keyword_args [("alpha", alpha); ("reduction_method", reduction_method); ("max_rank", max_rank)])
     |> of_pyobject
let aspreconditioner self =
   Py.Module.get_function_with_keywords (to_pyobject self) "aspreconditioner"
     [||]
     []

let matvec ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matvec"
     [||]
     (Wrap_utils.keyword_args [("f", Some(f ))])

let rmatvec ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatvec"
     [||]
     (Wrap_utils.keyword_args [("f", Some(f ))])

let rsolve ?tol ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rsolve"
     [||]
     (Wrap_utils.keyword_args [("tol", tol); ("f", Some(f ))])

let setup ~x ~f ~func self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setup"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x )); ("F", Some(f )); ("func", Some(func ))])

let solve ?tol ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "solve"
     [||]
     (Wrap_utils.keyword_args [("tol", tol); ("f", Some(f ))])

let todense self =
   Py.Module.get_function_with_keywords (to_pyobject self) "todense"
     [||]
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let update ~x ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "update"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x )); ("f", Some(f ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module BroydenSecond = struct
type tag = [`BroydenSecond]
type t = [`BroydenSecond | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?alpha ?reduction_method ?max_rank () =
   Py.Module.get_function_with_keywords __wrap_namespace "BroydenSecond"
     [||]
     (Wrap_utils.keyword_args [("alpha", alpha); ("reduction_method", reduction_method); ("max_rank", max_rank)])
     |> of_pyobject
let aspreconditioner self =
   Py.Module.get_function_with_keywords (to_pyobject self) "aspreconditioner"
     [||]
     []

let matvec ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matvec"
     [||]
     (Wrap_utils.keyword_args [("f", Some(f ))])

let rmatvec ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatvec"
     [||]
     (Wrap_utils.keyword_args [("f", Some(f ))])

let rsolve ?tol ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rsolve"
     [||]
     (Wrap_utils.keyword_args [("tol", tol); ("f", Some(f ))])

let setup ~x ~f ~func self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setup"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x )); ("F", Some(f )); ("func", Some(func ))])

let solve ?tol ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "solve"
     [||]
     (Wrap_utils.keyword_args [("tol", tol); ("f", Some(f ))])

let todense self =
   Py.Module.get_function_with_keywords (to_pyobject self) "todense"
     [||]
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let update ~x ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "update"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x )); ("f", Some(f ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module DiagBroyden = struct
type tag = [`DiagBroyden]
type t = [`DiagBroyden | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?alpha () =
   Py.Module.get_function_with_keywords __wrap_namespace "DiagBroyden"
     [||]
     (Wrap_utils.keyword_args [("alpha", alpha)])
     |> of_pyobject
let aspreconditioner self =
   Py.Module.get_function_with_keywords (to_pyobject self) "aspreconditioner"
     [||]
     []

let matvec ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matvec"
     [||]
     (Wrap_utils.keyword_args [("f", Some(f ))])

let rmatvec ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatvec"
     [||]
     (Wrap_utils.keyword_args [("f", Some(f ))])

let rsolve ?tol ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rsolve"
     [||]
     (Wrap_utils.keyword_args [("tol", tol); ("f", Some(f ))])

let setup ~x ~f ~func self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setup"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x )); ("F", Some(f )); ("func", Some(func ))])

let solve ?tol ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "solve"
     [||]
     (Wrap_utils.keyword_args [("tol", tol); ("f", Some(f ))])

let todense self =
   Py.Module.get_function_with_keywords (to_pyobject self) "todense"
     [||]
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let update ~x ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "update"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x )); ("f", Some(f ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ExcitingMixing = struct
type tag = [`ExcitingMixing]
type t = [`ExcitingMixing | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?alpha ?alphamax () =
   Py.Module.get_function_with_keywords __wrap_namespace "ExcitingMixing"
     [||]
     (Wrap_utils.keyword_args [("alpha", alpha); ("alphamax", Wrap_utils.Option.map alphamax Py.Float.of_float)])
     |> of_pyobject
let aspreconditioner self =
   Py.Module.get_function_with_keywords (to_pyobject self) "aspreconditioner"
     [||]
     []

let matvec ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matvec"
     [||]
     (Wrap_utils.keyword_args [("f", Some(f ))])

let rmatvec ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatvec"
     [||]
     (Wrap_utils.keyword_args [("f", Some(f ))])

let rsolve ?tol ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rsolve"
     [||]
     (Wrap_utils.keyword_args [("tol", tol); ("f", Some(f ))])

let setup ~x ~f ~func self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setup"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x )); ("F", Some(f )); ("func", Some(func ))])

let solve ?tol ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "solve"
     [||]
     (Wrap_utils.keyword_args [("tol", tol); ("f", Some(f ))])

let todense self =
   Py.Module.get_function_with_keywords (to_pyobject self) "todense"
     [||]
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let update ~x ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "update"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x )); ("f", Some(f ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module GenericBroyden = struct
type tag = [`GenericBroyden]
type t = [`GenericBroyden | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kw () =
   Py.Module.get_function_with_keywords __wrap_namespace "GenericBroyden"
     [||]
     (match kw with None -> [] | Some x -> x)
     |> of_pyobject
let aspreconditioner self =
   Py.Module.get_function_with_keywords (to_pyobject self) "aspreconditioner"
     [||]
     []

let setup ~x0 ~f0 ~func self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setup"
     [||]
     (Wrap_utils.keyword_args [("x0", Some(x0 )); ("f0", Some(f0 )); ("func", Some(func ))])

let solve ?tol ~v self =
   Py.Module.get_function_with_keywords (to_pyobject self) "solve"
     [||]
     (Wrap_utils.keyword_args [("tol", tol); ("v", Some(v ))])

let update ~x ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "update"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x )); ("f", Some(f ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module InverseJacobian = struct
type tag = [`InverseJacobian]
type t = [`InverseJacobian | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create jacobian =
   Py.Module.get_function_with_keywords __wrap_namespace "InverseJacobian"
     [||]
     (Wrap_utils.keyword_args [("jacobian", Some(jacobian ))])
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Jacobian = struct
type tag = [`Jacobian]
type t = [`Jacobian | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kw () =
   Py.Module.get_function_with_keywords __wrap_namespace "Jacobian"
     [||]
     (match kw with None -> [] | Some x -> x)
     |> of_pyobject
let aspreconditioner self =
   Py.Module.get_function_with_keywords (to_pyobject self) "aspreconditioner"
     [||]
     []

let setup ~x ~f ~func self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setup"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x )); ("F", Some(f )); ("func", Some(func ))])

let solve ?tol ~v self =
   Py.Module.get_function_with_keywords (to_pyobject self) "solve"
     [||]
     (Wrap_utils.keyword_args [("tol", tol); ("v", Some(v ))])

let update ~x ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "update"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x )); ("F", Some(f ))])


let shape_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "shape" with
  | None -> failwith "attribute shape not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let shape self = match shape_opt self with
  | None -> raise Not_found
  | Some x -> x

let dtype_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dtype" with
  | None -> failwith "attribute dtype not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let dtype self = match dtype_opt self with
  | None -> raise Not_found
  | Some x -> x

let func_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "func" with
  | None -> failwith "attribute func not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let func self = match func_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module KrylovJacobian = struct
type tag = [`KrylovJacobian]
type t = [`KrylovJacobian | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?rdiff ?method_ ?inner_maxiter ?inner_M ?outer_k ?kw () =
                     Py.Module.get_function_with_keywords __wrap_namespace "KrylovJacobian"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("rdiff", rdiff); ("method", Wrap_utils.Option.map method_ (function
| `Gmres -> Py.String.of_string "gmres"
| `Cgs -> Py.String.of_string "cgs"
| `Lgmres -> Py.String.of_string "lgmres"
| `Minres -> Py.String.of_string "minres"
| `Bicgstab -> Py.String.of_string "bicgstab"
| `Callable x -> Wrap_utils.id x
)); ("inner_maxiter", Wrap_utils.Option.map inner_maxiter Py.Int.of_int); ("inner_M", inner_M); ("outer_k", Wrap_utils.Option.map outer_k Py.Int.of_int)]) (match kw with None -> [] | Some x -> x))
                       |> of_pyobject
let aspreconditioner self =
   Py.Module.get_function_with_keywords (to_pyobject self) "aspreconditioner"
     [||]
     []

let matvec ~v self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matvec"
     [||]
     (Wrap_utils.keyword_args [("v", Some(v ))])

let setup ~x ~f ~func self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setup"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x )); ("f", Some(f )); ("func", Some(func ))])

let solve ?tol ~rhs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "solve"
     [||]
     (Wrap_utils.keyword_args [("tol", tol); ("rhs", Some(rhs ))])

let update ~x ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "update"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x )); ("f", Some(f ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LinearMixing = struct
type tag = [`LinearMixing]
type t = [`LinearMixing | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?alpha () =
   Py.Module.get_function_with_keywords __wrap_namespace "LinearMixing"
     [||]
     (Wrap_utils.keyword_args [("alpha", alpha)])
     |> of_pyobject
let aspreconditioner self =
   Py.Module.get_function_with_keywords (to_pyobject self) "aspreconditioner"
     [||]
     []

let matvec ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matvec"
     [||]
     (Wrap_utils.keyword_args [("f", Some(f ))])

let rmatvec ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatvec"
     [||]
     (Wrap_utils.keyword_args [("f", Some(f ))])

let rsolve ?tol ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rsolve"
     [||]
     (Wrap_utils.keyword_args [("tol", tol); ("f", Some(f ))])

let setup ~x0 ~f0 ~func self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setup"
     [||]
     (Wrap_utils.keyword_args [("x0", Some(x0 )); ("f0", Some(f0 )); ("func", Some(func ))])

let solve ?tol ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "solve"
     [||]
     (Wrap_utils.keyword_args [("tol", tol); ("f", Some(f ))])

let todense self =
   Py.Module.get_function_with_keywords (to_pyobject self) "todense"
     [||]
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let update ~x ~f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "update"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x )); ("f", Some(f ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LowRankMatrix = struct
type tag = [`LowRankMatrix]
type t = [`LowRankMatrix | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ~alpha ~n ~dtype () =
   Py.Module.get_function_with_keywords __wrap_namespace "LowRankMatrix"
     [||]
     (Wrap_utils.keyword_args [("alpha", Some(alpha )); ("n", Some(n )); ("dtype", Some(dtype ))])
     |> of_pyobject
let append ~c ~d self =
   Py.Module.get_function_with_keywords (to_pyobject self) "append"
     [||]
     (Wrap_utils.keyword_args [("c", Some(c )); ("d", Some(d ))])

let collapse self =
   Py.Module.get_function_with_keywords (to_pyobject self) "collapse"
     [||]
     []

let matvec ~v self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matvec"
     [||]
     (Wrap_utils.keyword_args [("v", Some(v ))])

let restart_reduce ~rank self =
   Py.Module.get_function_with_keywords (to_pyobject self) "restart_reduce"
     [||]
     (Wrap_utils.keyword_args [("rank", Some(rank ))])

let rmatvec ~v self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatvec"
     [||]
     (Wrap_utils.keyword_args [("v", Some(v ))])

let rsolve ?tol ~v self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rsolve"
     [||]
     (Wrap_utils.keyword_args [("tol", tol); ("v", Some(v ))])

let simple_reduce ~rank self =
   Py.Module.get_function_with_keywords (to_pyobject self) "simple_reduce"
     [||]
     (Wrap_utils.keyword_args [("rank", Some(rank ))])

let solve ?tol ~v self =
   Py.Module.get_function_with_keywords (to_pyobject self) "solve"
     [||]
     (Wrap_utils.keyword_args [("tol", tol); ("v", Some(v ))])

let svd_reduce ?to_retain ~max_rank self =
   Py.Module.get_function_with_keywords (to_pyobject self) "svd_reduce"
     [||]
     (Wrap_utils.keyword_args [("to_retain", Wrap_utils.Option.map to_retain Py.Int.of_int); ("max_rank", Some(max_rank |> Py.Int.of_int))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module NoConvergence = struct
type tag = [`NoConvergence]
type t = [`BaseException | `NoConvergence | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_exception x = (x :> [`BaseException] Obj.t)
let with_traceback ~tb self =
   Py.Module.get_function_with_keywords (to_pyobject self) "with_traceback"
     [||]
     (Wrap_utils.keyword_args [("tb", Some(tb ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module TerminationCondition = struct
type tag = [`TerminationCondition]
type t = [`Object | `TerminationCondition] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?f_tol ?f_rtol ?x_tol ?x_rtol ?iter ?norm () =
   Py.Module.get_function_with_keywords __wrap_namespace "TerminationCondition"
     [||]
     (Wrap_utils.keyword_args [("f_tol", f_tol); ("f_rtol", f_rtol); ("x_tol", x_tol); ("x_rtol", x_rtol); ("iter", iter); ("norm", norm)])
     |> of_pyobject
let check ~f ~x ~dx self =
   Py.Module.get_function_with_keywords (to_pyobject self) "check"
     [||]
     (Wrap_utils.keyword_args [("f", Some(f )); ("x", Some(x )); ("dx", Some(dx ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let anderson ?iter ?alpha ?w0 ?m ?verbose ?maxiter ?f_tol ?f_rtol ?x_tol ?x_rtol ?tol_norm ?line_search ?callback ?kw ~f ~xin () =
                     Py.Module.get_function_with_keywords __wrap_namespace "anderson"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("iter", Wrap_utils.Option.map iter Py.Int.of_int); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("w0", Wrap_utils.Option.map w0 Py.Float.of_float); ("M", Wrap_utils.Option.map m Py.Float.of_float); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("f_tol", Wrap_utils.Option.map f_tol Py.Float.of_float); ("f_rtol", Wrap_utils.Option.map f_rtol Py.Float.of_float); ("x_tol", Wrap_utils.Option.map x_tol Py.Float.of_float); ("x_rtol", Wrap_utils.Option.map x_rtol Py.Float.of_float); ("tol_norm", tol_norm); ("line_search", Wrap_utils.Option.map line_search (function
| `Wolfe -> Py.String.of_string "wolfe"
| `Armijo -> Py.String.of_string "armijo"
| `None -> Py.none
)); ("callback", callback); ("F", Some(f )); ("xin", Some(xin |> Np.Obj.to_pyobject))]) (match kw with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let asarray ?dtype ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let asjacobian j =
   Py.Module.get_function_with_keywords __wrap_namespace "asjacobian"
     [||]
     (Wrap_utils.keyword_args [("J", Some(j ))])

                  let broyden1 ?iter ?alpha ?reduction_method ?max_rank ?verbose ?maxiter ?f_tol ?f_rtol ?x_tol ?x_rtol ?tol_norm ?line_search ?callback ?kw ~f ~xin () =
                     Py.Module.get_function_with_keywords __wrap_namespace "broyden1"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("iter", Wrap_utils.Option.map iter Py.Int.of_int); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("reduction_method", Wrap_utils.Option.map reduction_method (function
| `S x -> Py.String.of_string x
| `Tuple x -> Wrap_utils.id x
)); ("max_rank", Wrap_utils.Option.map max_rank Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("f_tol", Wrap_utils.Option.map f_tol Py.Float.of_float); ("f_rtol", Wrap_utils.Option.map f_rtol Py.Float.of_float); ("x_tol", Wrap_utils.Option.map x_tol Py.Float.of_float); ("x_rtol", Wrap_utils.Option.map x_rtol Py.Float.of_float); ("tol_norm", tol_norm); ("line_search", Wrap_utils.Option.map line_search (function
| `Wolfe -> Py.String.of_string "wolfe"
| `Armijo -> Py.String.of_string "armijo"
| `None -> Py.none
)); ("callback", callback); ("F", Some(f )); ("xin", Some(xin |> Np.Obj.to_pyobject))]) (match kw with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let broyden2 ?iter ?alpha ?reduction_method ?max_rank ?verbose ?maxiter ?f_tol ?f_rtol ?x_tol ?x_rtol ?tol_norm ?line_search ?callback ?kw ~f ~xin () =
                     Py.Module.get_function_with_keywords __wrap_namespace "broyden2"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("iter", Wrap_utils.Option.map iter Py.Int.of_int); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("reduction_method", Wrap_utils.Option.map reduction_method (function
| `S x -> Py.String.of_string x
| `Tuple x -> Wrap_utils.id x
)); ("max_rank", Wrap_utils.Option.map max_rank Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("f_tol", Wrap_utils.Option.map f_tol Py.Float.of_float); ("f_rtol", Wrap_utils.Option.map f_rtol Py.Float.of_float); ("x_tol", Wrap_utils.Option.map x_tol Py.Float.of_float); ("x_rtol", Wrap_utils.Option.map x_rtol Py.Float.of_float); ("tol_norm", tol_norm); ("line_search", Wrap_utils.Option.map line_search (function
| `Wolfe -> Py.String.of_string "wolfe"
| `Armijo -> Py.String.of_string "armijo"
| `None -> Py.none
)); ("callback", callback); ("F", Some(f )); ("xin", Some(xin |> Np.Obj.to_pyobject))]) (match kw with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let diagbroyden ?iter ?alpha ?verbose ?maxiter ?f_tol ?f_rtol ?x_tol ?x_rtol ?tol_norm ?line_search ?callback ?kw ~f ~xin () =
                     Py.Module.get_function_with_keywords __wrap_namespace "diagbroyden"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("iter", Wrap_utils.Option.map iter Py.Int.of_int); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("f_tol", Wrap_utils.Option.map f_tol Py.Float.of_float); ("f_rtol", Wrap_utils.Option.map f_rtol Py.Float.of_float); ("x_tol", Wrap_utils.Option.map x_tol Py.Float.of_float); ("x_rtol", Wrap_utils.Option.map x_rtol Py.Float.of_float); ("tol_norm", tol_norm); ("line_search", Wrap_utils.Option.map line_search (function
| `Wolfe -> Py.String.of_string "wolfe"
| `Armijo -> Py.String.of_string "armijo"
| `None -> Py.none
)); ("callback", callback); ("F", Some(f )); ("xin", Some(xin |> Np.Obj.to_pyobject))]) (match kw with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let dot ?out ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "dot"
     [||]
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let excitingmixing ?iter ?alpha ?alphamax ?verbose ?maxiter ?f_tol ?f_rtol ?x_tol ?x_rtol ?tol_norm ?line_search ?callback ?kw ~f ~xin () =
                     Py.Module.get_function_with_keywords __wrap_namespace "excitingmixing"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("iter", Wrap_utils.Option.map iter Py.Int.of_int); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("alphamax", Wrap_utils.Option.map alphamax Py.Float.of_float); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("f_tol", Wrap_utils.Option.map f_tol Py.Float.of_float); ("f_rtol", Wrap_utils.Option.map f_rtol Py.Float.of_float); ("x_tol", Wrap_utils.Option.map x_tol Py.Float.of_float); ("x_rtol", Wrap_utils.Option.map x_rtol Py.Float.of_float); ("tol_norm", tol_norm); ("line_search", Wrap_utils.Option.map line_search (function
| `Wolfe -> Py.String.of_string "wolfe"
| `Armijo -> Py.String.of_string "armijo"
| `None -> Py.none
)); ("callback", callback); ("F", Some(f )); ("xin", Some(xin |> Np.Obj.to_pyobject))]) (match kw with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let get_blas_funcs ?arrays ?dtype ~names () =
                     Py.Module.get_function_with_keywords __wrap_namespace "get_blas_funcs"
                       [||]
                       (Wrap_utils.keyword_args [("arrays", Wrap_utils.Option.map arrays (fun ml -> Py.List.of_list_map Np.Obj.to_pyobject ml)); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("names", Some(names |> (function
| `Sequence_of_str x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let inv ?overwrite_a ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "inv"
     [||]
     (Wrap_utils.keyword_args [("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let linearmixing ?iter ?alpha ?verbose ?maxiter ?f_tol ?f_rtol ?x_tol ?x_rtol ?tol_norm ?line_search ?callback ?kw ~f ~xin () =
                     Py.Module.get_function_with_keywords __wrap_namespace "linearmixing"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("iter", Wrap_utils.Option.map iter Py.Int.of_int); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("f_tol", Wrap_utils.Option.map f_tol Py.Float.of_float); ("f_rtol", Wrap_utils.Option.map f_rtol Py.Float.of_float); ("x_tol", Wrap_utils.Option.map x_tol Py.Float.of_float); ("x_rtol", Wrap_utils.Option.map x_rtol Py.Float.of_float); ("tol_norm", tol_norm); ("line_search", Wrap_utils.Option.map line_search (function
| `Wolfe -> Py.String.of_string "wolfe"
| `Armijo -> Py.String.of_string "armijo"
| `None -> Py.none
)); ("callback", callback); ("F", Some(f )); ("xin", Some(xin |> Np.Obj.to_pyobject))]) (match kw with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let maxnorm x =
   Py.Module.get_function_with_keywords __wrap_namespace "maxnorm"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

                  let newton_krylov ?iter ?rdiff ?method_ ?inner_maxiter ?inner_M ?outer_k ?verbose ?maxiter ?f_tol ?f_rtol ?x_tol ?x_rtol ?tol_norm ?line_search ?callback ?kw ~f ~xin () =
                     Py.Module.get_function_with_keywords __wrap_namespace "newton_krylov"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("iter", Wrap_utils.Option.map iter Py.Int.of_int); ("rdiff", Wrap_utils.Option.map rdiff Py.Float.of_float); ("method", Wrap_utils.Option.map method_ (function
| `Gmres -> Py.String.of_string "gmres"
| `Cgs -> Py.String.of_string "cgs"
| `Lgmres -> Py.String.of_string "lgmres"
| `Minres -> Py.String.of_string "minres"
| `Bicgstab -> Py.String.of_string "bicgstab"
| `Callable x -> Wrap_utils.id x
)); ("inner_maxiter", Wrap_utils.Option.map inner_maxiter Py.Int.of_int); ("inner_M", inner_M); ("outer_k", Wrap_utils.Option.map outer_k Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("f_tol", Wrap_utils.Option.map f_tol Py.Float.of_float); ("f_rtol", Wrap_utils.Option.map f_rtol Py.Float.of_float); ("x_tol", Wrap_utils.Option.map x_tol Py.Float.of_float); ("x_rtol", Wrap_utils.Option.map x_rtol Py.Float.of_float); ("tol_norm", tol_norm); ("line_search", Wrap_utils.Option.map line_search (function
| `Wolfe -> Py.String.of_string "wolfe"
| `Armijo -> Py.String.of_string "armijo"
| `None -> Py.none
)); ("callback", callback); ("F", Some(f )); ("xin", Some(xin |> Np.Obj.to_pyobject))]) (match kw with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let nonlin_solve ?jacobian ?iter ?verbose ?maxiter ?f_tol ?f_rtol ?x_tol ?x_rtol ?tol_norm ?line_search ?callback ?full_output ?raise_exception ~f ~x0 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "nonlin_solve"
                       [||]
                       (Wrap_utils.keyword_args [("jacobian", jacobian); ("iter", Wrap_utils.Option.map iter Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("f_tol", Wrap_utils.Option.map f_tol Py.Float.of_float); ("f_rtol", Wrap_utils.Option.map f_rtol Py.Float.of_float); ("x_tol", Wrap_utils.Option.map x_tol Py.Float.of_float); ("x_rtol", Wrap_utils.Option.map x_rtol Py.Float.of_float); ("tol_norm", tol_norm); ("line_search", Wrap_utils.Option.map line_search (function
| `Wolfe -> Py.String.of_string "wolfe"
| `Armijo -> Py.String.of_string "armijo"
| `None -> Py.none
)); ("callback", callback); ("full_output", full_output); ("raise_exception", raise_exception); ("F", Some(f )); ("x0", Some(x0 ))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let norm ?ord ?axis ?keepdims ?check_finite ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "norm"
                       [||]
                       (Wrap_utils.keyword_args [("ord", Wrap_utils.Option.map ord (function
| `PyObject x -> Wrap_utils.id x
| `Fro -> Py.String.of_string "fro"
)); ("axis", Wrap_utils.Option.map axis (function
| `T2_tuple_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a ))])

                  let qr ?overwrite_a ?lwork ?mode ?pivoting ?check_finite ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "qr"
                       [||]
                       (Wrap_utils.keyword_args [("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("lwork", Wrap_utils.Option.map lwork Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Full -> Py.String.of_string "full"
| `R -> Py.String.of_string "r"
| `Economic -> Py.String.of_string "economic"
| `Raw -> Py.String.of_string "raw"
)); ("pivoting", Wrap_utils.Option.map pivoting Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
let scalar_search_armijo ?c1 ?alpha0 ?amin ~phi ~phi0 ~derphi0 () =
   Py.Module.get_function_with_keywords __wrap_namespace "scalar_search_armijo"
     [||]
     (Wrap_utils.keyword_args [("c1", c1); ("alpha0", alpha0); ("amin", amin); ("phi", Some(phi )); ("phi0", Some(phi0 )); ("derphi0", Some(derphi0 ))])

let scalar_search_wolfe1 ?phi0 ?old_phi0 ?derphi0 ?c1 ?c2 ?amax ?amin ?xtol ~phi ~derphi () =
   Py.Module.get_function_with_keywords __wrap_namespace "scalar_search_wolfe1"
     [||]
     (Wrap_utils.keyword_args [("phi0", Wrap_utils.Option.map phi0 Py.Float.of_float); ("old_phi0", Wrap_utils.Option.map old_phi0 Py.Float.of_float); ("derphi0", Wrap_utils.Option.map derphi0 Py.Float.of_float); ("c1", Wrap_utils.Option.map c1 Py.Float.of_float); ("c2", Wrap_utils.Option.map c2 Py.Float.of_float); ("amax", amax); ("amin", amin); ("xtol", Wrap_utils.Option.map xtol Py.Float.of_float); ("phi", Some(phi )); ("derphi", Some(derphi ))])
     |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let solve ?sym_pos ?lower ?overwrite_a ?overwrite_b ?debug ?check_finite ?assume_a ?transposed ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "solve"
     [||]
     (Wrap_utils.keyword_args [("sym_pos", Wrap_utils.Option.map sym_pos Py.Bool.of_bool); ("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("debug", debug); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("assume_a", Wrap_utils.Option.map assume_a Py.String.of_string); ("transposed", Wrap_utils.Option.map transposed Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let svd ?full_matrices ?compute_uv ?overwrite_a ?check_finite ?lapack_driver ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "svd"
                       [||]
                       (Wrap_utils.keyword_args [("full_matrices", Wrap_utils.Option.map full_matrices Py.Bool.of_bool); ("compute_uv", Wrap_utils.Option.map compute_uv Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("lapack_driver", Wrap_utils.Option.map lapack_driver (function
| `Gesdd -> Py.String.of_string "gesdd"
| `Gesvd -> Py.String.of_string "gesvd"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
let vdot ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "vdot"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
module Optimize = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.optimize.optimize"

let get_py name = Py.Module.get __wrap_namespace name
module Brent = struct
type tag = [`Brent]
type t = [`Brent | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?args ?tol ?maxiter ?full_output ~func () =
   Py.Module.get_function_with_keywords __wrap_namespace "Brent"
     [||]
     (Wrap_utils.keyword_args [("args", args); ("tol", tol); ("maxiter", maxiter); ("full_output", full_output); ("func", Some(func ))])
     |> of_pyobject
let get_bracket_info self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_bracket_info"
     [||]
     []

let get_result ?full_output self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_result"
     [||]
     (Wrap_utils.keyword_args [("full_output", full_output)])

let optimize self =
   Py.Module.get_function_with_keywords (to_pyobject self) "optimize"
     [||]
     []

let set_bracket ?brack self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_bracket"
     [||]
     (Wrap_utils.keyword_args [("brack", brack)])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LineSearchWarning = struct
type tag = [`LineSearchWarning]
type t = [`BaseException | `LineSearchWarning | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_exception x = (x :> [`BaseException] Obj.t)
let with_traceback ~tb self =
   Py.Module.get_function_with_keywords (to_pyobject self) "with_traceback"
     [||]
     (Wrap_utils.keyword_args [("tb", Some(tb ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MapWrapper = struct
type tag = [`MapWrapper]
type t = [`MapWrapper | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?pool () =
                     Py.Module.get_function_with_keywords __wrap_namespace "MapWrapper"
                       [||]
                       (Wrap_utils.keyword_args [("pool", Wrap_utils.Option.map pool (function
| `I x -> Py.Int.of_int x
| `Map_like_callable x -> Wrap_utils.id x
))])
                       |> of_pyobject
let close self =
   Py.Module.get_function_with_keywords (to_pyobject self) "close"
     [||]
     []

let join self =
   Py.Module.get_function_with_keywords (to_pyobject self) "join"
     [||]
     []

let terminate self =
   Py.Module.get_function_with_keywords (to_pyobject self) "terminate"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ScalarFunction = struct
type tag = [`ScalarFunction]
type t = [`Object | `ScalarFunction] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?epsilon ~fun_ ~x0 ~args ~grad ~hess ~finite_diff_rel_step ~finite_diff_bounds () =
   Py.Module.get_function_with_keywords __wrap_namespace "ScalarFunction"
     [||]
     (Wrap_utils.keyword_args [("epsilon", epsilon); ("fun", Some(fun_ )); ("x0", Some(x0 )); ("args", Some(args )); ("grad", Some(grad )); ("hess", Some(hess )); ("finite_diff_rel_step", Some(finite_diff_rel_step )); ("finite_diff_bounds", Some(finite_diff_bounds ))])
     |> of_pyobject
let fun_ ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fun"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let fun_and_grad ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fun_and_grad"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let grad ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "grad"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let hess ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "hess"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let approx_derivative ?method_ ?rel_step ?abs_step ?f0 ?bounds ?sparsity ?as_linear_operator ?args ?kwargs ~fun_ ~x0 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "approx_derivative"
                       [||]
                       (Wrap_utils.keyword_args [("method", Wrap_utils.Option.map method_ (function
| `T3_point -> Py.String.of_string "3-point"
| `T2_point -> Py.String.of_string "2-point"
| `Cs -> Py.String.of_string "cs"
)); ("rel_step", Wrap_utils.Option.map rel_step Np.Obj.to_pyobject); ("abs_step", Wrap_utils.Option.map abs_step Np.Obj.to_pyobject); ("f0", Wrap_utils.Option.map f0 Np.Obj.to_pyobject); ("bounds", bounds); ("sparsity", Wrap_utils.Option.map sparsity (function
| `Arr x -> Np.Obj.to_pyobject x
| `T2_tuple x -> Wrap_utils.id x
)); ("as_linear_operator", Wrap_utils.Option.map as_linear_operator Py.Bool.of_bool); ("args", args); ("kwargs", kwargs); ("fun", Some(fun_ )); ("x0", Some(x0 |> (function
| `F x -> Py.Float.of_float x
| `Ndarray x -> Np.Obj.to_pyobject x
)))])

let approx_fhess_p ~x0 ~p ~fprime ~epsilon args =
   Py.Module.get_function_with_keywords __wrap_namespace "approx_fhess_p"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (Wrap_utils.keyword_args [("x0", Some(x0 )); ("p", Some(p )); ("fprime", Some(fprime )); ("epsilon", Some(epsilon ))])

let approx_fprime ~xk ~f ~epsilon args =
   Py.Module.get_function_with_keywords __wrap_namespace "approx_fprime"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (Wrap_utils.keyword_args [("xk", Some(xk |> Np.Obj.to_pyobject)); ("f", Some(f )); ("epsilon", Some(epsilon |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let argmin ?axis ?out ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "argmin"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("a", Some(a |> Np.Obj.to_pyobject))])

                  let asarray ?dtype ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let asfarray ?dtype ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asfarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype_object x -> Wrap_utils.id x
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let atleast_1d arys =
   Py.Module.get_function_with_keywords __wrap_namespace "atleast_1d"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arys)])
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let bracket ?xa ?xb ?args ?grow_limit ?maxiter ~func () =
   Py.Module.get_function_with_keywords __wrap_namespace "bracket"
     [||]
     (Wrap_utils.keyword_args [("xa", xa); ("xb", xb); ("args", args); ("grow_limit", Wrap_utils.Option.map grow_limit Py.Float.of_float); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("func", Some(func ))])
     |> Py.Int.to_int
let brent ?args ?brack ?tol ?full_output ?maxiter ~func () =
   Py.Module.get_function_with_keywords __wrap_namespace "brent"
     [||]
     (Wrap_utils.keyword_args [("args", args); ("brack", brack); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("func", Some(func ))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Py.Int.to_int (Py.Tuple.get x 3))))
                  let brute ?args ?ns ?full_output ?finish ?disp ?workers ~func ~ranges () =
                     Py.Module.get_function_with_keywords __wrap_namespace "brute"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("Ns", Wrap_utils.Option.map ns Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("finish", finish); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers (function
| `I x -> Py.Int.of_int x
| `Map_like_callable x -> Wrap_utils.id x
)); ("func", Some(func )); ("ranges", Some(ranges ))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 3))))
let check_grad ?kwargs ~func ~grad ~x0 args =
   Py.Module.get_function_with_keywords __wrap_namespace "check_grad"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("func", Some(func )); ("grad", Some(grad )); ("x0", Some(x0 |> Np.Obj.to_pyobject))]) (match kwargs with None -> [] | Some x -> x))
     |> Py.Float.to_float
                  let eye ?m ?k ?dtype ?order ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "eye"
                       [||]
                       (Wrap_utils.keyword_args [("M", Wrap_utils.Option.map m Py.Int.of_int); ("k", Wrap_utils.Option.map k Py.Int.of_int); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("N", Some(n |> Py.Int.of_int))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let fmin ?args ?xtol ?ftol ?maxiter ?maxfun ?full_output ?disp ?retall ?callback ?initial_simplex ~func ~x0 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fmin"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("xtol", Wrap_utils.Option.map xtol Py.Float.of_float); ("ftol", Wrap_utils.Option.map ftol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("maxfun", Wrap_utils.Option.map maxfun (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("retall", Wrap_utils.Option.map retall Py.Bool.of_bool); ("callback", callback); ("initial_simplex", Wrap_utils.Option.map initial_simplex Np.Obj.to_pyobject); ("func", Some(func )); ("x0", Some(x0 |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Py.Int.to_int (Py.Tuple.get x 3)), (Py.Int.to_int (Py.Tuple.get x 4)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 5))))
                  let fmin_bfgs ?fprime ?args ?gtol ?norm ?epsilon ?maxiter ?full_output ?disp ?retall ?callback ~f ~x0 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fmin_bfgs"
                       [||]
                       (Wrap_utils.keyword_args [("fprime", fprime); ("args", args); ("gtol", Wrap_utils.Option.map gtol Py.Float.of_float); ("norm", Wrap_utils.Option.map norm Py.Float.of_float); ("epsilon", Wrap_utils.Option.map epsilon (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("retall", Wrap_utils.Option.map retall Py.Bool.of_bool); ("callback", callback); ("f", Some(f )); ("x0", Some(x0 |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 3)), (Py.Int.to_int (Py.Tuple.get x 4)), (Py.Int.to_int (Py.Tuple.get x 5)), (Py.Int.to_int (Py.Tuple.get x 6)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 7))))
                  let fmin_cg ?fprime ?args ?gtol ?norm ?epsilon ?maxiter ?full_output ?disp ?retall ?callback ~f ~x0 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fmin_cg"
                       [||]
                       (Wrap_utils.keyword_args [("fprime", Wrap_utils.Option.map fprime (function
| `Callable x -> Wrap_utils.id x
| `T_fprime_x_args_ x -> Wrap_utils.id x
)); ("args", args); ("gtol", Wrap_utils.Option.map gtol Py.Float.of_float); ("norm", Wrap_utils.Option.map norm Py.Float.of_float); ("epsilon", Wrap_utils.Option.map epsilon (function
| `F x -> Py.Float.of_float x
| `Ndarray x -> Np.Obj.to_pyobject x
)); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("retall", Wrap_utils.Option.map retall Py.Bool.of_bool); ("callback", callback); ("f", Some(f |> (function
| `Callable x -> Wrap_utils.id x
| `T_f_x_args_ x -> Wrap_utils.id x
))); ("x0", Some(x0 |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Py.Int.to_int (Py.Tuple.get x 3)), (Py.Int.to_int (Py.Tuple.get x 4)), (Wrap_utils.id (Py.Tuple.get x 5)), (Wrap_utils.id (Py.Tuple.get x 6)), (Wrap_utils.id (Py.Tuple.get x 7)), (Wrap_utils.id (Py.Tuple.get x 8)), (Wrap_utils.id (Py.Tuple.get x 9))))
                  let fmin_ncg ?fhess_p ?fhess ?args ?avextol ?epsilon ?maxiter ?full_output ?disp ?retall ?callback ~f ~x0 ~fprime () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fmin_ncg"
                       [||]
                       (Wrap_utils.keyword_args [("fhess_p", fhess_p); ("fhess", fhess); ("args", args); ("avextol", Wrap_utils.Option.map avextol Py.Float.of_float); ("epsilon", Wrap_utils.Option.map epsilon (function
| `F x -> Py.Float.of_float x
| `Ndarray x -> Np.Obj.to_pyobject x
)); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("retall", Wrap_utils.Option.map retall Py.Bool.of_bool); ("callback", callback); ("f", Some(f )); ("x0", Some(x0 |> Np.Obj.to_pyobject)); ("fprime", Some(fprime ))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Py.Int.to_int (Py.Tuple.get x 3)), (Py.Int.to_int (Py.Tuple.get x 4)), (Py.Int.to_int (Py.Tuple.get x 5)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 6))))
let fmin_powell ?args ?xtol ?ftol ?maxiter ?maxfun ?full_output ?disp ?retall ?callback ?direc ~func ~x0 () =
   Py.Module.get_function_with_keywords __wrap_namespace "fmin_powell"
     [||]
     (Wrap_utils.keyword_args [("args", args); ("xtol", Wrap_utils.Option.map xtol Py.Float.of_float); ("ftol", Wrap_utils.Option.map ftol Py.Float.of_float); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("maxfun", Wrap_utils.Option.map maxfun Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("retall", Wrap_utils.Option.map retall Py.Bool.of_bool); ("callback", callback); ("direc", Wrap_utils.Option.map direc Np.Obj.to_pyobject); ("func", Some(func )); ("x0", Some(x0 |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun x -> if Wrap_utils.check_float x then `F (Py.Float.to_float x) else if Wrap_utils.check_int x then `I (Py.Int.to_int x) else failwith (Printf.sprintf "Sklearn: could not identify type from Python value %s (%s)"
                                (Py.Object.to_string x) (Wrap_utils.type_string x))) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2)), (Py.Int.to_int (Py.Tuple.get x 3)), (Py.Int.to_int (Py.Tuple.get x 4)), (Py.Int.to_int (Py.Tuple.get x 5)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 6))))
let fminbound ?args ?xtol ?maxfun ?full_output ?disp ~func ~x1 ~x2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "fminbound"
     [||]
     (Wrap_utils.keyword_args [("args", args); ("xtol", Wrap_utils.Option.map xtol Py.Float.of_float); ("maxfun", Wrap_utils.Option.map maxfun Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("disp", Wrap_utils.Option.map disp Py.Int.of_int); ("func", Some(func )); ("x1", Some(x1 )); ("x2", Some(x2 ))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun x -> if Wrap_utils.check_float x then `F (Py.Float.to_float x) else if Wrap_utils.check_int x then `I (Py.Int.to_int x) else failwith (Printf.sprintf "Sklearn: could not identify type from Python value %s (%s)"
                                (Py.Object.to_string x) (Wrap_utils.type_string x))) (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Py.Int.to_int (Py.Tuple.get x 3))))
let golden ?args ?brack ?tol ?full_output ?maxiter ~func () =
   Py.Module.get_function_with_keywords __wrap_namespace "golden"
     [||]
     (Wrap_utils.keyword_args [("args", args); ("brack", brack); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("func", Some(func ))])

let is_array_scalar x =
   Py.Module.get_function_with_keywords __wrap_namespace "is_array_scalar"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

                  let isinf ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "isinf"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])

let line_search ?gfk ?old_fval ?old_old_fval ?args ?c1 ?c2 ?amax ?extra_condition ?maxiter ~f ~myfprime ~xk ~pk () =
   Py.Module.get_function_with_keywords __wrap_namespace "line_search"
     [||]
     (Wrap_utils.keyword_args [("gfk", Wrap_utils.Option.map gfk Np.Obj.to_pyobject); ("old_fval", Wrap_utils.Option.map old_fval Py.Float.of_float); ("old_old_fval", Wrap_utils.Option.map old_old_fval Py.Float.of_float); ("args", args); ("c1", Wrap_utils.Option.map c1 Py.Float.of_float); ("c2", Wrap_utils.Option.map c2 Py.Float.of_float); ("amax", Wrap_utils.Option.map amax Py.Float.of_float); ("extra_condition", extra_condition); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("f", Some(f )); ("myfprime", Some(myfprime )); ("xk", Some(xk |> Np.Obj.to_pyobject)); ("pk", Some(pk |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> if Py.is_none py then None else Some (Py.Float.to_float py)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), ((fun py -> if Py.is_none py then None else Some (Py.Float.to_float py)) (Py.Tuple.get x 3)), (Py.Float.to_float (Py.Tuple.get x 4)), ((fun py -> if Py.is_none py then None else Some (Py.Float.to_float py)) (Py.Tuple.get x 5))))
let line_search_wolfe1 ?gfk ?old_fval ?old_old_fval ?args ?c1 ?c2 ?amax ?amin ?xtol ~f ~fprime ~xk ~pk () =
   Py.Module.get_function_with_keywords __wrap_namespace "line_search_wolfe1"
     [||]
     (Wrap_utils.keyword_args [("gfk", Wrap_utils.Option.map gfk Np.Obj.to_pyobject); ("old_fval", Wrap_utils.Option.map old_fval Py.Float.of_float); ("old_old_fval", Wrap_utils.Option.map old_old_fval Py.Float.of_float); ("args", args); ("c1", c1); ("c2", c2); ("amax", amax); ("amin", amin); ("xtol", xtol); ("f", Some(f )); ("fprime", Some(fprime )); ("xk", Some(xk |> Np.Obj.to_pyobject)); ("pk", Some(pk |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let line_search_wolfe2 ?gfk ?old_fval ?old_old_fval ?args ?c1 ?c2 ?amax ?extra_condition ?maxiter ~f ~myfprime ~xk ~pk () =
   Py.Module.get_function_with_keywords __wrap_namespace "line_search_wolfe2"
     [||]
     (Wrap_utils.keyword_args [("gfk", Wrap_utils.Option.map gfk Np.Obj.to_pyobject); ("old_fval", Wrap_utils.Option.map old_fval Py.Float.of_float); ("old_old_fval", Wrap_utils.Option.map old_old_fval Py.Float.of_float); ("args", args); ("c1", Wrap_utils.Option.map c1 Py.Float.of_float); ("c2", Wrap_utils.Option.map c2 Py.Float.of_float); ("amax", Wrap_utils.Option.map amax Py.Float.of_float); ("extra_condition", extra_condition); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("f", Some(f )); ("myfprime", Some(myfprime )); ("xk", Some(xk |> Np.Obj.to_pyobject)); ("pk", Some(pk |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> if Py.is_none py then None else Some (Py.Float.to_float py)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), ((fun py -> if Py.is_none py then None else Some (Py.Float.to_float py)) (Py.Tuple.get x 3)), (Py.Float.to_float (Py.Tuple.get x 4)), ((fun py -> if Py.is_none py then None else Some (Py.Float.to_float py)) (Py.Tuple.get x 5))))
let main () =
   Py.Module.get_function_with_keywords __wrap_namespace "main"
     [||]
     []

let rosen x =
   Py.Module.get_function_with_keywords __wrap_namespace "rosen"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let rosen_der x =
   Py.Module.get_function_with_keywords __wrap_namespace "rosen_der"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let rosen_hess x =
   Py.Module.get_function_with_keywords __wrap_namespace "rosen_hess"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let rosen_hess_prod ~x ~p () =
   Py.Module.get_function_with_keywords __wrap_namespace "rosen_hess_prod"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
        let shape a =
           Py.Module.get_function_with_keywords __wrap_namespace "shape"
             [||]
             (Wrap_utils.keyword_args [("a", Some(a |> Np.Obj.to_pyobject))])
             |> (fun py -> let len = Py.Sequence.length py in Array.init len
(fun i -> Py.Int.to_int (Py.Sequence.get_item py i)))
let show_options ?solver ?method_ ?disp () =
   Py.Module.get_function_with_keywords __wrap_namespace "show_options"
     [||]
     (Wrap_utils.keyword_args [("solver", Wrap_utils.Option.map solver Py.String.of_string); ("method", Wrap_utils.Option.map method_ Py.String.of_string); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool)])

                  let sqrt ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "sqrt"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let squeeze ?axis ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "squeeze"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let vecnorm ?ord ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "vecnorm"
     [||]
     (Wrap_utils.keyword_args [("ord", ord); ("x", Some(x ))])

let wrap_function ~function_ ~args () =
   Py.Module.get_function_with_keywords __wrap_namespace "wrap_function"
     [||]
     (Wrap_utils.keyword_args [("function", Some(function_ )); ("args", Some(args ))])

                  let zeros ?dtype ?order ~shape () =
                     Py.Module.get_function_with_keywords __wrap_namespace "zeros"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
module Slsqp = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.optimize.slsqp"

let get_py name = Py.Module.get __wrap_namespace name
module Finfo = struct
type tag = [`Finfo]
type t = [`Finfo | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create dtype =
                     Py.Module.get_function_with_keywords __wrap_namespace "finfo"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Some(dtype |> (function
| `F x -> Py.Float.of_float x
| `Instance x -> Wrap_utils.id x
| `Dtype x -> Np.Dtype.to_pyobject x
)))])
                       |> of_pyobject

let bits_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "bits" with
  | None -> failwith "attribute bits not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let bits self = match bits_opt self with
  | None -> raise Not_found
  | Some x -> x

let eps_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "eps" with
  | None -> failwith "attribute eps not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let eps self = match eps_opt self with
  | None -> raise Not_found
  | Some x -> x

let epsneg_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "epsneg" with
  | None -> failwith "attribute epsneg not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let epsneg self = match epsneg_opt self with
  | None -> raise Not_found
  | Some x -> x

let iexp_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "iexp" with
  | None -> failwith "attribute iexp not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let iexp self = match iexp_opt self with
  | None -> raise Not_found
  | Some x -> x

let machar_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "machar" with
  | None -> failwith "attribute machar not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let machar self = match machar_opt self with
  | None -> raise Not_found
  | Some x -> x

let machep_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "machep" with
  | None -> failwith "attribute machep not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let machep self = match machep_opt self with
  | None -> raise Not_found
  | Some x -> x

let max_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "max" with
  | None -> failwith "attribute max not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let max self = match max_opt self with
  | None -> raise Not_found
  | Some x -> x

let maxexp_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "maxexp" with
  | None -> failwith "attribute maxexp not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let maxexp self = match maxexp_opt self with
  | None -> raise Not_found
  | Some x -> x

let min_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "min" with
  | None -> failwith "attribute min not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let min self = match min_opt self with
  | None -> raise Not_found
  | Some x -> x

let minexp_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "minexp" with
  | None -> failwith "attribute minexp not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let minexp self = match minexp_opt self with
  | None -> raise Not_found
  | Some x -> x

let negep_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "negep" with
  | None -> failwith "attribute negep not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let negep self = match negep_opt self with
  | None -> raise Not_found
  | Some x -> x

let nexp_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nexp" with
  | None -> failwith "attribute nexp not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let nexp self = match nexp_opt self with
  | None -> raise Not_found
  | Some x -> x

let nmant_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nmant" with
  | None -> failwith "attribute nmant not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let nmant self = match nmant_opt self with
  | None -> raise Not_found
  | Some x -> x

let precision_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "precision" with
  | None -> failwith "attribute precision not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let precision self = match precision_opt self with
  | None -> raise Not_found
  | Some x -> x

let resolution_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "resolution" with
  | None -> failwith "attribute resolution not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let resolution self = match resolution_opt self with
  | None -> raise Not_found
  | Some x -> x

let tiny_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "tiny" with
  | None -> failwith "attribute tiny not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let tiny self = match tiny_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let append ?axis ~arr ~values () =
   Py.Module.get_function_with_keywords __wrap_namespace "append"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("arr", Some(arr |> Np.Obj.to_pyobject)); ("values", Some(values |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let approx_derivative ?method_ ?rel_step ?abs_step ?f0 ?bounds ?sparsity ?as_linear_operator ?args ?kwargs ~fun_ ~x0 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "approx_derivative"
                       [||]
                       (Wrap_utils.keyword_args [("method", Wrap_utils.Option.map method_ (function
| `T3_point -> Py.String.of_string "3-point"
| `T2_point -> Py.String.of_string "2-point"
| `Cs -> Py.String.of_string "cs"
)); ("rel_step", Wrap_utils.Option.map rel_step Np.Obj.to_pyobject); ("abs_step", Wrap_utils.Option.map abs_step Np.Obj.to_pyobject); ("f0", Wrap_utils.Option.map f0 Np.Obj.to_pyobject); ("bounds", bounds); ("sparsity", Wrap_utils.Option.map sparsity (function
| `Arr x -> Np.Obj.to_pyobject x
| `T2_tuple x -> Wrap_utils.id x
)); ("as_linear_operator", Wrap_utils.Option.map as_linear_operator Py.Bool.of_bool); ("args", args); ("kwargs", kwargs); ("fun", Some(fun_ )); ("x0", Some(x0 |> (function
| `F x -> Py.Float.of_float x
| `Ndarray x -> Np.Obj.to_pyobject x
)))])

let approx_jacobian ~x ~func ~epsilon args =
   Py.Module.get_function_with_keywords __wrap_namespace "approx_jacobian"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject)); ("func", Some(func )); ("epsilon", Some(epsilon |> Py.Float.of_float))])

                  let array ?dtype ?copy ?order ?subok ?ndmin ~object_ () =
                     Py.Module.get_function_with_keywords __wrap_namespace "array"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("order", Wrap_utils.Option.map order (function
| `K -> Py.String.of_string "K"
| `A -> Py.String.of_string "A"
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("ndmin", Wrap_utils.Option.map ndmin Py.Int.of_int); ("object", Some(object_ |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let asfarray ?dtype ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asfarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype_object x -> Wrap_utils.id x
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let atleast_1d arys =
   Py.Module.get_function_with_keywords __wrap_namespace "atleast_1d"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arys)])
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let concatenate ?axis ?out ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "concatenate"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("a", Some(a ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let exp ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "exp"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let fmin_slsqp ?eqcons ?f_eqcons ?ieqcons ?f_ieqcons ?bounds ?fprime ?fprime_eqcons ?fprime_ieqcons ?args ?iter ?acc ?iprint ?disp ?full_output ?epsilon ?callback ~func ~x0 () =
   Py.Module.get_function_with_keywords __wrap_namespace "fmin_slsqp"
     [||]
     (Wrap_utils.keyword_args [("eqcons", Wrap_utils.Option.map eqcons Np.Obj.to_pyobject); ("f_eqcons", f_eqcons); ("ieqcons", Wrap_utils.Option.map ieqcons Np.Obj.to_pyobject); ("f_ieqcons", f_ieqcons); ("bounds", Wrap_utils.Option.map bounds Np.Obj.to_pyobject); ("fprime", fprime); ("fprime_eqcons", fprime_eqcons); ("fprime_ieqcons", fprime_ieqcons); ("args", args); ("iter", Wrap_utils.Option.map iter Py.Int.of_int); ("acc", Wrap_utils.Option.map acc Py.Float.of_float); ("iprint", Wrap_utils.Option.map iprint Py.Int.of_int); ("disp", Wrap_utils.Option.map disp Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("epsilon", Wrap_utils.Option.map epsilon Py.Float.of_float); ("callback", callback); ("func", Some(func )); ("x0", Some(x0 ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3)), (Wrap_utils.id (Py.Tuple.get x 4))))
                  let isfinite ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "isfinite"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let old_bound_to_new bounds =
   Py.Module.get_function_with_keywords __wrap_namespace "old_bound_to_new"
     [||]
     (Wrap_utils.keyword_args [("bounds", Some(bounds ))])

                  let sqrt ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "sqrt"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let vstack tup =
   Py.Module.get_function_with_keywords __wrap_namespace "vstack"
     [||]
     (Wrap_utils.keyword_args [("tup", Some(tup |> (fun ml -> Py.List.of_list_map Np.Obj.to_pyobject ml)))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let zeros ?dtype ?order ~shape () =
                     Py.Module.get_function_with_keywords __wrap_namespace "zeros"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
module Tnc = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.optimize.tnc"

let get_py name = Py.Module.get __wrap_namespace name
module MemoizeJac = struct
type tag = [`MemoizeJac]
type t = [`MemoizeJac | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create fun_ =
   Py.Module.get_function_with_keywords __wrap_namespace "MemoizeJac"
     [||]
     (Wrap_utils.keyword_args [("fun", Some(fun_ ))])
     |> of_pyobject
let derivative ~x args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "derivative"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (Wrap_utils.keyword_args [("x", Some(x ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let array ?dtype ?copy ?order ?subok ?ndmin ~object_ () =
                     Py.Module.get_function_with_keywords __wrap_namespace "array"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("order", Wrap_utils.Option.map order (function
| `K -> Py.String.of_string "K"
| `A -> Py.String.of_string "A"
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("ndmin", Wrap_utils.Option.map ndmin Py.Int.of_int); ("object", Some(object_ |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let asfarray ?dtype ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asfarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype_object x -> Wrap_utils.id x
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let fmin_tnc ?fprime ?args ?approx_grad ?bounds ?epsilon ?scale ?offset ?messages ?maxCGit ?maxfun ?eta ?stepmx ?accuracy ?fmin ?ftol ?xtol ?pgtol ?rescale ?disp ?callback ~func ~x0 () =
   Py.Module.get_function_with_keywords __wrap_namespace "fmin_tnc"
     [||]
     (Wrap_utils.keyword_args [("fprime", fprime); ("args", args); ("approx_grad", Wrap_utils.Option.map approx_grad Py.Bool.of_bool); ("bounds", Wrap_utils.Option.map bounds Np.Obj.to_pyobject); ("epsilon", Wrap_utils.Option.map epsilon Py.Float.of_float); ("scale", Wrap_utils.Option.map scale Py.Float.of_float); ("offset", Wrap_utils.Option.map offset Np.Obj.to_pyobject); ("messages", Wrap_utils.Option.map messages Py.Int.of_int); ("maxCGit", Wrap_utils.Option.map maxCGit Py.Int.of_int); ("maxfun", Wrap_utils.Option.map maxfun Py.Int.of_int); ("eta", Wrap_utils.Option.map eta Py.Float.of_float); ("stepmx", Wrap_utils.Option.map stepmx Py.Float.of_float); ("accuracy", Wrap_utils.Option.map accuracy Py.Float.of_float); ("fmin", Wrap_utils.Option.map fmin Py.Float.of_float); ("ftol", Wrap_utils.Option.map ftol Py.Float.of_float); ("xtol", Wrap_utils.Option.map xtol Py.Float.of_float); ("pgtol", Wrap_utils.Option.map pgtol Py.Float.of_float); ("rescale", Wrap_utils.Option.map rescale Py.Float.of_float); ("disp", Wrap_utils.Option.map disp Py.Int.of_int); ("callback", callback); ("func", Some(func )); ("x0", Some(x0 |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2))))
let old_bound_to_new bounds =
   Py.Module.get_function_with_keywords __wrap_namespace "old_bound_to_new"
     [||]
     (Wrap_utils.keyword_args [("bounds", Some(bounds ))])

                  let zeros ?dtype ?order ~shape () =
                     Py.Module.get_function_with_keywords __wrap_namespace "zeros"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
module Zeros = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.optimize.zeros"

let get_py name = Py.Module.get __wrap_namespace name
module TOMS748Solver = struct
type tag = [`TOMS748Solver]
type t = [`Object | `TOMS748Solver] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "TOMS748Solver"
     [||]
     []
     |> of_pyobject
let configure ~xtol ~rtol ~maxiter ~disp ~k self =
   Py.Module.get_function_with_keywords (to_pyobject self) "configure"
     [||]
     (Wrap_utils.keyword_args [("xtol", Some(xtol )); ("rtol", Some(rtol )); ("maxiter", Some(maxiter )); ("disp", Some(disp )); ("k", Some(k ))])

let get_result ?flag ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_result"
     [||]
     (Wrap_utils.keyword_args [("flag", flag); ("x", Some(x ))])

let get_status self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_status"
     [||]
     []

let iterate self =
   Py.Module.get_function_with_keywords (to_pyobject self) "iterate"
     [||]
     []

let solve ?args ?xtol ?rtol ?k ?maxiter ?disp ~f ~a ~b self =
   Py.Module.get_function_with_keywords (to_pyobject self) "solve"
     [||]
     (Wrap_utils.keyword_args [("args", args); ("xtol", xtol); ("rtol", rtol); ("k", k); ("maxiter", maxiter); ("disp", disp); ("f", Some(f )); ("a", Some(a )); ("b", Some(b ))])

let start ?args ~f ~a ~b self =
   Py.Module.get_function_with_keywords (to_pyobject self) "start"
     [||]
     (Wrap_utils.keyword_args [("args", args); ("f", Some(f )); ("a", Some(a )); ("b", Some(b ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let bisect ?args ?xtol ?rtol ?maxiter ?full_output ?disp ~f ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "bisect"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("xtol", Wrap_utils.Option.map xtol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("rtol", Wrap_utils.Option.map rtol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("f", Some(f )); ("a", Some(a |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
))); ("b", Some(b |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let brenth ?args ?xtol ?rtol ?maxiter ?full_output ?disp ~f ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "brenth"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("xtol", Wrap_utils.Option.map xtol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("rtol", Wrap_utils.Option.map rtol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("f", Some(f )); ("a", Some(a |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
))); ("b", Some(b |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let brentq ?args ?xtol ?rtol ?maxiter ?full_output ?disp ~f ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "brentq"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("xtol", Wrap_utils.Option.map xtol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("rtol", Wrap_utils.Option.map rtol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("f", Some(f )); ("a", Some(a |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
))); ("b", Some(b |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let namedtuple ?rename ?defaults ?module_ ~typename ~field_names () =
   Py.Module.get_function_with_keywords __wrap_namespace "namedtuple"
     [||]
     (Wrap_utils.keyword_args [("rename", rename); ("defaults", defaults); ("module", module_); ("typename", Some(typename )); ("field_names", Some(field_names ))])

                  let newton ?fprime ?args ?tol ?maxiter ?fprime2 ?x1 ?rtol ?full_output ?disp ~func ~x0 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "newton"
                       [||]
                       (Wrap_utils.keyword_args [("fprime", fprime); ("args", args); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("fprime2", fprime2); ("x1", Wrap_utils.Option.map x1 Py.Float.of_float); ("rtol", Wrap_utils.Option.map rtol Py.Float.of_float); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("func", Some(func )); ("x0", Some(x0 |> (function
| `F x -> Py.Float.of_float x
| `Ndarray x -> Np.Obj.to_pyobject x
| `Sequence x -> Wrap_utils.id x
)))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3))))
let results_c ~full_output ~r () =
   Py.Module.get_function_with_keywords __wrap_namespace "results_c"
     [||]
     (Wrap_utils.keyword_args [("full_output", Some(full_output )); ("r", Some(r ))])

                  let ridder ?args ?xtol ?rtol ?maxiter ?full_output ?disp ~f ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ridder"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("xtol", Wrap_utils.Option.map xtol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("rtol", Wrap_utils.Option.map rtol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("f", Some(f )); ("a", Some(a |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
))); ("b", Some(b |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let toms748 ?args ?k ?xtol ?rtol ?maxiter ?full_output ?disp ~f ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "toms748"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("k", Wrap_utils.Option.map k Py.Int.of_int); ("xtol", Wrap_utils.Option.map xtol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("rtol", Wrap_utils.Option.map rtol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("f", Some(f )); ("a", Some(a |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
))); ("b", Some(b |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))

end
                  let anderson ?iter ?alpha ?w0 ?m ?verbose ?maxiter ?f_tol ?f_rtol ?x_tol ?x_rtol ?tol_norm ?line_search ?callback ?kw ~f ~xin () =
                     Py.Module.get_function_with_keywords __wrap_namespace "anderson"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("iter", Wrap_utils.Option.map iter Py.Int.of_int); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("w0", Wrap_utils.Option.map w0 Py.Float.of_float); ("M", Wrap_utils.Option.map m Py.Float.of_float); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("f_tol", Wrap_utils.Option.map f_tol Py.Float.of_float); ("f_rtol", Wrap_utils.Option.map f_rtol Py.Float.of_float); ("x_tol", Wrap_utils.Option.map x_tol Py.Float.of_float); ("x_rtol", Wrap_utils.Option.map x_rtol Py.Float.of_float); ("tol_norm", tol_norm); ("line_search", Wrap_utils.Option.map line_search (function
| `Wolfe -> Py.String.of_string "wolfe"
| `Armijo -> Py.String.of_string "armijo"
| `None -> Py.none
)); ("callback", callback); ("F", Some(f )); ("xin", Some(xin |> Np.Obj.to_pyobject))]) (match kw with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let approx_fprime ~xk ~f ~epsilon args =
   Py.Module.get_function_with_keywords __wrap_namespace "approx_fprime"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (Wrap_utils.keyword_args [("xk", Some(xk |> Np.Obj.to_pyobject)); ("f", Some(f )); ("epsilon", Some(epsilon |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let basinhopping ?niter ?t ?stepsize ?minimizer_kwargs ?take_step ?accept_test ?callback ?interval ?disp ?niter_success ?seed ~func ~x0 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "basinhopping"
                       [||]
                       (Wrap_utils.keyword_args [("niter", Wrap_utils.Option.map niter Py.Int.of_int); ("T", Wrap_utils.Option.map t Py.Float.of_float); ("stepsize", Wrap_utils.Option.map stepsize Py.Float.of_float); ("minimizer_kwargs", minimizer_kwargs); ("take_step", take_step); ("accept_test", Wrap_utils.Option.map accept_test (function
| `T_accept_test_f_new_f_new_x_new_x_new_f_old_fold_x_old_x_old_ x -> Wrap_utils.id x
| `Callable x -> Wrap_utils.id x
)); ("callback", Wrap_utils.Option.map callback (function
| `T_callback_x_f_accept_ x -> Wrap_utils.id x
| `Callable x -> Wrap_utils.id x
)); ("interval", Wrap_utils.Option.map interval Py.Int.of_int); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("niter_success", Wrap_utils.Option.map niter_success Py.Int.of_int); ("seed", Wrap_utils.Option.map seed (function
| `I x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)); ("func", Some(func )); ("x0", Some(x0 |> Np.Obj.to_pyobject))])

                  let bisect ?args ?xtol ?rtol ?maxiter ?full_output ?disp ~f ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "bisect"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("xtol", Wrap_utils.Option.map xtol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("rtol", Wrap_utils.Option.map rtol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("f", Some(f )); ("a", Some(a |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
))); ("b", Some(b |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let bracket ?xa ?xb ?args ?grow_limit ?maxiter ~func () =
   Py.Module.get_function_with_keywords __wrap_namespace "bracket"
     [||]
     (Wrap_utils.keyword_args [("xa", xa); ("xb", xb); ("args", args); ("grow_limit", Wrap_utils.Option.map grow_limit Py.Float.of_float); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("func", Some(func ))])
     |> Py.Int.to_int
let brent ?args ?brack ?tol ?full_output ?maxiter ~func () =
   Py.Module.get_function_with_keywords __wrap_namespace "brent"
     [||]
     (Wrap_utils.keyword_args [("args", args); ("brack", brack); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("func", Some(func ))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Py.Int.to_int (Py.Tuple.get x 3))))
                  let brenth ?args ?xtol ?rtol ?maxiter ?full_output ?disp ~f ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "brenth"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("xtol", Wrap_utils.Option.map xtol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("rtol", Wrap_utils.Option.map rtol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("f", Some(f )); ("a", Some(a |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
))); ("b", Some(b |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let brentq ?args ?xtol ?rtol ?maxiter ?full_output ?disp ~f ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "brentq"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("xtol", Wrap_utils.Option.map xtol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("rtol", Wrap_utils.Option.map rtol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("f", Some(f )); ("a", Some(a |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
))); ("b", Some(b |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let broyden1 ?iter ?alpha ?reduction_method ?max_rank ?verbose ?maxiter ?f_tol ?f_rtol ?x_tol ?x_rtol ?tol_norm ?line_search ?callback ?kw ~f ~xin () =
                     Py.Module.get_function_with_keywords __wrap_namespace "broyden1"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("iter", Wrap_utils.Option.map iter Py.Int.of_int); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("reduction_method", Wrap_utils.Option.map reduction_method (function
| `S x -> Py.String.of_string x
| `Tuple x -> Wrap_utils.id x
)); ("max_rank", Wrap_utils.Option.map max_rank Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("f_tol", Wrap_utils.Option.map f_tol Py.Float.of_float); ("f_rtol", Wrap_utils.Option.map f_rtol Py.Float.of_float); ("x_tol", Wrap_utils.Option.map x_tol Py.Float.of_float); ("x_rtol", Wrap_utils.Option.map x_rtol Py.Float.of_float); ("tol_norm", tol_norm); ("line_search", Wrap_utils.Option.map line_search (function
| `Wolfe -> Py.String.of_string "wolfe"
| `Armijo -> Py.String.of_string "armijo"
| `None -> Py.none
)); ("callback", callback); ("F", Some(f )); ("xin", Some(xin |> Np.Obj.to_pyobject))]) (match kw with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let broyden2 ?iter ?alpha ?reduction_method ?max_rank ?verbose ?maxiter ?f_tol ?f_rtol ?x_tol ?x_rtol ?tol_norm ?line_search ?callback ?kw ~f ~xin () =
                     Py.Module.get_function_with_keywords __wrap_namespace "broyden2"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("iter", Wrap_utils.Option.map iter Py.Int.of_int); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("reduction_method", Wrap_utils.Option.map reduction_method (function
| `S x -> Py.String.of_string x
| `Tuple x -> Wrap_utils.id x
)); ("max_rank", Wrap_utils.Option.map max_rank Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("f_tol", Wrap_utils.Option.map f_tol Py.Float.of_float); ("f_rtol", Wrap_utils.Option.map f_rtol Py.Float.of_float); ("x_tol", Wrap_utils.Option.map x_tol Py.Float.of_float); ("x_rtol", Wrap_utils.Option.map x_rtol Py.Float.of_float); ("tol_norm", tol_norm); ("line_search", Wrap_utils.Option.map line_search (function
| `Wolfe -> Py.String.of_string "wolfe"
| `Armijo -> Py.String.of_string "armijo"
| `None -> Py.none
)); ("callback", callback); ("F", Some(f )); ("xin", Some(xin |> Np.Obj.to_pyobject))]) (match kw with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let brute ?args ?ns ?full_output ?finish ?disp ?workers ~func ~ranges () =
                     Py.Module.get_function_with_keywords __wrap_namespace "brute"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("Ns", Wrap_utils.Option.map ns Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("finish", finish); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers (function
| `I x -> Py.Int.of_int x
| `Map_like_callable x -> Wrap_utils.id x
)); ("func", Some(func )); ("ranges", Some(ranges ))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 3))))
let check_grad ?kwargs ~func ~grad ~x0 args =
   Py.Module.get_function_with_keywords __wrap_namespace "check_grad"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("func", Some(func )); ("grad", Some(grad )); ("x0", Some(x0 |> Np.Obj.to_pyobject))]) (match kwargs with None -> [] | Some x -> x))
     |> Py.Float.to_float
                  let curve_fit ?p0 ?sigma ?absolute_sigma ?check_finite ?bounds ?method_ ?jac ?kwargs ~f ~xdata ~ydata () =
                     Py.Module.get_function_with_keywords __wrap_namespace "curve_fit"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("p0", Wrap_utils.Option.map p0 Np.Obj.to_pyobject); ("sigma", sigma); ("absolute_sigma", Wrap_utils.Option.map absolute_sigma Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("bounds", bounds); ("method", Wrap_utils.Option.map method_ (function
| `Dogbox -> Py.String.of_string "dogbox"
| `Trf -> Py.String.of_string "trf"
| `Lm -> Py.String.of_string "lm"
)); ("jac", Wrap_utils.Option.map jac (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("f", Some(f )); ("xdata", Some(xdata |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("ydata", Some(ydata |> Np.Obj.to_pyobject))]) (match kwargs with None -> [] | Some x -> x))
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let diagbroyden ?iter ?alpha ?verbose ?maxiter ?f_tol ?f_rtol ?x_tol ?x_rtol ?tol_norm ?line_search ?callback ?kw ~f ~xin () =
                     Py.Module.get_function_with_keywords __wrap_namespace "diagbroyden"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("iter", Wrap_utils.Option.map iter Py.Int.of_int); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("f_tol", Wrap_utils.Option.map f_tol Py.Float.of_float); ("f_rtol", Wrap_utils.Option.map f_rtol Py.Float.of_float); ("x_tol", Wrap_utils.Option.map x_tol Py.Float.of_float); ("x_rtol", Wrap_utils.Option.map x_rtol Py.Float.of_float); ("tol_norm", tol_norm); ("line_search", Wrap_utils.Option.map line_search (function
| `Wolfe -> Py.String.of_string "wolfe"
| `Armijo -> Py.String.of_string "armijo"
| `None -> Py.none
)); ("callback", callback); ("F", Some(f )); ("xin", Some(xin |> Np.Obj.to_pyobject))]) (match kw with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let differential_evolution ?args ?strategy ?maxiter ?popsize ?tol ?mutation ?recombination ?seed ?callback ?disp ?polish ?init ?atol ?updating ?workers ?constraints ~func ~bounds () =
                     Py.Module.get_function_with_keywords __wrap_namespace "differential_evolution"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("strategy", Wrap_utils.Option.map strategy Py.String.of_string); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("popsize", Wrap_utils.Option.map popsize Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("mutation", Wrap_utils.Option.map mutation (function
| `F x -> Py.Float.of_float x
| `Tuple_float_float_ x -> Wrap_utils.id x
)); ("recombination", Wrap_utils.Option.map recombination Py.Float.of_float); ("seed", Wrap_utils.Option.map seed (function
| `I x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)); ("callback", Wrap_utils.Option.map callback (function
| `Callable x -> Wrap_utils.id x
| `T_callback_xk_convergence_val_ x -> Wrap_utils.id x
)); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("polish", Wrap_utils.Option.map polish Py.Bool.of_bool); ("init", Wrap_utils.Option.map init (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `S x -> Py.String.of_string x
)); ("atol", Wrap_utils.Option.map atol Py.Float.of_float); ("updating", Wrap_utils.Option.map updating (function
| `Immediate -> Py.String.of_string "immediate"
| `Deferred -> Py.String.of_string "deferred"
)); ("workers", Wrap_utils.Option.map workers (function
| `I x -> Py.Int.of_int x
| `Map_like_callable x -> Wrap_utils.id x
)); ("constraints", constraints); ("func", Some(func )); ("bounds", Some(bounds ))])

                  let dual_annealing ?args ?maxiter ?local_search_options ?initial_temp ?restart_temp_ratio ?visit ?accept ?maxfun ?seed ?no_local_search ?callback ?x0 ~func ~bounds () =
                     Py.Module.get_function_with_keywords __wrap_namespace "dual_annealing"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("local_search_options", local_search_options); ("initial_temp", Wrap_utils.Option.map initial_temp Py.Float.of_float); ("restart_temp_ratio", Wrap_utils.Option.map restart_temp_ratio Py.Float.of_float); ("visit", Wrap_utils.Option.map visit Py.Float.of_float); ("accept", Wrap_utils.Option.map accept Py.Float.of_float); ("maxfun", Wrap_utils.Option.map maxfun Py.Int.of_int); ("seed", Wrap_utils.Option.map seed (function
| `I x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)); ("no_local_search", Wrap_utils.Option.map no_local_search Py.Bool.of_bool); ("callback", callback); ("x0", Wrap_utils.Option.map x0 Np.Obj.to_pyobject); ("func", Some(func )); ("bounds", Some(bounds ))])

                  let excitingmixing ?iter ?alpha ?alphamax ?verbose ?maxiter ?f_tol ?f_rtol ?x_tol ?x_rtol ?tol_norm ?line_search ?callback ?kw ~f ~xin () =
                     Py.Module.get_function_with_keywords __wrap_namespace "excitingmixing"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("iter", Wrap_utils.Option.map iter Py.Int.of_int); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("alphamax", Wrap_utils.Option.map alphamax Py.Float.of_float); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("f_tol", Wrap_utils.Option.map f_tol Py.Float.of_float); ("f_rtol", Wrap_utils.Option.map f_rtol Py.Float.of_float); ("x_tol", Wrap_utils.Option.map x_tol Py.Float.of_float); ("x_rtol", Wrap_utils.Option.map x_rtol Py.Float.of_float); ("tol_norm", tol_norm); ("line_search", Wrap_utils.Option.map line_search (function
| `Wolfe -> Py.String.of_string "wolfe"
| `Armijo -> Py.String.of_string "armijo"
| `None -> Py.none
)); ("callback", callback); ("F", Some(f )); ("xin", Some(xin |> Np.Obj.to_pyobject))]) (match kw with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let fixed_point ?args ?xtol ?maxiter ?method_ ~func ~x0 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fixed_point"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("xtol", Wrap_utils.Option.map xtol Py.Float.of_float); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("method", Wrap_utils.Option.map method_ (function
| `Del2 -> Py.String.of_string "del2"
| `Iteration -> Py.String.of_string "iteration"
)); ("func", Some(func )); ("x0", Some(x0 |> Np.Obj.to_pyobject))])

                  let fmin ?args ?xtol ?ftol ?maxiter ?maxfun ?full_output ?disp ?retall ?callback ?initial_simplex ~func ~x0 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fmin"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("xtol", Wrap_utils.Option.map xtol Py.Float.of_float); ("ftol", Wrap_utils.Option.map ftol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("maxfun", Wrap_utils.Option.map maxfun (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("retall", Wrap_utils.Option.map retall Py.Bool.of_bool); ("callback", callback); ("initial_simplex", Wrap_utils.Option.map initial_simplex Np.Obj.to_pyobject); ("func", Some(func )); ("x0", Some(x0 |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Py.Int.to_int (Py.Tuple.get x 3)), (Py.Int.to_int (Py.Tuple.get x 4)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 5))))
                  let fmin_bfgs ?fprime ?args ?gtol ?norm ?epsilon ?maxiter ?full_output ?disp ?retall ?callback ~f ~x0 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fmin_bfgs"
                       [||]
                       (Wrap_utils.keyword_args [("fprime", fprime); ("args", args); ("gtol", Wrap_utils.Option.map gtol Py.Float.of_float); ("norm", Wrap_utils.Option.map norm Py.Float.of_float); ("epsilon", Wrap_utils.Option.map epsilon (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("retall", Wrap_utils.Option.map retall Py.Bool.of_bool); ("callback", callback); ("f", Some(f )); ("x0", Some(x0 |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 3)), (Py.Int.to_int (Py.Tuple.get x 4)), (Py.Int.to_int (Py.Tuple.get x 5)), (Py.Int.to_int (Py.Tuple.get x 6)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 7))))
                  let fmin_cg ?fprime ?args ?gtol ?norm ?epsilon ?maxiter ?full_output ?disp ?retall ?callback ~f ~x0 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fmin_cg"
                       [||]
                       (Wrap_utils.keyword_args [("fprime", Wrap_utils.Option.map fprime (function
| `Callable x -> Wrap_utils.id x
| `T_fprime_x_args_ x -> Wrap_utils.id x
)); ("args", args); ("gtol", Wrap_utils.Option.map gtol Py.Float.of_float); ("norm", Wrap_utils.Option.map norm Py.Float.of_float); ("epsilon", Wrap_utils.Option.map epsilon (function
| `F x -> Py.Float.of_float x
| `Ndarray x -> Np.Obj.to_pyobject x
)); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("retall", Wrap_utils.Option.map retall Py.Bool.of_bool); ("callback", callback); ("f", Some(f |> (function
| `Callable x -> Wrap_utils.id x
| `T_f_x_args_ x -> Wrap_utils.id x
))); ("x0", Some(x0 |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Py.Int.to_int (Py.Tuple.get x 3)), (Py.Int.to_int (Py.Tuple.get x 4)), (Wrap_utils.id (Py.Tuple.get x 5)), (Wrap_utils.id (Py.Tuple.get x 6)), (Wrap_utils.id (Py.Tuple.get x 7)), (Wrap_utils.id (Py.Tuple.get x 8)), (Wrap_utils.id (Py.Tuple.get x 9))))
                  let fmin_cobyla ?args ?consargs ?rhobeg ?rhoend ?maxfun ?disp ?catol ~func ~x0 ~cons () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fmin_cobyla"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("consargs", consargs); ("rhobeg", Wrap_utils.Option.map rhobeg Py.Float.of_float); ("rhoend", Wrap_utils.Option.map rhoend Py.Float.of_float); ("maxfun", Wrap_utils.Option.map maxfun Py.Int.of_int); ("disp", Wrap_utils.Option.map disp (function
| `Three -> Py.Int.of_int 3
| `Two -> Py.Int.of_int 2
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
)); ("catol", Wrap_utils.Option.map catol Py.Float.of_float); ("func", Some(func )); ("x0", Some(x0 |> Np.Obj.to_pyobject)); ("cons", Some(cons ))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let fmin_l_bfgs_b ?fprime ?args ?approx_grad ?bounds ?m ?factr ?pgtol ?epsilon ?iprint ?maxfun ?maxiter ?disp ?callback ?maxls ~func ~x0 () =
   Py.Module.get_function_with_keywords __wrap_namespace "fmin_l_bfgs_b"
     [||]
     (Wrap_utils.keyword_args [("fprime", fprime); ("args", args); ("approx_grad", Wrap_utils.Option.map approx_grad Py.Bool.of_bool); ("bounds", Wrap_utils.Option.map bounds Np.Obj.to_pyobject); ("m", Wrap_utils.Option.map m Py.Int.of_int); ("factr", Wrap_utils.Option.map factr Py.Float.of_float); ("pgtol", Wrap_utils.Option.map pgtol Py.Float.of_float); ("epsilon", Wrap_utils.Option.map epsilon Py.Float.of_float); ("iprint", Wrap_utils.Option.map iprint Py.Int.of_int); ("maxfun", Wrap_utils.Option.map maxfun Py.Int.of_int); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("disp", Wrap_utils.Option.map disp Py.Int.of_int); ("callback", callback); ("maxls", Wrap_utils.Option.map maxls Py.Int.of_int); ("func", Some(func )); ("x0", Some(x0 |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
                  let fmin_ncg ?fhess_p ?fhess ?args ?avextol ?epsilon ?maxiter ?full_output ?disp ?retall ?callback ~f ~x0 ~fprime () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fmin_ncg"
                       [||]
                       (Wrap_utils.keyword_args [("fhess_p", fhess_p); ("fhess", fhess); ("args", args); ("avextol", Wrap_utils.Option.map avextol Py.Float.of_float); ("epsilon", Wrap_utils.Option.map epsilon (function
| `F x -> Py.Float.of_float x
| `Ndarray x -> Np.Obj.to_pyobject x
)); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("retall", Wrap_utils.Option.map retall Py.Bool.of_bool); ("callback", callback); ("f", Some(f )); ("x0", Some(x0 |> Np.Obj.to_pyobject)); ("fprime", Some(fprime ))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Py.Int.to_int (Py.Tuple.get x 3)), (Py.Int.to_int (Py.Tuple.get x 4)), (Py.Int.to_int (Py.Tuple.get x 5)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 6))))
let fmin_powell ?args ?xtol ?ftol ?maxiter ?maxfun ?full_output ?disp ?retall ?callback ?direc ~func ~x0 () =
   Py.Module.get_function_with_keywords __wrap_namespace "fmin_powell"
     [||]
     (Wrap_utils.keyword_args [("args", args); ("xtol", Wrap_utils.Option.map xtol Py.Float.of_float); ("ftol", Wrap_utils.Option.map ftol Py.Float.of_float); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("maxfun", Wrap_utils.Option.map maxfun Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("retall", Wrap_utils.Option.map retall Py.Bool.of_bool); ("callback", callback); ("direc", Wrap_utils.Option.map direc Np.Obj.to_pyobject); ("func", Some(func )); ("x0", Some(x0 |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun x -> if Wrap_utils.check_float x then `F (Py.Float.to_float x) else if Wrap_utils.check_int x then `I (Py.Int.to_int x) else failwith (Printf.sprintf "Sklearn: could not identify type from Python value %s (%s)"
                                (Py.Object.to_string x) (Wrap_utils.type_string x))) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2)), (Py.Int.to_int (Py.Tuple.get x 3)), (Py.Int.to_int (Py.Tuple.get x 4)), (Py.Int.to_int (Py.Tuple.get x 5)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 6))))
let fmin_slsqp ?eqcons ?f_eqcons ?ieqcons ?f_ieqcons ?bounds ?fprime ?fprime_eqcons ?fprime_ieqcons ?args ?iter ?acc ?iprint ?disp ?full_output ?epsilon ?callback ~func ~x0 () =
   Py.Module.get_function_with_keywords __wrap_namespace "fmin_slsqp"
     [||]
     (Wrap_utils.keyword_args [("eqcons", Wrap_utils.Option.map eqcons Np.Obj.to_pyobject); ("f_eqcons", f_eqcons); ("ieqcons", Wrap_utils.Option.map ieqcons Np.Obj.to_pyobject); ("f_ieqcons", f_ieqcons); ("bounds", Wrap_utils.Option.map bounds Np.Obj.to_pyobject); ("fprime", fprime); ("fprime_eqcons", fprime_eqcons); ("fprime_ieqcons", fprime_ieqcons); ("args", args); ("iter", Wrap_utils.Option.map iter Py.Int.of_int); ("acc", Wrap_utils.Option.map acc Py.Float.of_float); ("iprint", Wrap_utils.Option.map iprint Py.Int.of_int); ("disp", Wrap_utils.Option.map disp Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("epsilon", Wrap_utils.Option.map epsilon Py.Float.of_float); ("callback", callback); ("func", Some(func )); ("x0", Some(x0 ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3)), (Wrap_utils.id (Py.Tuple.get x 4))))
let fmin_tnc ?fprime ?args ?approx_grad ?bounds ?epsilon ?scale ?offset ?messages ?maxCGit ?maxfun ?eta ?stepmx ?accuracy ?fmin ?ftol ?xtol ?pgtol ?rescale ?disp ?callback ~func ~x0 () =
   Py.Module.get_function_with_keywords __wrap_namespace "fmin_tnc"
     [||]
     (Wrap_utils.keyword_args [("fprime", fprime); ("args", args); ("approx_grad", Wrap_utils.Option.map approx_grad Py.Bool.of_bool); ("bounds", Wrap_utils.Option.map bounds Np.Obj.to_pyobject); ("epsilon", Wrap_utils.Option.map epsilon Py.Float.of_float); ("scale", Wrap_utils.Option.map scale Py.Float.of_float); ("offset", Wrap_utils.Option.map offset Np.Obj.to_pyobject); ("messages", Wrap_utils.Option.map messages Py.Int.of_int); ("maxCGit", Wrap_utils.Option.map maxCGit Py.Int.of_int); ("maxfun", Wrap_utils.Option.map maxfun Py.Int.of_int); ("eta", Wrap_utils.Option.map eta Py.Float.of_float); ("stepmx", Wrap_utils.Option.map stepmx Py.Float.of_float); ("accuracy", Wrap_utils.Option.map accuracy Py.Float.of_float); ("fmin", Wrap_utils.Option.map fmin Py.Float.of_float); ("ftol", Wrap_utils.Option.map ftol Py.Float.of_float); ("xtol", Wrap_utils.Option.map xtol Py.Float.of_float); ("pgtol", Wrap_utils.Option.map pgtol Py.Float.of_float); ("rescale", Wrap_utils.Option.map rescale Py.Float.of_float); ("disp", Wrap_utils.Option.map disp Py.Int.of_int); ("callback", callback); ("func", Some(func )); ("x0", Some(x0 |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2))))
let fminbound ?args ?xtol ?maxfun ?full_output ?disp ~func ~x1 ~x2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "fminbound"
     [||]
     (Wrap_utils.keyword_args [("args", args); ("xtol", Wrap_utils.Option.map xtol Py.Float.of_float); ("maxfun", Wrap_utils.Option.map maxfun Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("disp", Wrap_utils.Option.map disp Py.Int.of_int); ("func", Some(func )); ("x1", Some(x1 )); ("x2", Some(x2 ))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun x -> if Wrap_utils.check_float x then `F (Py.Float.to_float x) else if Wrap_utils.check_int x then `I (Py.Int.to_int x) else failwith (Printf.sprintf "Sklearn: could not identify type from Python value %s (%s)"
                                (Py.Object.to_string x) (Wrap_utils.type_string x))) (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Py.Int.to_int (Py.Tuple.get x 3))))
let fsolve ?args ?fprime ?full_output ?col_deriv ?xtol ?maxfev ?band ?epsfcn ?factor ?diag ~func ~x0 () =
   Py.Module.get_function_with_keywords __wrap_namespace "fsolve"
     [||]
     (Wrap_utils.keyword_args [("args", args); ("fprime", fprime); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("col_deriv", Wrap_utils.Option.map col_deriv Py.Bool.of_bool); ("xtol", Wrap_utils.Option.map xtol Py.Float.of_float); ("maxfev", Wrap_utils.Option.map maxfev Py.Int.of_int); ("band", band); ("epsfcn", Wrap_utils.Option.map epsfcn Py.Float.of_float); ("factor", Wrap_utils.Option.map factor Py.Float.of_float); ("diag", diag); ("func", Some(func )); ("x0", Some(x0 |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Py.String.to_string (Py.Tuple.get x 3))))
let golden ?args ?brack ?tol ?full_output ?maxiter ~func () =
   Py.Module.get_function_with_keywords __wrap_namespace "golden"
     [||]
     (Wrap_utils.keyword_args [("args", args); ("brack", brack); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("func", Some(func ))])

                  let least_squares ?jac ?bounds ?method_ ?ftol ?xtol ?gtol ?x_scale ?loss ?f_scale ?diff_step ?tr_solver ?tr_options ?jac_sparsity ?max_nfev ?verbose ?args ?kwargs ~fun_ ~x0 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "least_squares"
                       [||]
                       (Wrap_utils.keyword_args [("jac", Wrap_utils.Option.map jac (function
| `T2_point -> Py.String.of_string "2-point"
| `Callable x -> Wrap_utils.id x
| `Cs -> Py.String.of_string "cs"
| `T3_point -> Py.String.of_string "3-point"
)); ("bounds", bounds); ("method", Wrap_utils.Option.map method_ (function
| `Trf -> Py.String.of_string "trf"
| `Dogbox -> Py.String.of_string "dogbox"
| `Lm -> Py.String.of_string "lm"
)); ("ftol", Wrap_utils.Option.map ftol (function
| `F x -> Py.Float.of_float x
| `None -> Py.none
)); ("xtol", Wrap_utils.Option.map xtol (function
| `F x -> Py.Float.of_float x
| `None -> Py.none
)); ("gtol", Wrap_utils.Option.map gtol (function
| `F x -> Py.Float.of_float x
| `None -> Py.none
)); ("x_scale", Wrap_utils.Option.map x_scale (function
| `Jac -> Py.String.of_string "jac"
| `Ndarray x -> Np.Obj.to_pyobject x
)); ("loss", Wrap_utils.Option.map loss (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("f_scale", Wrap_utils.Option.map f_scale Py.Float.of_float); ("diff_step", Wrap_utils.Option.map diff_step Np.Obj.to_pyobject); ("tr_solver", Wrap_utils.Option.map tr_solver (function
| `Exact -> Py.String.of_string "exact"
| `Lsmr -> Py.String.of_string "lsmr"
)); ("tr_options", tr_options); ("jac_sparsity", Wrap_utils.Option.map jac_sparsity Np.Obj.to_pyobject); ("max_nfev", Wrap_utils.Option.map max_nfev Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose (function
| `Two -> Py.Int.of_int 2
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
)); ("args", args); ("kwargs", kwargs); ("fun", Some(fun_ )); ("x0", Some(x0 |> (function
| `F x -> Py.Float.of_float x
| `Ndarray x -> Np.Obj.to_pyobject x
)))])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 3)), (Py.Float.to_float (Py.Tuple.get x 4)), (Wrap_utils.id (Py.Tuple.get x 5)), (Py.Int.to_int (Py.Tuple.get x 6)), ((fun py -> if Py.is_none py then None else Some (Py.Int.to_int py)) (Py.Tuple.get x 7)), (Py.Int.to_int (Py.Tuple.get x 8)), (Py.String.to_string (Py.Tuple.get x 9)), (Py.Bool.to_bool (Py.Tuple.get x 10))))
let leastsq ?args ?dfun ?full_output ?col_deriv ?ftol ?xtol ?gtol ?maxfev ?epsfcn ?factor ?diag ~func ~x0 () =
   Py.Module.get_function_with_keywords __wrap_namespace "leastsq"
     [||]
     (Wrap_utils.keyword_args [("args", args); ("Dfun", dfun); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("col_deriv", Wrap_utils.Option.map col_deriv Py.Bool.of_bool); ("ftol", Wrap_utils.Option.map ftol Py.Float.of_float); ("xtol", Wrap_utils.Option.map xtol Py.Float.of_float); ("gtol", Wrap_utils.Option.map gtol Py.Float.of_float); ("maxfev", Wrap_utils.Option.map maxfev Py.Int.of_int); ("epsfcn", Wrap_utils.Option.map epsfcn Py.Float.of_float); ("factor", Wrap_utils.Option.map factor Py.Float.of_float); ("diag", diag); ("func", Some(func )); ("x0", Some(x0 |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), (Py.String.to_string (Py.Tuple.get x 3)), (Py.Int.to_int (Py.Tuple.get x 4))))
let line_search ?gfk ?old_fval ?old_old_fval ?args ?c1 ?c2 ?amax ?extra_condition ?maxiter ~f ~myfprime ~xk ~pk () =
   Py.Module.get_function_with_keywords __wrap_namespace "line_search"
     [||]
     (Wrap_utils.keyword_args [("gfk", Wrap_utils.Option.map gfk Np.Obj.to_pyobject); ("old_fval", Wrap_utils.Option.map old_fval Py.Float.of_float); ("old_old_fval", Wrap_utils.Option.map old_old_fval Py.Float.of_float); ("args", args); ("c1", Wrap_utils.Option.map c1 Py.Float.of_float); ("c2", Wrap_utils.Option.map c2 Py.Float.of_float); ("amax", Wrap_utils.Option.map amax Py.Float.of_float); ("extra_condition", extra_condition); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("f", Some(f )); ("myfprime", Some(myfprime )); ("xk", Some(xk |> Np.Obj.to_pyobject)); ("pk", Some(pk |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> if Py.is_none py then None else Some (Py.Float.to_float py)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), ((fun py -> if Py.is_none py then None else Some (Py.Float.to_float py)) (Py.Tuple.get x 3)), (Py.Float.to_float (Py.Tuple.get x 4)), ((fun py -> if Py.is_none py then None else Some (Py.Float.to_float py)) (Py.Tuple.get x 5))))
let linear_sum_assignment ?maximize ~cost_matrix () =
   Py.Module.get_function_with_keywords __wrap_namespace "linear_sum_assignment"
     [||]
     (Wrap_utils.keyword_args [("maximize", Wrap_utils.Option.map maximize Py.Bool.of_bool); ("cost_matrix", Some(cost_matrix |> Np.Obj.to_pyobject))])

                  let linearmixing ?iter ?alpha ?verbose ?maxiter ?f_tol ?f_rtol ?x_tol ?x_rtol ?tol_norm ?line_search ?callback ?kw ~f ~xin () =
                     Py.Module.get_function_with_keywords __wrap_namespace "linearmixing"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("iter", Wrap_utils.Option.map iter Py.Int.of_int); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("f_tol", Wrap_utils.Option.map f_tol Py.Float.of_float); ("f_rtol", Wrap_utils.Option.map f_rtol Py.Float.of_float); ("x_tol", Wrap_utils.Option.map x_tol Py.Float.of_float); ("x_rtol", Wrap_utils.Option.map x_rtol Py.Float.of_float); ("tol_norm", tol_norm); ("line_search", Wrap_utils.Option.map line_search (function
| `Wolfe -> Py.String.of_string "wolfe"
| `Armijo -> Py.String.of_string "armijo"
| `None -> Py.none
)); ("callback", callback); ("F", Some(f )); ("xin", Some(xin |> Np.Obj.to_pyobject))]) (match kw with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let linprog ?a_ub ?b_ub ?a_eq ?b_eq ?bounds ?method_ ?callback ?options ?x0 ~c () =
                     Py.Module.get_function_with_keywords __wrap_namespace "linprog"
                       [||]
                       (Wrap_utils.keyword_args [("A_ub", a_ub); ("b_ub", b_ub); ("A_eq", a_eq); ("b_eq", b_eq); ("bounds", bounds); ("method", Wrap_utils.Option.map method_ (function
| `Interior_point -> Py.String.of_string "interior-point"
| `Revised_simplex -> Py.String.of_string "revised simplex"
| `Simplex -> Py.String.of_string "simplex"
)); ("callback", callback); ("options", options); ("x0", x0); ("c", Some(c ))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3)), (Wrap_utils.id (Py.Tuple.get x 4)), (Py.Bool.to_bool (Py.Tuple.get x 5)), (Py.Int.to_int (Py.Tuple.get x 6)), (Py.Int.to_int (Py.Tuple.get x 7)), (Py.String.to_string (Py.Tuple.get x 8))))
let linprog_verbose_callback res =
   Py.Module.get_function_with_keywords __wrap_namespace "linprog_verbose_callback"
     [||]
     (Wrap_utils.keyword_args [("res", Some(res ))])

                  let lsq_linear ?bounds ?method_ ?tol ?lsq_solver ?lsmr_tol ?max_iter ?verbose ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "lsq_linear"
                       [||]
                       (Wrap_utils.keyword_args [("bounds", bounds); ("method", Wrap_utils.Option.map method_ (function
| `Trf -> Py.String.of_string "trf"
| `Bvls -> Py.String.of_string "bvls"
)); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("lsq_solver", Wrap_utils.Option.map lsq_solver (function
| `Exact -> Py.String.of_string "exact"
| `Lsmr -> Py.String.of_string "lsmr"
)); ("lsmr_tol", Wrap_utils.Option.map lsmr_tol (function
| `F x -> Py.Float.of_float x
| `Auto -> Py.String.of_string "auto"
)); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose (function
| `Two -> Py.Int.of_int 2
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
)); ("A", Some(a |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Sparse_matrix_of_LinearOperator x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3)), (Py.Int.to_int (Py.Tuple.get x 4)), (Py.Int.to_int (Py.Tuple.get x 5)), (Py.String.to_string (Py.Tuple.get x 6)), (Py.Bool.to_bool (Py.Tuple.get x 7))))
                  let minimize ?args ?method_ ?jac ?hess ?hessp ?bounds ?constraints ?tol ?callback ?options ~fun_ ~x0 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "minimize"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("method", Wrap_utils.Option.map method_ (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("jac", jac); ("hess", Wrap_utils.Option.map hess (function
| `T2_point -> Py.String.of_string "2-point"
| `HessianUpdateStrategy x -> Wrap_utils.id x
| `Cs -> Py.String.of_string "cs"
| `Callable x -> Wrap_utils.id x
| `T3_point -> Py.String.of_string "3-point"
)); ("hessp", hessp); ("bounds", bounds); ("constraints", constraints); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("callback", callback); ("options", options); ("fun", Some(fun_ )); ("x0", Some(x0 |> Np.Obj.to_pyobject))])

                  let minimize_scalar ?bracket ?bounds ?args ?method_ ?tol ?options ~fun_ () =
                     Py.Module.get_function_with_keywords __wrap_namespace "minimize_scalar"
                       [||]
                       (Wrap_utils.keyword_args [("bracket", bracket); ("bounds", bounds); ("args", args); ("method", Wrap_utils.Option.map method_ (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("options", options); ("fun", Some(fun_ ))])

                  let newton ?fprime ?args ?tol ?maxiter ?fprime2 ?x1 ?rtol ?full_output ?disp ~func ~x0 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "newton"
                       [||]
                       (Wrap_utils.keyword_args [("fprime", fprime); ("args", args); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("fprime2", fprime2); ("x1", Wrap_utils.Option.map x1 Py.Float.of_float); ("rtol", Wrap_utils.Option.map rtol Py.Float.of_float); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("func", Some(func )); ("x0", Some(x0 |> (function
| `F x -> Py.Float.of_float x
| `Ndarray x -> Np.Obj.to_pyobject x
| `Sequence x -> Wrap_utils.id x
)))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3))))
                  let newton_krylov ?iter ?rdiff ?method_ ?inner_maxiter ?inner_M ?outer_k ?verbose ?maxiter ?f_tol ?f_rtol ?x_tol ?x_rtol ?tol_norm ?line_search ?callback ?kw ~f ~xin () =
                     Py.Module.get_function_with_keywords __wrap_namespace "newton_krylov"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("iter", Wrap_utils.Option.map iter Py.Int.of_int); ("rdiff", Wrap_utils.Option.map rdiff Py.Float.of_float); ("method", Wrap_utils.Option.map method_ (function
| `Gmres -> Py.String.of_string "gmres"
| `Cgs -> Py.String.of_string "cgs"
| `Lgmres -> Py.String.of_string "lgmres"
| `Minres -> Py.String.of_string "minres"
| `Bicgstab -> Py.String.of_string "bicgstab"
| `Callable x -> Wrap_utils.id x
)); ("inner_maxiter", Wrap_utils.Option.map inner_maxiter Py.Int.of_int); ("inner_M", inner_M); ("outer_k", Wrap_utils.Option.map outer_k Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("f_tol", Wrap_utils.Option.map f_tol Py.Float.of_float); ("f_rtol", Wrap_utils.Option.map f_rtol Py.Float.of_float); ("x_tol", Wrap_utils.Option.map x_tol Py.Float.of_float); ("x_rtol", Wrap_utils.Option.map x_rtol Py.Float.of_float); ("tol_norm", tol_norm); ("line_search", Wrap_utils.Option.map line_search (function
| `Wolfe -> Py.String.of_string "wolfe"
| `Armijo -> Py.String.of_string "armijo"
| `None -> Py.none
)); ("callback", callback); ("F", Some(f )); ("xin", Some(xin |> Np.Obj.to_pyobject))]) (match kw with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let nnls ?maxiter ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "nnls"
     [||]
     (Wrap_utils.keyword_args [("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("A", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))
                  let ridder ?args ?xtol ?rtol ?maxiter ?full_output ?disp ~f ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ridder"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("xtol", Wrap_utils.Option.map xtol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("rtol", Wrap_utils.Option.map rtol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("f", Some(f )); ("a", Some(a |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
))); ("b", Some(b |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let root ?args ?method_ ?jac ?tol ?callback ?options ~fun_ ~x0 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "root"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("method", Wrap_utils.Option.map method_ Py.String.of_string); ("jac", Wrap_utils.Option.map jac (function
| `Callable x -> Wrap_utils.id x
| `Bool x -> Py.Bool.of_bool x
)); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("callback", callback); ("options", options); ("fun", Some(fun_ )); ("x0", Some(x0 |> Np.Obj.to_pyobject))])

                  let root_scalar ?args ?method_ ?bracket ?fprime ?fprime2 ?x0 ?x1 ?xtol ?rtol ?maxiter ?options ~f () =
                     Py.Module.get_function_with_keywords __wrap_namespace "root_scalar"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("method", Wrap_utils.Option.map method_ Py.String.of_string); ("bracket", bracket); ("fprime", Wrap_utils.Option.map fprime (function
| `Callable x -> Wrap_utils.id x
| `Bool x -> Py.Bool.of_bool x
)); ("fprime2", Wrap_utils.Option.map fprime2 (function
| `Callable x -> Wrap_utils.id x
| `Bool x -> Py.Bool.of_bool x
)); ("x0", Wrap_utils.Option.map x0 Py.Float.of_float); ("x1", Wrap_utils.Option.map x1 Py.Float.of_float); ("xtol", Wrap_utils.Option.map xtol Py.Float.of_float); ("rtol", Wrap_utils.Option.map rtol Py.Float.of_float); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("options", options); ("f", Some(f ))])

let rosen x =
   Py.Module.get_function_with_keywords __wrap_namespace "rosen"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let rosen_der x =
   Py.Module.get_function_with_keywords __wrap_namespace "rosen_der"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let rosen_hess x =
   Py.Module.get_function_with_keywords __wrap_namespace "rosen_hess"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let rosen_hess_prod ~x ~p () =
   Py.Module.get_function_with_keywords __wrap_namespace "rosen_hess_prod"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let shgo ?args ?constraints ?n ?iters ?callback ?minimizer_kwargs ?options ?sampling_method ~func ~bounds () =
                     Py.Module.get_function_with_keywords __wrap_namespace "shgo"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("constraints", constraints); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("iters", Wrap_utils.Option.map iters Py.Int.of_int); ("callback", callback); ("minimizer_kwargs", minimizer_kwargs); ("options", options); ("sampling_method", Wrap_utils.Option.map sampling_method (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("func", Some(func )); ("bounds", Some(bounds ))])

let show_options ?solver ?method_ ?disp () =
   Py.Module.get_function_with_keywords __wrap_namespace "show_options"
     [||]
     (Wrap_utils.keyword_args [("solver", Wrap_utils.Option.map solver Py.String.of_string); ("method", Wrap_utils.Option.map method_ Py.String.of_string); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool)])

                  let toms748 ?args ?k ?xtol ?rtol ?maxiter ?full_output ?disp ~f ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "toms748"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("k", Wrap_utils.Option.map k Py.Int.of_int); ("xtol", Wrap_utils.Option.map xtol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("rtol", Wrap_utils.Option.map rtol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("f", Some(f )); ("a", Some(a |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
))); ("b", Some(b |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
