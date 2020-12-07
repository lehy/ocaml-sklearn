let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.integrate"

let get_py name = Py.Module.get __wrap_namespace name
module AccuracyWarning = struct
type tag = [`AccuracyWarning]
type t = [`AccuracyWarning | `BaseException | `Object] Obj.t
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
module BDF = struct
type tag = [`BDF]
type t = [`BDF | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?max_step ?rtol ?atol ?jac ?jac_sparsity ?vectorized ?first_step ?extraneous ~fun_ ~t0 ~y0 ~t_bound () =
                     Py.Module.get_function_with_keywords __wrap_namespace "BDF"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("max_step", Wrap_utils.Option.map max_step Py.Float.of_float); ("rtol", rtol); ("atol", atol); ("jac", Wrap_utils.Option.map jac (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Callable x -> Wrap_utils.id x
| `Sparse_matrix x -> Wrap_utils.id x
)); ("jac_sparsity", Wrap_utils.Option.map jac_sparsity Np.Obj.to_pyobject); ("vectorized", Wrap_utils.Option.map vectorized Py.Bool.of_bool); ("first_step", Wrap_utils.Option.map first_step Py.Float.of_float); ("fun", Some(fun_ )); ("t0", Some(t0 |> Py.Float.of_float)); ("y0", Some(y0 |> Np.Obj.to_pyobject)); ("t_bound", Some(t_bound |> Py.Float.of_float))]) (match extraneous with None -> [] | Some x -> x))
                       |> of_pyobject
let dense_output self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dense_output"
     [||]
     []

let step self =
   Py.Module.get_function_with_keywords (to_pyobject self) "step"
     [||]
     []
     |> (fun py -> if Py.is_none py then None else Some (Py.String.to_string py))

let n_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n" with
  | None -> failwith "attribute n not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n self = match n_opt self with
  | None -> raise Not_found
  | Some x -> x

let status_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "status" with
  | None -> failwith "attribute status not found"
  | Some x -> if Py.is_none x then None else Some (Py.String.to_string x)

let status self = match status_opt self with
  | None -> raise Not_found
  | Some x -> x

let t_bound_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t_bound" with
  | None -> failwith "attribute t_bound not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let t_bound self = match t_bound_opt self with
  | None -> raise Not_found
  | Some x -> x

let direction_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "direction" with
  | None -> failwith "attribute direction not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let direction self = match direction_opt self with
  | None -> raise Not_found
  | Some x -> x

let t_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t" with
  | None -> failwith "attribute t not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let t self = match t_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "y" with
  | None -> failwith "attribute y not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let y self = match y_opt self with
  | None -> raise Not_found
  | Some x -> x

let t_old_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t_old" with
  | None -> failwith "attribute t_old not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let t_old self = match t_old_opt self with
  | None -> raise Not_found
  | Some x -> x

let step_size_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "step_size" with
  | None -> failwith "attribute step_size not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let step_size self = match step_size_opt self with
  | None -> raise Not_found
  | Some x -> x

let nfev_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nfev" with
  | None -> failwith "attribute nfev not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let nfev self = match nfev_opt self with
  | None -> raise Not_found
  | Some x -> x

let njev_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "njev" with
  | None -> failwith "attribute njev not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let njev self = match njev_opt self with
  | None -> raise Not_found
  | Some x -> x

let nlu_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nlu" with
  | None -> failwith "attribute nlu not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let nlu self = match nlu_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module DOP853 = struct
type tag = [`DOP853]
type t = [`DOP853 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?max_step ?rtol ?atol ?vectorized ?first_step ?extraneous ~fun_ ~t0 ~y0 ~t_bound () =
   Py.Module.get_function_with_keywords __wrap_namespace "DOP853"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("max_step", Wrap_utils.Option.map max_step Py.Float.of_float); ("rtol", rtol); ("atol", atol); ("vectorized", Wrap_utils.Option.map vectorized Py.Bool.of_bool); ("first_step", Wrap_utils.Option.map first_step Py.Float.of_float); ("fun", Some(fun_ )); ("t0", Some(t0 |> Py.Float.of_float)); ("y0", Some(y0 |> Np.Obj.to_pyobject)); ("t_bound", Some(t_bound |> Py.Float.of_float))]) (match extraneous with None -> [] | Some x -> x))
     |> of_pyobject
let dense_output self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dense_output"
     [||]
     []

let step self =
   Py.Module.get_function_with_keywords (to_pyobject self) "step"
     [||]
     []
     |> (fun py -> if Py.is_none py then None else Some (Py.String.to_string py))

let n_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n" with
  | None -> failwith "attribute n not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n self = match n_opt self with
  | None -> raise Not_found
  | Some x -> x

let status_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "status" with
  | None -> failwith "attribute status not found"
  | Some x -> if Py.is_none x then None else Some (Py.String.to_string x)

let status self = match status_opt self with
  | None -> raise Not_found
  | Some x -> x

let t_bound_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t_bound" with
  | None -> failwith "attribute t_bound not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let t_bound self = match t_bound_opt self with
  | None -> raise Not_found
  | Some x -> x

let direction_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "direction" with
  | None -> failwith "attribute direction not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let direction self = match direction_opt self with
  | None -> raise Not_found
  | Some x -> x

let t_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t" with
  | None -> failwith "attribute t not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let t self = match t_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "y" with
  | None -> failwith "attribute y not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let y self = match y_opt self with
  | None -> raise Not_found
  | Some x -> x

let t_old_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t_old" with
  | None -> failwith "attribute t_old not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let t_old self = match t_old_opt self with
  | None -> raise Not_found
  | Some x -> x

let step_size_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "step_size" with
  | None -> failwith "attribute step_size not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let step_size self = match step_size_opt self with
  | None -> raise Not_found
  | Some x -> x

let nfev_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nfev" with
  | None -> failwith "attribute nfev not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let nfev self = match nfev_opt self with
  | None -> raise Not_found
  | Some x -> x

let njev_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "njev" with
  | None -> failwith "attribute njev not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let njev self = match njev_opt self with
  | None -> raise Not_found
  | Some x -> x

let nlu_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nlu" with
  | None -> failwith "attribute nlu not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let nlu self = match nlu_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module DenseOutput = struct
type tag = [`DenseOutput]
type t = [`DenseOutput | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ~t_old ~t () =
   Py.Module.get_function_with_keywords __wrap_namespace "DenseOutput"
     [||]
     (Wrap_utils.keyword_args [("t_old", Some(t_old )); ("t", Some(t ))])
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module IntegrationWarning = struct
type tag = [`IntegrationWarning]
type t = [`BaseException | `IntegrationWarning | `Object] Obj.t
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
module LSODA = struct
type tag = [`LSODA]
type t = [`LSODA | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?first_step ?min_step ?max_step ?rtol ?atol ?jac ?lband ?uband ?vectorized ?extraneous ~fun_ ~t0 ~y0 ~t_bound () =
   Py.Module.get_function_with_keywords __wrap_namespace "LSODA"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("first_step", Wrap_utils.Option.map first_step Py.Float.of_float); ("min_step", Wrap_utils.Option.map min_step Py.Float.of_float); ("max_step", Wrap_utils.Option.map max_step Py.Float.of_float); ("rtol", rtol); ("atol", atol); ("jac", jac); ("lband", lband); ("uband", uband); ("vectorized", Wrap_utils.Option.map vectorized Py.Bool.of_bool); ("fun", Some(fun_ )); ("t0", Some(t0 |> Py.Float.of_float)); ("y0", Some(y0 |> Np.Obj.to_pyobject)); ("t_bound", Some(t_bound |> Py.Float.of_float))]) (match extraneous with None -> [] | Some x -> x))
     |> of_pyobject
let dense_output self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dense_output"
     [||]
     []

let step self =
   Py.Module.get_function_with_keywords (to_pyobject self) "step"
     [||]
     []
     |> (fun py -> if Py.is_none py then None else Some (Py.String.to_string py))

let n_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n" with
  | None -> failwith "attribute n not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n self = match n_opt self with
  | None -> raise Not_found
  | Some x -> x

let status_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "status" with
  | None -> failwith "attribute status not found"
  | Some x -> if Py.is_none x then None else Some (Py.String.to_string x)

let status self = match status_opt self with
  | None -> raise Not_found
  | Some x -> x

let t_bound_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t_bound" with
  | None -> failwith "attribute t_bound not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let t_bound self = match t_bound_opt self with
  | None -> raise Not_found
  | Some x -> x

let direction_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "direction" with
  | None -> failwith "attribute direction not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let direction self = match direction_opt self with
  | None -> raise Not_found
  | Some x -> x

let t_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t" with
  | None -> failwith "attribute t not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let t self = match t_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "y" with
  | None -> failwith "attribute y not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let y self = match y_opt self with
  | None -> raise Not_found
  | Some x -> x

let t_old_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t_old" with
  | None -> failwith "attribute t_old not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let t_old self = match t_old_opt self with
  | None -> raise Not_found
  | Some x -> x

let nfev_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nfev" with
  | None -> failwith "attribute nfev not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let nfev self = match nfev_opt self with
  | None -> raise Not_found
  | Some x -> x

let njev_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "njev" with
  | None -> failwith "attribute njev not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let njev self = match njev_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module OdeSolution = struct
type tag = [`OdeSolution]
type t = [`Object | `OdeSolution] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ~ts ~interpolants () =
   Py.Module.get_function_with_keywords __wrap_namespace "OdeSolution"
     [||]
     (Wrap_utils.keyword_args [("ts", Some(ts |> Np.Obj.to_pyobject)); ("interpolants", Some(interpolants ))])
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module OdeSolver = struct
type tag = [`OdeSolver]
type t = [`Object | `OdeSolver] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?support_complex ~fun_ ~t0 ~y0 ~t_bound ~vectorized () =
   Py.Module.get_function_with_keywords __wrap_namespace "OdeSolver"
     [||]
     (Wrap_utils.keyword_args [("support_complex", Wrap_utils.Option.map support_complex Py.Bool.of_bool); ("fun", Some(fun_ )); ("t0", Some(t0 |> Py.Float.of_float)); ("y0", Some(y0 |> Np.Obj.to_pyobject)); ("t_bound", Some(t_bound |> Py.Float.of_float)); ("vectorized", Some(vectorized |> Py.Bool.of_bool))])
     |> of_pyobject
let dense_output self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dense_output"
     [||]
     []

let step self =
   Py.Module.get_function_with_keywords (to_pyobject self) "step"
     [||]
     []
     |> (fun py -> if Py.is_none py then None else Some (Py.String.to_string py))

let n_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n" with
  | None -> failwith "attribute n not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n self = match n_opt self with
  | None -> raise Not_found
  | Some x -> x

let status_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "status" with
  | None -> failwith "attribute status not found"
  | Some x -> if Py.is_none x then None else Some (Py.String.to_string x)

let status self = match status_opt self with
  | None -> raise Not_found
  | Some x -> x

let t_bound_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t_bound" with
  | None -> failwith "attribute t_bound not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let t_bound self = match t_bound_opt self with
  | None -> raise Not_found
  | Some x -> x

let direction_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "direction" with
  | None -> failwith "attribute direction not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let direction self = match direction_opt self with
  | None -> raise Not_found
  | Some x -> x

let t_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t" with
  | None -> failwith "attribute t not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let t self = match t_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "y" with
  | None -> failwith "attribute y not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let y self = match y_opt self with
  | None -> raise Not_found
  | Some x -> x

let t_old_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t_old" with
  | None -> failwith "attribute t_old not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let t_old self = match t_old_opt self with
  | None -> raise Not_found
  | Some x -> x

let step_size_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "step_size" with
  | None -> failwith "attribute step_size not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let step_size self = match step_size_opt self with
  | None -> raise Not_found
  | Some x -> x

let nfev_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nfev" with
  | None -> failwith "attribute nfev not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let nfev self = match nfev_opt self with
  | None -> raise Not_found
  | Some x -> x

let njev_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "njev" with
  | None -> failwith "attribute njev not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let njev self = match njev_opt self with
  | None -> raise Not_found
  | Some x -> x

let nlu_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nlu" with
  | None -> failwith "attribute nlu not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let nlu self = match nlu_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RK23 = struct
type tag = [`RK23]
type t = [`Object | `RK23] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?max_step ?rtol ?atol ?vectorized ?first_step ?extraneous ~fun_ ~t0 ~y0 ~t_bound () =
   Py.Module.get_function_with_keywords __wrap_namespace "RK23"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("max_step", Wrap_utils.Option.map max_step Py.Float.of_float); ("rtol", rtol); ("atol", atol); ("vectorized", Wrap_utils.Option.map vectorized Py.Bool.of_bool); ("first_step", Wrap_utils.Option.map first_step Py.Float.of_float); ("fun", Some(fun_ )); ("t0", Some(t0 |> Py.Float.of_float)); ("y0", Some(y0 |> Np.Obj.to_pyobject)); ("t_bound", Some(t_bound |> Py.Float.of_float))]) (match extraneous with None -> [] | Some x -> x))
     |> of_pyobject
let dense_output self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dense_output"
     [||]
     []

let step self =
   Py.Module.get_function_with_keywords (to_pyobject self) "step"
     [||]
     []
     |> (fun py -> if Py.is_none py then None else Some (Py.String.to_string py))

let n_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n" with
  | None -> failwith "attribute n not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n self = match n_opt self with
  | None -> raise Not_found
  | Some x -> x

let status_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "status" with
  | None -> failwith "attribute status not found"
  | Some x -> if Py.is_none x then None else Some (Py.String.to_string x)

let status self = match status_opt self with
  | None -> raise Not_found
  | Some x -> x

let t_bound_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t_bound" with
  | None -> failwith "attribute t_bound not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let t_bound self = match t_bound_opt self with
  | None -> raise Not_found
  | Some x -> x

let direction_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "direction" with
  | None -> failwith "attribute direction not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let direction self = match direction_opt self with
  | None -> raise Not_found
  | Some x -> x

let t_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t" with
  | None -> failwith "attribute t not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let t self = match t_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "y" with
  | None -> failwith "attribute y not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let y self = match y_opt self with
  | None -> raise Not_found
  | Some x -> x

let t_old_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t_old" with
  | None -> failwith "attribute t_old not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let t_old self = match t_old_opt self with
  | None -> raise Not_found
  | Some x -> x

let step_size_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "step_size" with
  | None -> failwith "attribute step_size not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let step_size self = match step_size_opt self with
  | None -> raise Not_found
  | Some x -> x

let nfev_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nfev" with
  | None -> failwith "attribute nfev not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let nfev self = match nfev_opt self with
  | None -> raise Not_found
  | Some x -> x

let njev_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "njev" with
  | None -> failwith "attribute njev not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let njev self = match njev_opt self with
  | None -> raise Not_found
  | Some x -> x

let nlu_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nlu" with
  | None -> failwith "attribute nlu not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let nlu self = match nlu_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RK45 = struct
type tag = [`RK45]
type t = [`Object | `RK45] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?max_step ?rtol ?atol ?vectorized ?first_step ?extraneous ~fun_ ~t0 ~y0 ~t_bound () =
   Py.Module.get_function_with_keywords __wrap_namespace "RK45"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("max_step", Wrap_utils.Option.map max_step Py.Float.of_float); ("rtol", rtol); ("atol", atol); ("vectorized", Wrap_utils.Option.map vectorized Py.Bool.of_bool); ("first_step", Wrap_utils.Option.map first_step Py.Float.of_float); ("fun", Some(fun_ )); ("t0", Some(t0 |> Py.Float.of_float)); ("y0", Some(y0 |> Np.Obj.to_pyobject)); ("t_bound", Some(t_bound |> Py.Float.of_float))]) (match extraneous with None -> [] | Some x -> x))
     |> of_pyobject
let dense_output self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dense_output"
     [||]
     []

let step self =
   Py.Module.get_function_with_keywords (to_pyobject self) "step"
     [||]
     []
     |> (fun py -> if Py.is_none py then None else Some (Py.String.to_string py))

let n_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n" with
  | None -> failwith "attribute n not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n self = match n_opt self with
  | None -> raise Not_found
  | Some x -> x

let status_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "status" with
  | None -> failwith "attribute status not found"
  | Some x -> if Py.is_none x then None else Some (Py.String.to_string x)

let status self = match status_opt self with
  | None -> raise Not_found
  | Some x -> x

let t_bound_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t_bound" with
  | None -> failwith "attribute t_bound not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let t_bound self = match t_bound_opt self with
  | None -> raise Not_found
  | Some x -> x

let direction_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "direction" with
  | None -> failwith "attribute direction not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let direction self = match direction_opt self with
  | None -> raise Not_found
  | Some x -> x

let t_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t" with
  | None -> failwith "attribute t not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let t self = match t_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "y" with
  | None -> failwith "attribute y not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let y self = match y_opt self with
  | None -> raise Not_found
  | Some x -> x

let t_old_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t_old" with
  | None -> failwith "attribute t_old not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let t_old self = match t_old_opt self with
  | None -> raise Not_found
  | Some x -> x

let step_size_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "step_size" with
  | None -> failwith "attribute step_size not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let step_size self = match step_size_opt self with
  | None -> raise Not_found
  | Some x -> x

let nfev_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nfev" with
  | None -> failwith "attribute nfev not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let nfev self = match nfev_opt self with
  | None -> raise Not_found
  | Some x -> x

let njev_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "njev" with
  | None -> failwith "attribute njev not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let njev self = match njev_opt self with
  | None -> raise Not_found
  | Some x -> x

let nlu_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nlu" with
  | None -> failwith "attribute nlu not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let nlu self = match nlu_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Radau = struct
type tag = [`Radau]
type t = [`Object | `Radau] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?max_step ?rtol ?atol ?jac ?jac_sparsity ?vectorized ?first_step ?extraneous ~fun_ ~t0 ~y0 ~t_bound () =
                     Py.Module.get_function_with_keywords __wrap_namespace "Radau"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("max_step", Wrap_utils.Option.map max_step Py.Float.of_float); ("rtol", rtol); ("atol", atol); ("jac", Wrap_utils.Option.map jac (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Callable x -> Wrap_utils.id x
| `Sparse_matrix x -> Wrap_utils.id x
)); ("jac_sparsity", Wrap_utils.Option.map jac_sparsity Np.Obj.to_pyobject); ("vectorized", Wrap_utils.Option.map vectorized Py.Bool.of_bool); ("first_step", Wrap_utils.Option.map first_step Py.Float.of_float); ("fun", Some(fun_ )); ("t0", Some(t0 |> Py.Float.of_float)); ("y0", Some(y0 |> Np.Obj.to_pyobject)); ("t_bound", Some(t_bound |> Py.Float.of_float))]) (match extraneous with None -> [] | Some x -> x))
                       |> of_pyobject
let dense_output self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dense_output"
     [||]
     []

let step self =
   Py.Module.get_function_with_keywords (to_pyobject self) "step"
     [||]
     []
     |> (fun py -> if Py.is_none py then None else Some (Py.String.to_string py))

let n_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n" with
  | None -> failwith "attribute n not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n self = match n_opt self with
  | None -> raise Not_found
  | Some x -> x

let status_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "status" with
  | None -> failwith "attribute status not found"
  | Some x -> if Py.is_none x then None else Some (Py.String.to_string x)

let status self = match status_opt self with
  | None -> raise Not_found
  | Some x -> x

let t_bound_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t_bound" with
  | None -> failwith "attribute t_bound not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let t_bound self = match t_bound_opt self with
  | None -> raise Not_found
  | Some x -> x

let direction_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "direction" with
  | None -> failwith "attribute direction not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let direction self = match direction_opt self with
  | None -> raise Not_found
  | Some x -> x

let t_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t" with
  | None -> failwith "attribute t not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let t self = match t_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "y" with
  | None -> failwith "attribute y not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let y self = match y_opt self with
  | None -> raise Not_found
  | Some x -> x

let t_old_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t_old" with
  | None -> failwith "attribute t_old not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let t_old self = match t_old_opt self with
  | None -> raise Not_found
  | Some x -> x

let step_size_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "step_size" with
  | None -> failwith "attribute step_size not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let step_size self = match step_size_opt self with
  | None -> raise Not_found
  | Some x -> x

let nfev_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nfev" with
  | None -> failwith "attribute nfev not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let nfev self = match nfev_opt self with
  | None -> raise Not_found
  | Some x -> x

let njev_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "njev" with
  | None -> failwith "attribute njev not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let njev self = match njev_opt self with
  | None -> raise Not_found
  | Some x -> x

let nlu_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nlu" with
  | None -> failwith "attribute nlu not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let nlu self = match nlu_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Complex_ode = struct
type tag = [`Complex_ode]
type t = [`Complex_ode | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?jac ~f () =
   Py.Module.get_function_with_keywords __wrap_namespace "complex_ode"
     [||]
     (Wrap_utils.keyword_args [("jac", jac); ("f", Some(f ))])
     |> of_pyobject
let get_return_code self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_return_code"
     [||]
     []

let integrate ?step ?relax ~t self =
   Py.Module.get_function_with_keywords (to_pyobject self) "integrate"
     [||]
     (Wrap_utils.keyword_args [("step", Wrap_utils.Option.map step Py.Bool.of_bool); ("relax", Wrap_utils.Option.map relax Py.Bool.of_bool); ("t", Some(t |> Py.Float.of_float))])
     |> Py.Float.to_float
let set_f_params args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_f_params"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

let set_initial_value ?t ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_initial_value"
     [||]
     (Wrap_utils.keyword_args [("t", t); ("y", Some(y ))])

let set_integrator ?integrator_params ~name self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_integrator"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("name", Some(name |> Py.String.of_string))]) (match integrator_params with None -> [] | Some x -> x))

let set_jac_params args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_jac_params"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

let set_solout ~solout self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_solout"
     [||]
     (Wrap_utils.keyword_args [("solout", Some(solout ))])

let successful self =
   Py.Module.get_function_with_keywords (to_pyobject self) "successful"
     [||]
     []


let t_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t" with
  | None -> failwith "attribute t not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let t self = match t_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "y" with
  | None -> failwith "attribute y not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let y self = match y_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Ode = struct
type tag = [`Ode]
type t = [`Object | `Ode] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?jac ~f () =
   Py.Module.get_function_with_keywords __wrap_namespace "ode"
     [||]
     (Wrap_utils.keyword_args [("jac", jac); ("f", Some(f ))])
     |> of_pyobject
let get_return_code self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_return_code"
     [||]
     []

let integrate ?step ?relax ~t self =
   Py.Module.get_function_with_keywords (to_pyobject self) "integrate"
     [||]
     (Wrap_utils.keyword_args [("step", Wrap_utils.Option.map step Py.Bool.of_bool); ("relax", Wrap_utils.Option.map relax Py.Bool.of_bool); ("t", Some(t |> Py.Float.of_float))])
     |> Py.Float.to_float
let set_f_params args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_f_params"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

let set_initial_value ?t ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_initial_value"
     [||]
     (Wrap_utils.keyword_args [("t", t); ("y", Some(y ))])

let set_integrator ?integrator_params ~name self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_integrator"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("name", Some(name |> Py.String.of_string))]) (match integrator_params with None -> [] | Some x -> x))

let set_jac_params args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_jac_params"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

let set_solout ~solout self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_solout"
     [||]
     (Wrap_utils.keyword_args [("solout", Some(solout ))])

let successful self =
   Py.Module.get_function_with_keywords (to_pyobject self) "successful"
     [||]
     []


let t_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t" with
  | None -> failwith "attribute t not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let t self = match t_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "y" with
  | None -> failwith "attribute y not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let y self = match y_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Lsoda = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.integrate.lsoda"

let get_py name = Py.Module.get __wrap_namespace name

end
module Odepack = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.integrate.odepack"

let get_py name = Py.Module.get __wrap_namespace name
module ODEintWarning = struct
type tag = [`ODEintWarning]
type t = [`BaseException | `ODEintWarning | `Object] Obj.t
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
let copy x =
   Py.Module.get_function_with_keywords __wrap_namespace "copy"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let odeint ?args ?dfun ?col_deriv ?full_output ?ml ?mu ?rtol ?atol ?tcrit ?h0 ?hmax ?hmin ?ixpr ?mxstep ?mxhnil ?mxordn ?mxords ?printmessg ?tfirst ~func ~y0 ~t () =
   Py.Module.get_function_with_keywords __wrap_namespace "odeint"
     [||]
     (Wrap_utils.keyword_args [("args", args); ("Dfun", dfun); ("col_deriv", Wrap_utils.Option.map col_deriv Py.Bool.of_bool); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("ml", ml); ("mu", mu); ("rtol", rtol); ("atol", atol); ("tcrit", tcrit); ("h0", h0); ("hmax", hmax); ("hmin", hmin); ("ixpr", ixpr); ("mxstep", mxstep); ("mxhnil", mxhnil); ("mxordn", mxordn); ("mxords", mxords); ("printmessg", Wrap_utils.Option.map printmessg Py.Bool.of_bool); ("tfirst", Wrap_utils.Option.map tfirst Py.Bool.of_bool); ("func", Some(func )); ("y0", Some(y0 |> Np.Obj.to_pyobject)); ("t", Some(t |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))

end
module Quadpack = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.integrate.quadpack"

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
module Partial = struct
type tag = [`Partial]
type t = [`Object | `Partial] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?keywords ~func args =
   Py.Module.get_function_with_keywords __wrap_namespace "partial"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("func", Some(func ))]) (match keywords with None -> [] | Some x -> x))
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let dblquad ?args ?epsabs ?epsrel ~func ~a ~b ~gfun ~hfun () =
                     Py.Module.get_function_with_keywords __wrap_namespace "dblquad"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("epsabs", Wrap_utils.Option.map epsabs Py.Float.of_float); ("epsrel", Wrap_utils.Option.map epsrel Py.Float.of_float); ("func", Some(func )); ("a", Some(a )); ("b", Some(b )); ("gfun", Some(gfun |> (function
| `F x -> Py.Float.of_float x
| `Callable x -> Wrap_utils.id x
))); ("hfun", Some(hfun |> (function
| `F x -> Py.Float.of_float x
| `Callable x -> Wrap_utils.id x
)))])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))
                  let nquad ?args ?opts ?full_output ~func ~ranges () =
                     Py.Module.get_function_with_keywords __wrap_namespace "nquad"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("opts", opts); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("func", Some(func |> (function
| `Scipy_LowLevelCallable x -> Wrap_utils.id x
| `Callable x -> Wrap_utils.id x
))); ("ranges", Some(ranges ))])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
                  let quad ?args ?full_output ?epsabs ?epsrel ?limit ?points ?weight ?wvar ?wopts ?maxp1 ?limlst ~func ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "quad"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("full_output", Wrap_utils.Option.map full_output Py.Int.of_int); ("epsabs", epsabs); ("epsrel", epsrel); ("limit", limit); ("points", points); ("weight", weight); ("wvar", wvar); ("wopts", wopts); ("maxp1", maxp1); ("limlst", limlst); ("func", Some(func |> (function
| `Scipy_LowLevelCallable x -> Wrap_utils.id x
| `Callable x -> Wrap_utils.id x
))); ("a", Some(a |> Py.Float.of_float)); ("b", Some(b |> Py.Float.of_float))])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3)), (Wrap_utils.id (Py.Tuple.get x 4))))
let quad_explain ?output () =
   Py.Module.get_function_with_keywords __wrap_namespace "quad_explain"
     [||]
     (Wrap_utils.keyword_args [("output", output)])

                  let tplquad ?args ?epsabs ?epsrel ~func ~a ~b ~gfun ~hfun ~qfun ~rfun () =
                     Py.Module.get_function_with_keywords __wrap_namespace "tplquad"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("epsabs", Wrap_utils.Option.map epsabs Py.Float.of_float); ("epsrel", Wrap_utils.Option.map epsrel Py.Float.of_float); ("func", Some(func )); ("a", Some(a )); ("b", Some(b )); ("gfun", Some(gfun |> (function
| `F x -> Py.Float.of_float x
| `Callable x -> Wrap_utils.id x
))); ("hfun", Some(hfun |> (function
| `F x -> Py.Float.of_float x
| `Callable x -> Wrap_utils.id x
))); ("qfun", Some(qfun |> (function
| `F x -> Py.Float.of_float x
| `Callable x -> Wrap_utils.id x
))); ("rfun", Some(rfun |> (function
| `F x -> Py.Float.of_float x
| `Callable x -> Wrap_utils.id x
)))])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))

end
module Vode = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.integrate.vode"

let get_py name = Py.Module.get __wrap_namespace name

end
                  let cumtrapz ?x ?dx ?axis ?initial ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cumtrapz"
                       [||]
                       (Wrap_utils.keyword_args [("x", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("dx", Wrap_utils.Option.map dx Py.Float.of_float); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("initial", Wrap_utils.Option.map initial (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `Bool x -> Py.Bool.of_bool x
)); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let dblquad ?args ?epsabs ?epsrel ~func ~a ~b ~gfun ~hfun () =
                     Py.Module.get_function_with_keywords __wrap_namespace "dblquad"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("epsabs", Wrap_utils.Option.map epsabs Py.Float.of_float); ("epsrel", Wrap_utils.Option.map epsrel Py.Float.of_float); ("func", Some(func )); ("a", Some(a )); ("b", Some(b )); ("gfun", Some(gfun |> (function
| `F x -> Py.Float.of_float x
| `Callable x -> Wrap_utils.id x
))); ("hfun", Some(hfun |> (function
| `F x -> Py.Float.of_float x
| `Callable x -> Wrap_utils.id x
)))])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))
let fixed_quad ?args ?n ~func ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "fixed_quad"
     [||]
     (Wrap_utils.keyword_args [("args", args); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("func", Some(func )); ("a", Some(a |> Py.Float.of_float)); ("b", Some(b |> Py.Float.of_float))])
     |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let newton_cotes ?equal ~rn () =
   Py.Module.get_function_with_keywords __wrap_namespace "newton_cotes"
     [||]
     (Wrap_utils.keyword_args [("equal", Wrap_utils.Option.map equal Py.Int.of_int); ("rn", Some(rn |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))
                  let nquad ?args ?opts ?full_output ~func ~ranges () =
                     Py.Module.get_function_with_keywords __wrap_namespace "nquad"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("opts", opts); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("func", Some(func |> (function
| `Scipy_LowLevelCallable x -> Wrap_utils.id x
| `Callable x -> Wrap_utils.id x
))); ("ranges", Some(ranges ))])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
let odeint ?args ?dfun ?col_deriv ?full_output ?ml ?mu ?rtol ?atol ?tcrit ?h0 ?hmax ?hmin ?ixpr ?mxstep ?mxhnil ?mxordn ?mxords ?printmessg ?tfirst ~func ~y0 ~t () =
   Py.Module.get_function_with_keywords __wrap_namespace "odeint"
     [||]
     (Wrap_utils.keyword_args [("args", args); ("Dfun", dfun); ("col_deriv", Wrap_utils.Option.map col_deriv Py.Bool.of_bool); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("ml", ml); ("mu", mu); ("rtol", rtol); ("atol", atol); ("tcrit", tcrit); ("h0", h0); ("hmax", hmax); ("hmin", hmin); ("ixpr", ixpr); ("mxstep", mxstep); ("mxhnil", mxhnil); ("mxordn", mxordn); ("mxords", mxords); ("printmessg", Wrap_utils.Option.map printmessg Py.Bool.of_bool); ("tfirst", Wrap_utils.Option.map tfirst Py.Bool.of_bool); ("func", Some(func )); ("y0", Some(y0 |> Np.Obj.to_pyobject)); ("t", Some(t |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let quad ?args ?full_output ?epsabs ?epsrel ?limit ?points ?weight ?wvar ?wopts ?maxp1 ?limlst ~func ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "quad"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("full_output", Wrap_utils.Option.map full_output Py.Int.of_int); ("epsabs", epsabs); ("epsrel", epsrel); ("limit", limit); ("points", points); ("weight", weight); ("wvar", wvar); ("wopts", wopts); ("maxp1", maxp1); ("limlst", limlst); ("func", Some(func |> (function
| `Scipy_LowLevelCallable x -> Wrap_utils.id x
| `Callable x -> Wrap_utils.id x
))); ("a", Some(a |> Py.Float.of_float)); ("b", Some(b |> Py.Float.of_float))])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3)), (Wrap_utils.id (Py.Tuple.get x 4))))
let quad_explain ?output () =
   Py.Module.get_function_with_keywords __wrap_namespace "quad_explain"
     [||]
     (Wrap_utils.keyword_args [("output", output)])

                  let quad_vec ?epsabs ?epsrel ?norm ?cache_size ?limit ?workers ?points ?quadrature ?full_output ~f ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "quad_vec"
                       [||]
                       (Wrap_utils.keyword_args [("epsabs", Wrap_utils.Option.map epsabs Py.Float.of_float); ("epsrel", Wrap_utils.Option.map epsrel Py.Float.of_float); ("norm", Wrap_utils.Option.map norm (function
| `Max -> Py.String.of_string "max"
| `T2 -> Py.String.of_string "2"
)); ("cache_size", Wrap_utils.Option.map cache_size Py.Int.of_int); ("limit", limit); ("workers", Wrap_utils.Option.map workers (function
| `I x -> Py.Int.of_int x
| `Map_like_callable x -> Wrap_utils.id x
)); ("points", Wrap_utils.Option.map points Np.Obj.to_pyobject); ("quadrature", Wrap_utils.Option.map quadrature (function
| `Trapz -> Py.String.of_string "trapz"
| `Gk21 -> Py.String.of_string "gk21"
| `Gk15 -> Py.String.of_string "gk15"
)); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("f", Some(f )); ("a", Some(a |> Py.Float.of_float)); ("b", Some(b |> Py.Float.of_float))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), (Py.Bool.to_bool (Py.Tuple.get x 3)), (Py.Int.to_int (Py.Tuple.get x 4)), (Py.Int.to_int (Py.Tuple.get x 5)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 6)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 7)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 8))))
let quadrature ?args ?tol ?rtol ?maxiter ?vec_func ?miniter ~func ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "quadrature"
     [||]
     (Wrap_utils.keyword_args [("args", args); ("tol", tol); ("rtol", rtol); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("vec_func", Wrap_utils.Option.map vec_func Py.Bool.of_bool); ("miniter", Wrap_utils.Option.map miniter Py.Int.of_int); ("func", Some(func )); ("a", Some(a |> Py.Float.of_float)); ("b", Some(b |> Py.Float.of_float))])
     |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))
let romb ?dx ?axis ?show ~y () =
   Py.Module.get_function_with_keywords __wrap_namespace "romb"
     [||]
     (Wrap_utils.keyword_args [("dx", Wrap_utils.Option.map dx Py.Float.of_float); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("show", Wrap_utils.Option.map show Py.Bool.of_bool); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let romberg ?args ?tol ?rtol ?show ?divmax ?vec_func ~function_ ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "romberg"
     [||]
     (Wrap_utils.keyword_args [("args", args); ("tol", tol); ("rtol", rtol); ("show", show); ("divmax", divmax); ("vec_func", vec_func); ("function", Some(function_ )); ("a", Some(a |> Py.Float.of_float)); ("b", Some(b |> Py.Float.of_float))])
     |> Py.Float.to_float
                  let simps ?x ?dx ?axis ?even ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "simps"
                       [||]
                       (Wrap_utils.keyword_args [("x", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("dx", Wrap_utils.Option.map dx Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("even", Wrap_utils.Option.map even (function
| `Avg -> Py.String.of_string "avg"
| `First -> Py.String.of_string "first"
| `Last -> Py.String.of_string "last"
)); ("y", Some(y |> Np.Obj.to_pyobject))])

                  let solve_bvp ?p ?s ?fun_jac ?bc_jac ?tol ?max_nodes ?verbose ?bc_tol ~fun_ ~bc ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "solve_bvp"
                       [||]
                       (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p Np.Obj.to_pyobject); ("S", Wrap_utils.Option.map s Np.Obj.to_pyobject); ("fun_jac", fun_jac); ("bc_jac", bc_jac); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("max_nodes", Wrap_utils.Option.map max_nodes Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose (function
| `Two -> Py.Int.of_int 2
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
)); ("bc_tol", Wrap_utils.Option.map bc_tol Py.Float.of_float); ("fun", Some(fun_ )); ("bc", Some(bc )); ("x", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> if Py.is_none py then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) py)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 3)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 4)), (Py.Int.to_int (Py.Tuple.get x 5)), (Py.Int.to_int (Py.Tuple.get x 6)), (Py.String.to_string (Py.Tuple.get x 7)), (Py.Bool.to_bool (Py.Tuple.get x 8))))
                  let solve_ivp ?method_ ?t_eval ?dense_output ?events ?vectorized ?args ?options ~fun_ ~t_span ~y0 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "solve_ivp"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("method", Wrap_utils.Option.map method_ (function
| `T_OdeSolver_ x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("t_eval", Wrap_utils.Option.map t_eval Np.Obj.to_pyobject); ("dense_output", Wrap_utils.Option.map dense_output Py.Bool.of_bool); ("events", Wrap_utils.Option.map events (function
| `Callable x -> Wrap_utils.id x
| `List_of_callables x -> Wrap_utils.id x
)); ("vectorized", Wrap_utils.Option.map vectorized Py.Bool.of_bool); ("args", args); ("fun", Some(fun_ )); ("t_span", Some(t_span )); ("y0", Some(y0 |> Np.Obj.to_pyobject))]) (match options with None -> [] | Some x -> x))
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> if Py.is_none py then None else Some (Wrap_utils.id py)) (Py.Tuple.get x 1)), ((fun py -> if Py.is_none py then None else Some (Wrap_utils.id py)) (Py.Tuple.get x 2)), ((fun py -> if Py.is_none py then None else Some (Wrap_utils.id py)) (Py.Tuple.get x 3)), (Py.Int.to_int (Py.Tuple.get x 4)), (Py.Int.to_int (Py.Tuple.get x 5)), (Py.Int.to_int (Py.Tuple.get x 6)), (Py.Int.to_int (Py.Tuple.get x 7)), (Py.String.to_string (Py.Tuple.get x 8)), (Py.Bool.to_bool (Py.Tuple.get x 9))))
                  let tplquad ?args ?epsabs ?epsrel ~func ~a ~b ~gfun ~hfun ~qfun ~rfun () =
                     Py.Module.get_function_with_keywords __wrap_namespace "tplquad"
                       [||]
                       (Wrap_utils.keyword_args [("args", args); ("epsabs", Wrap_utils.Option.map epsabs Py.Float.of_float); ("epsrel", Wrap_utils.Option.map epsrel Py.Float.of_float); ("func", Some(func )); ("a", Some(a )); ("b", Some(b )); ("gfun", Some(gfun |> (function
| `F x -> Py.Float.of_float x
| `Callable x -> Wrap_utils.id x
))); ("hfun", Some(hfun |> (function
| `F x -> Py.Float.of_float x
| `Callable x -> Wrap_utils.id x
))); ("qfun", Some(qfun |> (function
| `F x -> Py.Float.of_float x
| `Callable x -> Wrap_utils.id x
))); ("rfun", Some(rfun |> (function
| `F x -> Py.Float.of_float x
| `Callable x -> Wrap_utils.id x
)))])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))
                  let trapz ?x ?dx ?axis ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "trapz"
                       [||]
                       (Wrap_utils.keyword_args [("x", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("dx", Wrap_utils.Option.map dx (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> Py.Float.to_float
