let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.interpolate"

let get_py name = Py.Module.get __wrap_namespace name
module Akima1DInterpolator = struct
type tag = [`Akima1DInterpolator]
type t = [`Akima1DInterpolator | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?axis ~x ~y () =
   Py.Module.get_function_with_keywords __wrap_namespace "Akima1DInterpolator"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
let antiderivative ?nu self =
   Py.Module.get_function_with_keywords (to_pyobject self) "antiderivative"
     [||]
     (Wrap_utils.keyword_args [("nu", Wrap_utils.Option.map nu Py.Int.of_int)])

let construct_fast ?extrapolate ?axis ~c ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "construct_fast"
     [||]
     (Wrap_utils.keyword_args [("extrapolate", extrapolate); ("axis", axis); ("c", Some(c )); ("x", Some(x ))])

let derivative ?nu self =
   Py.Module.get_function_with_keywords (to_pyobject self) "derivative"
     [||]
     (Wrap_utils.keyword_args [("nu", Wrap_utils.Option.map nu Py.Int.of_int)])

let extend ?right ~c ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "extend"
     [||]
     (Wrap_utils.keyword_args [("right", right); ("c", Some(c )); ("x", Some(x ))])

let from_bernstein_basis ?extrapolate ~bp self =
   Py.Module.get_function_with_keywords (to_pyobject self) "from_bernstein_basis"
     [||]
     (Wrap_utils.keyword_args [("extrapolate", extrapolate); ("bp", Some(bp ))])

let from_spline ?extrapolate ~tck self =
   Py.Module.get_function_with_keywords (to_pyobject self) "from_spline"
     [||]
     (Wrap_utils.keyword_args [("extrapolate", extrapolate); ("tck", Some(tck ))])

                  let integrate ?extrapolate ~a ~b self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "integrate"
                       [||]
                       (Wrap_utils.keyword_args [("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
)); ("a", Some(a |> Py.Float.of_float)); ("b", Some(b |> Py.Float.of_float))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let roots ?discontinuity ?extrapolate self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "roots"
                       [||]
                       (Wrap_utils.keyword_args [("discontinuity", Wrap_utils.Option.map discontinuity Py.Bool.of_bool); ("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let solve ?y ?discontinuity ?extrapolate self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "solve"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Py.Float.of_float); ("discontinuity", Wrap_utils.Option.map discontinuity Py.Bool.of_bool); ("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module BPoly = struct
type tag = [`BPoly]
type t = [`BPoly | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?extrapolate ?axis ~c ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "BPoly"
     [||]
     (Wrap_utils.keyword_args [("extrapolate", Wrap_utils.Option.map extrapolate Py.Bool.of_bool); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("c", Some(c |> Np.Obj.to_pyobject)); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> of_pyobject
let antiderivative ?nu self =
   Py.Module.get_function_with_keywords (to_pyobject self) "antiderivative"
     [||]
     (Wrap_utils.keyword_args [("nu", Wrap_utils.Option.map nu Py.Int.of_int)])

let construct_fast ?extrapolate ?axis ~c ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "construct_fast"
     [||]
     (Wrap_utils.keyword_args [("extrapolate", extrapolate); ("axis", axis); ("c", Some(c )); ("x", Some(x ))])

let derivative ?nu self =
   Py.Module.get_function_with_keywords (to_pyobject self) "derivative"
     [||]
     (Wrap_utils.keyword_args [("nu", Wrap_utils.Option.map nu Py.Int.of_int)])

                  let extend ?right ~c ~x self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "extend"
                       [||]
                       (Wrap_utils.keyword_args [("right", right); ("c", Some(c |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Size_k_m_ x -> Wrap_utils.id x
))); ("x", Some(x |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Size x -> Wrap_utils.id x
)))])

                  let from_derivatives ?orders ?extrapolate ~xi ~yi self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "from_derivatives"
                       [||]
                       (Wrap_utils.keyword_args [("orders", Wrap_utils.Option.map orders (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
)); ("xi", Some(xi |> Np.Obj.to_pyobject)); ("yi", Some(yi |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `List_of_array_likes x -> Wrap_utils.id x
)))])

                  let from_power_basis ?extrapolate ~pp self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "from_power_basis"
                       [||]
                       (Wrap_utils.keyword_args [("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
)); ("pp", Some(pp ))])

                  let integrate ?extrapolate ~a ~b self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "integrate"
                       [||]
                       (Wrap_utils.keyword_args [("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
)); ("a", Some(a |> Py.Float.of_float)); ("b", Some(b |> Py.Float.of_float))])


let x_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "x" with
  | None -> failwith "attribute x not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let x self = match x_opt self with
  | None -> raise Not_found
  | Some x -> x

let c_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "c" with
  | None -> failwith "attribute c not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let c self = match c_opt self with
  | None -> raise Not_found
  | Some x -> x

let axis_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "axis" with
  | None -> failwith "attribute axis not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let axis self = match axis_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module BSpline = struct
type tag = [`BSpline]
type t = [`BSpline | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?extrapolate ?axis ~t ~c ~k () =
                     Py.Module.get_function_with_keywords __wrap_namespace "BSpline"
                       [||]
                       (Wrap_utils.keyword_args [("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("t", Some(t |> Np.Obj.to_pyobject)); ("c", Some(c |> Np.Obj.to_pyobject)); ("k", Some(k |> Py.Int.of_int))])
                       |> of_pyobject
let antiderivative ?nu self =
   Py.Module.get_function_with_keywords (to_pyobject self) "antiderivative"
     [||]
     (Wrap_utils.keyword_args [("nu", Wrap_utils.Option.map nu Py.Int.of_int)])

                  let basis_element ?extrapolate ~t self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "basis_element"
                       [||]
                       (Wrap_utils.keyword_args [("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
)); ("t", Some(t |> Np.Obj.to_pyobject))])

let construct_fast ?extrapolate ?axis ~t ~c ~k self =
   Py.Module.get_function_with_keywords (to_pyobject self) "construct_fast"
     [||]
     (Wrap_utils.keyword_args [("extrapolate", extrapolate); ("axis", axis); ("t", Some(t )); ("c", Some(c )); ("k", Some(k ))])

let derivative ?nu self =
   Py.Module.get_function_with_keywords (to_pyobject self) "derivative"
     [||]
     (Wrap_utils.keyword_args [("nu", Wrap_utils.Option.map nu Py.Int.of_int)])

                  let integrate ?extrapolate ~a ~b self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "integrate"
                       [||]
                       (Wrap_utils.keyword_args [("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
)); ("a", Some(a |> Py.Float.of_float)); ("b", Some(b |> Py.Float.of_float))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

let t_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "t" with
  | None -> failwith "attribute t not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let t self = match t_opt self with
  | None -> raise Not_found
  | Some x -> x

let c_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "c" with
  | None -> failwith "attribute c not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let c self = match c_opt self with
  | None -> raise Not_found
  | Some x -> x

let k_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "k" with
  | None -> failwith "attribute k not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let k self = match k_opt self with
  | None -> raise Not_found
  | Some x -> x

let extrapolate_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "extrapolate" with
  | None -> failwith "attribute extrapolate not found"
  | Some x -> if Py.is_none x then None else Some (Py.Bool.to_bool x)

let extrapolate self = match extrapolate_opt self with
  | None -> raise Not_found
  | Some x -> x

let axis_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "axis" with
  | None -> failwith "attribute axis not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let axis self = match axis_opt self with
  | None -> raise Not_found
  | Some x -> x

let tck_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "tck" with
  | None -> failwith "attribute tck not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let tck self = match tck_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module BarycentricInterpolator = struct
type tag = [`BarycentricInterpolator]
type t = [`BarycentricInterpolator | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?yi ?axis ~xi () =
   Py.Module.get_function_with_keywords __wrap_namespace "BarycentricInterpolator"
     [||]
     (Wrap_utils.keyword_args [("yi", Wrap_utils.Option.map yi Np.Obj.to_pyobject); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("xi", Some(xi |> Np.Obj.to_pyobject))])
     |> of_pyobject
let add_xi ?yi ~xi self =
   Py.Module.get_function_with_keywords (to_pyobject self) "add_xi"
     [||]
     (Wrap_utils.keyword_args [("yi", Wrap_utils.Option.map yi Np.Obj.to_pyobject); ("xi", Some(xi |> Np.Obj.to_pyobject))])

let set_yi ?axis ~yi self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_yi"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("yi", Some(yi |> Np.Obj.to_pyobject))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module BivariateSpline = struct
type tag = [`BivariateSpline]
type t = [`BivariateSpline | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "BivariateSpline"
     [||]
     []
     |> of_pyobject
let ev ?dx ?dy ~xi ~yi self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ev"
     [||]
     (Wrap_utils.keyword_args [("dx", Wrap_utils.Option.map dx Py.Int.of_int); ("dy", Wrap_utils.Option.map dy Py.Int.of_int); ("xi", Some(xi )); ("yi", Some(yi ))])

let get_coeffs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_coeffs"
     [||]
     []

let get_knots self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_knots"
     [||]
     []

let get_residual self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_residual"
     [||]
     []

let integral ~xa ~xb ~ya ~yb self =
   Py.Module.get_function_with_keywords (to_pyobject self) "integral"
     [||]
     (Wrap_utils.keyword_args [("xa", Some(xa )); ("xb", Some(xb )); ("ya", Some(ya )); ("yb", Some(yb ))])
     |> Py.Float.to_float
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module CloughTocher2DInterpolator = struct
type tag = [`CloughTocher2DInterpolator]
type t = [`CloughTocher2DInterpolator | `NDInterpolatorBase | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_nd_interpolator x = (x :> [`NDInterpolatorBase] Obj.t)
                  let create ?fill_value ?tol ?maxiter ?rescale ~points ~values () =
                     Py.Module.get_function_with_keywords __wrap_namespace "CloughTocher2DInterpolator"
                       [||]
                       (Wrap_utils.keyword_args [("fill_value", Wrap_utils.Option.map fill_value Py.Float.of_float); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("rescale", Wrap_utils.Option.map rescale Py.Bool.of_bool); ("points", Some(points |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Delaunay x -> Wrap_utils.id x
))); ("values", Some(values ))])
                       |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module CubicHermiteSpline = struct
type tag = [`CubicHermiteSpline]
type t = [`CubicHermiteSpline | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?axis ?extrapolate ~x ~y ~dydx () =
                     Py.Module.get_function_with_keywords __wrap_namespace "CubicHermiteSpline"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
)); ("x", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject)); ("dydx", Some(dydx |> Np.Obj.to_pyobject))])
                       |> of_pyobject
let antiderivative ?nu self =
   Py.Module.get_function_with_keywords (to_pyobject self) "antiderivative"
     [||]
     (Wrap_utils.keyword_args [("nu", Wrap_utils.Option.map nu Py.Int.of_int)])

let construct_fast ?extrapolate ?axis ~c ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "construct_fast"
     [||]
     (Wrap_utils.keyword_args [("extrapolate", extrapolate); ("axis", axis); ("c", Some(c )); ("x", Some(x ))])

let derivative ?nu self =
   Py.Module.get_function_with_keywords (to_pyobject self) "derivative"
     [||]
     (Wrap_utils.keyword_args [("nu", Wrap_utils.Option.map nu Py.Int.of_int)])

                  let extend ?right ~c ~x self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "extend"
                       [||]
                       (Wrap_utils.keyword_args [("right", right); ("c", Some(c |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Size_k_m_ x -> Wrap_utils.id x
))); ("x", Some(x |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Size x -> Wrap_utils.id x
)))])

                  let from_bernstein_basis ?extrapolate ~bp self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "from_bernstein_basis"
                       [||]
                       (Wrap_utils.keyword_args [("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
)); ("bp", Some(bp ))])

                  let from_spline ?extrapolate ~tck self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "from_spline"
                       [||]
                       (Wrap_utils.keyword_args [("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
)); ("tck", Some(tck ))])

                  let integrate ?extrapolate ~a ~b self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "integrate"
                       [||]
                       (Wrap_utils.keyword_args [("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
)); ("a", Some(a |> Py.Float.of_float)); ("b", Some(b |> Py.Float.of_float))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let roots ?discontinuity ?extrapolate self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "roots"
                       [||]
                       (Wrap_utils.keyword_args [("discontinuity", Wrap_utils.Option.map discontinuity Py.Bool.of_bool); ("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let solve ?y ?discontinuity ?extrapolate self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "solve"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Py.Float.of_float); ("discontinuity", Wrap_utils.Option.map discontinuity Py.Bool.of_bool); ("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

let x_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "x" with
  | None -> failwith "attribute x not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let x self = match x_opt self with
  | None -> raise Not_found
  | Some x -> x

let c_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "c" with
  | None -> failwith "attribute c not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let c self = match c_opt self with
  | None -> raise Not_found
  | Some x -> x

let axis_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "axis" with
  | None -> failwith "attribute axis not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let axis self = match axis_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module CubicSpline = struct
type tag = [`CubicSpline]
type t = [`CubicSpline | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?axis ?bc_type ?extrapolate ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "CubicSpline"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("bc_type", Wrap_utils.Option.map bc_type (function
| `S x -> Py.String.of_string x
| `T2_tuple x -> Wrap_utils.id x
)); ("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
)); ("x", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> of_pyobject
let antiderivative ?nu self =
   Py.Module.get_function_with_keywords (to_pyobject self) "antiderivative"
     [||]
     (Wrap_utils.keyword_args [("nu", Wrap_utils.Option.map nu Py.Int.of_int)])

let construct_fast ?extrapolate ?axis ~c ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "construct_fast"
     [||]
     (Wrap_utils.keyword_args [("extrapolate", extrapolate); ("axis", axis); ("c", Some(c )); ("x", Some(x ))])

let derivative ?nu self =
   Py.Module.get_function_with_keywords (to_pyobject self) "derivative"
     [||]
     (Wrap_utils.keyword_args [("nu", Wrap_utils.Option.map nu Py.Int.of_int)])

                  let extend ?right ~c ~x self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "extend"
                       [||]
                       (Wrap_utils.keyword_args [("right", right); ("c", Some(c |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Size_k_m_ x -> Wrap_utils.id x
))); ("x", Some(x |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Size x -> Wrap_utils.id x
)))])

                  let from_bernstein_basis ?extrapolate ~bp self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "from_bernstein_basis"
                       [||]
                       (Wrap_utils.keyword_args [("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
)); ("bp", Some(bp ))])

                  let from_spline ?extrapolate ~tck self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "from_spline"
                       [||]
                       (Wrap_utils.keyword_args [("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
)); ("tck", Some(tck ))])

                  let integrate ?extrapolate ~a ~b self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "integrate"
                       [||]
                       (Wrap_utils.keyword_args [("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
)); ("a", Some(a |> Py.Float.of_float)); ("b", Some(b |> Py.Float.of_float))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let roots ?discontinuity ?extrapolate self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "roots"
                       [||]
                       (Wrap_utils.keyword_args [("discontinuity", Wrap_utils.Option.map discontinuity Py.Bool.of_bool); ("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let solve ?y ?discontinuity ?extrapolate self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "solve"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Py.Float.of_float); ("discontinuity", Wrap_utils.Option.map discontinuity Py.Bool.of_bool); ("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

let x_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "x" with
  | None -> failwith "attribute x not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let x self = match x_opt self with
  | None -> raise Not_found
  | Some x -> x

let c_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "c" with
  | None -> failwith "attribute c not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let c self = match c_opt self with
  | None -> raise Not_found
  | Some x -> x

let axis_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "axis" with
  | None -> failwith "attribute axis not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let axis self = match axis_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module InterpolatedUnivariateSpline = struct
type tag = [`InterpolatedUnivariateSpline]
type t = [`InterpolatedUnivariateSpline | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?w ?bbox ?k ?ext ?check_finite ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "InterpolatedUnivariateSpline"
                       [||]
                       (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("bbox", bbox); ("k", Wrap_utils.Option.map k Py.Int.of_int); ("ext", Wrap_utils.Option.map ext (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> of_pyobject
let antiderivative ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "antiderivative"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int)])

let derivative ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "derivative"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int)])

let derivatives ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "derivatives"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let get_coeffs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_coeffs"
     [||]
     []

let get_knots self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_knots"
     [||]
     []

let get_residual self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_residual"
     [||]
     []

let integral ~a ~b self =
   Py.Module.get_function_with_keywords (to_pyobject self) "integral"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Py.Float.of_float)); ("b", Some(b |> Py.Float.of_float))])
     |> Py.Float.to_float
let roots self =
   Py.Module.get_function_with_keywords (to_pyobject self) "roots"
     [||]
     []

let set_smoothing_factor ~s self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_smoothing_factor"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module KroghInterpolator = struct
type tag = [`KroghInterpolator]
type t = [`KroghInterpolator | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?axis ~xi ~yi () =
                     Py.Module.get_function_with_keywords __wrap_namespace "KroghInterpolator"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("xi", Some(xi |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Length_N x -> Wrap_utils.id x
))); ("yi", Some(yi |> Np.Obj.to_pyobject))])
                       |> of_pyobject
let derivative ?der ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "derivative"
     [||]
     (Wrap_utils.keyword_args [("der", Wrap_utils.Option.map der Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let derivatives ?der ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "derivatives"
     [||]
     (Wrap_utils.keyword_args [("der", Wrap_utils.Option.map der Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LSQBivariateSpline = struct
type tag = [`LSQBivariateSpline]
type t = [`LSQBivariateSpline | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?w ?bbox ?kx ?ky ?eps ~x ~y ~z ~tx ~ty () =
   Py.Module.get_function_with_keywords __wrap_namespace "LSQBivariateSpline"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("bbox", bbox); ("kx", kx); ("ky", ky); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("x", Some(x )); ("y", Some(y )); ("z", Some(z )); ("tx", Some(tx )); ("ty", Some(ty ))])
     |> of_pyobject
let ev ?dx ?dy ~xi ~yi self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ev"
     [||]
     (Wrap_utils.keyword_args [("dx", Wrap_utils.Option.map dx Py.Int.of_int); ("dy", Wrap_utils.Option.map dy Py.Int.of_int); ("xi", Some(xi )); ("yi", Some(yi ))])

let get_coeffs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_coeffs"
     [||]
     []

let get_knots self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_knots"
     [||]
     []

let get_residual self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_residual"
     [||]
     []

let integral ~xa ~xb ~ya ~yb self =
   Py.Module.get_function_with_keywords (to_pyobject self) "integral"
     [||]
     (Wrap_utils.keyword_args [("xa", Some(xa )); ("xb", Some(xb )); ("ya", Some(ya )); ("yb", Some(yb ))])
     |> Py.Float.to_float
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LSQSphereBivariateSpline = struct
type tag = [`LSQSphereBivariateSpline]
type t = [`LSQSphereBivariateSpline | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?w ?eps ~theta ~phi ~r ~tt ~tp () =
   Py.Module.get_function_with_keywords __wrap_namespace "LSQSphereBivariateSpline"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("theta", Some(theta )); ("phi", Some(phi )); ("r", Some(r )); ("tt", Some(tt )); ("tp", Some(tp ))])
     |> of_pyobject
let ev ?dtheta ?dphi ~theta ~phi self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ev"
     [||]
     (Wrap_utils.keyword_args [("dtheta", Wrap_utils.Option.map dtheta Py.Int.of_int); ("dphi", Wrap_utils.Option.map dphi Py.Int.of_int); ("theta", Some(theta )); ("phi", Some(phi ))])

let get_coeffs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_coeffs"
     [||]
     []

let get_knots self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_knots"
     [||]
     []

let get_residual self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_residual"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LSQUnivariateSpline = struct
type tag = [`LSQUnivariateSpline]
type t = [`LSQUnivariateSpline | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?w ?bbox ?k ?ext ?check_finite ~x ~y ~t () =
                     Py.Module.get_function_with_keywords __wrap_namespace "LSQUnivariateSpline"
                       [||]
                       (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("bbox", bbox); ("k", Wrap_utils.Option.map k Py.Int.of_int); ("ext", Wrap_utils.Option.map ext (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject)); ("t", Some(t |> Np.Obj.to_pyobject))])
                       |> of_pyobject
let antiderivative ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "antiderivative"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int)])

let derivative ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "derivative"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int)])

let derivatives ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "derivatives"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let get_coeffs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_coeffs"
     [||]
     []

let get_knots self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_knots"
     [||]
     []

let get_residual self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_residual"
     [||]
     []

let integral ~a ~b self =
   Py.Module.get_function_with_keywords (to_pyobject self) "integral"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Py.Float.of_float)); ("b", Some(b |> Py.Float.of_float))])
     |> Py.Float.to_float
let roots self =
   Py.Module.get_function_with_keywords (to_pyobject self) "roots"
     [||]
     []

let set_smoothing_factor ~s self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_smoothing_factor"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LinearNDInterpolator = struct
type tag = [`LinearNDInterpolator]
type t = [`LinearNDInterpolator | `NDInterpolatorBase | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_nd_interpolator x = (x :> [`NDInterpolatorBase] Obj.t)
                  let create ?fill_value ?rescale ~points ~values () =
                     Py.Module.get_function_with_keywords __wrap_namespace "LinearNDInterpolator"
                       [||]
                       (Wrap_utils.keyword_args [("fill_value", Wrap_utils.Option.map fill_value Py.Float.of_float); ("rescale", Wrap_utils.Option.map rescale Py.Bool.of_bool); ("points", Some(points |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Delaunay x -> Wrap_utils.id x
))); ("values", Some(values ))])
                       |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module NdPPoly = struct
type tag = [`NdPPoly]
type t = [`NdPPoly | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?extrapolate ~c ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "NdPPoly"
     [||]
     (Wrap_utils.keyword_args [("extrapolate", Wrap_utils.Option.map extrapolate Py.Bool.of_bool); ("c", Some(c |> Np.Obj.to_pyobject)); ("x", Some(x ))])
     |> of_pyobject
let antiderivative ~nu self =
   Py.Module.get_function_with_keywords (to_pyobject self) "antiderivative"
     [||]
     (Wrap_utils.keyword_args [("nu", Some(nu ))])

let construct_fast ?extrapolate ~c ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "construct_fast"
     [||]
     (Wrap_utils.keyword_args [("extrapolate", extrapolate); ("c", Some(c )); ("x", Some(x ))])

let derivative ~nu self =
   Py.Module.get_function_with_keywords (to_pyobject self) "derivative"
     [||]
     (Wrap_utils.keyword_args [("nu", Some(nu ))])

let integrate ?extrapolate ~ranges self =
   Py.Module.get_function_with_keywords (to_pyobject self) "integrate"
     [||]
     (Wrap_utils.keyword_args [("extrapolate", Wrap_utils.Option.map extrapolate Py.Bool.of_bool); ("ranges", Some(ranges ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let integrate_1d ?extrapolate ~a ~b ~axis self =
   Py.Module.get_function_with_keywords (to_pyobject self) "integrate_1d"
     [||]
     (Wrap_utils.keyword_args [("extrapolate", Wrap_utils.Option.map extrapolate Py.Bool.of_bool); ("a", Some(a )); ("b", Some(b )); ("axis", Some(axis |> Py.Int.of_int))])


let x_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "x" with
  | None -> failwith "attribute x not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let x self = match x_opt self with
  | None -> raise Not_found
  | Some x -> x

let c_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "c" with
  | None -> failwith "attribute c not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let c self = match c_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module NearestNDInterpolator = struct
type tag = [`NearestNDInterpolator]
type t = [`NDInterpolatorBase | `NearestNDInterpolator | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_nd_interpolator x = (x :> [`NDInterpolatorBase] Obj.t)
let create ?rescale ?tree_options ~x ~y () =
   Py.Module.get_function_with_keywords __wrap_namespace "NearestNDInterpolator"
     [||]
     (Wrap_utils.keyword_args [("rescale", Wrap_utils.Option.map rescale Py.Bool.of_bool); ("tree_options", tree_options); ("x", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y ))])
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module PPoly = struct
type tag = [`PPoly]
type t = [`Object | `PPoly] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?extrapolate ?axis ~c ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "PPoly"
                       [||]
                       (Wrap_utils.keyword_args [("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("c", Some(c |> Np.Obj.to_pyobject)); ("x", Some(x |> Np.Obj.to_pyobject))])
                       |> of_pyobject
let antiderivative ?nu self =
   Py.Module.get_function_with_keywords (to_pyobject self) "antiderivative"
     [||]
     (Wrap_utils.keyword_args [("nu", Wrap_utils.Option.map nu Py.Int.of_int)])

let construct_fast ?extrapolate ?axis ~c ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "construct_fast"
     [||]
     (Wrap_utils.keyword_args [("extrapolate", extrapolate); ("axis", axis); ("c", Some(c )); ("x", Some(x ))])

let derivative ?nu self =
   Py.Module.get_function_with_keywords (to_pyobject self) "derivative"
     [||]
     (Wrap_utils.keyword_args [("nu", Wrap_utils.Option.map nu Py.Int.of_int)])

                  let extend ?right ~c ~x self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "extend"
                       [||]
                       (Wrap_utils.keyword_args [("right", right); ("c", Some(c |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Size_k_m_ x -> Wrap_utils.id x
))); ("x", Some(x |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Size x -> Wrap_utils.id x
)))])

                  let from_bernstein_basis ?extrapolate ~bp self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "from_bernstein_basis"
                       [||]
                       (Wrap_utils.keyword_args [("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
)); ("bp", Some(bp ))])

                  let from_spline ?extrapolate ~tck self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "from_spline"
                       [||]
                       (Wrap_utils.keyword_args [("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
)); ("tck", Some(tck ))])

                  let integrate ?extrapolate ~a ~b self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "integrate"
                       [||]
                       (Wrap_utils.keyword_args [("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
)); ("a", Some(a |> Py.Float.of_float)); ("b", Some(b |> Py.Float.of_float))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let roots ?discontinuity ?extrapolate self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "roots"
                       [||]
                       (Wrap_utils.keyword_args [("discontinuity", Wrap_utils.Option.map discontinuity Py.Bool.of_bool); ("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let solve ?y ?discontinuity ?extrapolate self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "solve"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Py.Float.of_float); ("discontinuity", Wrap_utils.Option.map discontinuity Py.Bool.of_bool); ("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

let x_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "x" with
  | None -> failwith "attribute x not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let x self = match x_opt self with
  | None -> raise Not_found
  | Some x -> x

let c_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "c" with
  | None -> failwith "attribute c not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let c self = match c_opt self with
  | None -> raise Not_found
  | Some x -> x

let axis_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "axis" with
  | None -> failwith "attribute axis not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let axis self = match axis_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module PchipInterpolator = struct
type tag = [`PchipInterpolator]
type t = [`Object | `PchipInterpolator] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?axis ?extrapolate ~x ~y () =
   Py.Module.get_function_with_keywords __wrap_namespace "PchipInterpolator"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("extrapolate", Wrap_utils.Option.map extrapolate Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
let antiderivative ?nu self =
   Py.Module.get_function_with_keywords (to_pyobject self) "antiderivative"
     [||]
     (Wrap_utils.keyword_args [("nu", Wrap_utils.Option.map nu Py.Int.of_int)])

let construct_fast ?extrapolate ?axis ~c ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "construct_fast"
     [||]
     (Wrap_utils.keyword_args [("extrapolate", extrapolate); ("axis", axis); ("c", Some(c )); ("x", Some(x ))])

let derivative ?nu self =
   Py.Module.get_function_with_keywords (to_pyobject self) "derivative"
     [||]
     (Wrap_utils.keyword_args [("nu", Wrap_utils.Option.map nu Py.Int.of_int)])

                  let extend ?right ~c ~x self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "extend"
                       [||]
                       (Wrap_utils.keyword_args [("right", right); ("c", Some(c |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Size_k_m_ x -> Wrap_utils.id x
))); ("x", Some(x |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Size x -> Wrap_utils.id x
)))])

                  let from_bernstein_basis ?extrapolate ~bp self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "from_bernstein_basis"
                       [||]
                       (Wrap_utils.keyword_args [("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
)); ("bp", Some(bp ))])

                  let from_spline ?extrapolate ~tck self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "from_spline"
                       [||]
                       (Wrap_utils.keyword_args [("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
)); ("tck", Some(tck ))])

                  let integrate ?extrapolate ~a ~b self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "integrate"
                       [||]
                       (Wrap_utils.keyword_args [("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
)); ("a", Some(a |> Py.Float.of_float)); ("b", Some(b |> Py.Float.of_float))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let roots ?discontinuity ?extrapolate self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "roots"
                       [||]
                       (Wrap_utils.keyword_args [("discontinuity", Wrap_utils.Option.map discontinuity Py.Bool.of_bool); ("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let solve ?y ?discontinuity ?extrapolate self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "solve"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Py.Float.of_float); ("discontinuity", Wrap_utils.Option.map discontinuity Py.Bool.of_bool); ("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Rbf = struct
type tag = [`Rbf]
type t = [`Object | `Rbf] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "Rbf"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject

let n_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "N" with
  | None -> failwith "attribute N not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n self = match n_opt self with
  | None -> raise Not_found
  | Some x -> x

let di_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "di" with
  | None -> failwith "attribute di not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let di self = match di_opt self with
  | None -> raise Not_found
  | Some x -> x

let xi_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "xi" with
  | None -> failwith "attribute xi not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let xi self = match xi_opt self with
  | None -> raise Not_found
  | Some x -> x

let function_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "function" with
  | None -> failwith "attribute function not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let function_ self = match function_opt self with
  | None -> raise Not_found
  | Some x -> x

let epsilon_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "epsilon" with
  | None -> failwith "attribute epsilon not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let epsilon self = match epsilon_opt self with
  | None -> raise Not_found
  | Some x -> x

let smooth_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "smooth" with
  | None -> failwith "attribute smooth not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let smooth self = match smooth_opt self with
  | None -> raise Not_found
  | Some x -> x

let norm_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "norm" with
  | None -> failwith "attribute norm not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let norm self = match norm_opt self with
  | None -> raise Not_found
  | Some x -> x

let mode_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "mode" with
  | None -> failwith "attribute mode not found"
  | Some x -> if Py.is_none x then None else Some (Py.String.to_string x)

let mode self = match mode_opt self with
  | None -> raise Not_found
  | Some x -> x

let nodes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nodes" with
  | None -> failwith "attribute nodes not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let nodes self = match nodes_opt self with
  | None -> raise Not_found
  | Some x -> x

let a_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "A" with
  | None -> failwith "attribute A not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let a self = match a_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RectBivariateSpline = struct
type tag = [`RectBivariateSpline]
type t = [`Object | `RectBivariateSpline] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?bbox ?kx ?ky ?s ~x ~y ~z () =
   Py.Module.get_function_with_keywords __wrap_namespace "RectBivariateSpline"
     [||]
     (Wrap_utils.keyword_args [("bbox", Wrap_utils.Option.map bbox Np.Obj.to_pyobject); ("kx", kx); ("ky", ky); ("s", Wrap_utils.Option.map s Py.Float.of_float); ("x", Some(x )); ("y", Some(y )); ("z", Some(z |> Np.Obj.to_pyobject))])
     |> of_pyobject
let ev ?dx ?dy ~xi ~yi self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ev"
     [||]
     (Wrap_utils.keyword_args [("dx", Wrap_utils.Option.map dx Py.Int.of_int); ("dy", Wrap_utils.Option.map dy Py.Int.of_int); ("xi", Some(xi )); ("yi", Some(yi ))])

let get_coeffs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_coeffs"
     [||]
     []

let get_knots self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_knots"
     [||]
     []

let get_residual self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_residual"
     [||]
     []

let integral ~xa ~xb ~ya ~yb self =
   Py.Module.get_function_with_keywords (to_pyobject self) "integral"
     [||]
     (Wrap_utils.keyword_args [("xa", Some(xa )); ("xb", Some(xb )); ("ya", Some(ya )); ("yb", Some(yb ))])
     |> Py.Float.to_float
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RectSphereBivariateSpline = struct
type tag = [`RectSphereBivariateSpline]
type t = [`Object | `RectSphereBivariateSpline] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?s ?pole_continuity ?pole_values ?pole_exact ?pole_flat ~u ~v ~r () =
                     Py.Module.get_function_with_keywords __wrap_namespace "RectSphereBivariateSpline"
                       [||]
                       (Wrap_utils.keyword_args [("s", Wrap_utils.Option.map s Py.Float.of_float); ("pole_continuity", Wrap_utils.Option.map pole_continuity (function
| `T_bool_bool_ x -> Wrap_utils.id x
| `Bool x -> Py.Bool.of_bool x
)); ("pole_values", Wrap_utils.Option.map pole_values (function
| `F x -> Py.Float.of_float x
| `Tuple x -> (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.Float.of_float ml_0); (Py.Float.of_float ml_1)]) x
)); ("pole_exact", Wrap_utils.Option.map pole_exact (function
| `T_bool_bool_ x -> Wrap_utils.id x
| `Bool x -> Py.Bool.of_bool x
)); ("pole_flat", Wrap_utils.Option.map pole_flat (function
| `T_bool_bool_ x -> Wrap_utils.id x
| `Bool x -> Py.Bool.of_bool x
)); ("u", Some(u |> Np.Obj.to_pyobject)); ("v", Some(v |> Np.Obj.to_pyobject)); ("r", Some(r |> Np.Obj.to_pyobject))])
                       |> of_pyobject
let ev ?dtheta ?dphi ~theta ~phi self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ev"
     [||]
     (Wrap_utils.keyword_args [("dtheta", Wrap_utils.Option.map dtheta Py.Int.of_int); ("dphi", Wrap_utils.Option.map dphi Py.Int.of_int); ("theta", Some(theta )); ("phi", Some(phi ))])

let get_coeffs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_coeffs"
     [||]
     []

let get_knots self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_knots"
     [||]
     []

let get_residual self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_residual"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RegularGridInterpolator = struct
type tag = [`RegularGridInterpolator]
type t = [`Object | `RegularGridInterpolator] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?method_ ?bounds_error ?fill_value ~points ~values () =
                     Py.Module.get_function_with_keywords __wrap_namespace "RegularGridInterpolator"
                       [||]
                       (Wrap_utils.keyword_args [("method", Wrap_utils.Option.map method_ Py.String.of_string); ("bounds_error", Wrap_utils.Option.map bounds_error Py.Bool.of_bool); ("fill_value", Wrap_utils.Option.map fill_value (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("points", Some(points )); ("values", Some(values |> Np.Obj.to_pyobject))])
                       |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SmoothBivariateSpline = struct
type tag = [`SmoothBivariateSpline]
type t = [`Object | `SmoothBivariateSpline] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?w ?bbox ?kx ?ky ?s ?eps ~x ~y ~z () =
   Py.Module.get_function_with_keywords __wrap_namespace "SmoothBivariateSpline"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("bbox", Wrap_utils.Option.map bbox Np.Obj.to_pyobject); ("kx", kx); ("ky", ky); ("s", Wrap_utils.Option.map s Py.Float.of_float); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("x", Some(x )); ("y", Some(y )); ("z", Some(z ))])
     |> of_pyobject
let ev ?dx ?dy ~xi ~yi self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ev"
     [||]
     (Wrap_utils.keyword_args [("dx", Wrap_utils.Option.map dx Py.Int.of_int); ("dy", Wrap_utils.Option.map dy Py.Int.of_int); ("xi", Some(xi )); ("yi", Some(yi ))])

let get_coeffs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_coeffs"
     [||]
     []

let get_knots self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_knots"
     [||]
     []

let get_residual self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_residual"
     [||]
     []

let integral ~xa ~xb ~ya ~yb self =
   Py.Module.get_function_with_keywords (to_pyobject self) "integral"
     [||]
     (Wrap_utils.keyword_args [("xa", Some(xa )); ("xb", Some(xb )); ("ya", Some(ya )); ("yb", Some(yb ))])
     |> Py.Float.to_float
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SmoothSphereBivariateSpline = struct
type tag = [`SmoothSphereBivariateSpline]
type t = [`Object | `SmoothSphereBivariateSpline] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?w ?s ?eps ~theta ~phi ~r () =
   Py.Module.get_function_with_keywords __wrap_namespace "SmoothSphereBivariateSpline"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("s", Wrap_utils.Option.map s Py.Float.of_float); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("theta", Some(theta )); ("phi", Some(phi )); ("r", Some(r ))])
     |> of_pyobject
let ev ?dtheta ?dphi ~theta ~phi self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ev"
     [||]
     (Wrap_utils.keyword_args [("dtheta", Wrap_utils.Option.map dtheta Py.Int.of_int); ("dphi", Wrap_utils.Option.map dphi Py.Int.of_int); ("theta", Some(theta )); ("phi", Some(phi ))])

let get_coeffs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_coeffs"
     [||]
     []

let get_knots self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_knots"
     [||]
     []

let get_residual self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_residual"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module UnivariateSpline = struct
type tag = [`UnivariateSpline]
type t = [`Object | `UnivariateSpline] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?w ?bbox ?k ?s ?ext ?check_finite ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "UnivariateSpline"
                       [||]
                       (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("bbox", bbox); ("k", Wrap_utils.Option.map k Py.Int.of_int); ("s", Wrap_utils.Option.map s Py.Float.of_float); ("ext", Wrap_utils.Option.map ext (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> of_pyobject
let antiderivative ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "antiderivative"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int)])

let derivative ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "derivative"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int)])

let derivatives ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "derivatives"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let get_coeffs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_coeffs"
     [||]
     []

let get_knots self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_knots"
     [||]
     []

let get_residual self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_residual"
     [||]
     []

let integral ~a ~b self =
   Py.Module.get_function_with_keywords (to_pyobject self) "integral"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Py.Float.of_float)); ("b", Some(b |> Py.Float.of_float))])
     |> Py.Float.to_float
let roots self =
   Py.Module.get_function_with_keywords (to_pyobject self) "roots"
     [||]
     []

let set_smoothing_factor ~s self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_smoothing_factor"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Interp1d = struct
type tag = [`Interp1d]
type t = [`Interp1d | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?kind ?axis ?copy ?bounds_error ?fill_value ?assume_sorted ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "interp1d"
                       [||]
                       (Wrap_utils.keyword_args [("kind", Wrap_utils.Option.map kind (function
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("bounds_error", Wrap_utils.Option.map bounds_error Py.Bool.of_bool); ("fill_value", Wrap_utils.Option.map fill_value (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Extrapolate -> Py.String.of_string "extrapolate"
| `T_array_like_array_like_ x -> Wrap_utils.id x
)); ("assume_sorted", Wrap_utils.Option.map assume_sorted Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y ))])
                       |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Interp2d = struct
type tag = [`Interp2d]
type t = [`Interp2d | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?kind ?copy ?bounds_error ?fill_value ~x ~y ~z () =
                     Py.Module.get_function_with_keywords __wrap_namespace "interp2d"
                       [||]
                       (Wrap_utils.keyword_args [("kind", Wrap_utils.Option.map kind (function
| `Linear -> Py.String.of_string "linear"
| `Cubic -> Py.String.of_string "cubic"
| `Quintic -> Py.String.of_string "quintic"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("bounds_error", Wrap_utils.Option.map bounds_error Py.Bool.of_bool); ("fill_value", Wrap_utils.Option.map fill_value (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("x", Some(x )); ("y", Some(y )); ("z", Some(z |> Np.Obj.to_pyobject))])
                       |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Pchip = struct
type tag = [`PchipInterpolator]
type t = [`Object | `PchipInterpolator] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?axis ?extrapolate ~x ~y () =
   Py.Module.get_function_with_keywords __wrap_namespace "pchip"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("extrapolate", Wrap_utils.Option.map extrapolate Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
let antiderivative ?nu self =
   Py.Module.get_function_with_keywords (to_pyobject self) "antiderivative"
     [||]
     (Wrap_utils.keyword_args [("nu", Wrap_utils.Option.map nu Py.Int.of_int)])

let construct_fast ?extrapolate ?axis ~c ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "construct_fast"
     [||]
     (Wrap_utils.keyword_args [("extrapolate", extrapolate); ("axis", axis); ("c", Some(c )); ("x", Some(x ))])

let derivative ?nu self =
   Py.Module.get_function_with_keywords (to_pyobject self) "derivative"
     [||]
     (Wrap_utils.keyword_args [("nu", Wrap_utils.Option.map nu Py.Int.of_int)])

                  let extend ?right ~c ~x self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "extend"
                       [||]
                       (Wrap_utils.keyword_args [("right", right); ("c", Some(c |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Size_k_m_ x -> Wrap_utils.id x
))); ("x", Some(x |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Size x -> Wrap_utils.id x
)))])

                  let from_bernstein_basis ?extrapolate ~bp self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "from_bernstein_basis"
                       [||]
                       (Wrap_utils.keyword_args [("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
)); ("bp", Some(bp ))])

                  let from_spline ?extrapolate ~tck self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "from_spline"
                       [||]
                       (Wrap_utils.keyword_args [("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
)); ("tck", Some(tck ))])

                  let integrate ?extrapolate ~a ~b self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "integrate"
                       [||]
                       (Wrap_utils.keyword_args [("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
)); ("a", Some(a |> Py.Float.of_float)); ("b", Some(b |> Py.Float.of_float))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let roots ?discontinuity ?extrapolate self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "roots"
                       [||]
                       (Wrap_utils.keyword_args [("discontinuity", Wrap_utils.Option.map discontinuity Py.Bool.of_bool); ("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let solve ?y ?discontinuity ?extrapolate self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "solve"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Py.Float.of_float); ("discontinuity", Wrap_utils.Option.map discontinuity Py.Bool.of_bool); ("extrapolate", Wrap_utils.Option.map extrapolate (function
| `Bool x -> Py.Bool.of_bool x
| `Periodic -> Py.String.of_string "periodic"
))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Dfitpack = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.interpolate.dfitpack"

let get_py name = Py.Module.get __wrap_namespace name
let splev ?e ~t ~c ~k ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "splev"
     [||]
     (Wrap_utils.keyword_args [("e", e); ("t", Some(t )); ("c", Some(c )); ("k", Some(k )); ("x", Some(x ))])


end
module Fitpack = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.interpolate.fitpack"

let get_py name = Py.Module.get __wrap_namespace name
let bisplev ?dx ?dy ~x ~y ~tck () =
   Py.Module.get_function_with_keywords __wrap_namespace "bisplev"
     [||]
     (Wrap_utils.keyword_args [("dx", dx); ("dy", dy); ("x", Some(x )); ("y", Some(y )); ("tck", Some(tck ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let bisplrep ?w ?xb ?xe ?yb ?ye ?kx ?ky ?task ?s ?eps ?tx ?ty ?full_output ?nxest ?nyest ?quiet ~x ~y ~z () =
   Py.Module.get_function_with_keywords __wrap_namespace "bisplrep"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("xb", xb); ("xe", xe); ("yb", yb); ("ye", ye); ("kx", kx); ("ky", ky); ("task", Wrap_utils.Option.map task Py.Int.of_int); ("s", Wrap_utils.Option.map s Py.Float.of_float); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("tx", tx); ("ty", ty); ("full_output", Wrap_utils.Option.map full_output Py.Int.of_int); ("nxest", nxest); ("nyest", nyest); ("quiet", Wrap_utils.Option.map quiet Py.Int.of_int); ("x", Some(x )); ("y", Some(y )); ("z", Some(z ))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Py.String.to_string (Py.Tuple.get x 3))))
let dblint ~xa ~xb ~ya ~yb ~tck () =
   Py.Module.get_function_with_keywords __wrap_namespace "dblint"
     [||]
     (Wrap_utils.keyword_args [("xa", Some(xa )); ("xb", Some(xb )); ("ya", Some(ya )); ("yb", Some(yb )); ("tck", Some(tck ))])
     |> Py.Float.to_float
let insert ?m ?per ~x ~tck () =
   Py.Module.get_function_with_keywords __wrap_namespace "insert"
     [||]
     (Wrap_utils.keyword_args [("m", Wrap_utils.Option.map m Py.Int.of_int); ("per", Wrap_utils.Option.map per Py.Int.of_int); ("x", Some(x )); ("tck", Some(tck ))])

let spalde ~x ~tck () =
   Py.Module.get_function_with_keywords __wrap_namespace "spalde"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject)); ("tck", Some(tck ))])

let splantider ?n ~tck () =
   Py.Module.get_function_with_keywords __wrap_namespace "splantider"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("tck", Some(tck ))])

let splder ?n ~tck () =
   Py.Module.get_function_with_keywords __wrap_namespace "splder"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("tck", Some(tck ))])

let splev ?der ?ext ~x ~tck () =
   Py.Module.get_function_with_keywords __wrap_namespace "splev"
     [||]
     (Wrap_utils.keyword_args [("der", Wrap_utils.Option.map der Py.Int.of_int); ("ext", Wrap_utils.Option.map ext Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject)); ("tck", Some(tck ))])

let splint ?full_output ~a ~b ~tck () =
   Py.Module.get_function_with_keywords __wrap_namespace "splint"
     [||]
     (Wrap_utils.keyword_args [("full_output", Wrap_utils.Option.map full_output Py.Int.of_int); ("a", Some(a )); ("b", Some(b )); ("tck", Some(tck ))])
     |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let splprep ?w ?u ?ub ?ue ?k ?task ?s ?t ?full_output ?nest ?per ?quiet ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "splprep"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("u", Wrap_utils.Option.map u Np.Obj.to_pyobject); ("ub", ub); ("ue", ue); ("k", Wrap_utils.Option.map k Py.Int.of_int); ("task", Wrap_utils.Option.map task Py.Int.of_int); ("s", Wrap_utils.Option.map s Py.Float.of_float); ("t", Wrap_utils.Option.map t Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Int.of_int); ("nest", Wrap_utils.Option.map nest Py.Int.of_int); ("per", Wrap_utils.Option.map per Py.Int.of_int); ("quiet", Wrap_utils.Option.map quiet Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2)), (Py.Int.to_int (Py.Tuple.get x 3)), (Py.String.to_string (Py.Tuple.get x 4))))
                  let splrep ?w ?xb ?xe ?k ?task ?s ?t ?full_output ?per ?quiet ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "splrep"
                       [||]
                       (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("xb", xb); ("xe", xe); ("k", Wrap_utils.Option.map k Py.Int.of_int); ("task", Wrap_utils.Option.map task (function
| `One -> Py.Int.of_int 1
| `T_1 x -> Wrap_utils.id x
| `Zero -> Py.Int.of_int 0
)); ("s", Wrap_utils.Option.map s Py.Float.of_float); ("t", Wrap_utils.Option.map t Np.Obj.to_pyobject); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("per", Wrap_utils.Option.map per Py.Bool.of_bool); ("quiet", Wrap_utils.Option.map quiet Py.Bool.of_bool); ("x", Some(x )); ("y", Some(y ))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Py.String.to_string (Py.Tuple.get x 3))))
let sproot ?mest ~tck () =
   Py.Module.get_function_with_keywords __wrap_namespace "sproot"
     [||]
     (Wrap_utils.keyword_args [("mest", Wrap_utils.Option.map mest Py.Int.of_int); ("tck", Some(tck ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
module Fitpack2 = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.interpolate.fitpack2"

let get_py name = Py.Module.get __wrap_namespace name
module SphereBivariateSpline = struct
type tag = [`SphereBivariateSpline]
type t = [`Object | `SphereBivariateSpline] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "SphereBivariateSpline"
     [||]
     []
     |> of_pyobject
let ev ?dtheta ?dphi ~theta ~phi self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ev"
     [||]
     (Wrap_utils.keyword_args [("dtheta", Wrap_utils.Option.map dtheta Py.Int.of_int); ("dphi", Wrap_utils.Option.map dphi Py.Int.of_int); ("theta", Some(theta )); ("phi", Some(phi ))])

let get_coeffs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_coeffs"
     [||]
     []

let get_knots self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_knots"
     [||]
     []

let get_residual self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_residual"
     [||]
     []

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
let concatenate ?axis ?out ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "concatenate"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("a", Some(a ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let diff ?n ?axis ?prepend ?append ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "diff"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("prepend", prepend); ("append", append); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let ones ?dtype ?order ~shape () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ones"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let ravel ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ravel"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
| `A -> Py.String.of_string "A"
| `K -> Py.String.of_string "K"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
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
module Interpnd = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.interpolate.interpnd"

let get_py name = Py.Module.get __wrap_namespace name
module GradientEstimationWarning = struct
type tag = [`GradientEstimationWarning]
type t = [`BaseException | `GradientEstimationWarning | `Object] Obj.t
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
module NDInterpolatorBase = struct
type tag = [`NDInterpolatorBase]
type t = [`NDInterpolatorBase | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?fill_value ?ndim ?rescale ?need_contiguous ?need_values ~points ~values () =
   Py.Module.get_function_with_keywords __wrap_namespace "NDInterpolatorBase"
     [||]
     (Wrap_utils.keyword_args [("fill_value", fill_value); ("ndim", ndim); ("rescale", rescale); ("need_contiguous", need_contiguous); ("need_values", need_values); ("points", Some(points )); ("values", Some(values ))])
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end

end
module Interpolate = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.interpolate.interpolate"

let get_py name = Py.Module.get __wrap_namespace name
module Intp = struct
type tag = [`Int64]
type t = [`Int64 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
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
module Poly1d = struct
type tag = [`Poly1d]
type t = [`Object | `Poly1d] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?r ?variable ~c_or_r () =
   Py.Module.get_function_with_keywords __wrap_namespace "poly1d"
     [||]
     (Wrap_utils.keyword_args [("r", Wrap_utils.Option.map r Py.Bool.of_bool); ("variable", Wrap_utils.Option.map variable Py.String.of_string); ("c_or_r", Some(c_or_r |> Np.Obj.to_pyobject))])
     |> of_pyobject
let __getitem__ ~val_ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("val", Some(val_ ))])

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let __setitem__ ~key ~val_ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key )); ("val", Some(val_ ))])

let deriv ?m self =
   Py.Module.get_function_with_keywords (to_pyobject self) "deriv"
     [||]
     (Wrap_utils.keyword_args [("m", m)])

let integ ?m ?k self =
   Py.Module.get_function_with_keywords (to_pyobject self) "integ"
     [||]
     (Wrap_utils.keyword_args [("m", m); ("k", k)])

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
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let atleast_1d arys =
   Py.Module.get_function_with_keywords __wrap_namespace "atleast_1d"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arys)])
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let atleast_2d arys =
   Py.Module.get_function_with_keywords __wrap_namespace "atleast_2d"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arys)])
     []

                  let comb ?exact ?repetition ~n ~k () =
                     Py.Module.get_function_with_keywords __wrap_namespace "comb"
                       [||]
                       (Wrap_utils.keyword_args [("exact", Wrap_utils.Option.map exact Py.Bool.of_bool); ("repetition", Wrap_utils.Option.map repetition Py.Bool.of_bool); ("N", Some(n |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
))); ("k", Some(k |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)))])

                  let interpn ?method_ ?bounds_error ?fill_value ~points ~values ~xi () =
                     Py.Module.get_function_with_keywords __wrap_namespace "interpn"
                       [||]
                       (Wrap_utils.keyword_args [("method", Wrap_utils.Option.map method_ Py.String.of_string); ("bounds_error", Wrap_utils.Option.map bounds_error Py.Bool.of_bool); ("fill_value", Wrap_utils.Option.map fill_value (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("points", Some(points )); ("values", Some(values |> Np.Obj.to_pyobject)); ("xi", Some(xi |> Np.Obj.to_pyobject))])

let lagrange ~x ~w () =
   Py.Module.get_function_with_keywords __wrap_namespace "lagrange"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject)); ("w", Some(w |> Np.Obj.to_pyobject))])

let make_interp_spline ?k ?t ?bc_type ?axis ?check_finite ~x ~y () =
   Py.Module.get_function_with_keywords __wrap_namespace "make_interp_spline"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("t", Wrap_utils.Option.map t Np.Obj.to_pyobject); ("bc_type", bc_type); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])

let prod x =
   Py.Module.get_function_with_keywords __wrap_namespace "prod"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

                  let ravel ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ravel"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
| `A -> Py.String.of_string "A"
| `K -> Py.String.of_string "K"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let searchsorted ?side ?sorter ~a ~v () =
                     Py.Module.get_function_with_keywords __wrap_namespace "searchsorted"
                       [||]
                       (Wrap_utils.keyword_args [("side", Wrap_utils.Option.map side (function
| `Left -> Py.String.of_string "left"
| `Right -> Py.String.of_string "right"
)); ("sorter", sorter); ("a", Some(a )); ("v", Some(v |> Np.Obj.to_pyobject))])

let transpose ?axes ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "transpose"
     [||]
     (Wrap_utils.keyword_args [("axes", axes); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
module Ndgriddata = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.interpolate.ndgriddata"

let get_py name = Py.Module.get __wrap_namespace name
module CKDTree = struct
type tag = [`CKDTree]
type t = [`CKDTree | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?leafsize ?compact_nodes ?copy_data ?balanced_tree ?boxsize ~data () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cKDTree"
                       [||]
                       (Wrap_utils.keyword_args [("leafsize", leafsize); ("compact_nodes", Wrap_utils.Option.map compact_nodes Py.Bool.of_bool); ("copy_data", Wrap_utils.Option.map copy_data Py.Bool.of_bool); ("balanced_tree", Wrap_utils.Option.map balanced_tree Py.Bool.of_bool); ("boxsize", Wrap_utils.Option.map boxsize (function
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
)); ("data", Some(data |> Np.Obj.to_pyobject))])
                       |> of_pyobject
                  let count_neighbors ?p ?weights ?cumulative ~other ~r self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "count_neighbors"
                       [||]
                       (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p Py.Float.of_float); ("weights", Wrap_utils.Option.map weights (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple x -> Wrap_utils.id x
)); ("cumulative", Wrap_utils.Option.map cumulative Py.Bool.of_bool); ("other", Some(other )); ("r", Some(r |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])

                  let query_ball_point ?p ?eps ~x ~r self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "query_ball_point"
                       [||]
                       (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p Py.Float.of_float); ("eps", eps); ("x", Some(x |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Shape_tuple_self_m_ x -> Wrap_utils.id x
))); ("r", Some(r |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])

let query_ball_tree ?p ?eps ~other ~r self =
   Py.Module.get_function_with_keywords (to_pyobject self) "query_ball_tree"
     [||]
     (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p Py.Float.of_float); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("other", Some(other )); ("r", Some(r |> Py.Float.of_float))])

let query_pairs ?p ?eps ~r self =
   Py.Module.get_function_with_keywords (to_pyobject self) "query_pairs"
     [||]
     (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p Py.Float.of_float); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("r", Some(r |> Py.Float.of_float))])

                  let sparse_distance_matrix ?p ~other ~max_distance self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "sparse_distance_matrix"
                       [||]
                       (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p (function
| `T1_p_infinity x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)); ("other", Some(other )); ("max_distance", Some(max_distance |> Py.Float.of_float))])


let data_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "data" with
  | None -> failwith "attribute data not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let data self = match data_opt self with
  | None -> raise Not_found
  | Some x -> x

let leafsize_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "leafsize" with
  | None -> failwith "attribute leafsize not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let leafsize self = match leafsize_opt self with
  | None -> raise Not_found
  | Some x -> x

let m_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "m" with
  | None -> failwith "attribute m not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let m self = match m_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n" with
  | None -> failwith "attribute n not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n self = match n_opt self with
  | None -> raise Not_found
  | Some x -> x

let maxes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "maxes" with
  | None -> failwith "attribute maxes not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let maxes self = match maxes_opt self with
  | None -> raise Not_found
  | Some x -> x

let mins_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "mins" with
  | None -> failwith "attribute mins not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let mins self = match mins_opt self with
  | None -> raise Not_found
  | Some x -> x

let tree_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "tree" with
  | None -> failwith "attribute tree not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let tree self = match tree_opt self with
  | None -> raise Not_found
  | Some x -> x

let size_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "size" with
  | None -> failwith "attribute size not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let size self = match size_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let griddata ?method_ ?fill_value ?rescale ~points ~values ~xi () =
                     Py.Module.get_function_with_keywords __wrap_namespace "griddata"
                       [||]
                       (Wrap_utils.keyword_args [("method", Wrap_utils.Option.map method_ (function
| `Linear -> Py.String.of_string "linear"
| `Nearest -> Py.String.of_string "nearest"
| `Cubic -> Py.String.of_string "cubic"
)); ("fill_value", Wrap_utils.Option.map fill_value Py.Float.of_float); ("rescale", Wrap_utils.Option.map rescale Py.Bool.of_bool); ("points", Some(points )); ("values", Some(values )); ("xi", Some(xi ))])


end
module Polyint = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.interpolate.polyint"

let get_py name = Py.Module.get __wrap_namespace name
                  let approximate_taylor_polynomial ?order ~f ~x ~degree ~scale () =
                     Py.Module.get_function_with_keywords __wrap_namespace "approximate_taylor_polynomial"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order Py.Int.of_int); ("f", Some(f )); ("x", Some(x |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
))); ("degree", Some(degree |> Py.Int.of_int)); ("scale", Some(scale |> Py.Float.of_float))])

                  let barycentric_interpolate ?axis ~xi ~yi ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "barycentric_interpolate"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("xi", Some(xi |> Np.Obj.to_pyobject)); ("yi", Some(yi |> Np.Obj.to_pyobject)); ("x", Some(x |> (function
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
)))])

                  let factorial ?exact ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "factorial"
                       [||]
                       (Wrap_utils.keyword_args [("exact", Wrap_utils.Option.map exact Py.Bool.of_bool); ("n", Some(n |> (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)))])

                  let krogh_interpolate ?der ?axis ~xi ~yi ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "krogh_interpolate"
                       [||]
                       (Wrap_utils.keyword_args [("der", Wrap_utils.Option.map der (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("xi", Some(xi |> Np.Obj.to_pyobject)); ("yi", Some(yi |> Np.Obj.to_pyobject)); ("x", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
module Rbf' = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.interpolate.rbf"

let get_py name = Py.Module.get __wrap_namespace name
let callable obj =
   Py.Module.get_function_with_keywords __wrap_namespace "callable"
     [||]
     (Wrap_utils.keyword_args [("obj", Some(obj ))])

                  let cdist ?metric ?kwargs ~xa ~xb args =
                     Py.Module.get_function_with_keywords __wrap_namespace "cdist"
                       (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
                       (List.rev_append (Wrap_utils.keyword_args [("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("XA", Some(xa |> Np.Obj.to_pyobject)); ("XB", Some(xb |> Np.Obj.to_pyobject))]) (match kwargs with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let pdist ?metric ?kwargs ~x args =
                     Py.Module.get_function_with_keywords __wrap_namespace "pdist"
                       (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
                       (List.rev_append (Wrap_utils.keyword_args [("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("X", Some(x |> Np.Obj.to_pyobject))]) (match kwargs with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let squareform ?force ?checks ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "squareform"
     [||]
     (Wrap_utils.keyword_args [("force", Wrap_utils.Option.map force Py.String.of_string); ("checks", Wrap_utils.Option.map checks Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let xlogy ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "xlogy"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
                  let approximate_taylor_polynomial ?order ~f ~x ~degree ~scale () =
                     Py.Module.get_function_with_keywords __wrap_namespace "approximate_taylor_polynomial"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order Py.Int.of_int); ("f", Some(f )); ("x", Some(x |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
))); ("degree", Some(degree |> Py.Int.of_int)); ("scale", Some(scale |> Py.Float.of_float))])

                  let barycentric_interpolate ?axis ~xi ~yi ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "barycentric_interpolate"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("xi", Some(xi |> Np.Obj.to_pyobject)); ("yi", Some(yi |> Np.Obj.to_pyobject)); ("x", Some(x |> (function
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
)))])

let bisplev ?dx ?dy ~x ~y ~tck () =
   Py.Module.get_function_with_keywords __wrap_namespace "bisplev"
     [||]
     (Wrap_utils.keyword_args [("dx", dx); ("dy", dy); ("x", Some(x )); ("y", Some(y )); ("tck", Some(tck ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let bisplrep ?w ?xb ?xe ?yb ?ye ?kx ?ky ?task ?s ?eps ?tx ?ty ?full_output ?nxest ?nyest ?quiet ~x ~y ~z () =
   Py.Module.get_function_with_keywords __wrap_namespace "bisplrep"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("xb", xb); ("xe", xe); ("yb", yb); ("ye", ye); ("kx", kx); ("ky", ky); ("task", Wrap_utils.Option.map task Py.Int.of_int); ("s", Wrap_utils.Option.map s Py.Float.of_float); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("tx", tx); ("ty", ty); ("full_output", Wrap_utils.Option.map full_output Py.Int.of_int); ("nxest", nxest); ("nyest", nyest); ("quiet", Wrap_utils.Option.map quiet Py.Int.of_int); ("x", Some(x )); ("y", Some(y )); ("z", Some(z ))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Py.String.to_string (Py.Tuple.get x 3))))
                  let griddata ?method_ ?fill_value ?rescale ~points ~values ~xi () =
                     Py.Module.get_function_with_keywords __wrap_namespace "griddata"
                       [||]
                       (Wrap_utils.keyword_args [("method", Wrap_utils.Option.map method_ (function
| `Linear -> Py.String.of_string "linear"
| `Nearest -> Py.String.of_string "nearest"
| `Cubic -> Py.String.of_string "cubic"
)); ("fill_value", Wrap_utils.Option.map fill_value Py.Float.of_float); ("rescale", Wrap_utils.Option.map rescale Py.Bool.of_bool); ("points", Some(points )); ("values", Some(values )); ("xi", Some(xi ))])

let insert ?m ?per ~x ~tck () =
   Py.Module.get_function_with_keywords __wrap_namespace "insert"
     [||]
     (Wrap_utils.keyword_args [("m", Wrap_utils.Option.map m Py.Int.of_int); ("per", Wrap_utils.Option.map per Py.Int.of_int); ("x", Some(x )); ("tck", Some(tck ))])

                  let interpn ?method_ ?bounds_error ?fill_value ~points ~values ~xi () =
                     Py.Module.get_function_with_keywords __wrap_namespace "interpn"
                       [||]
                       (Wrap_utils.keyword_args [("method", Wrap_utils.Option.map method_ Py.String.of_string); ("bounds_error", Wrap_utils.Option.map bounds_error Py.Bool.of_bool); ("fill_value", Wrap_utils.Option.map fill_value (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("points", Some(points )); ("values", Some(values |> Np.Obj.to_pyobject)); ("xi", Some(xi |> Np.Obj.to_pyobject))])

                  let krogh_interpolate ?der ?axis ~xi ~yi ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "krogh_interpolate"
                       [||]
                       (Wrap_utils.keyword_args [("der", Wrap_utils.Option.map der (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("xi", Some(xi |> Np.Obj.to_pyobject)); ("yi", Some(yi |> Np.Obj.to_pyobject)); ("x", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let lagrange ~x ~w () =
   Py.Module.get_function_with_keywords __wrap_namespace "lagrange"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject)); ("w", Some(w |> Np.Obj.to_pyobject))])

let make_interp_spline ?k ?t ?bc_type ?axis ?check_finite ~x ~y () =
   Py.Module.get_function_with_keywords __wrap_namespace "make_interp_spline"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("t", Wrap_utils.Option.map t Np.Obj.to_pyobject); ("bc_type", bc_type); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])

let make_lsq_spline ?k ?w ?axis ?check_finite ~x ~y ~t () =
   Py.Module.get_function_with_keywords __wrap_namespace "make_lsq_spline"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject)); ("t", Some(t |> Np.Obj.to_pyobject))])

let pade ?n ~an ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "pade"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("an", Some(an |> Np.Obj.to_pyobject)); ("m", Some(m |> Py.Int.of_int))])

                  let pchip_interpolate ?der ?axis ~xi ~yi ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "pchip_interpolate"
                       [||]
                       (Wrap_utils.keyword_args [("der", Wrap_utils.Option.map der (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("xi", Some(xi |> Np.Obj.to_pyobject)); ("yi", Some(yi |> Np.Obj.to_pyobject)); ("x", Some(x |> (function
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
)))])

let spalde ~x ~tck () =
   Py.Module.get_function_with_keywords __wrap_namespace "spalde"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject)); ("tck", Some(tck ))])

let splantider ?n ~tck () =
   Py.Module.get_function_with_keywords __wrap_namespace "splantider"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("tck", Some(tck ))])

let splder ?n ~tck () =
   Py.Module.get_function_with_keywords __wrap_namespace "splder"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("tck", Some(tck ))])

let splev ?der ?ext ~x ~tck () =
   Py.Module.get_function_with_keywords __wrap_namespace "splev"
     [||]
     (Wrap_utils.keyword_args [("der", Wrap_utils.Option.map der Py.Int.of_int); ("ext", Wrap_utils.Option.map ext Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject)); ("tck", Some(tck ))])

let splint ?full_output ~a ~b ~tck () =
   Py.Module.get_function_with_keywords __wrap_namespace "splint"
     [||]
     (Wrap_utils.keyword_args [("full_output", Wrap_utils.Option.map full_output Py.Int.of_int); ("a", Some(a )); ("b", Some(b )); ("tck", Some(tck ))])
     |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let splprep ?w ?u ?ub ?ue ?k ?task ?s ?t ?full_output ?nest ?per ?quiet ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "splprep"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("u", Wrap_utils.Option.map u Np.Obj.to_pyobject); ("ub", ub); ("ue", ue); ("k", Wrap_utils.Option.map k Py.Int.of_int); ("task", Wrap_utils.Option.map task Py.Int.of_int); ("s", Wrap_utils.Option.map s Py.Float.of_float); ("t", Wrap_utils.Option.map t Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Int.of_int); ("nest", Wrap_utils.Option.map nest Py.Int.of_int); ("per", Wrap_utils.Option.map per Py.Int.of_int); ("quiet", Wrap_utils.Option.map quiet Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2)), (Py.Int.to_int (Py.Tuple.get x 3)), (Py.String.to_string (Py.Tuple.get x 4))))
                  let splrep ?w ?xb ?xe ?k ?task ?s ?t ?full_output ?per ?quiet ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "splrep"
                       [||]
                       (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("xb", xb); ("xe", xe); ("k", Wrap_utils.Option.map k Py.Int.of_int); ("task", Wrap_utils.Option.map task (function
| `One -> Py.Int.of_int 1
| `T_1 x -> Wrap_utils.id x
| `Zero -> Py.Int.of_int 0
)); ("s", Wrap_utils.Option.map s Py.Float.of_float); ("t", Wrap_utils.Option.map t Np.Obj.to_pyobject); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("per", Wrap_utils.Option.map per Py.Bool.of_bool); ("quiet", Wrap_utils.Option.map quiet Py.Bool.of_bool); ("x", Some(x )); ("y", Some(y ))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Py.String.to_string (Py.Tuple.get x 3))))
let sproot ?mest ~tck () =
   Py.Module.get_function_with_keywords __wrap_namespace "sproot"
     [||]
     (Wrap_utils.keyword_args [("mest", Wrap_utils.Option.map mest Py.Int.of_int); ("tck", Some(tck ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
