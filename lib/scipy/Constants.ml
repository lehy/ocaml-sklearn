let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.constants"

let get_py name = Py.Module.get __wrap_namespace name
module ConstantWarning = struct
type tag = [`ConstantWarning]
type t = [`BaseException | `ConstantWarning | `Object] Obj.t
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
module Codata = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.constants.codata"

let get_py name = Py.Module.get __wrap_namespace name
let find ?sub ?disp () =
   Py.Module.get_function_with_keywords __wrap_namespace "find"
     [||]
     (Wrap_utils.keyword_args [("sub", Wrap_utils.Option.map sub Py.String.of_string); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool)])
     |> (fun py -> if Py.is_none py then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) py))
let parse_constants_2002to2014 d =
   Py.Module.get_function_with_keywords __wrap_namespace "parse_constants_2002to2014"
     [||]
     (Wrap_utils.keyword_args [("d", Some(d ))])

let parse_constants_2018toXXXX d =
   Py.Module.get_function_with_keywords __wrap_namespace "parse_constants_2018toXXXX"
     [||]
     (Wrap_utils.keyword_args [("d", Some(d ))])

                  let precision key =
                     Py.Module.get_function_with_keywords __wrap_namespace "precision"
                       [||]
                       (Wrap_utils.keyword_args [("key", Some(key |> (function
| `S x -> Py.String.of_string x
| `Python_string x -> Wrap_utils.id x
)))])
                       |> Py.Float.to_float
let sqrt x =
   Py.Module.get_function_with_keywords __wrap_namespace "sqrt"
     (Array.of_list @@ List.concat [[x ]])
     []

                  let unit key =
                     Py.Module.get_function_with_keywords __wrap_namespace "unit"
                       [||]
                       (Wrap_utils.keyword_args [("key", Some(key |> (function
| `S x -> Py.String.of_string x
| `Python_string x -> Wrap_utils.id x
)))])

                  let value key =
                     Py.Module.get_function_with_keywords __wrap_namespace "value"
                       [||]
                       (Wrap_utils.keyword_args [("key", Some(key |> (function
| `S x -> Py.String.of_string x
| `Python_string x -> Wrap_utils.id x
)))])
                       |> Py.Float.to_float

end
module Constants = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.constants.constants"

let get_py name = Py.Module.get __wrap_namespace name
let convert_temperature ~val_ ~old_scale ~new_scale () =
   Py.Module.get_function_with_keywords __wrap_namespace "convert_temperature"
     [||]
     (Wrap_utils.keyword_args [("val", Some(val_ |> Np.Obj.to_pyobject)); ("old_scale", Some(old_scale |> Py.String.of_string)); ("new_scale", Some(new_scale |> Py.String.of_string))])

let lambda2nu lambda_ =
   Py.Module.get_function_with_keywords __wrap_namespace "lambda2nu"
     [||]
     (Wrap_utils.keyword_args [("lambda_", Some(lambda_ |> Np.Obj.to_pyobject))])

let nu2lambda nu =
   Py.Module.get_function_with_keywords __wrap_namespace "nu2lambda"
     [||]
     (Wrap_utils.keyword_args [("nu", Some(nu |> Np.Obj.to_pyobject))])


end
let convert_temperature ~val_ ~old_scale ~new_scale () =
   Py.Module.get_function_with_keywords __wrap_namespace "convert_temperature"
     [||]
     (Wrap_utils.keyword_args [("val", Some(val_ |> Np.Obj.to_pyobject)); ("old_scale", Some(old_scale |> Py.String.of_string)); ("new_scale", Some(new_scale |> Py.String.of_string))])

let find ?sub ?disp () =
   Py.Module.get_function_with_keywords __wrap_namespace "find"
     [||]
     (Wrap_utils.keyword_args [("sub", Wrap_utils.Option.map sub Py.String.of_string); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool)])
     |> (fun py -> if Py.is_none py then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) py))
let lambda2nu lambda_ =
   Py.Module.get_function_with_keywords __wrap_namespace "lambda2nu"
     [||]
     (Wrap_utils.keyword_args [("lambda_", Some(lambda_ |> Np.Obj.to_pyobject))])

let nu2lambda nu =
   Py.Module.get_function_with_keywords __wrap_namespace "nu2lambda"
     [||]
     (Wrap_utils.keyword_args [("nu", Some(nu |> Np.Obj.to_pyobject))])

                  let precision key =
                     Py.Module.get_function_with_keywords __wrap_namespace "precision"
                       [||]
                       (Wrap_utils.keyword_args [("key", Some(key |> (function
| `S x -> Py.String.of_string x
| `Python_string x -> Wrap_utils.id x
)))])
                       |> Py.Float.to_float
                  let unit key =
                     Py.Module.get_function_with_keywords __wrap_namespace "unit"
                       [||]
                       (Wrap_utils.keyword_args [("key", Some(key |> (function
| `S x -> Py.String.of_string x
| `Python_string x -> Wrap_utils.id x
)))])

                  let value key =
                     Py.Module.get_function_with_keywords __wrap_namespace "value"
                       [||]
                       (Wrap_utils.keyword_args [("key", Some(key |> (function
| `S x -> Py.String.of_string x
| `Python_string x -> Wrap_utils.id x
)))])
                       |> Py.Float.to_float
