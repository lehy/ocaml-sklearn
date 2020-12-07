let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.conftest"

let get_py name = Py.Module.get __wrap_namespace name
module FPUModeChangeWarning = struct
type tag = [`FPUModeChangeWarning]
type t = [`BaseException | `FPUModeChangeWarning | `Object] Obj.t
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
module LooseVersion = struct
type tag = [`LooseVersion]
type t = [`LooseVersion | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?vstring () =
   Py.Module.get_function_with_keywords __wrap_namespace "LooseVersion"
     [||]
     (Wrap_utils.keyword_args [("vstring", vstring)])
     |> of_pyobject
let parse ~vstring self =
   Py.Module.get_function_with_keywords (to_pyobject self) "parse"
     [||]
     (Wrap_utils.keyword_args [("vstring", Some(vstring ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let check_fpu_mode request =
   Py.Module.get_function_with_keywords __wrap_namespace "check_fpu_mode"
     [||]
     (Wrap_utils.keyword_args [("request", Some(request ))])

let pytest_configure config =
   Py.Module.get_function_with_keywords __wrap_namespace "pytest_configure"
     [||]
     (Wrap_utils.keyword_args [("config", Some(config ))])

let pytest_runtest_setup item =
   Py.Module.get_function_with_keywords __wrap_namespace "pytest_runtest_setup"
     [||]
     (Wrap_utils.keyword_args [("item", Some(item ))])

