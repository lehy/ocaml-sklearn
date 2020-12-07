let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.conftest"

let get_py name = Py.Module.get __wrap_namespace name
let pyplot () =
   Py.Module.get_function_with_keywords __wrap_namespace "pyplot"
     [||]
     []

