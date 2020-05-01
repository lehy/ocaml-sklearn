let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.conftest"

let get_py name = Py.Module.get ns name
let pyplot () =
   Py.Module.get_function_with_keywords ns "pyplot"
     [||]
     []

