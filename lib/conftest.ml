let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.conftest"

let pyplot () =
   Py.Module.get_function_with_keywords ns "pyplot"
     [||]
     []

