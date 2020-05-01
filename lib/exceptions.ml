let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.exceptions"

let get_py name = Py.Module.get ns name
