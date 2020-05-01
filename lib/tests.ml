let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.tests"

let get_py name = Py.Module.get ns name
