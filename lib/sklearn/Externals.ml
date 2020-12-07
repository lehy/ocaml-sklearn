let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.externals"

let get_py name = Py.Module.get __wrap_namespace name
