let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.setup"

let get_py name = Py.Module.get __wrap_namespace name
let configuration ?parent_package ?top_path () =
   Py.Module.get_function_with_keywords __wrap_namespace "configuration"
     [||]
     (Wrap_utils.keyword_args [("parent_package", parent_package); ("top_path", top_path)])

let cythonize_extensions ~top_path ~config () =
   Py.Module.get_function_with_keywords __wrap_namespace "cythonize_extensions"
     [||]
     (Wrap_utils.keyword_args [("top_path", Some(top_path )); ("config", Some(config ))])

