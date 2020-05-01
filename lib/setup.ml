let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.setup"

let get_py name = Py.Module.get ns name
let configuration ?parent_package ?top_path () =
   Py.Module.get_function_with_keywords ns "configuration"
     [||]
     (Wrap_utils.keyword_args [("parent_package", parent_package); ("top_path", top_path)])

let cythonize_extensions ~top_path ~config () =
   Py.Module.get_function_with_keywords ns "cythonize_extensions"
     [||]
     (Wrap_utils.keyword_args [("top_path", Some(top_path )); ("config", Some(config ))])

