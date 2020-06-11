let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.misc"

let get_py name = Py.Module.get __wrap_namespace name
module Doccer = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.misc.doccer"

let get_py name = Py.Module.get __wrap_namespace name
let docformat ?kwds args =
   Py.Module.get_function_with_keywords __wrap_namespace "docformat"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwds with None -> [] | Some x -> x)

let extend_notes_in_docstring ?kwds args =
   Py.Module.get_function_with_keywords __wrap_namespace "extend_notes_in_docstring"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwds with None -> [] | Some x -> x)

let filldoc ?kwds args =
   Py.Module.get_function_with_keywords __wrap_namespace "filldoc"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwds with None -> [] | Some x -> x)

let indentcount_lines ?kwds args =
   Py.Module.get_function_with_keywords __wrap_namespace "indentcount_lines"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwds with None -> [] | Some x -> x)

let inherit_docstring_from ?kwds args =
   Py.Module.get_function_with_keywords __wrap_namespace "inherit_docstring_from"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwds with None -> [] | Some x -> x)

let replace_notes_in_docstring ?kwds args =
   Py.Module.get_function_with_keywords __wrap_namespace "replace_notes_in_docstring"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwds with None -> [] | Some x -> x)

let unindent_dict ?kwds args =
   Py.Module.get_function_with_keywords __wrap_namespace "unindent_dict"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwds with None -> [] | Some x -> x)

let unindent_string ?kwds args =
   Py.Module.get_function_with_keywords __wrap_namespace "unindent_string"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwds with None -> [] | Some x -> x)


end
let ascent () =
   Py.Module.get_function_with_keywords __wrap_namespace "ascent"
     [||]
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let central_diff_weights ?ndiv ~np () =
   Py.Module.get_function_with_keywords __wrap_namespace "central_diff_weights"
     [||]
     (Wrap_utils.keyword_args [("ndiv", Wrap_utils.Option.map ndiv Py.Int.of_int); ("Np", Some(np |> Py.Int.of_int))])

let derivative ?dx ?n ?args ?order ~func ~x0 () =
   Py.Module.get_function_with_keywords __wrap_namespace "derivative"
     [||]
     (Wrap_utils.keyword_args [("dx", Wrap_utils.Option.map dx Py.Float.of_float); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("args", args); ("order", Wrap_utils.Option.map order Py.Int.of_int); ("func", Some(func )); ("x0", Some(x0 |> Py.Float.of_float))])

let electrocardiogram () =
   Py.Module.get_function_with_keywords __wrap_namespace "electrocardiogram"
     [||]
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let face ?gray () =
   Py.Module.get_function_with_keywords __wrap_namespace "face"
     [||]
     (Wrap_utils.keyword_args [("gray", Wrap_utils.Option.map gray Py.Bool.of_bool)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
