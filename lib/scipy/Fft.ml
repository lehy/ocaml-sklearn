let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.fft"

let get_py name = Py.Module.get __wrap_namespace name
                  let dct ?type_ ?n ?axis ?norm ?overwrite_x ?workers ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "dct"
                       [||]
                       (Wrap_utils.keyword_args [("type", Wrap_utils.Option.map type_ (function
| `Three -> Py.Int.of_int 3
| `One -> Py.Int.of_int 1
| `Four -> Py.Int.of_int 4
| `Two -> Py.Int.of_int 2
)); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])

                  let dctn ?type_ ?s ?axes ?norm ?overwrite_x ?workers ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "dctn"
                       [||]
                       (Wrap_utils.keyword_args [("type", Wrap_utils.Option.map type_ (function
| `Three -> Py.Int.of_int 3
| `One -> Py.Int.of_int 1
| `Four -> Py.Int.of_int 4
| `Two -> Py.Int.of_int 2
)); ("s", Wrap_utils.Option.map s (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("axes", Wrap_utils.Option.map axes (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])

                  let dst ?type_ ?n ?axis ?norm ?overwrite_x ?workers ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "dst"
                       [||]
                       (Wrap_utils.keyword_args [("type", Wrap_utils.Option.map type_ (function
| `Three -> Py.Int.of_int 3
| `One -> Py.Int.of_int 1
| `Four -> Py.Int.of_int 4
| `Two -> Py.Int.of_int 2
)); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let dstn ?type_ ?s ?axes ?norm ?overwrite_x ?workers ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "dstn"
                       [||]
                       (Wrap_utils.keyword_args [("type", Wrap_utils.Option.map type_ (function
| `Three -> Py.Int.of_int 3
| `One -> Py.Int.of_int 1
| `Four -> Py.Int.of_int 4
| `Two -> Py.Int.of_int 2
)); ("s", Wrap_utils.Option.map s (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("axes", Wrap_utils.Option.map axes (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])

let fft ?n ?axis ?norm ?overwrite_x ?workers ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "fft"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])

let fft2 ?s ?axes ?norm ?overwrite_x ?workers ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "fft2"
     [||]
     (Wrap_utils.keyword_args [("s", Wrap_utils.Option.map s (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("axes", Wrap_utils.Option.map axes (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])

                  let fftfreq ?d ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fftfreq"
                       [||]
                       (Wrap_utils.keyword_args [("d", Wrap_utils.Option.map d (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("n", Some(n |> Py.Int.of_int))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let fftn ?s ?axes ?norm ?overwrite_x ?workers ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "fftn"
     [||]
     (Wrap_utils.keyword_args [("s", Wrap_utils.Option.map s (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("axes", Wrap_utils.Option.map axes (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])

                  let fftshift ?axes ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fftshift"
                       [||]
                       (Wrap_utils.keyword_args [("axes", Wrap_utils.Option.map axes (function
| `Shape_tuple x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("x", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let get_workers () =
   Py.Module.get_function_with_keywords __wrap_namespace "get_workers"
     [||]
     []

let hfft ?n ?axis ?norm ?overwrite_x ?workers ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "hfft"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let hfft2 ?s ?axes ?norm ?overwrite_x ?workers ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "hfft2"
     [||]
     (Wrap_utils.keyword_args [("s", Wrap_utils.Option.map s (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("axes", Wrap_utils.Option.map axes (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let hfftn ?s ?axes ?norm ?overwrite_x ?workers ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "hfftn"
     [||]
     (Wrap_utils.keyword_args [("s", Wrap_utils.Option.map s (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("axes", Wrap_utils.Option.map axes (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let idct ?type_ ?n ?axis ?norm ?overwrite_x ?workers ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "idct"
                       [||]
                       (Wrap_utils.keyword_args [("type", Wrap_utils.Option.map type_ (function
| `Three -> Py.Int.of_int 3
| `One -> Py.Int.of_int 1
| `Four -> Py.Int.of_int 4
| `Two -> Py.Int.of_int 2
)); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])

                  let idctn ?type_ ?s ?axes ?norm ?overwrite_x ?workers ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "idctn"
                       [||]
                       (Wrap_utils.keyword_args [("type", Wrap_utils.Option.map type_ (function
| `Three -> Py.Int.of_int 3
| `One -> Py.Int.of_int 1
| `Four -> Py.Int.of_int 4
| `Two -> Py.Int.of_int 2
)); ("s", Wrap_utils.Option.map s (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("axes", Wrap_utils.Option.map axes (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])

                  let idst ?type_ ?n ?axis ?norm ?overwrite_x ?workers ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "idst"
                       [||]
                       (Wrap_utils.keyword_args [("type", Wrap_utils.Option.map type_ (function
| `Three -> Py.Int.of_int 3
| `One -> Py.Int.of_int 1
| `Four -> Py.Int.of_int 4
| `Two -> Py.Int.of_int 2
)); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])

                  let idstn ?type_ ?s ?axes ?norm ?overwrite_x ?workers ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "idstn"
                       [||]
                       (Wrap_utils.keyword_args [("type", Wrap_utils.Option.map type_ (function
| `Three -> Py.Int.of_int 3
| `One -> Py.Int.of_int 1
| `Four -> Py.Int.of_int 4
| `Two -> Py.Int.of_int 2
)); ("s", Wrap_utils.Option.map s (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("axes", Wrap_utils.Option.map axes (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])

let ifft ?n ?axis ?norm ?overwrite_x ?workers ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ifft"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])

let ifft2 ?s ?axes ?norm ?overwrite_x ?workers ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ifft2"
     [||]
     (Wrap_utils.keyword_args [("s", Wrap_utils.Option.map s (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("axes", Wrap_utils.Option.map axes (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])

let ifftn ?s ?axes ?norm ?overwrite_x ?workers ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ifftn"
     [||]
     (Wrap_utils.keyword_args [("s", Wrap_utils.Option.map s (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("axes", Wrap_utils.Option.map axes (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])

                  let ifftshift ?axes ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ifftshift"
                       [||]
                       (Wrap_utils.keyword_args [("axes", Wrap_utils.Option.map axes (function
| `Shape_tuple x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("x", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let ihfft ?n ?axis ?norm ?overwrite_x ?workers ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ihfft"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])

let ihfft2 ?s ?axes ?norm ?overwrite_x ?workers ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ihfft2"
     [||]
     (Wrap_utils.keyword_args [("s", Wrap_utils.Option.map s (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("axes", Wrap_utils.Option.map axes (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let ihfftn ?s ?axes ?norm ?overwrite_x ?workers ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ihfftn"
     [||]
     (Wrap_utils.keyword_args [("s", Wrap_utils.Option.map s (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("axes", Wrap_utils.Option.map axes (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])

let irfft ?n ?axis ?norm ?overwrite_x ?workers ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "irfft"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let irfft2 ?s ?axes ?norm ?overwrite_x ?workers ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "irfft2"
     [||]
     (Wrap_utils.keyword_args [("s", Wrap_utils.Option.map s (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("axes", Wrap_utils.Option.map axes (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let irfftn ?s ?axes ?norm ?overwrite_x ?workers ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "irfftn"
     [||]
     (Wrap_utils.keyword_args [("s", Wrap_utils.Option.map s (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("axes", Wrap_utils.Option.map axes (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let register_backend backend =
                     Py.Module.get_function_with_keywords __wrap_namespace "register_backend"
                       [||]
                       (Wrap_utils.keyword_args [("backend", Some(backend |> (function
| `Scipy -> Py.String.of_string "scipy"
| `PyObject x -> Wrap_utils.id x
)))])

let rfft ?n ?axis ?norm ?overwrite_x ?workers ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "rfft"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers Py.Int.of_int); ("x", Some(x ))])

let rfft2 ?s ?axes ?norm ?overwrite_x ?workers ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "rfft2"
     [||]
     (Wrap_utils.keyword_args [("s", Wrap_utils.Option.map s (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("axes", Wrap_utils.Option.map axes (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let rfftfreq ?d ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "rfftfreq"
                       [||]
                       (Wrap_utils.keyword_args [("d", Wrap_utils.Option.map d (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("n", Some(n |> Py.Int.of_int))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let rfftn ?s ?axes ?norm ?overwrite_x ?workers ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "rfftn"
     [||]
     (Wrap_utils.keyword_args [("s", Wrap_utils.Option.map s (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("axes", Wrap_utils.Option.map axes (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("workers", Wrap_utils.Option.map workers Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])

                  let set_backend ?coerce ?only ~backend () =
                     Py.Module.get_function_with_keywords __wrap_namespace "set_backend"
                       [||]
                       (Wrap_utils.keyword_args [("coerce", Wrap_utils.Option.map coerce Py.Bool.of_bool); ("only", Wrap_utils.Option.map only Py.Bool.of_bool); ("backend", Some(backend |> (function
| `Scipy -> Py.String.of_string "scipy"
| `PyObject x -> Wrap_utils.id x
)))])

                  let set_global_backend backend =
                     Py.Module.get_function_with_keywords __wrap_namespace "set_global_backend"
                       [||]
                       (Wrap_utils.keyword_args [("backend", Some(backend |> (function
| `Scipy -> Py.String.of_string "scipy"
| `PyObject x -> Wrap_utils.id x
)))])

let set_workers workers =
   Py.Module.get_function_with_keywords __wrap_namespace "set_workers"
     [||]
     (Wrap_utils.keyword_args [("workers", Some(workers |> Py.Int.of_int))])

                  let skip_backend backend =
                     Py.Module.get_function_with_keywords __wrap_namespace "skip_backend"
                       [||]
                       (Wrap_utils.keyword_args [("backend", Some(backend |> (function
| `Scipy -> Py.String.of_string "scipy"
| `PyObject x -> Wrap_utils.id x
)))])

