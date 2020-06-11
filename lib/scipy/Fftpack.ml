let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.fftpack"

let get_py name = Py.Module.get __wrap_namespace name
module Basic = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.fftpack.basic"

let get_py name = Py.Module.get __wrap_namespace name
let fft ?n ?axis ?overwrite_x ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "fft"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let fft2 ?shape ?axes ?overwrite_x ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "fft2"
     [||]
     (Wrap_utils.keyword_args [("shape", shape); ("axes", axes); ("overwrite_x", overwrite_x); ("x", Some(x ))])

                  let fftn ?shape ?axes ?overwrite_x ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fftn"
                       [||]
                       (Wrap_utils.keyword_args [("shape", Wrap_utils.Option.map shape (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("axes", Wrap_utils.Option.map axes (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])

let ifft ?n ?axis ?overwrite_x ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ifft"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let ifft2 ?shape ?axes ?overwrite_x ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ifft2"
     [||]
     (Wrap_utils.keyword_args [("shape", shape); ("axes", axes); ("overwrite_x", overwrite_x); ("x", Some(x ))])

let ifftn ?shape ?axes ?overwrite_x ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ifftn"
     [||]
     (Wrap_utils.keyword_args [("shape", shape); ("axes", axes); ("overwrite_x", overwrite_x); ("x", Some(x ))])

let irfft ?n ?axis ?overwrite_x ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "irfft"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let rfft ?n ?axis ?overwrite_x ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "rfft"
                       [||]
                       (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("x", Some(x |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Real_valued x -> Wrap_utils.id x
)))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))

end
module Convolve = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.fftpack.convolve"

let get_py name = Py.Module.get __wrap_namespace name
let convolve_z ?overwrite_x ~x ~omega_real ~omega_imag () =
   Py.Module.get_function_with_keywords __wrap_namespace "convolve_z"
     [||]
     (Wrap_utils.keyword_args [("overwrite_x", overwrite_x); ("x", Some(x )); ("omega_real", Some(omega_real )); ("omega_imag", Some(omega_imag ))])


end
module Helper = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.fftpack.helper"

let get_py name = Py.Module.get __wrap_namespace name
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
                  let fftshift ?axes ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fftshift"
                       [||]
                       (Wrap_utils.keyword_args [("axes", Wrap_utils.Option.map axes (function
| `Shape_tuple x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("x", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let ifftshift ?axes ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ifftshift"
                       [||]
                       (Wrap_utils.keyword_args [("axes", Wrap_utils.Option.map axes (function
| `Shape_tuple x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("x", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let next_fast_len target =
   Py.Module.get_function_with_keywords __wrap_namespace "next_fast_len"
     [||]
     (Wrap_utils.keyword_args [("target", Some(target |> Py.Int.of_int))])
     |> Py.Int.to_int
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

end
module Pseudo_diffs = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.fftpack.pseudo_diffs"

let get_py name = Py.Module.get __wrap_namespace name
                  let asarray ?dtype ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let cc_diff ?period ?_cache ~x ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "cc_diff"
     [||]
     (Wrap_utils.keyword_args [("period", Wrap_utils.Option.map period Py.Float.of_float); ("_cache", _cache); ("x", Some(x |> Np.Obj.to_pyobject)); ("a", Some(a )); ("b", Some(b ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let cos ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cos"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let cosh ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cosh"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let cs_diff ?period ?_cache ~x ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "cs_diff"
     [||]
     (Wrap_utils.keyword_args [("period", Wrap_utils.Option.map period Py.Float.of_float); ("_cache", _cache); ("x", Some(x |> Np.Obj.to_pyobject)); ("a", Some(a )); ("b", Some(b ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let diff ?order ?period ?_cache ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "diff"
     [||]
     (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order Py.Int.of_int); ("period", Wrap_utils.Option.map period Py.Float.of_float); ("_cache", _cache); ("x", Some(x |> Np.Obj.to_pyobject))])

let hilbert ?_cache ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "hilbert"
     [||]
     (Wrap_utils.keyword_args [("_cache", _cache); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let ihilbert x =
   Py.Module.get_function_with_keywords __wrap_namespace "ihilbert"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let iscomplexobj x =
   Py.Module.get_function_with_keywords __wrap_namespace "iscomplexobj"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])
     |> Py.Bool.to_bool
let itilbert ?period ?_cache ~x ~h () =
   Py.Module.get_function_with_keywords __wrap_namespace "itilbert"
     [||]
     (Wrap_utils.keyword_args [("period", period); ("_cache", _cache); ("x", Some(x )); ("h", Some(h ))])

let sc_diff ?period ?_cache ~x ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "sc_diff"
     [||]
     (Wrap_utils.keyword_args [("period", Wrap_utils.Option.map period Py.Float.of_float); ("_cache", _cache); ("x", Some(x |> Np.Obj.to_pyobject)); ("a", Some(a )); ("b", Some(b ))])

let shift ?period ?_cache ~x ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "shift"
     [||]
     (Wrap_utils.keyword_args [("period", Wrap_utils.Option.map period Py.Float.of_float); ("_cache", _cache); ("x", Some(x |> Np.Obj.to_pyobject)); ("a", Some(a |> Py.Float.of_float))])

                  let sin ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "sin"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let sinh ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "sinh"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let ss_diff ?period ?_cache ~x ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "ss_diff"
     [||]
     (Wrap_utils.keyword_args [("period", Wrap_utils.Option.map period Py.Float.of_float); ("_cache", _cache); ("x", Some(x |> Np.Obj.to_pyobject)); ("a", Some(a )); ("b", Some(b ))])

                  let tanh ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "tanh"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let tilbert ?period ?_cache ~x ~h () =
   Py.Module.get_function_with_keywords __wrap_namespace "tilbert"
     [||]
     (Wrap_utils.keyword_args [("period", Wrap_utils.Option.map period Py.Float.of_float); ("_cache", _cache); ("x", Some(x |> Np.Obj.to_pyobject)); ("h", Some(h |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
module Realtransforms = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.fftpack.realtransforms"

let get_py name = Py.Module.get __wrap_namespace name
                  let dct ?type_ ?n ?axis ?norm ?overwrite_x ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "dct"
                       [||]
                       (Wrap_utils.keyword_args [("type", Wrap_utils.Option.map type_ (function
| `Three -> Py.Int.of_int 3
| `One -> Py.Int.of_int 1
| `Four -> Py.Int.of_int 4
| `Two -> Py.Int.of_int 2
)); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])

                  let dctn ?type_ ?shape ?axes ?norm ?overwrite_x ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "dctn"
                       [||]
                       (Wrap_utils.keyword_args [("type", Wrap_utils.Option.map type_ (function
| `Three -> Py.Int.of_int 3
| `One -> Py.Int.of_int 1
| `Four -> Py.Int.of_int 4
| `Two -> Py.Int.of_int 2
)); ("shape", Wrap_utils.Option.map shape (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("axes", Wrap_utils.Option.map axes (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])

                  let dst ?type_ ?n ?axis ?norm ?overwrite_x ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "dst"
                       [||]
                       (Wrap_utils.keyword_args [("type", Wrap_utils.Option.map type_ (function
| `Three -> Py.Int.of_int 3
| `One -> Py.Int.of_int 1
| `Four -> Py.Int.of_int 4
| `Two -> Py.Int.of_int 2
)); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let dstn ?type_ ?shape ?axes ?norm ?overwrite_x ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "dstn"
                       [||]
                       (Wrap_utils.keyword_args [("type", Wrap_utils.Option.map type_ (function
| `Three -> Py.Int.of_int 3
| `One -> Py.Int.of_int 1
| `Four -> Py.Int.of_int 4
| `Two -> Py.Int.of_int 2
)); ("shape", Wrap_utils.Option.map shape (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("axes", Wrap_utils.Option.map axes (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])

                  let idct ?type_ ?n ?axis ?norm ?overwrite_x ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "idct"
                       [||]
                       (Wrap_utils.keyword_args [("type", Wrap_utils.Option.map type_ (function
| `Three -> Py.Int.of_int 3
| `One -> Py.Int.of_int 1
| `Four -> Py.Int.of_int 4
| `Two -> Py.Int.of_int 2
)); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])

                  let idctn ?type_ ?shape ?axes ?norm ?overwrite_x ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "idctn"
                       [||]
                       (Wrap_utils.keyword_args [("type", Wrap_utils.Option.map type_ (function
| `Three -> Py.Int.of_int 3
| `One -> Py.Int.of_int 1
| `Four -> Py.Int.of_int 4
| `Two -> Py.Int.of_int 2
)); ("shape", Wrap_utils.Option.map shape (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("axes", Wrap_utils.Option.map axes (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])

                  let idst ?type_ ?n ?axis ?norm ?overwrite_x ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "idst"
                       [||]
                       (Wrap_utils.keyword_args [("type", Wrap_utils.Option.map type_ (function
| `Three -> Py.Int.of_int 3
| `One -> Py.Int.of_int 1
| `Four -> Py.Int.of_int 4
| `Two -> Py.Int.of_int 2
)); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])

                  let idstn ?type_ ?shape ?axes ?norm ?overwrite_x ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "idstn"
                       [||]
                       (Wrap_utils.keyword_args [("type", Wrap_utils.Option.map type_ (function
| `Three -> Py.Int.of_int 3
| `One -> Py.Int.of_int 1
| `Four -> Py.Int.of_int 4
| `Two -> Py.Int.of_int 2
)); ("shape", Wrap_utils.Option.map shape (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("axes", Wrap_utils.Option.map axes (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])


end
let cc_diff ?period ?_cache ~x ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "cc_diff"
     [||]
     (Wrap_utils.keyword_args [("period", Wrap_utils.Option.map period Py.Float.of_float); ("_cache", _cache); ("x", Some(x |> Np.Obj.to_pyobject)); ("a", Some(a )); ("b", Some(b ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let cs_diff ?period ?_cache ~x ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "cs_diff"
     [||]
     (Wrap_utils.keyword_args [("period", Wrap_utils.Option.map period Py.Float.of_float); ("_cache", _cache); ("x", Some(x |> Np.Obj.to_pyobject)); ("a", Some(a )); ("b", Some(b ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let dct ?type_ ?n ?axis ?norm ?overwrite_x ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "dct"
                       [||]
                       (Wrap_utils.keyword_args [("type", Wrap_utils.Option.map type_ (function
| `Three -> Py.Int.of_int 3
| `One -> Py.Int.of_int 1
| `Four -> Py.Int.of_int 4
| `Two -> Py.Int.of_int 2
)); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])

                  let dctn ?type_ ?shape ?axes ?norm ?overwrite_x ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "dctn"
                       [||]
                       (Wrap_utils.keyword_args [("type", Wrap_utils.Option.map type_ (function
| `Three -> Py.Int.of_int 3
| `One -> Py.Int.of_int 1
| `Four -> Py.Int.of_int 4
| `Two -> Py.Int.of_int 2
)); ("shape", Wrap_utils.Option.map shape (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("axes", Wrap_utils.Option.map axes (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])

let diff ?order ?period ?_cache ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "diff"
     [||]
     (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order Py.Int.of_int); ("period", Wrap_utils.Option.map period Py.Float.of_float); ("_cache", _cache); ("x", Some(x |> Np.Obj.to_pyobject))])

                  let dst ?type_ ?n ?axis ?norm ?overwrite_x ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "dst"
                       [||]
                       (Wrap_utils.keyword_args [("type", Wrap_utils.Option.map type_ (function
| `Three -> Py.Int.of_int 3
| `One -> Py.Int.of_int 1
| `Four -> Py.Int.of_int 4
| `Two -> Py.Int.of_int 2
)); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let dstn ?type_ ?shape ?axes ?norm ?overwrite_x ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "dstn"
                       [||]
                       (Wrap_utils.keyword_args [("type", Wrap_utils.Option.map type_ (function
| `Three -> Py.Int.of_int 3
| `One -> Py.Int.of_int 1
| `Four -> Py.Int.of_int 4
| `Two -> Py.Int.of_int 2
)); ("shape", Wrap_utils.Option.map shape (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("axes", Wrap_utils.Option.map axes (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])

let fft ?n ?axis ?overwrite_x ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "fft"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let fft2 ?shape ?axes ?overwrite_x ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "fft2"
     [||]
     (Wrap_utils.keyword_args [("shape", shape); ("axes", axes); ("overwrite_x", overwrite_x); ("x", Some(x ))])

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
                  let fftn ?shape ?axes ?overwrite_x ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fftn"
                       [||]
                       (Wrap_utils.keyword_args [("shape", Wrap_utils.Option.map shape (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("axes", Wrap_utils.Option.map axes (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])

                  let fftshift ?axes ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fftshift"
                       [||]
                       (Wrap_utils.keyword_args [("axes", Wrap_utils.Option.map axes (function
| `Shape_tuple x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("x", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let hilbert ?_cache ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "hilbert"
     [||]
     (Wrap_utils.keyword_args [("_cache", _cache); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let idct ?type_ ?n ?axis ?norm ?overwrite_x ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "idct"
                       [||]
                       (Wrap_utils.keyword_args [("type", Wrap_utils.Option.map type_ (function
| `Three -> Py.Int.of_int 3
| `One -> Py.Int.of_int 1
| `Four -> Py.Int.of_int 4
| `Two -> Py.Int.of_int 2
)); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])

                  let idctn ?type_ ?shape ?axes ?norm ?overwrite_x ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "idctn"
                       [||]
                       (Wrap_utils.keyword_args [("type", Wrap_utils.Option.map type_ (function
| `Three -> Py.Int.of_int 3
| `One -> Py.Int.of_int 1
| `Four -> Py.Int.of_int 4
| `Two -> Py.Int.of_int 2
)); ("shape", Wrap_utils.Option.map shape (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("axes", Wrap_utils.Option.map axes (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])

                  let idst ?type_ ?n ?axis ?norm ?overwrite_x ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "idst"
                       [||]
                       (Wrap_utils.keyword_args [("type", Wrap_utils.Option.map type_ (function
| `Three -> Py.Int.of_int 3
| `One -> Py.Int.of_int 1
| `Four -> Py.Int.of_int 4
| `Two -> Py.Int.of_int 2
)); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])

                  let idstn ?type_ ?shape ?axes ?norm ?overwrite_x ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "idstn"
                       [||]
                       (Wrap_utils.keyword_args [("type", Wrap_utils.Option.map type_ (function
| `Three -> Py.Int.of_int 3
| `One -> Py.Int.of_int 1
| `Four -> Py.Int.of_int 4
| `Two -> Py.Int.of_int 2
)); ("shape", Wrap_utils.Option.map shape (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("axes", Wrap_utils.Option.map axes (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])

let ifft ?n ?axis ?overwrite_x ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ifft"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let ifft2 ?shape ?axes ?overwrite_x ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ifft2"
     [||]
     (Wrap_utils.keyword_args [("shape", shape); ("axes", axes); ("overwrite_x", overwrite_x); ("x", Some(x ))])

let ifftn ?shape ?axes ?overwrite_x ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ifftn"
     [||]
     (Wrap_utils.keyword_args [("shape", shape); ("axes", axes); ("overwrite_x", overwrite_x); ("x", Some(x ))])

                  let ifftshift ?axes ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ifftshift"
                       [||]
                       (Wrap_utils.keyword_args [("axes", Wrap_utils.Option.map axes (function
| `Shape_tuple x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("x", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let ihilbert x =
   Py.Module.get_function_with_keywords __wrap_namespace "ihilbert"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let irfft ?n ?axis ?overwrite_x ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "irfft"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let itilbert ?period ?_cache ~x ~h () =
   Py.Module.get_function_with_keywords __wrap_namespace "itilbert"
     [||]
     (Wrap_utils.keyword_args [("period", period); ("_cache", _cache); ("x", Some(x )); ("h", Some(h ))])

let next_fast_len target =
   Py.Module.get_function_with_keywords __wrap_namespace "next_fast_len"
     [||]
     (Wrap_utils.keyword_args [("target", Some(target |> Py.Int.of_int))])
     |> Py.Int.to_int
                  let rfft ?n ?axis ?overwrite_x ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "rfft"
                       [||]
                       (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("overwrite_x", Wrap_utils.Option.map overwrite_x Py.Bool.of_bool); ("x", Some(x |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Real_valued x -> Wrap_utils.id x
)))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
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
let sc_diff ?period ?_cache ~x ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "sc_diff"
     [||]
     (Wrap_utils.keyword_args [("period", Wrap_utils.Option.map period Py.Float.of_float); ("_cache", _cache); ("x", Some(x |> Np.Obj.to_pyobject)); ("a", Some(a )); ("b", Some(b ))])

let shift ?period ?_cache ~x ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "shift"
     [||]
     (Wrap_utils.keyword_args [("period", Wrap_utils.Option.map period Py.Float.of_float); ("_cache", _cache); ("x", Some(x |> Np.Obj.to_pyobject)); ("a", Some(a |> Py.Float.of_float))])

let ss_diff ?period ?_cache ~x ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "ss_diff"
     [||]
     (Wrap_utils.keyword_args [("period", Wrap_utils.Option.map period Py.Float.of_float); ("_cache", _cache); ("x", Some(x |> Np.Obj.to_pyobject)); ("a", Some(a )); ("b", Some(b ))])

let tilbert ?period ?_cache ~x ~h () =
   Py.Module.get_function_with_keywords __wrap_namespace "tilbert"
     [||]
     (Wrap_utils.keyword_args [("period", Wrap_utils.Option.map period Py.Float.of_float); ("_cache", _cache); ("x", Some(x |> Np.Obj.to_pyobject)); ("h", Some(h |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
