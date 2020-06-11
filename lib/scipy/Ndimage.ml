let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.ndimage"

let get_py name = Py.Module.get __wrap_namespace name
module Filters = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.ndimage.filters"

let get_py name = Py.Module.get __wrap_namespace name
                  let convolve ?output ?mode ?cval ?origin ~input ~weights () =
                     Py.Module.get_function_with_keywords __wrap_namespace "convolve"
                       [||]
                       (Wrap_utils.keyword_args [("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("input", Some(input |> Np.Obj.to_pyobject)); ("weights", Some(weights |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let convolve1d ?axis ?output ?mode ?cval ?origin ~input ~weights () =
                     Py.Module.get_function_with_keywords __wrap_namespace "convolve1d"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin Py.Int.of_int); ("input", Some(input |> Np.Obj.to_pyobject)); ("weights", Some(weights |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let correlate ?output ?mode ?cval ?origin ~input ~weights () =
                     Py.Module.get_function_with_keywords __wrap_namespace "correlate"
                       [||]
                       (Wrap_utils.keyword_args [("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("input", Some(input |> Np.Obj.to_pyobject)); ("weights", Some(weights |> Np.Obj.to_pyobject))])

                  let correlate1d ?axis ?output ?mode ?cval ?origin ~input ~weights () =
                     Py.Module.get_function_with_keywords __wrap_namespace "correlate1d"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin Py.Int.of_int); ("input", Some(input |> Np.Obj.to_pyobject)); ("weights", Some(weights |> Np.Obj.to_pyobject))])

                  let gaussian_filter ?order ?output ?mode ?cval ?truncate ~input ~sigma () =
                     Py.Module.get_function_with_keywords __wrap_namespace "gaussian_filter"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("truncate", Wrap_utils.Option.map truncate Py.Float.of_float); ("input", Some(input |> Np.Obj.to_pyobject)); ("sigma", Some(sigma |> (function
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Sequence_of_scalars x -> Wrap_utils.id x
| `Bool x -> Py.Bool.of_bool x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let gaussian_filter1d ?axis ?order ?output ?mode ?cval ?truncate ~input ~sigma () =
                     Py.Module.get_function_with_keywords __wrap_namespace "gaussian_filter1d"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("order", Wrap_utils.Option.map order Py.Int.of_int); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("truncate", Wrap_utils.Option.map truncate Py.Float.of_float); ("input", Some(input |> Np.Obj.to_pyobject)); ("sigma", Some(sigma |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let gaussian_gradient_magnitude ?output ?mode ?cval ?kwargs ~input ~sigma () =
                     Py.Module.get_function_with_keywords __wrap_namespace "gaussian_gradient_magnitude"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("input", Some(input |> Np.Obj.to_pyobject)); ("sigma", Some(sigma |> (function
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Sequence_of_scalars x -> Wrap_utils.id x
| `Bool x -> Py.Bool.of_bool x
)))]) (match kwargs with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let gaussian_laplace ?output ?mode ?cval ?kwargs ~input ~sigma () =
                     Py.Module.get_function_with_keywords __wrap_namespace "gaussian_laplace"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("input", Some(input |> Np.Obj.to_pyobject)); ("sigma", Some(sigma |> (function
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Sequence_of_scalars x -> Wrap_utils.id x
| `Bool x -> Py.Bool.of_bool x
)))]) (match kwargs with None -> [] | Some x -> x))

                  let generic_filter ?size ?footprint ?output ?mode ?cval ?origin ?extra_arguments ?extra_keywords ~input ~function_ () =
                     Py.Module.get_function_with_keywords __wrap_namespace "generic_filter"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (function
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
)); ("footprint", Wrap_utils.Option.map footprint Np.Obj.to_pyobject); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("extra_arguments", extra_arguments); ("extra_keywords", extra_keywords); ("input", Some(input |> Np.Obj.to_pyobject)); ("function", Some(function_ |> (function
| `Callable x -> Wrap_utils.id x
| `Scipy_LowLevelCallable x -> Wrap_utils.id x
)))])

                  let generic_filter1d ?axis ?output ?mode ?cval ?origin ?extra_arguments ?extra_keywords ~input ~function_ ~filter_size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "generic_filter1d"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin Py.Int.of_int); ("extra_arguments", extra_arguments); ("extra_keywords", extra_keywords); ("input", Some(input |> Np.Obj.to_pyobject)); ("function", Some(function_ |> (function
| `Callable x -> Wrap_utils.id x
| `Scipy_LowLevelCallable x -> Wrap_utils.id x
))); ("filter_size", Some(filter_size |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])

                  let generic_gradient_magnitude ?output ?mode ?cval ?extra_arguments ?extra_keywords ~input ~derivative () =
                     Py.Module.get_function_with_keywords __wrap_namespace "generic_gradient_magnitude"
                       [||]
                       (Wrap_utils.keyword_args [("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("extra_arguments", extra_arguments); ("extra_keywords", extra_keywords); ("input", Some(input |> Np.Obj.to_pyobject)); ("derivative", Some(derivative ))])

                  let generic_laplace ?output ?mode ?cval ?extra_arguments ?extra_keywords ~input ~derivative2 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "generic_laplace"
                       [||]
                       (Wrap_utils.keyword_args [("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("extra_arguments", extra_arguments); ("extra_keywords", extra_keywords); ("input", Some(input |> Np.Obj.to_pyobject)); ("derivative2", Some(derivative2 ))])

                  let laplace ?output ?mode ?cval ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "laplace"
                       [||]
                       (Wrap_utils.keyword_args [("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("input", Some(input |> Np.Obj.to_pyobject))])

                  let maximum_filter ?size ?footprint ?output ?mode ?cval ?origin ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "maximum_filter"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (function
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
)); ("footprint", Wrap_utils.Option.map footprint Np.Obj.to_pyobject); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let maximum_filter1d ?axis ?output ?mode ?cval ?origin ~input ~size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "maximum_filter1d"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin Py.Int.of_int); ("input", Some(input |> Np.Obj.to_pyobject)); ("size", Some(size |> Py.Int.of_int))])
                       |> (fun py -> if Py.is_none py then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) py))
                  let median_filter ?size ?footprint ?output ?mode ?cval ?origin ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "median_filter"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (function
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
)); ("footprint", Wrap_utils.Option.map footprint Np.Obj.to_pyobject); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let minimum_filter ?size ?footprint ?output ?mode ?cval ?origin ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "minimum_filter"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (function
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
)); ("footprint", Wrap_utils.Option.map footprint Np.Obj.to_pyobject); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let minimum_filter1d ?axis ?output ?mode ?cval ?origin ~input ~size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "minimum_filter1d"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin Py.Int.of_int); ("input", Some(input |> Np.Obj.to_pyobject)); ("size", Some(size |> Py.Int.of_int))])

                  let percentile_filter ?size ?footprint ?output ?mode ?cval ?origin ~input ~percentile () =
                     Py.Module.get_function_with_keywords __wrap_namespace "percentile_filter"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (function
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
)); ("footprint", Wrap_utils.Option.map footprint Np.Obj.to_pyobject); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("input", Some(input |> Np.Obj.to_pyobject)); ("percentile", Some(percentile |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let prewitt ?axis ?output ?mode ?cval ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "prewitt"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("input", Some(input |> Np.Obj.to_pyobject))])

                  let rank_filter ?size ?footprint ?output ?mode ?cval ?origin ~input ~rank () =
                     Py.Module.get_function_with_keywords __wrap_namespace "rank_filter"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (function
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
)); ("footprint", Wrap_utils.Option.map footprint Np.Obj.to_pyobject); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("input", Some(input |> Np.Obj.to_pyobject)); ("rank", Some(rank |> Py.Int.of_int))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let sobel ?axis ?output ?mode ?cval ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "sobel"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("input", Some(input |> Np.Obj.to_pyobject))])

                  let uniform_filter ?size ?output ?mode ?cval ?origin ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "uniform_filter"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let uniform_filter1d ?axis ?output ?mode ?cval ?origin ~input ~size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "uniform_filter1d"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin Py.Int.of_int); ("input", Some(input |> Np.Obj.to_pyobject)); ("size", Some(size |> Py.Int.of_int))])


end
module Fourier = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.ndimage.fourier"

let get_py name = Py.Module.get __wrap_namespace name
                  let fourier_ellipsoid ?n ?axis ?output ~input ~size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fourier_ellipsoid"
                       [||]
                       (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("input", Some(input |> Np.Obj.to_pyobject)); ("size", Some(size |> (function
| `Sequence x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let fourier_gaussian ?n ?axis ?output ~input ~sigma () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fourier_gaussian"
                       [||]
                       (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("input", Some(input |> Np.Obj.to_pyobject)); ("sigma", Some(sigma |> (function
| `Sequence x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let fourier_shift ?n ?axis ?output ~input ~shift () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fourier_shift"
                       [||]
                       (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("input", Some(input |> Np.Obj.to_pyobject)); ("shift", Some(shift |> (function
| `Sequence x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let fourier_uniform ?n ?axis ?output ~input ~size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fourier_uniform"
                       [||]
                       (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("input", Some(input |> Np.Obj.to_pyobject)); ("size", Some(size |> (function
| `Sequence x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
module Interpolation = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.ndimage.interpolation"

let get_py name = Py.Module.get __wrap_namespace name
                  let affine_transform ?offset ?output_shape ?output ?order ?mode ?cval ?prefilter ~input ~matrix () =
                     Py.Module.get_function_with_keywords __wrap_namespace "affine_transform"
                       [||]
                       (Wrap_utils.keyword_args [("offset", Wrap_utils.Option.map offset (function
| `Sequence x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)); ("output_shape", Wrap_utils.Option.map output_shape (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("order", Wrap_utils.Option.map order Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("prefilter", Wrap_utils.Option.map prefilter Py.Bool.of_bool); ("input", Some(input |> Np.Obj.to_pyobject)); ("matrix", Some(matrix |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let docfiller f =
   Py.Module.get_function_with_keywords __wrap_namespace "docfiller"
     [||]
     (Wrap_utils.keyword_args [("f", Some(f ))])

                  let geometric_transform ?output_shape ?output ?order ?mode ?cval ?prefilter ?extra_arguments ?extra_keywords ~input ~mapping () =
                     Py.Module.get_function_with_keywords __wrap_namespace "geometric_transform"
                       [||]
                       (Wrap_utils.keyword_args [("output_shape", Wrap_utils.Option.map output_shape (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("order", Wrap_utils.Option.map order Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("prefilter", Wrap_utils.Option.map prefilter Py.Bool.of_bool); ("extra_arguments", extra_arguments); ("extra_keywords", extra_keywords); ("input", Some(input |> Np.Obj.to_pyobject)); ("mapping", Some(mapping |> (function
| `Callable x -> Wrap_utils.id x
| `Scipy_LowLevelCallable x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let map_coordinates ?output ?order ?mode ?cval ?prefilter ~input ~coordinates () =
                     Py.Module.get_function_with_keywords __wrap_namespace "map_coordinates"
                       [||]
                       (Wrap_utils.keyword_args [("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("order", Wrap_utils.Option.map order Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("prefilter", Wrap_utils.Option.map prefilter Py.Bool.of_bool); ("input", Some(input |> Np.Obj.to_pyobject)); ("coordinates", Some(coordinates |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let rotate ?axes ?reshape ?output ?order ?mode ?cval ?prefilter ~input ~angle () =
                     Py.Module.get_function_with_keywords __wrap_namespace "rotate"
                       [||]
                       (Wrap_utils.keyword_args [("axes", axes); ("reshape", Wrap_utils.Option.map reshape Py.Bool.of_bool); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("order", Wrap_utils.Option.map order Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("prefilter", Wrap_utils.Option.map prefilter Py.Bool.of_bool); ("input", Some(input |> Np.Obj.to_pyobject)); ("angle", Some(angle |> Py.Float.of_float))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let shift ?output ?order ?mode ?cval ?prefilter ~input ~shift () =
                     Py.Module.get_function_with_keywords __wrap_namespace "shift"
                       [||]
                       (Wrap_utils.keyword_args [("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("order", Wrap_utils.Option.map order Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("prefilter", Wrap_utils.Option.map prefilter Py.Bool.of_bool); ("input", Some(input |> Np.Obj.to_pyobject)); ("shift", Some(shift |> (function
| `Sequence x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let spline_filter ?order ?output ?mode ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "spline_filter"
     [||]
     (Wrap_utils.keyword_args [("order", order); ("output", output); ("mode", mode); ("input", Some(input ))])

                  let spline_filter1d ?order ?axis ?output ?mode ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "spline_filter1d"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let zoom ?output ?order ?mode ?cval ?prefilter ~input ~zoom () =
                     Py.Module.get_function_with_keywords __wrap_namespace "zoom"
                       [||]
                       (Wrap_utils.keyword_args [("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("order", Wrap_utils.Option.map order Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("prefilter", Wrap_utils.Option.map prefilter Py.Bool.of_bool); ("input", Some(input |> Np.Obj.to_pyobject)); ("zoom", Some(zoom |> (function
| `Sequence x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
module Measurements = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.ndimage.measurements"

let get_py name = Py.Module.get __wrap_namespace name
                  let center_of_mass ?labels ?index ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "center_of_mass"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("index", Wrap_utils.Option.map index (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)); ("input", Some(input |> Np.Obj.to_pyobject))])

                  let extrema ?labels ?index ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "extrema"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("index", Wrap_utils.Option.map index (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)); ("input", Some(input |> Np.Obj.to_pyobject))])

let find_objects ?max_label ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "find_objects"
     [||]
     (Wrap_utils.keyword_args [("max_label", Wrap_utils.Option.map max_label Py.Int.of_int); ("input", Some(input ))])

                  let histogram ?labels ?index ~input ~min ~max ~bins () =
                     Py.Module.get_function_with_keywords __wrap_namespace "histogram"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("index", Wrap_utils.Option.map index (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)); ("input", Some(input |> Np.Obj.to_pyobject)); ("min", Some(min )); ("max", Some(max )); ("bins", Some(bins |> Py.Int.of_int))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let label ?structure ?output ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "label"
     [||]
     (Wrap_utils.keyword_args [("structure", Wrap_utils.Option.map structure Np.Obj.to_pyobject); ("output", output); ("input", Some(input |> Np.Obj.to_pyobject))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let labeled_comprehension ?pass_positions ~input ~labels ~index ~func ~out_dtype ~default () =
                     Py.Module.get_function_with_keywords __wrap_namespace "labeled_comprehension"
                       [||]
                       (Wrap_utils.keyword_args [("pass_positions", Wrap_utils.Option.map pass_positions Py.Bool.of_bool); ("input", Some(input |> Np.Obj.to_pyobject)); ("labels", Some(labels |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `None -> Py.none
))); ("index", Some(index |> (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
| `None -> Py.none
))); ("func", Some(func )); ("out_dtype", Some(out_dtype |> Np.Dtype.to_pyobject)); ("default", Some(default |> (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
| `None -> Py.none
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let maximum ?labels ?index ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "maximum"
     [||]
     (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("index", Wrap_utils.Option.map index Np.Obj.to_pyobject); ("input", Some(input |> Np.Obj.to_pyobject))])

let maximum_position ?labels ?index ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "maximum_position"
     [||]
     (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("index", Wrap_utils.Option.map index Np.Obj.to_pyobject); ("input", Some(input |> Np.Obj.to_pyobject))])

                  let mean ?labels ?index ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "mean"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("index", Wrap_utils.Option.map index (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let median ?labels ?index ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "median"
     [||]
     (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("index", Wrap_utils.Option.map index Np.Obj.to_pyobject); ("input", Some(input |> Np.Obj.to_pyobject))])

let minimum ?labels ?index ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "minimum"
     [||]
     (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("index", Wrap_utils.Option.map index Np.Obj.to_pyobject); ("input", Some(input |> Np.Obj.to_pyobject))])

let minimum_position ?labels ?index ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "minimum_position"
     [||]
     (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("index", Wrap_utils.Option.map index Np.Obj.to_pyobject); ("input", Some(input |> Np.Obj.to_pyobject))])

                  let standard_deviation ?labels ?index ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "standard_deviation"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("index", Wrap_utils.Option.map index (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)); ("input", Some(input |> Np.Obj.to_pyobject))])

let sum ?labels ?index ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "sum"
     [||]
     (Wrap_utils.keyword_args [("labels", labels); ("index", Wrap_utils.Option.map index Np.Obj.to_pyobject); ("input", Some(input |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let variance ?labels ?index ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "variance"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("index", Wrap_utils.Option.map index (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)); ("input", Some(input |> Np.Obj.to_pyobject))])

let watershed_ift ?structure ?output ~input ~markers () =
   Py.Module.get_function_with_keywords __wrap_namespace "watershed_ift"
     [||]
     (Wrap_utils.keyword_args [("structure", structure); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("input", Some(input |> Np.Obj.to_pyobject)); ("markers", Some(markers |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
module Morphology = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.ndimage.morphology"

let get_py name = Py.Module.get __wrap_namespace name
let binary_closing ?structure ?iterations ?output ?origin ?mask ?border_value ?brute_force ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "binary_closing"
     [||]
     (Wrap_utils.keyword_args [("structure", Wrap_utils.Option.map structure Np.Obj.to_pyobject); ("iterations", Wrap_utils.Option.map iterations Py.Int.of_int); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("origin", Wrap_utils.Option.map origin (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("mask", Wrap_utils.Option.map mask Np.Obj.to_pyobject); ("border_value", Wrap_utils.Option.map border_value Py.Int.of_int); ("brute_force", Wrap_utils.Option.map brute_force Py.Bool.of_bool); ("input", Some(input |> Np.Obj.to_pyobject))])

let binary_dilation ?structure ?iterations ?mask ?output ?border_value ?origin ?brute_force ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "binary_dilation"
     [||]
     (Wrap_utils.keyword_args [("structure", Wrap_utils.Option.map structure Np.Obj.to_pyobject); ("iterations", Wrap_utils.Option.map iterations Py.Int.of_int); ("mask", Wrap_utils.Option.map mask Np.Obj.to_pyobject); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("border_value", Wrap_utils.Option.map border_value Py.Int.of_int); ("origin", Wrap_utils.Option.map origin (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("brute_force", Wrap_utils.Option.map brute_force Py.Bool.of_bool); ("input", Some(input |> Np.Obj.to_pyobject))])

let binary_erosion ?structure ?iterations ?mask ?output ?border_value ?origin ?brute_force ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "binary_erosion"
     [||]
     (Wrap_utils.keyword_args [("structure", Wrap_utils.Option.map structure Np.Obj.to_pyobject); ("iterations", Wrap_utils.Option.map iterations Py.Int.of_int); ("mask", Wrap_utils.Option.map mask Np.Obj.to_pyobject); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("border_value", Wrap_utils.Option.map border_value Py.Int.of_int); ("origin", Wrap_utils.Option.map origin (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("brute_force", Wrap_utils.Option.map brute_force Py.Bool.of_bool); ("input", Some(input |> Np.Obj.to_pyobject))])

let binary_fill_holes ?structure ?output ?origin ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "binary_fill_holes"
     [||]
     (Wrap_utils.keyword_args [("structure", Wrap_utils.Option.map structure Np.Obj.to_pyobject); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("origin", Wrap_utils.Option.map origin (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("input", Some(input |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let binary_hit_or_miss ?structure1 ?structure2 ?output ?origin1 ?origin2 ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "binary_hit_or_miss"
     [||]
     (Wrap_utils.keyword_args [("structure1", Wrap_utils.Option.map structure1 Np.Obj.to_pyobject); ("structure2", Wrap_utils.Option.map structure2 Np.Obj.to_pyobject); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("origin1", Wrap_utils.Option.map origin1 (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("origin2", Wrap_utils.Option.map origin2 (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("input", Some(input |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let binary_opening ?structure ?iterations ?output ?origin ?mask ?border_value ?brute_force ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "binary_opening"
     [||]
     (Wrap_utils.keyword_args [("structure", Wrap_utils.Option.map structure Np.Obj.to_pyobject); ("iterations", Wrap_utils.Option.map iterations Py.Int.of_int); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("origin", Wrap_utils.Option.map origin (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("mask", Wrap_utils.Option.map mask Np.Obj.to_pyobject); ("border_value", Wrap_utils.Option.map border_value Py.Int.of_int); ("brute_force", Wrap_utils.Option.map brute_force Py.Bool.of_bool); ("input", Some(input |> Np.Obj.to_pyobject))])

let binary_propagation ?structure ?mask ?output ?border_value ?origin ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "binary_propagation"
     [||]
     (Wrap_utils.keyword_args [("structure", Wrap_utils.Option.map structure Np.Obj.to_pyobject); ("mask", Wrap_utils.Option.map mask Np.Obj.to_pyobject); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("border_value", Wrap_utils.Option.map border_value Py.Int.of_int); ("origin", Wrap_utils.Option.map origin (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("input", Some(input |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let black_tophat ?size ?footprint ?structure ?output ?mode ?cval ?origin ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "black_tophat"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("footprint", footprint); ("structure", structure); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let distance_transform_bf ?metric ?sampling ?return_distances ?return_indices ?distances ?indices ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "distance_transform_bf"
                       [||]
                       (Wrap_utils.keyword_args [("metric", Wrap_utils.Option.map metric Py.String.of_string); ("sampling", Wrap_utils.Option.map sampling (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)); ("return_distances", Wrap_utils.Option.map return_distances Py.Bool.of_bool); ("return_indices", Wrap_utils.Option.map return_indices Py.Bool.of_bool); ("distances", distances); ("indices", indices); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let distance_transform_cdt ?metric ?return_distances ?return_indices ?distances ?indices ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "distance_transform_cdt"
                       [||]
                       (Wrap_utils.keyword_args [("metric", Wrap_utils.Option.map metric (function
| `Chessboard -> Py.String.of_string "chessboard"
| `Taxicab -> Py.String.of_string "taxicab"
)); ("return_distances", return_distances); ("return_indices", return_indices); ("distances", distances); ("indices", indices); ("input", Some(input |> Np.Obj.to_pyobject))])

                  let distance_transform_edt ?sampling ?return_distances ?return_indices ?distances ?indices ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "distance_transform_edt"
                       [||]
                       (Wrap_utils.keyword_args [("sampling", Wrap_utils.Option.map sampling (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
| `Sequence_of_same x -> Wrap_utils.id x
)); ("return_distances", Wrap_utils.Option.map return_distances Py.Bool.of_bool); ("return_indices", Wrap_utils.Option.map return_indices Py.Bool.of_bool); ("distances", Wrap_utils.Option.map distances Np.Obj.to_pyobject); ("indices", Wrap_utils.Option.map indices Np.Obj.to_pyobject); ("input", Some(input |> Np.Obj.to_pyobject))])

let generate_binary_structure ~rank ~connectivity () =
   Py.Module.get_function_with_keywords __wrap_namespace "generate_binary_structure"
     [||]
     (Wrap_utils.keyword_args [("rank", Some(rank |> Py.Int.of_int)); ("connectivity", Some(connectivity |> Py.Int.of_int))])

                  let grey_closing ?size ?footprint ?structure ?output ?mode ?cval ?origin ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "grey_closing"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("footprint", footprint); ("structure", structure); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let grey_dilation ?size ?footprint ?structure ?output ?mode ?cval ?origin ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "grey_dilation"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("footprint", footprint); ("structure", structure); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let grey_erosion ?size ?footprint ?structure ?output ?mode ?cval ?origin ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "grey_erosion"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("footprint", footprint); ("structure", structure); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let grey_opening ?size ?footprint ?structure ?output ?mode ?cval ?origin ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "grey_opening"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("footprint", footprint); ("structure", structure); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let iterate_structure ?origin ~structure ~iterations () =
   Py.Module.get_function_with_keywords __wrap_namespace "iterate_structure"
     [||]
     (Wrap_utils.keyword_args [("origin", origin); ("structure", Some(structure |> Np.Obj.to_pyobject)); ("iterations", Some(iterations |> Py.Int.of_int))])

                  let morphological_gradient ?size ?footprint ?structure ?output ?mode ?cval ?origin ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "morphological_gradient"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("footprint", footprint); ("structure", structure); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let morphological_laplace ?size ?footprint ?structure ?output ?mode ?cval ?origin ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "morphological_laplace"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)); ("footprint", Wrap_utils.Option.map footprint (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Bool x -> Py.Bool.of_bool x
)); ("structure", structure); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", origin); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let white_tophat ?size ?footprint ?structure ?output ?mode ?cval ?origin ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "white_tophat"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("footprint", footprint); ("structure", structure); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
                  let affine_transform ?offset ?output_shape ?output ?order ?mode ?cval ?prefilter ~input ~matrix () =
                     Py.Module.get_function_with_keywords __wrap_namespace "affine_transform"
                       [||]
                       (Wrap_utils.keyword_args [("offset", Wrap_utils.Option.map offset (function
| `Sequence x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)); ("output_shape", Wrap_utils.Option.map output_shape (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("order", Wrap_utils.Option.map order Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("prefilter", Wrap_utils.Option.map prefilter Py.Bool.of_bool); ("input", Some(input |> Np.Obj.to_pyobject)); ("matrix", Some(matrix |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let binary_closing ?structure ?iterations ?output ?origin ?mask ?border_value ?brute_force ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "binary_closing"
     [||]
     (Wrap_utils.keyword_args [("structure", Wrap_utils.Option.map structure Np.Obj.to_pyobject); ("iterations", Wrap_utils.Option.map iterations Py.Int.of_int); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("origin", Wrap_utils.Option.map origin (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("mask", Wrap_utils.Option.map mask Np.Obj.to_pyobject); ("border_value", Wrap_utils.Option.map border_value Py.Int.of_int); ("brute_force", Wrap_utils.Option.map brute_force Py.Bool.of_bool); ("input", Some(input |> Np.Obj.to_pyobject))])

let binary_dilation ?structure ?iterations ?mask ?output ?border_value ?origin ?brute_force ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "binary_dilation"
     [||]
     (Wrap_utils.keyword_args [("structure", Wrap_utils.Option.map structure Np.Obj.to_pyobject); ("iterations", Wrap_utils.Option.map iterations Py.Int.of_int); ("mask", Wrap_utils.Option.map mask Np.Obj.to_pyobject); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("border_value", Wrap_utils.Option.map border_value Py.Int.of_int); ("origin", Wrap_utils.Option.map origin (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("brute_force", Wrap_utils.Option.map brute_force Py.Bool.of_bool); ("input", Some(input |> Np.Obj.to_pyobject))])

let binary_erosion ?structure ?iterations ?mask ?output ?border_value ?origin ?brute_force ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "binary_erosion"
     [||]
     (Wrap_utils.keyword_args [("structure", Wrap_utils.Option.map structure Np.Obj.to_pyobject); ("iterations", Wrap_utils.Option.map iterations Py.Int.of_int); ("mask", Wrap_utils.Option.map mask Np.Obj.to_pyobject); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("border_value", Wrap_utils.Option.map border_value Py.Int.of_int); ("origin", Wrap_utils.Option.map origin (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("brute_force", Wrap_utils.Option.map brute_force Py.Bool.of_bool); ("input", Some(input |> Np.Obj.to_pyobject))])

let binary_fill_holes ?structure ?output ?origin ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "binary_fill_holes"
     [||]
     (Wrap_utils.keyword_args [("structure", Wrap_utils.Option.map structure Np.Obj.to_pyobject); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("origin", Wrap_utils.Option.map origin (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("input", Some(input |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let binary_hit_or_miss ?structure1 ?structure2 ?output ?origin1 ?origin2 ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "binary_hit_or_miss"
     [||]
     (Wrap_utils.keyword_args [("structure1", Wrap_utils.Option.map structure1 Np.Obj.to_pyobject); ("structure2", Wrap_utils.Option.map structure2 Np.Obj.to_pyobject); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("origin1", Wrap_utils.Option.map origin1 (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("origin2", Wrap_utils.Option.map origin2 (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("input", Some(input |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let binary_opening ?structure ?iterations ?output ?origin ?mask ?border_value ?brute_force ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "binary_opening"
     [||]
     (Wrap_utils.keyword_args [("structure", Wrap_utils.Option.map structure Np.Obj.to_pyobject); ("iterations", Wrap_utils.Option.map iterations Py.Int.of_int); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("origin", Wrap_utils.Option.map origin (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("mask", Wrap_utils.Option.map mask Np.Obj.to_pyobject); ("border_value", Wrap_utils.Option.map border_value Py.Int.of_int); ("brute_force", Wrap_utils.Option.map brute_force Py.Bool.of_bool); ("input", Some(input |> Np.Obj.to_pyobject))])

let binary_propagation ?structure ?mask ?output ?border_value ?origin ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "binary_propagation"
     [||]
     (Wrap_utils.keyword_args [("structure", Wrap_utils.Option.map structure Np.Obj.to_pyobject); ("mask", Wrap_utils.Option.map mask Np.Obj.to_pyobject); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("border_value", Wrap_utils.Option.map border_value Py.Int.of_int); ("origin", Wrap_utils.Option.map origin (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("input", Some(input |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let black_tophat ?size ?footprint ?structure ?output ?mode ?cval ?origin ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "black_tophat"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("footprint", footprint); ("structure", structure); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let center_of_mass ?labels ?index ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "center_of_mass"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("index", Wrap_utils.Option.map index (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)); ("input", Some(input |> Np.Obj.to_pyobject))])

                  let convolve ?output ?mode ?cval ?origin ~input ~weights () =
                     Py.Module.get_function_with_keywords __wrap_namespace "convolve"
                       [||]
                       (Wrap_utils.keyword_args [("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("input", Some(input |> Np.Obj.to_pyobject)); ("weights", Some(weights |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let convolve1d ?axis ?output ?mode ?cval ?origin ~input ~weights () =
                     Py.Module.get_function_with_keywords __wrap_namespace "convolve1d"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin Py.Int.of_int); ("input", Some(input |> Np.Obj.to_pyobject)); ("weights", Some(weights |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let correlate ?output ?mode ?cval ?origin ~input ~weights () =
                     Py.Module.get_function_with_keywords __wrap_namespace "correlate"
                       [||]
                       (Wrap_utils.keyword_args [("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("input", Some(input |> Np.Obj.to_pyobject)); ("weights", Some(weights |> Np.Obj.to_pyobject))])

                  let correlate1d ?axis ?output ?mode ?cval ?origin ~input ~weights () =
                     Py.Module.get_function_with_keywords __wrap_namespace "correlate1d"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin Py.Int.of_int); ("input", Some(input |> Np.Obj.to_pyobject)); ("weights", Some(weights |> Np.Obj.to_pyobject))])

                  let distance_transform_bf ?metric ?sampling ?return_distances ?return_indices ?distances ?indices ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "distance_transform_bf"
                       [||]
                       (Wrap_utils.keyword_args [("metric", Wrap_utils.Option.map metric Py.String.of_string); ("sampling", Wrap_utils.Option.map sampling (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)); ("return_distances", Wrap_utils.Option.map return_distances Py.Bool.of_bool); ("return_indices", Wrap_utils.Option.map return_indices Py.Bool.of_bool); ("distances", distances); ("indices", indices); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let distance_transform_cdt ?metric ?return_distances ?return_indices ?distances ?indices ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "distance_transform_cdt"
                       [||]
                       (Wrap_utils.keyword_args [("metric", Wrap_utils.Option.map metric (function
| `Chessboard -> Py.String.of_string "chessboard"
| `Taxicab -> Py.String.of_string "taxicab"
)); ("return_distances", return_distances); ("return_indices", return_indices); ("distances", distances); ("indices", indices); ("input", Some(input |> Np.Obj.to_pyobject))])

                  let distance_transform_edt ?sampling ?return_distances ?return_indices ?distances ?indices ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "distance_transform_edt"
                       [||]
                       (Wrap_utils.keyword_args [("sampling", Wrap_utils.Option.map sampling (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
| `Sequence_of_same x -> Wrap_utils.id x
)); ("return_distances", Wrap_utils.Option.map return_distances Py.Bool.of_bool); ("return_indices", Wrap_utils.Option.map return_indices Py.Bool.of_bool); ("distances", Wrap_utils.Option.map distances Np.Obj.to_pyobject); ("indices", Wrap_utils.Option.map indices Np.Obj.to_pyobject); ("input", Some(input |> Np.Obj.to_pyobject))])

                  let extrema ?labels ?index ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "extrema"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("index", Wrap_utils.Option.map index (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)); ("input", Some(input |> Np.Obj.to_pyobject))])

let find_objects ?max_label ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "find_objects"
     [||]
     (Wrap_utils.keyword_args [("max_label", Wrap_utils.Option.map max_label Py.Int.of_int); ("input", Some(input ))])

                  let fourier_ellipsoid ?n ?axis ?output ~input ~size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fourier_ellipsoid"
                       [||]
                       (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("input", Some(input |> Np.Obj.to_pyobject)); ("size", Some(size |> (function
| `Sequence x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let fourier_gaussian ?n ?axis ?output ~input ~sigma () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fourier_gaussian"
                       [||]
                       (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("input", Some(input |> Np.Obj.to_pyobject)); ("sigma", Some(sigma |> (function
| `Sequence x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let fourier_shift ?n ?axis ?output ~input ~shift () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fourier_shift"
                       [||]
                       (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("input", Some(input |> Np.Obj.to_pyobject)); ("shift", Some(shift |> (function
| `Sequence x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let fourier_uniform ?n ?axis ?output ~input ~size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fourier_uniform"
                       [||]
                       (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("input", Some(input |> Np.Obj.to_pyobject)); ("size", Some(size |> (function
| `Sequence x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let gaussian_filter ?order ?output ?mode ?cval ?truncate ~input ~sigma () =
                     Py.Module.get_function_with_keywords __wrap_namespace "gaussian_filter"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("truncate", Wrap_utils.Option.map truncate Py.Float.of_float); ("input", Some(input |> Np.Obj.to_pyobject)); ("sigma", Some(sigma |> (function
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Sequence_of_scalars x -> Wrap_utils.id x
| `Bool x -> Py.Bool.of_bool x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let gaussian_filter1d ?axis ?order ?output ?mode ?cval ?truncate ~input ~sigma () =
                     Py.Module.get_function_with_keywords __wrap_namespace "gaussian_filter1d"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("order", Wrap_utils.Option.map order Py.Int.of_int); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("truncate", Wrap_utils.Option.map truncate Py.Float.of_float); ("input", Some(input |> Np.Obj.to_pyobject)); ("sigma", Some(sigma |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let gaussian_gradient_magnitude ?output ?mode ?cval ?kwargs ~input ~sigma () =
                     Py.Module.get_function_with_keywords __wrap_namespace "gaussian_gradient_magnitude"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("input", Some(input |> Np.Obj.to_pyobject)); ("sigma", Some(sigma |> (function
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Sequence_of_scalars x -> Wrap_utils.id x
| `Bool x -> Py.Bool.of_bool x
)))]) (match kwargs with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let gaussian_laplace ?output ?mode ?cval ?kwargs ~input ~sigma () =
                     Py.Module.get_function_with_keywords __wrap_namespace "gaussian_laplace"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("input", Some(input |> Np.Obj.to_pyobject)); ("sigma", Some(sigma |> (function
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Sequence_of_scalars x -> Wrap_utils.id x
| `Bool x -> Py.Bool.of_bool x
)))]) (match kwargs with None -> [] | Some x -> x))

let generate_binary_structure ~rank ~connectivity () =
   Py.Module.get_function_with_keywords __wrap_namespace "generate_binary_structure"
     [||]
     (Wrap_utils.keyword_args [("rank", Some(rank |> Py.Int.of_int)); ("connectivity", Some(connectivity |> Py.Int.of_int))])

                  let generic_filter ?size ?footprint ?output ?mode ?cval ?origin ?extra_arguments ?extra_keywords ~input ~function_ () =
                     Py.Module.get_function_with_keywords __wrap_namespace "generic_filter"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (function
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
)); ("footprint", Wrap_utils.Option.map footprint Np.Obj.to_pyobject); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("extra_arguments", extra_arguments); ("extra_keywords", extra_keywords); ("input", Some(input |> Np.Obj.to_pyobject)); ("function", Some(function_ |> (function
| `Callable x -> Wrap_utils.id x
| `Scipy_LowLevelCallable x -> Wrap_utils.id x
)))])

                  let generic_filter1d ?axis ?output ?mode ?cval ?origin ?extra_arguments ?extra_keywords ~input ~function_ ~filter_size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "generic_filter1d"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin Py.Int.of_int); ("extra_arguments", extra_arguments); ("extra_keywords", extra_keywords); ("input", Some(input |> Np.Obj.to_pyobject)); ("function", Some(function_ |> (function
| `Callable x -> Wrap_utils.id x
| `Scipy_LowLevelCallable x -> Wrap_utils.id x
))); ("filter_size", Some(filter_size |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])

                  let generic_gradient_magnitude ?output ?mode ?cval ?extra_arguments ?extra_keywords ~input ~derivative () =
                     Py.Module.get_function_with_keywords __wrap_namespace "generic_gradient_magnitude"
                       [||]
                       (Wrap_utils.keyword_args [("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("extra_arguments", extra_arguments); ("extra_keywords", extra_keywords); ("input", Some(input |> Np.Obj.to_pyobject)); ("derivative", Some(derivative ))])

                  let generic_laplace ?output ?mode ?cval ?extra_arguments ?extra_keywords ~input ~derivative2 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "generic_laplace"
                       [||]
                       (Wrap_utils.keyword_args [("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("extra_arguments", extra_arguments); ("extra_keywords", extra_keywords); ("input", Some(input |> Np.Obj.to_pyobject)); ("derivative2", Some(derivative2 ))])

                  let geometric_transform ?output_shape ?output ?order ?mode ?cval ?prefilter ?extra_arguments ?extra_keywords ~input ~mapping () =
                     Py.Module.get_function_with_keywords __wrap_namespace "geometric_transform"
                       [||]
                       (Wrap_utils.keyword_args [("output_shape", Wrap_utils.Option.map output_shape (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("order", Wrap_utils.Option.map order Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("prefilter", Wrap_utils.Option.map prefilter Py.Bool.of_bool); ("extra_arguments", extra_arguments); ("extra_keywords", extra_keywords); ("input", Some(input |> Np.Obj.to_pyobject)); ("mapping", Some(mapping |> (function
| `Callable x -> Wrap_utils.id x
| `Scipy_LowLevelCallable x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let grey_closing ?size ?footprint ?structure ?output ?mode ?cval ?origin ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "grey_closing"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("footprint", footprint); ("structure", structure); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let grey_dilation ?size ?footprint ?structure ?output ?mode ?cval ?origin ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "grey_dilation"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("footprint", footprint); ("structure", structure); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let grey_erosion ?size ?footprint ?structure ?output ?mode ?cval ?origin ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "grey_erosion"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("footprint", footprint); ("structure", structure); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let grey_opening ?size ?footprint ?structure ?output ?mode ?cval ?origin ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "grey_opening"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("footprint", footprint); ("structure", structure); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let histogram ?labels ?index ~input ~min ~max ~bins () =
                     Py.Module.get_function_with_keywords __wrap_namespace "histogram"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("index", Wrap_utils.Option.map index (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)); ("input", Some(input |> Np.Obj.to_pyobject)); ("min", Some(min )); ("max", Some(max )); ("bins", Some(bins |> Py.Int.of_int))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let iterate_structure ?origin ~structure ~iterations () =
   Py.Module.get_function_with_keywords __wrap_namespace "iterate_structure"
     [||]
     (Wrap_utils.keyword_args [("origin", origin); ("structure", Some(structure |> Np.Obj.to_pyobject)); ("iterations", Some(iterations |> Py.Int.of_int))])

let label ?structure ?output ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "label"
     [||]
     (Wrap_utils.keyword_args [("structure", Wrap_utils.Option.map structure Np.Obj.to_pyobject); ("output", output); ("input", Some(input |> Np.Obj.to_pyobject))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let labeled_comprehension ?pass_positions ~input ~labels ~index ~func ~out_dtype ~default () =
                     Py.Module.get_function_with_keywords __wrap_namespace "labeled_comprehension"
                       [||]
                       (Wrap_utils.keyword_args [("pass_positions", Wrap_utils.Option.map pass_positions Py.Bool.of_bool); ("input", Some(input |> Np.Obj.to_pyobject)); ("labels", Some(labels |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `None -> Py.none
))); ("index", Some(index |> (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
| `None -> Py.none
))); ("func", Some(func )); ("out_dtype", Some(out_dtype |> Np.Dtype.to_pyobject)); ("default", Some(default |> (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
| `None -> Py.none
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let laplace ?output ?mode ?cval ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "laplace"
                       [||]
                       (Wrap_utils.keyword_args [("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("input", Some(input |> Np.Obj.to_pyobject))])

                  let map_coordinates ?output ?order ?mode ?cval ?prefilter ~input ~coordinates () =
                     Py.Module.get_function_with_keywords __wrap_namespace "map_coordinates"
                       [||]
                       (Wrap_utils.keyword_args [("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("order", Wrap_utils.Option.map order Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("prefilter", Wrap_utils.Option.map prefilter Py.Bool.of_bool); ("input", Some(input |> Np.Obj.to_pyobject)); ("coordinates", Some(coordinates |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let maximum ?labels ?index ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "maximum"
     [||]
     (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("index", Wrap_utils.Option.map index Np.Obj.to_pyobject); ("input", Some(input |> Np.Obj.to_pyobject))])

                  let maximum_filter ?size ?footprint ?output ?mode ?cval ?origin ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "maximum_filter"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (function
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
)); ("footprint", Wrap_utils.Option.map footprint Np.Obj.to_pyobject); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let maximum_filter1d ?axis ?output ?mode ?cval ?origin ~input ~size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "maximum_filter1d"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin Py.Int.of_int); ("input", Some(input |> Np.Obj.to_pyobject)); ("size", Some(size |> Py.Int.of_int))])
                       |> (fun py -> if Py.is_none py then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) py))
let maximum_position ?labels ?index ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "maximum_position"
     [||]
     (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("index", Wrap_utils.Option.map index Np.Obj.to_pyobject); ("input", Some(input |> Np.Obj.to_pyobject))])

                  let mean ?labels ?index ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "mean"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("index", Wrap_utils.Option.map index (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let median ?labels ?index ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "median"
     [||]
     (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("index", Wrap_utils.Option.map index Np.Obj.to_pyobject); ("input", Some(input |> Np.Obj.to_pyobject))])

                  let median_filter ?size ?footprint ?output ?mode ?cval ?origin ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "median_filter"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (function
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
)); ("footprint", Wrap_utils.Option.map footprint Np.Obj.to_pyobject); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let minimum ?labels ?index ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "minimum"
     [||]
     (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("index", Wrap_utils.Option.map index Np.Obj.to_pyobject); ("input", Some(input |> Np.Obj.to_pyobject))])

                  let minimum_filter ?size ?footprint ?output ?mode ?cval ?origin ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "minimum_filter"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (function
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
)); ("footprint", Wrap_utils.Option.map footprint Np.Obj.to_pyobject); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let minimum_filter1d ?axis ?output ?mode ?cval ?origin ~input ~size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "minimum_filter1d"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin Py.Int.of_int); ("input", Some(input |> Np.Obj.to_pyobject)); ("size", Some(size |> Py.Int.of_int))])

let minimum_position ?labels ?index ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "minimum_position"
     [||]
     (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("index", Wrap_utils.Option.map index Np.Obj.to_pyobject); ("input", Some(input |> Np.Obj.to_pyobject))])

                  let morphological_gradient ?size ?footprint ?structure ?output ?mode ?cval ?origin ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "morphological_gradient"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("footprint", footprint); ("structure", structure); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let morphological_laplace ?size ?footprint ?structure ?output ?mode ?cval ?origin ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "morphological_laplace"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)); ("footprint", Wrap_utils.Option.map footprint (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Bool x -> Py.Bool.of_bool x
)); ("structure", structure); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", origin); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let percentile_filter ?size ?footprint ?output ?mode ?cval ?origin ~input ~percentile () =
                     Py.Module.get_function_with_keywords __wrap_namespace "percentile_filter"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (function
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
)); ("footprint", Wrap_utils.Option.map footprint Np.Obj.to_pyobject); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("input", Some(input |> Np.Obj.to_pyobject)); ("percentile", Some(percentile |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let prewitt ?axis ?output ?mode ?cval ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "prewitt"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("input", Some(input |> Np.Obj.to_pyobject))])

                  let rank_filter ?size ?footprint ?output ?mode ?cval ?origin ~input ~rank () =
                     Py.Module.get_function_with_keywords __wrap_namespace "rank_filter"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (function
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
)); ("footprint", Wrap_utils.Option.map footprint Np.Obj.to_pyobject); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("input", Some(input |> Np.Obj.to_pyobject)); ("rank", Some(rank |> Py.Int.of_int))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let rotate ?axes ?reshape ?output ?order ?mode ?cval ?prefilter ~input ~angle () =
                     Py.Module.get_function_with_keywords __wrap_namespace "rotate"
                       [||]
                       (Wrap_utils.keyword_args [("axes", axes); ("reshape", Wrap_utils.Option.map reshape Py.Bool.of_bool); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("order", Wrap_utils.Option.map order Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("prefilter", Wrap_utils.Option.map prefilter Py.Bool.of_bool); ("input", Some(input |> Np.Obj.to_pyobject)); ("angle", Some(angle |> Py.Float.of_float))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let shift ?output ?order ?mode ?cval ?prefilter ~input ~shift () =
                     Py.Module.get_function_with_keywords __wrap_namespace "shift"
                       [||]
                       (Wrap_utils.keyword_args [("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("order", Wrap_utils.Option.map order Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("prefilter", Wrap_utils.Option.map prefilter Py.Bool.of_bool); ("input", Some(input |> Np.Obj.to_pyobject)); ("shift", Some(shift |> (function
| `Sequence x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let sobel ?axis ?output ?mode ?cval ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "sobel"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("input", Some(input |> Np.Obj.to_pyobject))])

let spline_filter ?order ?output ?mode ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "spline_filter"
     [||]
     (Wrap_utils.keyword_args [("order", order); ("output", output); ("mode", mode); ("input", Some(input ))])

                  let spline_filter1d ?order ?axis ?output ?mode ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "spline_filter1d"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let standard_deviation ?labels ?index ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "standard_deviation"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("index", Wrap_utils.Option.map index (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)); ("input", Some(input |> Np.Obj.to_pyobject))])

let sum ?labels ?index ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "sum"
     [||]
     (Wrap_utils.keyword_args [("labels", labels); ("index", Wrap_utils.Option.map index Np.Obj.to_pyobject); ("input", Some(input |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let uniform_filter ?size ?output ?mode ?cval ?origin ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "uniform_filter"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Sequence x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let uniform_filter1d ?axis ?output ?mode ?cval ?origin ~input ~size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "uniform_filter1d"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin Py.Int.of_int); ("input", Some(input |> Np.Obj.to_pyobject)); ("size", Some(size |> Py.Int.of_int))])

                  let variance ?labels ?index ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "variance"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("index", Wrap_utils.Option.map index (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)); ("input", Some(input |> Np.Obj.to_pyobject))])

let watershed_ift ?structure ?output ~input ~markers () =
   Py.Module.get_function_with_keywords __wrap_namespace "watershed_ift"
     [||]
     (Wrap_utils.keyword_args [("structure", structure); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("input", Some(input |> Np.Obj.to_pyobject)); ("markers", Some(markers |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let white_tophat ?size ?footprint ?structure ?output ?mode ?cval ?origin ~input () =
                     Py.Module.get_function_with_keywords __wrap_namespace "white_tophat"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("footprint", footprint); ("structure", structure); ("output", Wrap_utils.Option.map output Np.Obj.to_pyobject); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("origin", Wrap_utils.Option.map origin (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("input", Some(input |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let zoom ?output ?order ?mode ?cval ?prefilter ~input ~zoom () =
                     Py.Module.get_function_with_keywords __wrap_namespace "zoom"
                       [||]
                       (Wrap_utils.keyword_args [("output", Wrap_utils.Option.map output (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("order", Wrap_utils.Option.map order Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Reflect -> Py.String.of_string "reflect"
| `Constant -> Py.String.of_string "constant"
| `Nearest -> Py.String.of_string "nearest"
| `Mirror -> Py.String.of_string "mirror"
| `Wrap -> Py.String.of_string "wrap"
)); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("prefilter", Wrap_utils.Option.map prefilter Py.Bool.of_bool); ("input", Some(input |> Np.Obj.to_pyobject)); ("zoom", Some(zoom |> (function
| `Sequence x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
