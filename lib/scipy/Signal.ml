let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.signal"

let get_py name = Py.Module.get __wrap_namespace name
module BadCoefficients = struct
type tag = [`BadCoefficients]
type t = [`BadCoefficients | `BaseException | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_exception x = (x :> [`BaseException] Obj.t)
let with_traceback ~tb self =
   Py.Module.get_function_with_keywords (to_pyobject self) "with_traceback"
     [||]
     (Wrap_utils.keyword_args [("tb", Some(tb ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module StateSpace = struct
type tag = [`StateSpace]
type t = [`Object | `StateSpace] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs system =
   Py.Module.get_function_with_keywords __wrap_namespace "StateSpace"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id system)])
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let to_ss self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_ss"
     [||]
     []

let to_tf ?kwargs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_tf"
     [||]
     (match kwargs with None -> [] | Some x -> x)

let to_zpk ?kwargs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_zpk"
     [||]
     (match kwargs with None -> [] | Some x -> x)

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module TransferFunction = struct
type tag = [`TransferFunction]
type t = [`Object | `TransferFunction] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs system =
   Py.Module.get_function_with_keywords __wrap_namespace "TransferFunction"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id system)])
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let to_ss self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_ss"
     [||]
     []

let to_tf self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_tf"
     [||]
     []

let to_zpk self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_zpk"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ZerosPolesGain = struct
type tag = [`ZerosPolesGain]
type t = [`Object | `ZerosPolesGain] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs system =
   Py.Module.get_function_with_keywords __wrap_namespace "ZerosPolesGain"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id system)])
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let to_ss self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_ss"
     [||]
     []

let to_tf self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_tf"
     [||]
     []

let to_zpk self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_zpk"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Dlti = struct
type tag = [`Dlti]
type t = [`Dlti | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs system =
   Py.Module.get_function_with_keywords __wrap_namespace "dlti"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id system)])
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let bode ?w ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "bode"
     [||]
     (Wrap_utils.keyword_args [("w", w); ("n", n)])

let freqresp ?w ?n ?whole self =
   Py.Module.get_function_with_keywords (to_pyobject self) "freqresp"
     [||]
     (Wrap_utils.keyword_args [("w", w); ("n", n); ("whole", whole)])

let impulse ?x0 ?t ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "impulse"
     [||]
     (Wrap_utils.keyword_args [("x0", x0); ("t", t); ("n", n)])

let output ?x0 ~u ~t self =
   Py.Module.get_function_with_keywords (to_pyobject self) "output"
     [||]
     (Wrap_utils.keyword_args [("x0", x0); ("u", Some(u )); ("t", Some(t ))])

let step ?x0 ?t ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "step"
     [||]
     (Wrap_utils.keyword_args [("x0", x0); ("t", t); ("n", n)])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Lti = struct
type tag = [`Lti]
type t = [`Lti | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create system =
   Py.Module.get_function_with_keywords __wrap_namespace "lti"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id system)])
     []
     |> of_pyobject
let bode ?w ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "bode"
     [||]
     (Wrap_utils.keyword_args [("w", w); ("n", n)])

let freqresp ?w ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "freqresp"
     [||]
     (Wrap_utils.keyword_args [("w", w); ("n", n)])

let impulse ?x0 ?t ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "impulse"
     [||]
     (Wrap_utils.keyword_args [("X0", x0); ("T", t); ("N", n)])

let output ?x0 ~u ~t self =
   Py.Module.get_function_with_keywords (to_pyobject self) "output"
     [||]
     (Wrap_utils.keyword_args [("X0", x0); ("U", Some(u )); ("T", Some(t ))])

let step ?x0 ?t ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "step"
     [||]
     (Wrap_utils.keyword_args [("X0", x0); ("T", t); ("N", n)])

let to_discrete ?method_ ?alpha ~dt self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_discrete"
     [||]
     (Wrap_utils.keyword_args [("method", method_); ("alpha", alpha); ("dt", Some(dt ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Bsplines = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.signal.bsplines"

let get_py name = Py.Module.get __wrap_namespace name
                  let add ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "add"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let arange ?start ?step ?dtype ~stop () =
                     Py.Module.get_function_with_keywords __wrap_namespace "arange"
                       [||]
                       (Wrap_utils.keyword_args [("start", Wrap_utils.Option.map start (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("step", Wrap_utils.Option.map step (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("stop", Some(stop |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let arctan2 ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "arctan2"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let array ?dtype ?copy ?order ?subok ?ndmin ~object_ () =
                     Py.Module.get_function_with_keywords __wrap_namespace "array"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("order", Wrap_utils.Option.map order (function
| `K -> Py.String.of_string "K"
| `A -> Py.String.of_string "A"
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("ndmin", Wrap_utils.Option.map ndmin Py.Int.of_int); ("object", Some(object_ |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let asarray ?dtype ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let bspline ~x ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "bspline"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x )); ("n", Some(n ))])

                  let comb ?exact ?repetition ~n ~k () =
                     Py.Module.get_function_with_keywords __wrap_namespace "comb"
                       [||]
                       (Wrap_utils.keyword_args [("exact", Wrap_utils.Option.map exact Py.Bool.of_bool); ("repetition", Wrap_utils.Option.map repetition Py.Bool.of_bool); ("N", Some(n |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
))); ("k", Some(k |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)))])

                  let cos ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cos"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let cspline1d ?lamb ~signal () =
   Py.Module.get_function_with_keywords __wrap_namespace "cspline1d"
     [||]
     (Wrap_utils.keyword_args [("lamb", Wrap_utils.Option.map lamb Py.Float.of_float); ("signal", Some(signal |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let cspline1d_eval ?dx ?x0 ~cj ~newx () =
   Py.Module.get_function_with_keywords __wrap_namespace "cspline1d_eval"
     [||]
     (Wrap_utils.keyword_args [("dx", dx); ("x0", x0); ("cj", Some(cj )); ("newx", Some(newx ))])

let cubic x =
   Py.Module.get_function_with_keywords __wrap_namespace "cubic"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

                  let exp ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "exp"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let factorial n =
   Py.Module.get_function_with_keywords __wrap_namespace "factorial"
     [||]
     (Wrap_utils.keyword_args [("n", Some(n ))])

                  let floor ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "floor"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let gamma ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "gamma"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let gauss_spline ~x ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "gauss_spline"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x )); ("n", Some(n |> Py.Int.of_int))])

                  let greater ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "greater"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let greater_equal ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "greater_equal"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])

                  let less ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "less"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let less_equal ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "less_equal"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let logical_and ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "logical_and"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])

let piecewise ?kw ~x ~condlist ~funclist args =
   Py.Module.get_function_with_keywords __wrap_namespace "piecewise"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject)); ("condlist", Some(condlist )); ("funclist", Some(funclist ))]) (match kw with None -> [] | Some x -> x))
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let qspline1d ?lamb ~signal () =
   Py.Module.get_function_with_keywords __wrap_namespace "qspline1d"
     [||]
     (Wrap_utils.keyword_args [("lamb", Wrap_utils.Option.map lamb Py.Float.of_float); ("signal", Some(signal |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let qspline1d_eval ?dx ?x0 ~cj ~newx () =
   Py.Module.get_function_with_keywords __wrap_namespace "qspline1d_eval"
     [||]
     (Wrap_utils.keyword_args [("dx", dx); ("x0", x0); ("cj", Some(cj )); ("newx", Some(newx ))])

let quadratic x =
   Py.Module.get_function_with_keywords __wrap_namespace "quadratic"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let sepfir2d ~input ~hrow ~hcol () =
   Py.Module.get_function_with_keywords __wrap_namespace "sepfir2d"
     [||]
     (Wrap_utils.keyword_args [("input", Some(input )); ("hrow", Some(hrow )); ("hcol", Some(hcol ))])

                  let sin ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "sin"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let spline_filter ?lmbda ~iin () =
   Py.Module.get_function_with_keywords __wrap_namespace "spline_filter"
     [||]
     (Wrap_utils.keyword_args [("lmbda", lmbda); ("Iin", Some(iin ))])

                  let sqrt ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "sqrt"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let tan ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "tan"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let zeros ?dtype ?order ~shape () =
                     Py.Module.get_function_with_keywords __wrap_namespace "zeros"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let zeros_like ?dtype ?order ?subok ?shape ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "zeros_like"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `A -> Py.String.of_string "A"
| `PyObject x -> Wrap_utils.id x
)); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("shape", Wrap_utils.Option.map shape (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
module Filter_design = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.signal.filter_design"

let get_py name = Py.Module.get __wrap_namespace name
module Sp_fft = struct
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


end
                  let abs ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "abs"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let absolute ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "absolute"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let append ?axis ~arr ~values () =
   Py.Module.get_function_with_keywords __wrap_namespace "append"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("arr", Some(arr |> Np.Obj.to_pyobject)); ("values", Some(values |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let arccosh ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "arccosh"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let arcsinh ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "arcsinh"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let arctan ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "arctan"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let array ?dtype ?copy ?order ?subok ?ndmin ~object_ () =
                     Py.Module.get_function_with_keywords __wrap_namespace "array"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("order", Wrap_utils.Option.map order (function
| `K -> Py.String.of_string "K"
| `A -> Py.String.of_string "A"
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("ndmin", Wrap_utils.Option.map ndmin Py.Int.of_int); ("object", Some(object_ |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let asarray ?dtype ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let atleast_1d arys =
   Py.Module.get_function_with_keywords __wrap_namespace "atleast_1d"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arys)])
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let band_stop_obj ~wp ~ind ~passb ~stopb ~gpass ~gstop ~type_ () =
                     Py.Module.get_function_with_keywords __wrap_namespace "band_stop_obj"
                       [||]
                       (Wrap_utils.keyword_args [("wp", Some(wp |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
))); ("ind", Some(ind |> (function
| `I x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
))); ("passb", Some(passb |> Np.Obj.to_pyobject)); ("stopb", Some(stopb |> Np.Obj.to_pyobject)); ("gpass", Some(gpass |> Py.Float.of_float)); ("gstop", Some(gstop |> Py.Float.of_float)); ("type", Some(type_ |> (function
| `Butter -> Py.String.of_string "butter"
| `Cheby -> Py.String.of_string "cheby"
| `Ellip -> Py.String.of_string "ellip"
)))])

                  let bessel ?btype ?analog ?output ?norm ?fs ~n ~wn () =
                     Py.Module.get_function_with_keywords __wrap_namespace "bessel"
                       [||]
                       (Wrap_utils.keyword_args [("btype", Wrap_utils.Option.map btype (function
| `Lowpass -> Py.String.of_string "lowpass"
| `Highpass -> Py.String.of_string "highpass"
| `Bandpass -> Py.String.of_string "bandpass"
| `Bandstop -> Py.String.of_string "bandstop"
)); ("analog", Wrap_utils.Option.map analog Py.Bool.of_bool); ("output", Wrap_utils.Option.map output (function
| `Ba -> Py.String.of_string "ba"
| `Zpk -> Py.String.of_string "zpk"
| `Sos -> Py.String.of_string "sos"
)); ("norm", Wrap_utils.Option.map norm (function
| `Phase -> Py.String.of_string "phase"
| `Delay -> Py.String.of_string "delay"
| `Mag -> Py.String.of_string "mag"
)); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("N", Some(n |> Py.Int.of_int)); ("Wn", Some(wn |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let besselap ?norm ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "besselap"
                       [||]
                       (Wrap_utils.keyword_args [("norm", Wrap_utils.Option.map norm (function
| `Phase -> Py.String.of_string "phase"
| `Delay -> Py.String.of_string "delay"
| `Mag -> Py.String.of_string "mag"
)); ("N", Some(n |> Py.Int.of_int))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
let bilinear ?fs ~b ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "bilinear"
     [||]
     (Wrap_utils.keyword_args [("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let bilinear_zpk ~z ~p ~k ~fs () =
   Py.Module.get_function_with_keywords __wrap_namespace "bilinear_zpk"
     [||]
     (Wrap_utils.keyword_args [("z", Some(z |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject)); ("k", Some(k |> Py.Float.of_float)); ("fs", Some(fs |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let buttap n =
   Py.Module.get_function_with_keywords __wrap_namespace "buttap"
     [||]
     (Wrap_utils.keyword_args [("N", Some(n ))])

                  let butter ?btype ?analog ?output ?fs ~n ~wn () =
                     Py.Module.get_function_with_keywords __wrap_namespace "butter"
                       [||]
                       (Wrap_utils.keyword_args [("btype", Wrap_utils.Option.map btype (function
| `Lowpass -> Py.String.of_string "lowpass"
| `Highpass -> Py.String.of_string "highpass"
| `Bandpass -> Py.String.of_string "bandpass"
| `Bandstop -> Py.String.of_string "bandstop"
)); ("analog", Wrap_utils.Option.map analog Py.Bool.of_bool); ("output", Wrap_utils.Option.map output (function
| `Ba -> Py.String.of_string "ba"
| `Zpk -> Py.String.of_string "zpk"
| `Sos -> Py.String.of_string "sos"
)); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("N", Some(n |> Py.Int.of_int)); ("Wn", Some(wn |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let buttord ?analog ?fs ~wp ~ws ~gpass ~gstop () =
   Py.Module.get_function_with_keywords __wrap_namespace "buttord"
     [||]
     (Wrap_utils.keyword_args [("analog", Wrap_utils.Option.map analog Py.Bool.of_bool); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("wp", Some(wp )); ("ws", Some(ws )); ("gpass", Some(gpass |> Py.Float.of_float)); ("gstop", Some(gstop |> Py.Float.of_float))])
     |> (fun x -> ((Py.Int.to_int (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let ceil ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ceil"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let cheb1ap ~n ~rp () =
   Py.Module.get_function_with_keywords __wrap_namespace "cheb1ap"
     [||]
     (Wrap_utils.keyword_args [("N", Some(n )); ("rp", Some(rp ))])

let cheb1ord ?analog ?fs ~wp ~ws ~gpass ~gstop () =
   Py.Module.get_function_with_keywords __wrap_namespace "cheb1ord"
     [||]
     (Wrap_utils.keyword_args [("analog", Wrap_utils.Option.map analog Py.Bool.of_bool); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("wp", Some(wp )); ("ws", Some(ws )); ("gpass", Some(gpass |> Py.Float.of_float)); ("gstop", Some(gstop |> Py.Float.of_float))])
     |> (fun x -> ((Py.Int.to_int (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let cheb2ap ~n ~rs () =
   Py.Module.get_function_with_keywords __wrap_namespace "cheb2ap"
     [||]
     (Wrap_utils.keyword_args [("N", Some(n )); ("rs", Some(rs ))])

let cheb2ord ?analog ?fs ~wp ~ws ~gpass ~gstop () =
   Py.Module.get_function_with_keywords __wrap_namespace "cheb2ord"
     [||]
     (Wrap_utils.keyword_args [("analog", Wrap_utils.Option.map analog Py.Bool.of_bool); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("wp", Some(wp )); ("ws", Some(ws )); ("gpass", Some(gpass |> Py.Float.of_float)); ("gstop", Some(gstop |> Py.Float.of_float))])
     |> (fun x -> ((Py.Int.to_int (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let cheby1 ?btype ?analog ?output ?fs ~n ~rp ~wn () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cheby1"
                       [||]
                       (Wrap_utils.keyword_args [("btype", Wrap_utils.Option.map btype (function
| `Lowpass -> Py.String.of_string "lowpass"
| `Highpass -> Py.String.of_string "highpass"
| `Bandpass -> Py.String.of_string "bandpass"
| `Bandstop -> Py.String.of_string "bandstop"
)); ("analog", Wrap_utils.Option.map analog Py.Bool.of_bool); ("output", Wrap_utils.Option.map output (function
| `Ba -> Py.String.of_string "ba"
| `Zpk -> Py.String.of_string "zpk"
| `Sos -> Py.String.of_string "sos"
)); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("N", Some(n |> Py.Int.of_int)); ("rp", Some(rp |> Py.Float.of_float)); ("Wn", Some(wn |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let cheby2 ?btype ?analog ?output ?fs ~n ~rs ~wn () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cheby2"
                       [||]
                       (Wrap_utils.keyword_args [("btype", Wrap_utils.Option.map btype (function
| `Lowpass -> Py.String.of_string "lowpass"
| `Highpass -> Py.String.of_string "highpass"
| `Bandpass -> Py.String.of_string "bandpass"
| `Bandstop -> Py.String.of_string "bandstop"
)); ("analog", Wrap_utils.Option.map analog Py.Bool.of_bool); ("output", Wrap_utils.Option.map output (function
| `Ba -> Py.String.of_string "ba"
| `Zpk -> Py.String.of_string "zpk"
| `Sos -> Py.String.of_string "sos"
)); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("N", Some(n |> Py.Int.of_int)); ("rs", Some(rs |> Py.Float.of_float)); ("Wn", Some(wn |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let comb ?exact ?repetition ~n ~k () =
                     Py.Module.get_function_with_keywords __wrap_namespace "comb"
                       [||]
                       (Wrap_utils.keyword_args [("exact", Wrap_utils.Option.map exact Py.Bool.of_bool); ("repetition", Wrap_utils.Option.map repetition Py.Bool.of_bool); ("N", Some(n |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
))); ("k", Some(k |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)))])

let concatenate ?axis ?out ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "concatenate"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("a", Some(a ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let conjugate ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "conjugate"
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
                  let ellip ?btype ?analog ?output ?fs ~n ~rp ~rs ~wn () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ellip"
                       [||]
                       (Wrap_utils.keyword_args [("btype", Wrap_utils.Option.map btype (function
| `Lowpass -> Py.String.of_string "lowpass"
| `Highpass -> Py.String.of_string "highpass"
| `Bandpass -> Py.String.of_string "bandpass"
| `Bandstop -> Py.String.of_string "bandstop"
)); ("analog", Wrap_utils.Option.map analog Py.Bool.of_bool); ("output", Wrap_utils.Option.map output (function
| `Ba -> Py.String.of_string "ba"
| `Zpk -> Py.String.of_string "zpk"
| `Sos -> Py.String.of_string "sos"
)); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("N", Some(n |> Py.Int.of_int)); ("rp", Some(rp |> Py.Float.of_float)); ("rs", Some(rs |> Py.Float.of_float)); ("Wn", Some(wn |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let ellipap ~n ~rp ~rs () =
   Py.Module.get_function_with_keywords __wrap_namespace "ellipap"
     [||]
     (Wrap_utils.keyword_args [("N", Some(n )); ("rp", Some(rp )); ("rs", Some(rs ))])

let ellipord ?analog ?fs ~wp ~ws ~gpass ~gstop () =
   Py.Module.get_function_with_keywords __wrap_namespace "ellipord"
     [||]
     (Wrap_utils.keyword_args [("analog", Wrap_utils.Option.map analog Py.Bool.of_bool); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("wp", Some(wp )); ("ws", Some(ws )); ("gpass", Some(gpass |> Py.Float.of_float)); ("gstop", Some(gstop |> Py.Float.of_float))])
     |> (fun x -> ((Py.Int.to_int (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let exp ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "exp"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let factorial ?exact ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "factorial"
                       [||]
                       (Wrap_utils.keyword_args [("exact", Wrap_utils.Option.map exact Py.Bool.of_bool); ("n", Some(n |> (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)))])

                  let findfreqs ?kind ~num ~den ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "findfreqs"
                       [||]
                       (Wrap_utils.keyword_args [("kind", Wrap_utils.Option.map kind (function
| `Ba -> Py.String.of_string "ba"
| `Zp -> Py.String.of_string "zp"
)); ("num", Some(num )); ("den", Some(den )); ("N", Some(n |> Py.Int.of_int))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let freqs ?worN ?plot ~b ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "freqs"
                       [||]
                       (Wrap_utils.keyword_args [("worN", Wrap_utils.Option.map worN (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("plot", plot); ("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let freqs_zpk ?worN ~z ~p ~k () =
                     Py.Module.get_function_with_keywords __wrap_namespace "freqs_zpk"
                       [||]
                       (Wrap_utils.keyword_args [("worN", Wrap_utils.Option.map worN (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("z", Some(z |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject)); ("k", Some(k |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let freqz ?a ?worN ?whole ?plot ?fs ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "freqz"
                       [||]
                       (Wrap_utils.keyword_args [("a", Wrap_utils.Option.map a Np.Obj.to_pyobject); ("worN", Wrap_utils.Option.map worN (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("whole", Wrap_utils.Option.map whole Py.Bool.of_bool); ("plot", plot); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let freqz_zpk ?worN ?whole ?fs ~z ~p ~k () =
                     Py.Module.get_function_with_keywords __wrap_namespace "freqz_zpk"
                       [||]
                       (Wrap_utils.keyword_args [("worN", Wrap_utils.Option.map worN (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("whole", Wrap_utils.Option.map whole Py.Bool.of_bool); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("z", Some(z |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject)); ("k", Some(k |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let full ?dtype ?order ~shape ~fill_value () =
                     Py.Module.get_function_with_keywords __wrap_namespace "full"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
))); ("fill_value", Some(fill_value |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let group_delay ?w ?whole ?fs ~system () =
                     Py.Module.get_function_with_keywords __wrap_namespace "group_delay"
                       [||]
                       (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("whole", Wrap_utils.Option.map whole Py.Bool.of_bool); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("system", Some(system ))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let iirdesign ?analog ?ftype ?output ?fs ~wp ~ws ~gpass ~gstop () =
                     Py.Module.get_function_with_keywords __wrap_namespace "iirdesign"
                       [||]
                       (Wrap_utils.keyword_args [("analog", Wrap_utils.Option.map analog Py.Bool.of_bool); ("ftype", Wrap_utils.Option.map ftype Py.String.of_string); ("output", Wrap_utils.Option.map output (function
| `Ba -> Py.String.of_string "ba"
| `Zpk -> Py.String.of_string "zpk"
| `Sos -> Py.String.of_string "sos"
)); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("wp", Some(wp )); ("ws", Some(ws )); ("gpass", Some(gpass |> Py.Float.of_float)); ("gstop", Some(gstop |> Py.Float.of_float))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let iirfilter ?rp ?rs ?btype ?analog ?ftype ?output ?fs ~n ~wn () =
                     Py.Module.get_function_with_keywords __wrap_namespace "iirfilter"
                       [||]
                       (Wrap_utils.keyword_args [("rp", Wrap_utils.Option.map rp Py.Float.of_float); ("rs", Wrap_utils.Option.map rs Py.Float.of_float); ("btype", Wrap_utils.Option.map btype (function
| `Bandpass -> Py.String.of_string "bandpass"
| `Lowpass -> Py.String.of_string "lowpass"
| `Highpass -> Py.String.of_string "highpass"
| `Bandstop -> Py.String.of_string "bandstop"
)); ("analog", Wrap_utils.Option.map analog Py.Bool.of_bool); ("ftype", Wrap_utils.Option.map ftype Py.String.of_string); ("output", Wrap_utils.Option.map output (function
| `Ba -> Py.String.of_string "ba"
| `Zpk -> Py.String.of_string "zpk"
| `Sos -> Py.String.of_string "sos"
)); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("N", Some(n |> Py.Int.of_int)); ("Wn", Some(wn |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let iirnotch ?fs ~w0 ~q () =
   Py.Module.get_function_with_keywords __wrap_namespace "iirnotch"
     [||]
     (Wrap_utils.keyword_args [("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("w0", Some(w0 |> Py.Float.of_float)); ("Q", Some(q |> Py.Float.of_float))])

let iirpeak ?fs ~w0 ~q () =
   Py.Module.get_function_with_keywords __wrap_namespace "iirpeak"
     [||]
     (Wrap_utils.keyword_args [("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("w0", Some(w0 |> Py.Float.of_float)); ("Q", Some(q |> Py.Float.of_float))])

                  let log10 ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "log10"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let logspace ?num ?endpoint ?base ?dtype ?axis ~start ~stop () =
   Py.Module.get_function_with_keywords __wrap_namespace "logspace"
     [||]
     (Wrap_utils.keyword_args [("num", Wrap_utils.Option.map num Py.Int.of_int); ("endpoint", Wrap_utils.Option.map endpoint Py.Bool.of_bool); ("base", Wrap_utils.Option.map base Py.Float.of_float); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("start", Some(start |> Np.Obj.to_pyobject)); ("stop", Some(stop |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let lp2bp ?wo ?bw ~b ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "lp2bp"
     [||]
     (Wrap_utils.keyword_args [("wo", Wrap_utils.Option.map wo Py.Float.of_float); ("bw", Wrap_utils.Option.map bw Py.Float.of_float); ("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let lp2bp_zpk ?wo ?bw ~z ~p ~k () =
   Py.Module.get_function_with_keywords __wrap_namespace "lp2bp_zpk"
     [||]
     (Wrap_utils.keyword_args [("wo", Wrap_utils.Option.map wo Py.Float.of_float); ("bw", Wrap_utils.Option.map bw Py.Float.of_float); ("z", Some(z |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject)); ("k", Some(k |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let lp2bs ?wo ?bw ~b ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "lp2bs"
     [||]
     (Wrap_utils.keyword_args [("wo", Wrap_utils.Option.map wo Py.Float.of_float); ("bw", Wrap_utils.Option.map bw Py.Float.of_float); ("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let lp2bs_zpk ?wo ?bw ~z ~p ~k () =
   Py.Module.get_function_with_keywords __wrap_namespace "lp2bs_zpk"
     [||]
     (Wrap_utils.keyword_args [("wo", Wrap_utils.Option.map wo Py.Float.of_float); ("bw", Wrap_utils.Option.map bw Py.Float.of_float); ("z", Some(z |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject)); ("k", Some(k |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let lp2hp ?wo ~b ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "lp2hp"
     [||]
     (Wrap_utils.keyword_args [("wo", Wrap_utils.Option.map wo Py.Float.of_float); ("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let lp2hp_zpk ?wo ~z ~p ~k () =
   Py.Module.get_function_with_keywords __wrap_namespace "lp2hp_zpk"
     [||]
     (Wrap_utils.keyword_args [("wo", Wrap_utils.Option.map wo Py.Float.of_float); ("z", Some(z |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject)); ("k", Some(k |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let lp2lp ?wo ~b ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "lp2lp"
     [||]
     (Wrap_utils.keyword_args [("wo", Wrap_utils.Option.map wo Py.Float.of_float); ("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let lp2lp_zpk ?wo ~z ~p ~k () =
   Py.Module.get_function_with_keywords __wrap_namespace "lp2lp_zpk"
     [||]
     (Wrap_utils.keyword_args [("wo", Wrap_utils.Option.map wo Py.Float.of_float); ("z", Some(z |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject)); ("k", Some(k |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let maxflat () =
   Py.Module.get_function_with_keywords __wrap_namespace "maxflat"
     [||]
     []

                  let mintypecode ?typeset ?default ~typechars () =
                     Py.Module.get_function_with_keywords __wrap_namespace "mintypecode"
                       [||]
                       (Wrap_utils.keyword_args [("typeset", Wrap_utils.Option.map typeset (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `S x -> Py.String.of_string x
)); ("default", Wrap_utils.Option.map default Py.String.of_string); ("typechars", Some(typechars |> (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `Ndarray x -> Np.Obj.to_pyobject x
)))])
                       |> Py.String.to_string
let normalize ~b ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "normalize"
     [||]
     (Wrap_utils.keyword_args [("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let npp_polyval ?tensor ~x ~c () =
                     Py.Module.get_function_with_keywords __wrap_namespace "npp_polyval"
                       [||]
                       (Wrap_utils.keyword_args [("tensor", Wrap_utils.Option.map tensor Py.Bool.of_bool); ("x", Some(x |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Compatible_object x -> Wrap_utils.id x
))); ("c", Some(c |> Np.Obj.to_pyobject))])

                  let ones ?dtype ?order ~shape () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ones"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let poly seq_of_zeros =
   Py.Module.get_function_with_keywords __wrap_namespace "poly"
     [||]
     (Wrap_utils.keyword_args [("seq_of_zeros", Some(seq_of_zeros |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let polyval ~p ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "polyval"
                       [||]
                       (Wrap_utils.keyword_args [("p", Some(p |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Poly1d_object x -> Wrap_utils.id x
))); ("x", Some(x |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Poly1d_object x -> Wrap_utils.id x
)))])

                  let polyvalfromroots ?tensor ~x ~r () =
                     Py.Module.get_function_with_keywords __wrap_namespace "polyvalfromroots"
                       [||]
                       (Wrap_utils.keyword_args [("tensor", Wrap_utils.Option.map tensor Py.Bool.of_bool); ("x", Some(x |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Compatible_object x -> Wrap_utils.id x
))); ("r", Some(r |> Np.Obj.to_pyobject))])

                  let prod ?axis ?dtype ?out ?keepdims ?initial ?where ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "prod"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("initial", Wrap_utils.Option.map initial (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("where", where); ("a", Some(a |> Np.Obj.to_pyobject))])

let real val_ =
   Py.Module.get_function_with_keywords __wrap_namespace "real"
     [||]
     (Wrap_utils.keyword_args [("val", Some(val_ |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let resize ~a ~new_shape () =
                     Py.Module.get_function_with_keywords __wrap_namespace "resize"
                       [||]
                       (Wrap_utils.keyword_args [("a", Some(a |> Np.Obj.to_pyobject)); ("new_shape", Some(new_shape |> (function
| `I x -> Py.Int.of_int x
| `Tuple_of_int x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let roots p =
   Py.Module.get_function_with_keywords __wrap_namespace "roots"
     [||]
     (Wrap_utils.keyword_args [("p", Some(p |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
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
let sos2tf sos =
   Py.Module.get_function_with_keywords __wrap_namespace "sos2tf"
     [||]
     (Wrap_utils.keyword_args [("sos", Some(sos |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let sos2zpk sos =
   Py.Module.get_function_with_keywords __wrap_namespace "sos2zpk"
     [||]
     (Wrap_utils.keyword_args [("sos", Some(sos |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
                  let sosfreqz ?worN ?whole ?fs ~sos () =
                     Py.Module.get_function_with_keywords __wrap_namespace "sosfreqz"
                       [||]
                       (Wrap_utils.keyword_args [("worN", Wrap_utils.Option.map worN (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("whole", Wrap_utils.Option.map whole Py.Bool.of_bool); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("sos", Some(sos |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let sp_fft ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "sp_fft"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

                  let sqrt ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "sqrt"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let tan ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "tan"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let tf2sos ?pairing ~b ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "tf2sos"
                       [||]
                       (Wrap_utils.keyword_args [("pairing", Wrap_utils.Option.map pairing (function
| `Nearest -> Py.String.of_string "nearest"
| `Keep_odd -> Py.String.of_string "keep_odd"
)); ("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let tf2zpk ~b ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "tf2zpk"
     [||]
     (Wrap_utils.keyword_args [("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let yulewalk () =
   Py.Module.get_function_with_keywords __wrap_namespace "yulewalk"
     [||]
     []

                  let zeros ?dtype ?order ~shape () =
                     Py.Module.get_function_with_keywords __wrap_namespace "zeros"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let zpk2sos ?pairing ~z ~p ~k () =
                     Py.Module.get_function_with_keywords __wrap_namespace "zpk2sos"
                       [||]
                       (Wrap_utils.keyword_args [("pairing", Wrap_utils.Option.map pairing (function
| `Nearest -> Py.String.of_string "nearest"
| `Keep_odd -> Py.String.of_string "keep_odd"
)); ("z", Some(z |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject)); ("k", Some(k |> Py.Float.of_float))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let zpk2tf ~z ~p ~k () =
   Py.Module.get_function_with_keywords __wrap_namespace "zpk2tf"
     [||]
     (Wrap_utils.keyword_args [("z", Some(z |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject)); ("k", Some(k |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))

end
module Fir_filter_design = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.signal.fir_filter_design"

let get_py name = Py.Module.get __wrap_namespace name
let ceil x =
   Py.Module.get_function_with_keywords __wrap_namespace "ceil"
     (Array.of_list @@ List.concat [[x ]])
     []

let fft ?n ?axis ?norm ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "fft"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("a", Some(a |> Np.Obj.to_pyobject))])

let firls ?weight ?nyq ?fs ~numtaps ~bands ~desired () =
   Py.Module.get_function_with_keywords __wrap_namespace "firls"
     [||]
     (Wrap_utils.keyword_args [("weight", Wrap_utils.Option.map weight Np.Obj.to_pyobject); ("nyq", Wrap_utils.Option.map nyq Py.Float.of_float); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("numtaps", Some(numtaps |> Py.Int.of_int)); ("bands", Some(bands |> Np.Obj.to_pyobject)); ("desired", Some(desired |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let firwin ?width ?window ?pass_zero ?scale ?nyq ?fs ~numtaps ~cutoff () =
                     Py.Module.get_function_with_keywords __wrap_namespace "firwin"
                       [||]
                       (Wrap_utils.keyword_args [("width", Wrap_utils.Option.map width Py.Float.of_float); ("window", Wrap_utils.Option.map window (function
| `Tuple_of_string_and_parameter_values x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("pass_zero", Wrap_utils.Option.map pass_zero (function
| `Highpass -> Py.String.of_string "highpass"
| `Lowpass -> Py.String.of_string "lowpass"
| `Bandstop -> Py.String.of_string "bandstop"
| `Bool x -> Py.Bool.of_bool x
| `Bandpass -> Py.String.of_string "bandpass"
)); ("scale", Wrap_utils.Option.map scale Py.Float.of_float); ("nyq", Wrap_utils.Option.map nyq Py.Float.of_float); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("numtaps", Some(numtaps |> Py.Int.of_int)); ("cutoff", Some(cutoff |> (function
| `F x -> Py.Float.of_float x
| `T1D_array_like x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let firwin2 ?nfreqs ?window ?nyq ?antisymmetric ?fs ~numtaps ~freq ~gain () =
                     Py.Module.get_function_with_keywords __wrap_namespace "firwin2"
                       [||]
                       (Wrap_utils.keyword_args [("nfreqs", Wrap_utils.Option.map nfreqs Py.Int.of_int); ("window", Wrap_utils.Option.map window (function
| `T_string_float_ x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `None -> Py.none
)); ("nyq", Wrap_utils.Option.map nyq Py.Float.of_float); ("antisymmetric", Wrap_utils.Option.map antisymmetric Py.Bool.of_bool); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("numtaps", Some(numtaps |> Py.Int.of_int)); ("freq", Some(freq |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `T1D x -> Wrap_utils.id x
))); ("gain", Some(gain |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let hankel ?r ~c () =
   Py.Module.get_function_with_keywords __wrap_namespace "hankel"
     [||]
     (Wrap_utils.keyword_args [("r", Wrap_utils.Option.map r Np.Obj.to_pyobject); ("c", Some(c |> Np.Obj.to_pyobject))])

let ifft ?n ?axis ?norm ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "ifft"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("a", Some(a |> Np.Obj.to_pyobject))])

let irfft ?n ?axis ?norm ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "irfft"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let kaiser_atten ~numtaps ~width () =
   Py.Module.get_function_with_keywords __wrap_namespace "kaiser_atten"
     [||]
     (Wrap_utils.keyword_args [("numtaps", Some(numtaps |> Py.Int.of_int)); ("width", Some(width |> Py.Float.of_float))])
     |> Py.Float.to_float
let kaiser_beta a =
   Py.Module.get_function_with_keywords __wrap_namespace "kaiser_beta"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Py.Float.of_float))])
     |> Py.Float.to_float
let kaiserord ~ripple ~width () =
   Py.Module.get_function_with_keywords __wrap_namespace "kaiserord"
     [||]
     (Wrap_utils.keyword_args [("ripple", Some(ripple |> Py.Float.of_float)); ("width", Some(width |> Py.Float.of_float))])
     |> (fun x -> ((Py.Int.to_int (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))
let lstsq ?cond ?overwrite_a ?overwrite_b ?check_finite ?lapack_driver ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "lstsq"
     [||]
     (Wrap_utils.keyword_args [("cond", Wrap_utils.Option.map cond Py.Float.of_float); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("lapack_driver", Wrap_utils.Option.map lapack_driver Py.String.of_string); ("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), ((fun py -> if Py.is_none py then None else Some (Wrap_utils.id py)) (Py.Tuple.get x 3))))
                  let minimum_phase ?method_ ?n_fft ~h () =
                     Py.Module.get_function_with_keywords __wrap_namespace "minimum_phase"
                       [||]
                       (Wrap_utils.keyword_args [("method", Wrap_utils.Option.map method_ (function
| `Hilbert -> Py.String.of_string "hilbert"
| `Homomorphic -> Py.String.of_string "homomorphic"
)); ("n_fft", Wrap_utils.Option.map n_fft Py.Int.of_int); ("h", Some(h |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let remez ?weight ?hz ?type_ ?maxiter ?grid_density ?fs ~numtaps ~bands ~desired () =
                     Py.Module.get_function_with_keywords __wrap_namespace "remez"
                       [||]
                       (Wrap_utils.keyword_args [("weight", Wrap_utils.Option.map weight Np.Obj.to_pyobject); ("Hz", Wrap_utils.Option.map hz (function
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("type", Wrap_utils.Option.map type_ (function
| `Bandpass -> Py.String.of_string "bandpass"
| `Differentiator -> Py.String.of_string "differentiator"
| `Hilbert -> Py.String.of_string "hilbert"
)); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("grid_density", Wrap_utils.Option.map grid_density Py.Int.of_int); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("numtaps", Some(numtaps |> Py.Int.of_int)); ("bands", Some(bands |> Np.Obj.to_pyobject)); ("desired", Some(desired |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let sinc x =
   Py.Module.get_function_with_keywords __wrap_namespace "sinc"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let solve ?sym_pos ?lower ?overwrite_a ?overwrite_b ?debug ?check_finite ?assume_a ?transposed ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "solve"
     [||]
     (Wrap_utils.keyword_args [("sym_pos", Wrap_utils.Option.map sym_pos Py.Bool.of_bool); ("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("debug", debug); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("assume_a", Wrap_utils.Option.map assume_a Py.String.of_string); ("transposed", Wrap_utils.Option.map transposed Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let toeplitz ?r ~c () =
   Py.Module.get_function_with_keywords __wrap_namespace "toeplitz"
     [||]
     (Wrap_utils.keyword_args [("r", Wrap_utils.Option.map r Np.Obj.to_pyobject); ("c", Some(c |> Np.Obj.to_pyobject))])


end
module Lti_conversion = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.signal.lti_conversion"

let get_py name = Py.Module.get __wrap_namespace name
let abcd_normalize ?a ?b ?c ?d () =
   Py.Module.get_function_with_keywords __wrap_namespace "abcd_normalize"
     [||]
     (Wrap_utils.keyword_args [("A", a); ("B", b); ("C", c); ("D", d)])

                  let array ?dtype ?copy ?order ?subok ?ndmin ~object_ () =
                     Py.Module.get_function_with_keywords __wrap_namespace "array"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("order", Wrap_utils.Option.map order (function
| `K -> Py.String.of_string "K"
| `A -> Py.String.of_string "A"
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("ndmin", Wrap_utils.Option.map ndmin Py.Int.of_int); ("object", Some(object_ |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let asarray ?dtype ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let atleast_2d arys =
   Py.Module.get_function_with_keywords __wrap_namespace "atleast_2d"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arys)])
     []

let cont2discrete ?method_ ?alpha ~system ~dt () =
   Py.Module.get_function_with_keywords __wrap_namespace "cont2discrete"
     [||]
     (Wrap_utils.keyword_args [("method", Wrap_utils.Option.map method_ Py.String.of_string); ("alpha", alpha); ("system", Some(system )); ("dt", Some(dt |> Py.Float.of_float))])

let dot ?out ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "dot"
     [||]
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let eye ?m ?k ?dtype ?order ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "eye"
                       [||]
                       (Wrap_utils.keyword_args [("M", Wrap_utils.Option.map m Py.Int.of_int); ("k", Wrap_utils.Option.map k Py.Int.of_int); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("N", Some(n |> Py.Int.of_int))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let normalize ~b ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "normalize"
     [||]
     (Wrap_utils.keyword_args [("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let outer ?out ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "outer"
     [||]
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let poly seq_of_zeros =
   Py.Module.get_function_with_keywords __wrap_namespace "poly"
     [||]
     (Wrap_utils.keyword_args [("seq_of_zeros", Some(seq_of_zeros |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let prod ?axis ?dtype ?out ?keepdims ?initial ?where ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "prod"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("initial", Wrap_utils.Option.map initial (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("where", where); ("a", Some(a |> Np.Obj.to_pyobject))])

let ss2tf ?input ~a ~b ~c ~d () =
   Py.Module.get_function_with_keywords __wrap_namespace "ss2tf"
     [||]
     (Wrap_utils.keyword_args [("input", Wrap_utils.Option.map input Py.Int.of_int); ("A", Some(a |> Np.Obj.to_pyobject)); ("B", Some(b |> Np.Obj.to_pyobject)); ("C", Some(c |> Np.Obj.to_pyobject)); ("D", Some(d |> Np.Obj.to_pyobject))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let ss2zpk ?input ~a ~b ~c ~d () =
   Py.Module.get_function_with_keywords __wrap_namespace "ss2zpk"
     [||]
     (Wrap_utils.keyword_args [("input", Wrap_utils.Option.map input Py.Int.of_int); ("A", Some(a |> Np.Obj.to_pyobject)); ("B", Some(b |> Np.Obj.to_pyobject)); ("C", Some(c |> Np.Obj.to_pyobject)); ("D", Some(d |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let tf2ss ~num ~den () =
   Py.Module.get_function_with_keywords __wrap_namespace "tf2ss"
     [||]
     (Wrap_utils.keyword_args [("num", Some(num )); ("den", Some(den ))])

let tf2zpk ~b ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "tf2zpk"
     [||]
     (Wrap_utils.keyword_args [("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
                  let zeros ?dtype ?order ~shape () =
                     Py.Module.get_function_with_keywords __wrap_namespace "zeros"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let zpk2ss ~z ~p ~k () =
   Py.Module.get_function_with_keywords __wrap_namespace "zpk2ss"
     [||]
     (Wrap_utils.keyword_args [("z", Some(z )); ("p", Some(p )); ("k", Some(k |> Py.Float.of_float))])

let zpk2tf ~z ~p ~k () =
   Py.Module.get_function_with_keywords __wrap_namespace "zpk2tf"
     [||]
     (Wrap_utils.keyword_args [("z", Some(z |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject)); ("k", Some(k |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))

end
module Ltisys = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.signal.ltisys"

let get_py name = Py.Module.get __wrap_namespace name
module Bunch = struct
type tag = [`Bunch]
type t = [`Bunch | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwds () =
   Py.Module.get_function_with_keywords __wrap_namespace "Bunch"
     [||]
     (match kwds with None -> [] | Some x -> x)
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LinearTimeInvariant = struct
type tag = [`LinearTimeInvariant]
type t = [`LinearTimeInvariant | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs system =
   Py.Module.get_function_with_keywords __wrap_namespace "LinearTimeInvariant"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id system)])
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module StateSpaceContinuous = struct
type tag = [`StateSpaceContinuous]
type t = [`Object | `StateSpaceContinuous] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs system =
   Py.Module.get_function_with_keywords __wrap_namespace "StateSpaceContinuous"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id system)])
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let bode ?w ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "bode"
     [||]
     (Wrap_utils.keyword_args [("w", w); ("n", n)])

let freqresp ?w ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "freqresp"
     [||]
     (Wrap_utils.keyword_args [("w", w); ("n", n)])

let impulse ?x0 ?t ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "impulse"
     [||]
     (Wrap_utils.keyword_args [("X0", x0); ("T", t); ("N", n)])

let output ?x0 ~u ~t self =
   Py.Module.get_function_with_keywords (to_pyobject self) "output"
     [||]
     (Wrap_utils.keyword_args [("X0", x0); ("U", Some(u )); ("T", Some(t ))])

let step ?x0 ?t ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "step"
     [||]
     (Wrap_utils.keyword_args [("X0", x0); ("T", t); ("N", n)])

let to_discrete ?method_ ?alpha ~dt self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_discrete"
     [||]
     (Wrap_utils.keyword_args [("method", method_); ("alpha", alpha); ("dt", Some(dt ))])

let to_ss self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_ss"
     [||]
     []

let to_tf ?kwargs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_tf"
     [||]
     (match kwargs with None -> [] | Some x -> x)

let to_zpk ?kwargs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_zpk"
     [||]
     (match kwargs with None -> [] | Some x -> x)

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module StateSpaceDiscrete = struct
type tag = [`StateSpaceDiscrete]
type t = [`Object | `StateSpaceDiscrete] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs system =
   Py.Module.get_function_with_keywords __wrap_namespace "StateSpaceDiscrete"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id system)])
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let bode ?w ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "bode"
     [||]
     (Wrap_utils.keyword_args [("w", w); ("n", n)])

let freqresp ?w ?n ?whole self =
   Py.Module.get_function_with_keywords (to_pyobject self) "freqresp"
     [||]
     (Wrap_utils.keyword_args [("w", w); ("n", n); ("whole", whole)])

let impulse ?x0 ?t ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "impulse"
     [||]
     (Wrap_utils.keyword_args [("x0", x0); ("t", t); ("n", n)])

let output ?x0 ~u ~t self =
   Py.Module.get_function_with_keywords (to_pyobject self) "output"
     [||]
     (Wrap_utils.keyword_args [("x0", x0); ("u", Some(u )); ("t", Some(t ))])

let step ?x0 ?t ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "step"
     [||]
     (Wrap_utils.keyword_args [("x0", x0); ("t", t); ("n", n)])

let to_ss self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_ss"
     [||]
     []

let to_tf ?kwargs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_tf"
     [||]
     (match kwargs with None -> [] | Some x -> x)

let to_zpk ?kwargs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_zpk"
     [||]
     (match kwargs with None -> [] | Some x -> x)

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module TransferFunctionContinuous = struct
type tag = [`TransferFunctionContinuous]
type t = [`Object | `TransferFunctionContinuous] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs system =
   Py.Module.get_function_with_keywords __wrap_namespace "TransferFunctionContinuous"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id system)])
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let bode ?w ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "bode"
     [||]
     (Wrap_utils.keyword_args [("w", w); ("n", n)])

let freqresp ?w ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "freqresp"
     [||]
     (Wrap_utils.keyword_args [("w", w); ("n", n)])

let impulse ?x0 ?t ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "impulse"
     [||]
     (Wrap_utils.keyword_args [("X0", x0); ("T", t); ("N", n)])

let output ?x0 ~u ~t self =
   Py.Module.get_function_with_keywords (to_pyobject self) "output"
     [||]
     (Wrap_utils.keyword_args [("X0", x0); ("U", Some(u )); ("T", Some(t ))])

let step ?x0 ?t ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "step"
     [||]
     (Wrap_utils.keyword_args [("X0", x0); ("T", t); ("N", n)])

let to_discrete ?method_ ?alpha ~dt self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_discrete"
     [||]
     (Wrap_utils.keyword_args [("method", method_); ("alpha", alpha); ("dt", Some(dt ))])

let to_ss self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_ss"
     [||]
     []

let to_tf self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_tf"
     [||]
     []

let to_zpk self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_zpk"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module TransferFunctionDiscrete = struct
type tag = [`TransferFunctionDiscrete]
type t = [`Object | `TransferFunctionDiscrete] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs system =
   Py.Module.get_function_with_keywords __wrap_namespace "TransferFunctionDiscrete"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id system)])
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let bode ?w ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "bode"
     [||]
     (Wrap_utils.keyword_args [("w", w); ("n", n)])

let freqresp ?w ?n ?whole self =
   Py.Module.get_function_with_keywords (to_pyobject self) "freqresp"
     [||]
     (Wrap_utils.keyword_args [("w", w); ("n", n); ("whole", whole)])

let impulse ?x0 ?t ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "impulse"
     [||]
     (Wrap_utils.keyword_args [("x0", x0); ("t", t); ("n", n)])

let output ?x0 ~u ~t self =
   Py.Module.get_function_with_keywords (to_pyobject self) "output"
     [||]
     (Wrap_utils.keyword_args [("x0", x0); ("u", Some(u )); ("t", Some(t ))])

let step ?x0 ?t ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "step"
     [||]
     (Wrap_utils.keyword_args [("x0", x0); ("t", t); ("n", n)])

let to_ss self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_ss"
     [||]
     []

let to_tf self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_tf"
     [||]
     []

let to_zpk self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_zpk"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ZerosPolesGainContinuous = struct
type tag = [`ZerosPolesGainContinuous]
type t = [`Object | `ZerosPolesGainContinuous] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs system =
   Py.Module.get_function_with_keywords __wrap_namespace "ZerosPolesGainContinuous"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id system)])
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let bode ?w ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "bode"
     [||]
     (Wrap_utils.keyword_args [("w", w); ("n", n)])

let freqresp ?w ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "freqresp"
     [||]
     (Wrap_utils.keyword_args [("w", w); ("n", n)])

let impulse ?x0 ?t ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "impulse"
     [||]
     (Wrap_utils.keyword_args [("X0", x0); ("T", t); ("N", n)])

let output ?x0 ~u ~t self =
   Py.Module.get_function_with_keywords (to_pyobject self) "output"
     [||]
     (Wrap_utils.keyword_args [("X0", x0); ("U", Some(u )); ("T", Some(t ))])

let step ?x0 ?t ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "step"
     [||]
     (Wrap_utils.keyword_args [("X0", x0); ("T", t); ("N", n)])

let to_discrete ?method_ ?alpha ~dt self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_discrete"
     [||]
     (Wrap_utils.keyword_args [("method", method_); ("alpha", alpha); ("dt", Some(dt ))])

let to_ss self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_ss"
     [||]
     []

let to_tf self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_tf"
     [||]
     []

let to_zpk self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_zpk"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ZerosPolesGainDiscrete = struct
type tag = [`ZerosPolesGainDiscrete]
type t = [`Object | `ZerosPolesGainDiscrete] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs system =
   Py.Module.get_function_with_keywords __wrap_namespace "ZerosPolesGainDiscrete"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id system)])
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let bode ?w ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "bode"
     [||]
     (Wrap_utils.keyword_args [("w", w); ("n", n)])

let freqresp ?w ?n ?whole self =
   Py.Module.get_function_with_keywords (to_pyobject self) "freqresp"
     [||]
     (Wrap_utils.keyword_args [("w", w); ("n", n); ("whole", whole)])

let impulse ?x0 ?t ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "impulse"
     [||]
     (Wrap_utils.keyword_args [("x0", x0); ("t", t); ("n", n)])

let output ?x0 ~u ~t self =
   Py.Module.get_function_with_keywords (to_pyobject self) "output"
     [||]
     (Wrap_utils.keyword_args [("x0", x0); ("u", Some(u )); ("t", Some(t ))])

let step ?x0 ?t ?n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "step"
     [||]
     (Wrap_utils.keyword_args [("x0", x0); ("t", t); ("n", n)])

let to_ss self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_ss"
     [||]
     []

let to_tf self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_tf"
     [||]
     []

let to_zpk self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_zpk"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Six = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy._lib.six"

let get_py name = Py.Module.get __wrap_namespace name
module Iterator = struct
type tag = [`Object]
type t = [`Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "Iterator"
     [||]
     []
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module StringIO = struct
type tag = [`StringIO]
type t = [`Object | `StringIO] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?initial_value ?newline () =
   Py.Module.get_function_with_keywords __wrap_namespace "StringIO"
     [||]
     (Wrap_utils.keyword_args [("initial_value", initial_value); ("newline", newline)])
     |> of_pyobject
let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let close self =
   Py.Module.get_function_with_keywords (to_pyobject self) "close"
     [||]
     []

let fileno self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fileno"
     [||]
     []

let flush self =
   Py.Module.get_function_with_keywords (to_pyobject self) "flush"
     [||]
     []

let getvalue self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getvalue"
     [||]
     []

let isatty self =
   Py.Module.get_function_with_keywords (to_pyobject self) "isatty"
     [||]
     []

let read ?size self =
   Py.Module.get_function_with_keywords (to_pyobject self) "read"
     (Array.of_list @@ List.concat [(match size with None -> [] | Some x -> [x ])])
     []

let readable self =
   Py.Module.get_function_with_keywords (to_pyobject self) "readable"
     [||]
     []

let readline ?size self =
   Py.Module.get_function_with_keywords (to_pyobject self) "readline"
     (Array.of_list @@ List.concat [(match size with None -> [] | Some x -> [x ])])
     []

let readlines ?hint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "readlines"
     (Array.of_list @@ List.concat [(match hint with None -> [] | Some x -> [x ])])
     []

let seek ?whence ~pos self =
   Py.Module.get_function_with_keywords (to_pyobject self) "seek"
     (Array.of_list @@ List.concat [(match whence with None -> [] | Some x -> [x ]);[pos ]])
     []

let seekable self =
   Py.Module.get_function_with_keywords (to_pyobject self) "seekable"
     [||]
     []

let tell self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tell"
     [||]
     []

let truncate ?pos self =
   Py.Module.get_function_with_keywords (to_pyobject self) "truncate"
     (Array.of_list @@ List.concat [(match pos with None -> [] | Some x -> [x ])])
     []

let writable self =
   Py.Module.get_function_with_keywords (to_pyobject self) "writable"
     [||]
     []

let write ~s self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write"
     (Array.of_list @@ List.concat [[s ]])
     []

let writelines ~lines self =
   Py.Module.get_function_with_keywords (to_pyobject self) "writelines"
     (Array.of_list @@ List.concat [[lines ]])
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let advance_iterator ?default ~iterator () =
   Py.Module.get_function_with_keywords __wrap_namespace "advance_iterator"
     [||]
     (Wrap_utils.keyword_args [("default", default); ("iterator", Some(iterator ))])

let b s =
   Py.Module.get_function_with_keywords __wrap_namespace "b"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s ))])

let callable obj =
   Py.Module.get_function_with_keywords __wrap_namespace "callable"
     [||]
     (Wrap_utils.keyword_args [("obj", Some(obj ))])

let get_unbound_function unbound =
   Py.Module.get_function_with_keywords __wrap_namespace "get_unbound_function"
     [||]
     (Wrap_utils.keyword_args [("unbound", Some(unbound ))])

let iteritems d =
   Py.Module.get_function_with_keywords __wrap_namespace "iteritems"
     [||]
     (Wrap_utils.keyword_args [("d", Some(d ))])

let iterkeys d =
   Py.Module.get_function_with_keywords __wrap_namespace "iterkeys"
     [||]
     (Wrap_utils.keyword_args [("d", Some(d ))])

let itervalues d =
   Py.Module.get_function_with_keywords __wrap_namespace "itervalues"
     [||]
     (Wrap_utils.keyword_args [("d", Some(d ))])

let next ?default ~iterator () =
   Py.Module.get_function_with_keywords __wrap_namespace "next"
     [||]
     (Wrap_utils.keyword_args [("default", default); ("iterator", Some(iterator ))])

let reduce ?initial ~function_ ~sequence () =
   Py.Module.get_function_with_keywords __wrap_namespace "reduce"
     [||]
     (Wrap_utils.keyword_args [("initial", initial); ("function", Some(function_ )); ("sequence", Some(sequence ))])

let reraise ?tb ~tp ~value () =
   Py.Module.get_function_with_keywords __wrap_namespace "reraise"
     [||]
     (Wrap_utils.keyword_args [("tb", tb); ("tp", Some(tp )); ("value", Some(value ))])

let u s =
   Py.Module.get_function_with_keywords __wrap_namespace "u"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s ))])

let with_metaclass ?base ~meta () =
   Py.Module.get_function_with_keywords __wrap_namespace "with_metaclass"
     [||]
     (Wrap_utils.keyword_args [("base", base); ("meta", Some(meta ))])


end
let abcd_normalize ?a ?b ?c ?d () =
   Py.Module.get_function_with_keywords __wrap_namespace "abcd_normalize"
     [||]
     (Wrap_utils.keyword_args [("A", a); ("B", b); ("C", c); ("D", d)])

                  let asarray ?dtype ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let atleast_1d arys =
   Py.Module.get_function_with_keywords __wrap_namespace "atleast_1d"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arys)])
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let atleast_2d arys =
   Py.Module.get_function_with_keywords __wrap_namespace "atleast_2d"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arys)])
     []

let bode ?w ?n ~system () =
   Py.Module.get_function_with_keywords __wrap_namespace "bode"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("system", Some(system ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
let cont2discrete ?method_ ?alpha ~system ~dt () =
   Py.Module.get_function_with_keywords __wrap_namespace "cont2discrete"
     [||]
     (Wrap_utils.keyword_args [("method", Wrap_utils.Option.map method_ Py.String.of_string); ("alpha", alpha); ("system", Some(system )); ("dt", Some(dt |> Py.Float.of_float))])

let dbode ?w ?n ~system () =
   Py.Module.get_function_with_keywords __wrap_namespace "dbode"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("system", Some(system ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
let dfreqresp ?w ?n ?whole ~system () =
   Py.Module.get_function_with_keywords __wrap_namespace "dfreqresp"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("whole", Wrap_utils.Option.map whole Py.Bool.of_bool); ("system", Some(system ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let dimpulse ?x0 ?t ?n ~system () =
   Py.Module.get_function_with_keywords __wrap_namespace "dimpulse"
     [||]
     (Wrap_utils.keyword_args [("x0", Wrap_utils.Option.map x0 Np.Obj.to_pyobject); ("t", Wrap_utils.Option.map t Np.Obj.to_pyobject); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("system", Some(system ))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let dlsim ?t ?x0 ~system ~u () =
   Py.Module.get_function_with_keywords __wrap_namespace "dlsim"
     [||]
     (Wrap_utils.keyword_args [("t", Wrap_utils.Option.map t Np.Obj.to_pyobject); ("x0", Wrap_utils.Option.map x0 Np.Obj.to_pyobject); ("system", Some(system )); ("u", Some(u |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
let dot ?out ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "dot"
     [||]
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let dstep ?x0 ?t ?n ~system () =
   Py.Module.get_function_with_keywords __wrap_namespace "dstep"
     [||]
     (Wrap_utils.keyword_args [("x0", Wrap_utils.Option.map x0 Np.Obj.to_pyobject); ("t", Wrap_utils.Option.map t Np.Obj.to_pyobject); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("system", Some(system ))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let freqresp ?w ?n ~system () =
   Py.Module.get_function_with_keywords __wrap_namespace "freqresp"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("system", Some(system ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let freqs ?worN ?plot ~b ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "freqs"
                       [||]
                       (Wrap_utils.keyword_args [("worN", Wrap_utils.Option.map worN (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("plot", plot); ("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let freqs_zpk ?worN ~z ~p ~k () =
                     Py.Module.get_function_with_keywords __wrap_namespace "freqs_zpk"
                       [||]
                       (Wrap_utils.keyword_args [("worN", Wrap_utils.Option.map worN (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("z", Some(z |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject)); ("k", Some(k |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let freqz ?a ?worN ?whole ?plot ?fs ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "freqz"
                       [||]
                       (Wrap_utils.keyword_args [("a", Wrap_utils.Option.map a Np.Obj.to_pyobject); ("worN", Wrap_utils.Option.map worN (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("whole", Wrap_utils.Option.map whole Py.Bool.of_bool); ("plot", plot); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let freqz_zpk ?worN ?whole ?fs ~z ~p ~k () =
                     Py.Module.get_function_with_keywords __wrap_namespace "freqz_zpk"
                       [||]
                       (Wrap_utils.keyword_args [("worN", Wrap_utils.Option.map worN (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("whole", Wrap_utils.Option.map whole Py.Bool.of_bool); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("z", Some(z |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject)); ("k", Some(k |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let impulse ?x0 ?t ?n ~system () =
   Py.Module.get_function_with_keywords __wrap_namespace "impulse"
     [||]
     (Wrap_utils.keyword_args [("X0", Wrap_utils.Option.map x0 Np.Obj.to_pyobject); ("T", Wrap_utils.Option.map t Np.Obj.to_pyobject); ("N", Wrap_utils.Option.map n Py.Int.of_int); ("system", Some(system ))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let impulse2 ?x0 ?t ?n ?kwargs ~system () =
   Py.Module.get_function_with_keywords __wrap_namespace "impulse2"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("X0", x0); ("T", t); ("N", Wrap_utils.Option.map n Py.Int.of_int); ("system", Some(system ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let linspace ?num ?endpoint ?retstep ?dtype ?axis ~start ~stop () =
   Py.Module.get_function_with_keywords __wrap_namespace "linspace"
     [||]
     (Wrap_utils.keyword_args [("num", Wrap_utils.Option.map num Py.Int.of_int); ("endpoint", Wrap_utils.Option.map endpoint Py.Bool.of_bool); ("retstep", Wrap_utils.Option.map retstep Py.Bool.of_bool); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("start", Some(start |> Np.Obj.to_pyobject)); ("stop", Some(stop |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))
let lsim ?x0 ?interp ~system ~u ~t () =
   Py.Module.get_function_with_keywords __wrap_namespace "lsim"
     [||]
     (Wrap_utils.keyword_args [("X0", Wrap_utils.Option.map x0 Np.Obj.to_pyobject); ("interp", Wrap_utils.Option.map interp Py.Bool.of_bool); ("system", Some(system )); ("U", Some(u |> Np.Obj.to_pyobject)); ("T", Some(t |> Np.Obj.to_pyobject))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
let lsim2 ?u ?t ?x0 ?kwargs ~system () =
   Py.Module.get_function_with_keywords __wrap_namespace "lsim2"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("U", Wrap_utils.Option.map u Np.Obj.to_pyobject); ("T", Wrap_utils.Option.map t Np.Obj.to_pyobject); ("X0", x0); ("system", Some(system ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
                  let nan_to_num ?copy ?nan ?posinf ?neginf ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "nan_to_num"
                       [||]
                       (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("nan", Wrap_utils.Option.map nan (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("posinf", Wrap_utils.Option.map posinf (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("neginf", Wrap_utils.Option.map neginf (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("x", Some(x |> (function
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let normalize ~b ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "normalize"
     [||]
     (Wrap_utils.keyword_args [("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let ones ?dtype ?order ~shape () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ones"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let place_poles ?method_ ?rtol ?maxiter ~a ~b ~poles () =
                     Py.Module.get_function_with_keywords __wrap_namespace "place_poles"
                       [||]
                       (Wrap_utils.keyword_args [("method", Wrap_utils.Option.map method_ (function
| `YT -> Py.String.of_string "YT"
| `KNV0 -> Py.String.of_string "KNV0"
)); ("rtol", Wrap_utils.Option.map rtol Py.Float.of_float); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("A", Some(a )); ("B", Some(b )); ("poles", Some(poles |> Np.Obj.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3)), (Py.Float.to_float (Py.Tuple.get x 4)), (Py.Int.to_int (Py.Tuple.get x 5))))
let real val_ =
   Py.Module.get_function_with_keywords __wrap_namespace "real"
     [||]
     (Wrap_utils.keyword_args [("val", Some(val_ |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let s_qr ?overwrite_a ?lwork ?mode ?pivoting ?check_finite ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "s_qr"
                       [||]
                       (Wrap_utils.keyword_args [("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("lwork", Wrap_utils.Option.map lwork Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Full -> Py.String.of_string "full"
| `R -> Py.String.of_string "r"
| `Economic -> Py.String.of_string "economic"
| `Raw -> Py.String.of_string "raw"
)); ("pivoting", Wrap_utils.Option.map pivoting Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
let squeeze ?axis ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "squeeze"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let ss2tf ?input ~a ~b ~c ~d () =
   Py.Module.get_function_with_keywords __wrap_namespace "ss2tf"
     [||]
     (Wrap_utils.keyword_args [("input", Wrap_utils.Option.map input Py.Int.of_int); ("A", Some(a |> Np.Obj.to_pyobject)); ("B", Some(b |> Np.Obj.to_pyobject)); ("C", Some(c |> Np.Obj.to_pyobject)); ("D", Some(d |> Np.Obj.to_pyobject))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let ss2zpk ?input ~a ~b ~c ~d () =
   Py.Module.get_function_with_keywords __wrap_namespace "ss2zpk"
     [||]
     (Wrap_utils.keyword_args [("input", Wrap_utils.Option.map input Py.Int.of_int); ("A", Some(a |> Np.Obj.to_pyobject)); ("B", Some(b |> Np.Obj.to_pyobject)); ("C", Some(c |> Np.Obj.to_pyobject)); ("D", Some(d |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let step ?x0 ?t ?n ~system () =
   Py.Module.get_function_with_keywords __wrap_namespace "step"
     [||]
     (Wrap_utils.keyword_args [("X0", Wrap_utils.Option.map x0 Np.Obj.to_pyobject); ("T", Wrap_utils.Option.map t Np.Obj.to_pyobject); ("N", Wrap_utils.Option.map n Py.Int.of_int); ("system", Some(system ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let step2 ?x0 ?t ?n ?kwargs ~system () =
   Py.Module.get_function_with_keywords __wrap_namespace "step2"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("X0", Wrap_utils.Option.map x0 Np.Obj.to_pyobject); ("T", Wrap_utils.Option.map t Np.Obj.to_pyobject); ("N", Wrap_utils.Option.map n Py.Int.of_int); ("system", Some(system ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let tf2ss ~num ~den () =
   Py.Module.get_function_with_keywords __wrap_namespace "tf2ss"
     [||]
     (Wrap_utils.keyword_args [("num", Some(num )); ("den", Some(den ))])

let tf2zpk ~b ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "tf2zpk"
     [||]
     (Wrap_utils.keyword_args [("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let transpose ?axes ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "transpose"
     [||]
     (Wrap_utils.keyword_args [("axes", axes); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let zeros ?dtype ?order ~shape () =
                     Py.Module.get_function_with_keywords __wrap_namespace "zeros"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let zeros_like ?dtype ?order ?subok ?shape ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "zeros_like"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `A -> Py.String.of_string "A"
| `PyObject x -> Wrap_utils.id x
)); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("shape", Wrap_utils.Option.map shape (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let zpk2ss ~z ~p ~k () =
   Py.Module.get_function_with_keywords __wrap_namespace "zpk2ss"
     [||]
     (Wrap_utils.keyword_args [("z", Some(z )); ("p", Some(p )); ("k", Some(k |> Py.Float.of_float))])

let zpk2tf ~z ~p ~k () =
   Py.Module.get_function_with_keywords __wrap_namespace "zpk2tf"
     [||]
     (Wrap_utils.keyword_args [("z", Some(z |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject)); ("k", Some(k |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))

end
module Signaltools = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.signal.signaltools"

let get_py name = Py.Module.get __wrap_namespace name
module CKDTree = struct
type tag = [`CKDTree]
type t = [`CKDTree | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?leafsize ?compact_nodes ?copy_data ?balanced_tree ?boxsize ~data () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cKDTree"
                       [||]
                       (Wrap_utils.keyword_args [("leafsize", leafsize); ("compact_nodes", Wrap_utils.Option.map compact_nodes Py.Bool.of_bool); ("copy_data", Wrap_utils.Option.map copy_data Py.Bool.of_bool); ("balanced_tree", Wrap_utils.Option.map balanced_tree Py.Bool.of_bool); ("boxsize", Wrap_utils.Option.map boxsize (function
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
)); ("data", Some(data |> Np.Obj.to_pyobject))])
                       |> of_pyobject
                  let count_neighbors ?p ?weights ?cumulative ~other ~r self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "count_neighbors"
                       [||]
                       (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p Py.Float.of_float); ("weights", Wrap_utils.Option.map weights (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple x -> Wrap_utils.id x
)); ("cumulative", Wrap_utils.Option.map cumulative Py.Bool.of_bool); ("other", Some(other )); ("r", Some(r |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])

                  let query_ball_point ?p ?eps ~x ~r self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "query_ball_point"
                       [||]
                       (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p Py.Float.of_float); ("eps", eps); ("x", Some(x |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Shape_tuple_self_m_ x -> Wrap_utils.id x
))); ("r", Some(r |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])

let query_ball_tree ?p ?eps ~other ~r self =
   Py.Module.get_function_with_keywords (to_pyobject self) "query_ball_tree"
     [||]
     (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p Py.Float.of_float); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("other", Some(other )); ("r", Some(r |> Py.Float.of_float))])

let query_pairs ?p ?eps ~r self =
   Py.Module.get_function_with_keywords (to_pyobject self) "query_pairs"
     [||]
     (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p Py.Float.of_float); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("r", Some(r |> Py.Float.of_float))])

                  let sparse_distance_matrix ?p ~other ~max_distance self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "sparse_distance_matrix"
                       [||]
                       (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p (function
| `T1_p_infinity x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)); ("other", Some(other )); ("max_distance", Some(max_distance |> Py.Float.of_float))])


let data_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "data" with
  | None -> failwith "attribute data not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let data self = match data_opt self with
  | None -> raise Not_found
  | Some x -> x

let leafsize_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "leafsize" with
  | None -> failwith "attribute leafsize not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let leafsize self = match leafsize_opt self with
  | None -> raise Not_found
  | Some x -> x

let m_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "m" with
  | None -> failwith "attribute m not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let m self = match m_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n" with
  | None -> failwith "attribute n not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n self = match n_opt self with
  | None -> raise Not_found
  | Some x -> x

let maxes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "maxes" with
  | None -> failwith "attribute maxes not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let maxes self = match maxes_opt self with
  | None -> raise Not_found
  | Some x -> x

let mins_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "mins" with
  | None -> failwith "attribute mins not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let mins self = match mins_opt self with
  | None -> raise Not_found
  | Some x -> x

let tree_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "tree" with
  | None -> failwith "attribute tree not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let tree self = match tree_opt self with
  | None -> raise Not_found
  | Some x -> x

let size_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "size" with
  | None -> failwith "attribute size not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let size self = match size_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Sp_fft = struct
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


end
let axis_reverse ?axis ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "axis_reverse"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("a", Some(a ))])

let axis_slice ?start ?stop ?step ?axis ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "axis_slice"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("stop", stop); ("step", step); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("a", Some(a |> Np.Obj.to_pyobject))])

let callable obj =
   Py.Module.get_function_with_keywords __wrap_namespace "callable"
     [||]
     (Wrap_utils.keyword_args [("obj", Some(obj ))])

                  let cheby1 ?btype ?analog ?output ?fs ~n ~rp ~wn () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cheby1"
                       [||]
                       (Wrap_utils.keyword_args [("btype", Wrap_utils.Option.map btype (function
| `Lowpass -> Py.String.of_string "lowpass"
| `Highpass -> Py.String.of_string "highpass"
| `Bandpass -> Py.String.of_string "bandpass"
| `Bandstop -> Py.String.of_string "bandstop"
)); ("analog", Wrap_utils.Option.map analog Py.Bool.of_bool); ("output", Wrap_utils.Option.map output (function
| `Ba -> Py.String.of_string "ba"
| `Zpk -> Py.String.of_string "zpk"
| `Sos -> Py.String.of_string "sos"
)); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("N", Some(n |> Py.Int.of_int)); ("rp", Some(rp |> Py.Float.of_float)); ("Wn", Some(wn |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let choose_conv_method ?mode ?measure ~in1 ~in2 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "choose_conv_method"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Full -> Py.String.of_string "full"
| `Valid -> Py.String.of_string "valid"
| `Same -> Py.String.of_string "same"
)); ("measure", Wrap_utils.Option.map measure Py.Bool.of_bool); ("in1", Some(in1 |> Np.Obj.to_pyobject)); ("in2", Some(in2 |> Np.Obj.to_pyobject))])
                       |> (fun x -> ((Py.String.to_string (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let cmplx_sort p =
   Py.Module.get_function_with_keywords __wrap_namespace "cmplx_sort"
     [||]
     (Wrap_utils.keyword_args [("p", Some(p |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let const_ext ?axis ~x ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "const_ext"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject)); ("n", Some(n |> Py.Int.of_int))])

                  let convolve ?mode ?method_ ~in1 ~in2 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "convolve"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Full -> Py.String.of_string "full"
| `Valid -> Py.String.of_string "valid"
| `Same -> Py.String.of_string "same"
)); ("method", Wrap_utils.Option.map method_ (function
| `Auto -> Py.String.of_string "auto"
| `Direct -> Py.String.of_string "direct"
| `Fft -> Py.String.of_string "fft"
)); ("in1", Some(in1 |> Np.Obj.to_pyobject)); ("in2", Some(in2 |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let convolve2d ?mode ?boundary ?fillvalue ~in1 ~in2 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "convolve2d"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Full -> Py.String.of_string "full"
| `Valid -> Py.String.of_string "valid"
| `Same -> Py.String.of_string "same"
)); ("boundary", Wrap_utils.Option.map boundary (function
| `Fill -> Py.String.of_string "fill"
| `Wrap -> Py.String.of_string "wrap"
| `Symm -> Py.String.of_string "symm"
)); ("fillvalue", Wrap_utils.Option.map fillvalue (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("in1", Some(in1 |> Np.Obj.to_pyobject)); ("in2", Some(in2 |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let correlate ?mode ?method_ ~in1 ~in2 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "correlate"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Full -> Py.String.of_string "full"
| `Valid -> Py.String.of_string "valid"
| `Same -> Py.String.of_string "same"
)); ("method", Wrap_utils.Option.map method_ (function
| `Auto -> Py.String.of_string "auto"
| `Direct -> Py.String.of_string "direct"
| `Fft -> Py.String.of_string "fft"
)); ("in1", Some(in1 |> Np.Obj.to_pyobject)); ("in2", Some(in2 |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let correlate2d ?mode ?boundary ?fillvalue ~in1 ~in2 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "correlate2d"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Full -> Py.String.of_string "full"
| `Valid -> Py.String.of_string "valid"
| `Same -> Py.String.of_string "same"
)); ("boundary", Wrap_utils.Option.map boundary (function
| `Fill -> Py.String.of_string "fill"
| `Wrap -> Py.String.of_string "wrap"
| `Symm -> Py.String.of_string "symm"
)); ("fillvalue", Wrap_utils.Option.map fillvalue (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("in1", Some(in1 |> Np.Obj.to_pyobject)); ("in2", Some(in2 |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let decimate ?n ?ftype ?axis ?zero_phase ~x ~q () =
                     Py.Module.get_function_with_keywords __wrap_namespace "decimate"
                       [||]
                       (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("ftype", Wrap_utils.Option.map ftype (function
| `Fir -> Py.String.of_string "fir"
| `Iir -> Py.String.of_string "iir"
| `T_dlti_instance x -> Wrap_utils.id x
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("zero_phase", Wrap_utils.Option.map zero_phase Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject)); ("q", Some(q |> Py.Int.of_int))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let deconvolve ~signal ~divisor () =
   Py.Module.get_function_with_keywords __wrap_namespace "deconvolve"
     [||]
     (Wrap_utils.keyword_args [("signal", Some(signal |> Np.Obj.to_pyobject)); ("divisor", Some(divisor |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let detrend ?axis ?type_ ?bp ?overwrite_data ~data () =
                     Py.Module.get_function_with_keywords __wrap_namespace "detrend"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("type", Wrap_utils.Option.map type_ (function
| `Linear -> Py.String.of_string "linear"
| `Constant -> Py.String.of_string "constant"
)); ("bp", bp); ("overwrite_data", Wrap_utils.Option.map overwrite_data Py.Bool.of_bool); ("data", Some(data |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let even_ext ?axis ~x ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "even_ext"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject)); ("n", Some(n |> Py.Int.of_int))])

                  let factorial ?exact ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "factorial"
                       [||]
                       (Wrap_utils.keyword_args [("exact", Wrap_utils.Option.map exact Py.Bool.of_bool); ("n", Some(n |> (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)))])

                  let fftconvolve ?mode ?axes ~in1 ~in2 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fftconvolve"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Full -> Py.String.of_string "full"
| `Valid -> Py.String.of_string "valid"
| `Same -> Py.String.of_string "same"
)); ("axes", Wrap_utils.Option.map axes (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("in1", Some(in1 |> Np.Obj.to_pyobject)); ("in2", Some(in2 |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let filtfilt ?axis ?padtype ?padlen ?method_ ?irlen ~b ~a ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "filtfilt"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("padtype", Wrap_utils.Option.map padtype (function
| `S x -> Py.String.of_string x
| `None -> Py.none
)); ("padlen", Wrap_utils.Option.map padlen Py.Int.of_int); ("method", Wrap_utils.Option.map method_ Py.String.of_string); ("irlen", Wrap_utils.Option.map irlen Py.Int.of_int); ("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject)); ("x", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let firwin ?width ?window ?pass_zero ?scale ?nyq ?fs ~numtaps ~cutoff () =
                     Py.Module.get_function_with_keywords __wrap_namespace "firwin"
                       [||]
                       (Wrap_utils.keyword_args [("width", Wrap_utils.Option.map width Py.Float.of_float); ("window", Wrap_utils.Option.map window (function
| `Tuple_of_string_and_parameter_values x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("pass_zero", Wrap_utils.Option.map pass_zero (function
| `Highpass -> Py.String.of_string "highpass"
| `Lowpass -> Py.String.of_string "lowpass"
| `Bandstop -> Py.String.of_string "bandstop"
| `Bool x -> Py.Bool.of_bool x
| `Bandpass -> Py.String.of_string "bandpass"
)); ("scale", Wrap_utils.Option.map scale Py.Float.of_float); ("nyq", Wrap_utils.Option.map nyq Py.Float.of_float); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("numtaps", Some(numtaps |> Py.Int.of_int)); ("cutoff", Some(cutoff |> (function
| `F x -> Py.Float.of_float x
| `T1D_array_like x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let gcd ~x ~y () =
   Py.Module.get_function_with_keywords __wrap_namespace "gcd"
     (Array.of_list @@ List.concat [[x ];[y ]])
     []

                  let get_window ?fftbins ~window ~nx () =
                     Py.Module.get_function_with_keywords __wrap_namespace "get_window"
                       [||]
                       (Wrap_utils.keyword_args [("fftbins", Wrap_utils.Option.map fftbins Py.Bool.of_bool); ("window", Some(window |> (function
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
))); ("Nx", Some(nx |> Py.Int.of_int))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let hilbert ?n ?axis ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "hilbert"
     [||]
     (Wrap_utils.keyword_args [("N", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let hilbert2 ?n ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "hilbert2"
                       [||]
                       (Wrap_utils.keyword_args [("N", Wrap_utils.Option.map n (function
| `Tuple_of_two_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("x", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let invres ?tol ?rtype ~r ~p ~k () =
                     Py.Module.get_function_with_keywords __wrap_namespace "invres"
                       [||]
                       (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("rtype", Wrap_utils.Option.map rtype (function
| `Avg -> Py.String.of_string "avg"
| `Min -> Py.String.of_string "min"
| `Max -> Py.String.of_string "max"
)); ("r", Some(r |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject)); ("k", Some(k |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let invresz ?tol ?rtype ~r ~p ~k () =
                     Py.Module.get_function_with_keywords __wrap_namespace "invresz"
                       [||]
                       (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("rtype", Wrap_utils.Option.map rtype (function
| `Avg -> Py.String.of_string "avg"
| `Min -> Py.String.of_string "min"
| `Max -> Py.String.of_string "max"
)); ("r", Some(r |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject)); ("k", Some(k |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let lambertw ?k ?tol ~z () =
   Py.Module.get_function_with_keywords __wrap_namespace "lambertw"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("z", Some(z |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let lfilter ?axis ?zi ~b ~a ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "lfilter"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("zi", Wrap_utils.Option.map zi Np.Obj.to_pyobject); ("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject)); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let lfilter_zi ~b ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "lfilter_zi"
     [||]
     (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))])

let lfiltic ?x ~b ~a ~y () =
   Py.Module.get_function_with_keywords __wrap_namespace "lfiltic"
     [||]
     (Wrap_utils.keyword_args [("x", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let medfilt ?kernel_size ~volume () =
   Py.Module.get_function_with_keywords __wrap_namespace "medfilt"
     [||]
     (Wrap_utils.keyword_args [("kernel_size", Wrap_utils.Option.map kernel_size Np.Obj.to_pyobject); ("volume", Some(volume |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let medfilt2d ?kernel_size ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "medfilt2d"
     [||]
     (Wrap_utils.keyword_args [("kernel_size", Wrap_utils.Option.map kernel_size Np.Obj.to_pyobject); ("input", Some(input |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let oaconvolve ?mode ?axes ~in1 ~in2 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "oaconvolve"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Full -> Py.String.of_string "full"
| `Valid -> Py.String.of_string "valid"
| `Same -> Py.String.of_string "same"
)); ("axes", Wrap_utils.Option.map axes (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("in1", Some(in1 |> Np.Obj.to_pyobject)); ("in2", Some(in2 |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let odd_ext ?axis ~x ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "odd_ext"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject)); ("n", Some(n |> Py.Int.of_int))])

let order_filter ~a ~domain ~rank () =
   Py.Module.get_function_with_keywords __wrap_namespace "order_filter"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Np.Obj.to_pyobject)); ("domain", Some(domain |> Np.Obj.to_pyobject)); ("rank", Some(rank |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let resample ?t ?axis ?window ~x ~num () =
                     Py.Module.get_function_with_keywords __wrap_namespace "resample"
                       [||]
                       (Wrap_utils.keyword_args [("t", Wrap_utils.Option.map t Np.Obj.to_pyobject); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("window", Wrap_utils.Option.map window (function
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Np.Obj.to_pyobject x
| `Callable x -> Wrap_utils.id x
)); ("x", Some(x |> Np.Obj.to_pyobject)); ("num", Some(num |> Py.Int.of_int))])

                  let resample_poly ?axis ?window ?padtype ?cval ~x ~up ~down () =
                     Py.Module.get_function_with_keywords __wrap_namespace "resample_poly"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("window", Wrap_utils.Option.map window (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("padtype", Wrap_utils.Option.map padtype Py.String.of_string); ("cval", Wrap_utils.Option.map cval Py.Float.of_float); ("x", Some(x |> Np.Obj.to_pyobject)); ("up", Some(up |> Py.Int.of_int)); ("down", Some(down |> Py.Int.of_int))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let residue ?tol ?rtype ~b ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "residue"
                       [||]
                       (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("rtype", Wrap_utils.Option.map rtype (function
| `Avg -> Py.String.of_string "avg"
| `Min -> Py.String.of_string "min"
| `Max -> Py.String.of_string "max"
)); ("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
                  let residuez ?tol ?rtype ~b ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "residuez"
                       [||]
                       (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("rtype", Wrap_utils.Option.map rtype (function
| `Avg -> Py.String.of_string "avg"
| `Min -> Py.String.of_string "min"
| `Max -> Py.String.of_string "max"
)); ("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
let sosfilt ?axis ?zi ~sos ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "sosfilt"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("zi", Wrap_utils.Option.map zi Np.Obj.to_pyobject); ("sos", Some(sos |> Np.Obj.to_pyobject)); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let sosfilt_zi sos =
   Py.Module.get_function_with_keywords __wrap_namespace "sosfilt_zi"
     [||]
     (Wrap_utils.keyword_args [("sos", Some(sos |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let sosfiltfilt ?axis ?padtype ?padlen ~sos ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "sosfiltfilt"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("padtype", Wrap_utils.Option.map padtype (function
| `S x -> Py.String.of_string x
| `None -> Py.none
)); ("padlen", Wrap_utils.Option.map padlen Py.Int.of_int); ("sos", Some(sos |> Np.Obj.to_pyobject)); ("x", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let sp_fft ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "sp_fft"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

                  let unique_roots ?tol ?rtype ~p () =
                     Py.Module.get_function_with_keywords __wrap_namespace "unique_roots"
                       [||]
                       (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("rtype", Wrap_utils.Option.map rtype (function
| `Max -> Py.String.of_string "max"
| `Maximum -> Py.String.of_string "maximum"
| `Min -> Py.String.of_string "min"
| `Minimum -> Py.String.of_string "minimum"
| `Avg -> Py.String.of_string "avg"
| `Mean -> Py.String.of_string "mean"
)); ("p", Some(p |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let upfirdn ?up ?down ?axis ?mode ?cval ~h ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "upfirdn"
     [||]
     (Wrap_utils.keyword_args [("up", Wrap_utils.Option.map up Py.Int.of_int); ("down", Wrap_utils.Option.map down Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("mode", Wrap_utils.Option.map mode Py.String.of_string); ("cval", Wrap_utils.Option.map cval Py.Float.of_float); ("h", Some(h |> Np.Obj.to_pyobject)); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let vectorstrength ~events ~period () =
                     Py.Module.get_function_with_keywords __wrap_namespace "vectorstrength"
                       [||]
                       (Wrap_utils.keyword_args [("events", Some(events )); ("period", Some(period |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let wiener ?mysize ?noise ~im () =
                     Py.Module.get_function_with_keywords __wrap_namespace "wiener"
                       [||]
                       (Wrap_utils.keyword_args [("mysize", Wrap_utils.Option.map mysize (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("noise", Wrap_utils.Option.map noise Py.Float.of_float); ("im", Some(im |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
module Sigtools = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.signal.sigtools"

let get_py name = Py.Module.get __wrap_namespace name

end
module Spectral = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.signal.spectral"

let get_py name = Py.Module.get __wrap_namespace name
module Sp_fft = struct
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


end
                  let check_COLA ?tol ~window ~nperseg ~noverlap () =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_COLA"
                       [||]
                       (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("window", Some(window |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
))); ("nperseg", Some(nperseg |> Py.Int.of_int)); ("noverlap", Some(noverlap |> Py.Int.of_int))])
                       |> Py.Bool.to_bool
                  let check_NOLA ?tol ~window ~nperseg ~noverlap () =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_NOLA"
                       [||]
                       (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("window", Some(window |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
))); ("nperseg", Some(nperseg |> Py.Int.of_int)); ("noverlap", Some(noverlap |> Py.Int.of_int))])
                       |> Py.Bool.to_bool
                  let coherence ?fs ?window ?nperseg ?noverlap ?nfft ?detrend ?axis ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "coherence"
                       [||]
                       (Wrap_utils.keyword_args [("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("window", Wrap_utils.Option.map window (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("nperseg", Wrap_utils.Option.map nperseg Py.Int.of_int); ("noverlap", Wrap_utils.Option.map noverlap Py.Int.of_int); ("nfft", Wrap_utils.Option.map nfft Py.Int.of_int); ("detrend", Wrap_utils.Option.map detrend (function
| `T_False_ x -> Wrap_utils.id x
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let const_ext ?axis ~x ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "const_ext"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject)); ("n", Some(n |> Py.Int.of_int))])

                  let csd ?fs ?window ?nperseg ?noverlap ?nfft ?detrend ?return_onesided ?scaling ?axis ?average ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "csd"
                       [||]
                       (Wrap_utils.keyword_args [("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("window", Wrap_utils.Option.map window (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("nperseg", Wrap_utils.Option.map nperseg Py.Int.of_int); ("noverlap", Wrap_utils.Option.map noverlap Py.Int.of_int); ("nfft", Wrap_utils.Option.map nfft Py.Int.of_int); ("detrend", Wrap_utils.Option.map detrend (function
| `T_False_ x -> Wrap_utils.id x
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("return_onesided", Wrap_utils.Option.map return_onesided Py.Bool.of_bool); ("scaling", Wrap_utils.Option.map scaling (function
| `Density -> Py.String.of_string "density"
| `Spectrum -> Py.String.of_string "spectrum"
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("average", Wrap_utils.Option.map average (function
| `Mean -> Py.String.of_string "mean"
| `Median -> Py.String.of_string "median"
)); ("x", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let even_ext ?axis ~x ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "even_ext"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject)); ("n", Some(n |> Py.Int.of_int))])

                  let get_window ?fftbins ~window ~nx () =
                     Py.Module.get_function_with_keywords __wrap_namespace "get_window"
                       [||]
                       (Wrap_utils.keyword_args [("fftbins", Wrap_utils.Option.map fftbins Py.Bool.of_bool); ("window", Some(window |> (function
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
))); ("Nx", Some(nx |> Py.Int.of_int))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let istft ?fs ?window ?nperseg ?noverlap ?nfft ?input_onesided ?boundary ?time_axis ?freq_axis ~zxx () =
                     Py.Module.get_function_with_keywords __wrap_namespace "istft"
                       [||]
                       (Wrap_utils.keyword_args [("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("window", Wrap_utils.Option.map window (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("nperseg", Wrap_utils.Option.map nperseg Py.Int.of_int); ("noverlap", Wrap_utils.Option.map noverlap Py.Int.of_int); ("nfft", Wrap_utils.Option.map nfft Py.Int.of_int); ("input_onesided", Wrap_utils.Option.map input_onesided Py.Bool.of_bool); ("boundary", Wrap_utils.Option.map boundary Py.Bool.of_bool); ("time_axis", Wrap_utils.Option.map time_axis Py.Int.of_int); ("freq_axis", Wrap_utils.Option.map freq_axis Py.Int.of_int); ("Zxx", Some(zxx |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let lombscargle ?precenter ?normalize ~x ~y ~freqs () =
   Py.Module.get_function_with_keywords __wrap_namespace "lombscargle"
     [||]
     (Wrap_utils.keyword_args [("precenter", Wrap_utils.Option.map precenter Py.Bool.of_bool); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject)); ("freqs", Some(freqs |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let odd_ext ?axis ~x ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "odd_ext"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject)); ("n", Some(n |> Py.Int.of_int))])

                  let periodogram ?fs ?window ?nfft ?detrend ?return_onesided ?scaling ?axis ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "periodogram"
                       [||]
                       (Wrap_utils.keyword_args [("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("window", Wrap_utils.Option.map window (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("nfft", Wrap_utils.Option.map nfft Py.Int.of_int); ("detrend", Wrap_utils.Option.map detrend (function
| `T_False_ x -> Wrap_utils.id x
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("return_onesided", Wrap_utils.Option.map return_onesided Py.Bool.of_bool); ("scaling", Wrap_utils.Option.map scaling (function
| `Density -> Py.String.of_string "density"
| `Spectrum -> Py.String.of_string "spectrum"
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let sp_fft ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "sp_fft"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

                  let spectrogram ?fs ?window ?nperseg ?noverlap ?nfft ?detrend ?return_onesided ?scaling ?axis ?mode ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "spectrogram"
                       [||]
                       (Wrap_utils.keyword_args [("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("window", Wrap_utils.Option.map window (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("nperseg", Wrap_utils.Option.map nperseg Py.Int.of_int); ("noverlap", Wrap_utils.Option.map noverlap Py.Int.of_int); ("nfft", Wrap_utils.Option.map nfft Py.Int.of_int); ("detrend", Wrap_utils.Option.map detrend (function
| `T_False_ x -> Wrap_utils.id x
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("return_onesided", Wrap_utils.Option.map return_onesided Py.Bool.of_bool); ("scaling", Wrap_utils.Option.map scaling (function
| `Density -> Py.String.of_string "density"
| `Spectrum -> Py.String.of_string "spectrum"
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("mode", Wrap_utils.Option.map mode Py.String.of_string); ("x", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
                  let stft ?fs ?window ?nperseg ?noverlap ?nfft ?detrend ?return_onesided ?boundary ?padded ?axis ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "stft"
                       [||]
                       (Wrap_utils.keyword_args [("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("window", Wrap_utils.Option.map window (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("nperseg", Wrap_utils.Option.map nperseg Py.Int.of_int); ("noverlap", Wrap_utils.Option.map noverlap Py.Int.of_int); ("nfft", Wrap_utils.Option.map nfft Py.Int.of_int); ("detrend", Wrap_utils.Option.map detrend (function
| `T_False_ x -> Wrap_utils.id x
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("return_onesided", Wrap_utils.Option.map return_onesided Py.Bool.of_bool); ("boundary", Wrap_utils.Option.map boundary (function
| `S x -> Py.String.of_string x
| `None -> Py.none
)); ("padded", Wrap_utils.Option.map padded Py.Bool.of_bool); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
                  let welch ?fs ?window ?nperseg ?noverlap ?nfft ?detrend ?return_onesided ?scaling ?axis ?average ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "welch"
                       [||]
                       (Wrap_utils.keyword_args [("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("window", Wrap_utils.Option.map window (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("nperseg", Wrap_utils.Option.map nperseg Py.Int.of_int); ("noverlap", Wrap_utils.Option.map noverlap Py.Int.of_int); ("nfft", Wrap_utils.Option.map nfft Py.Int.of_int); ("detrend", Wrap_utils.Option.map detrend (function
| `T_False_ x -> Wrap_utils.id x
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("return_onesided", Wrap_utils.Option.map return_onesided Py.Bool.of_bool); ("scaling", Wrap_utils.Option.map scaling (function
| `Density -> Py.String.of_string "density"
| `Spectrum -> Py.String.of_string "spectrum"
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("average", Wrap_utils.Option.map average (function
| `Mean -> Py.String.of_string "mean"
| `Median -> Py.String.of_string "median"
)); ("x", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let zero_ext ?axis ~x ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "zero_ext"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject)); ("n", Some(n |> Py.Int.of_int))])


end
module Spline = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.signal.spline"

let get_py name = Py.Module.get __wrap_namespace name
let sepfir2d ~input ~hrow ~hcol () =
   Py.Module.get_function_with_keywords __wrap_namespace "sepfir2d"
     [||]
     (Wrap_utils.keyword_args [("input", Some(input )); ("hrow", Some(hrow )); ("hcol", Some(hcol ))])


end
module Waveforms = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.signal.waveforms"

let get_py name = Py.Module.get __wrap_namespace name
                  let asarray ?dtype ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let chirp ?method_ ?phi ?vertex_zero ~t ~f0 ~t1 ~f1 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "chirp"
                       [||]
                       (Wrap_utils.keyword_args [("method", Wrap_utils.Option.map method_ (function
| `Linear -> Py.String.of_string "linear"
| `Quadratic -> Py.String.of_string "quadratic"
| `Logarithmic -> Py.String.of_string "logarithmic"
| `Hyperbolic -> Py.String.of_string "hyperbolic"
)); ("phi", Wrap_utils.Option.map phi Py.Float.of_float); ("vertex_zero", Wrap_utils.Option.map vertex_zero Py.Bool.of_bool); ("t", Some(t |> Np.Obj.to_pyobject)); ("f0", Some(f0 |> Py.Float.of_float)); ("t1", Some(t1 |> Py.Float.of_float)); ("f1", Some(f1 |> Py.Float.of_float))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let cos ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cos"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let exp ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "exp"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let extract ~condition ~arr () =
   Py.Module.get_function_with_keywords __wrap_namespace "extract"
     [||]
     (Wrap_utils.keyword_args [("condition", Some(condition |> Np.Obj.to_pyobject)); ("arr", Some(arr |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let gausspulse ?fc ?bw ?bwr ?tpr ?retquad ?retenv ~t () =
                     Py.Module.get_function_with_keywords __wrap_namespace "gausspulse"
                       [||]
                       (Wrap_utils.keyword_args [("fc", Wrap_utils.Option.map fc Py.Int.of_int); ("bw", Wrap_utils.Option.map bw Py.Float.of_float); ("bwr", Wrap_utils.Option.map bwr Py.Float.of_float); ("tpr", Wrap_utils.Option.map tpr Py.Float.of_float); ("retquad", Wrap_utils.Option.map retquad Py.Bool.of_bool); ("retenv", Wrap_utils.Option.map retenv Py.Bool.of_bool); ("t", Some(t |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `The_string_cutoff_ x -> Wrap_utils.id x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
                  let log ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "log"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let mod_ ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "mod"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let place ~arr ~mask ~vals () =
   Py.Module.get_function_with_keywords __wrap_namespace "place"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr |> Np.Obj.to_pyobject)); ("mask", Some(mask |> Np.Obj.to_pyobject)); ("vals", Some(vals ))])

                  let polyint ?m ?k ~p () =
                     Py.Module.get_function_with_keywords __wrap_namespace "polyint"
                       [||]
                       (Wrap_utils.keyword_args [("m", Wrap_utils.Option.map m Py.Int.of_int); ("k", Wrap_utils.Option.map k (function
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `List_of_m_scalars x -> Wrap_utils.id x
| `Bool x -> Py.Bool.of_bool x
)); ("p", Some(p |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Poly1d x -> Wrap_utils.id x
)))])

                  let polyval ~p ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "polyval"
                       [||]
                       (Wrap_utils.keyword_args [("p", Some(p |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Poly1d_object x -> Wrap_utils.id x
))); ("x", Some(x |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Poly1d_object x -> Wrap_utils.id x
)))])

let sawtooth ?width ~t () =
   Py.Module.get_function_with_keywords __wrap_namespace "sawtooth"
     [||]
     (Wrap_utils.keyword_args [("width", Wrap_utils.Option.map width Np.Obj.to_pyobject); ("t", Some(t |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let sin ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "sin"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let sqrt ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "sqrt"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let square ?duty ~t () =
   Py.Module.get_function_with_keywords __wrap_namespace "square"
     [||]
     (Wrap_utils.keyword_args [("duty", Wrap_utils.Option.map duty Np.Obj.to_pyobject); ("t", Some(t |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let sweep_poly ?phi ~t ~poly () =
   Py.Module.get_function_with_keywords __wrap_namespace "sweep_poly"
     [||]
     (Wrap_utils.keyword_args [("phi", Wrap_utils.Option.map phi Py.Float.of_float); ("t", Some(t |> Np.Obj.to_pyobject)); ("poly", Some(poly ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let unit_impulse ?idx ?dtype ~shape () =
                     Py.Module.get_function_with_keywords __wrap_namespace "unit_impulse"
                       [||]
                       (Wrap_utils.keyword_args [("idx", Wrap_utils.Option.map idx (function
| `I x -> Py.Int.of_int x
| `Tuple_of_int x -> Wrap_utils.id x
| `Mid -> Py.String.of_string "mid"
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("shape", Some(shape |> (function
| `I x -> Py.Int.of_int x
| `Tuple_of_int x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let zeros ?dtype ?order ~shape () =
                     Py.Module.get_function_with_keywords __wrap_namespace "zeros"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
module Wavelets = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.signal.wavelets"

let get_py name = Py.Module.get __wrap_namespace name
let cascade ?j ~hk () =
   Py.Module.get_function_with_keywords __wrap_namespace "cascade"
     [||]
     (Wrap_utils.keyword_args [("J", Wrap_utils.Option.map j Py.Int.of_int); ("hk", Some(hk |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
                  let comb ?exact ?repetition ~n ~k () =
                     Py.Module.get_function_with_keywords __wrap_namespace "comb"
                       [||]
                       (Wrap_utils.keyword_args [("exact", Wrap_utils.Option.map exact Py.Bool.of_bool); ("repetition", Wrap_utils.Option.map repetition Py.Bool.of_bool); ("N", Some(n |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
))); ("k", Some(k |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)))])

                  let convolve ?mode ?method_ ~in1 ~in2 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "convolve"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Full -> Py.String.of_string "full"
| `Valid -> Py.String.of_string "valid"
| `Same -> Py.String.of_string "same"
)); ("method", Wrap_utils.Option.map method_ (function
| `Auto -> Py.String.of_string "auto"
| `Direct -> Py.String.of_string "direct"
| `Fft -> Py.String.of_string "fft"
)); ("in1", Some(in1 |> Np.Obj.to_pyobject)); ("in2", Some(in2 |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let cwt ?dtype ?kwargs ~data ~wavelet ~widths () =
   Py.Module.get_function_with_keywords __wrap_namespace "cwt"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("data", Some(data |> Np.Obj.to_pyobject)); ("wavelet", Some(wavelet )); ("widths", Some(widths ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let daub p =
   Py.Module.get_function_with_keywords __wrap_namespace "daub"
     [||]
     (Wrap_utils.keyword_args [("p", Some(p |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eig ?b ?left ?right ?overwrite_a ?overwrite_b ?check_finite ?homogeneous_eigvals ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "eig"
     [||]
     (Wrap_utils.keyword_args [("b", Wrap_utils.Option.map b Np.Obj.to_pyobject); ("left", Wrap_utils.Option.map left Py.Bool.of_bool); ("right", Wrap_utils.Option.map right Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("homogeneous_eigvals", Wrap_utils.Option.map homogeneous_eigvals Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
let morlet ?w ?s ?complete ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "morlet"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Py.Float.of_float); ("s", Wrap_utils.Option.map s Py.Float.of_float); ("complete", Wrap_utils.Option.map complete Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let morlet2 ?w ~m ~s () =
   Py.Module.get_function_with_keywords __wrap_namespace "morlet2"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Py.Float.of_float); ("M", Some(m |> Py.Int.of_int)); ("s", Some(s |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let qmf hk =
   Py.Module.get_function_with_keywords __wrap_namespace "qmf"
     [||]
     (Wrap_utils.keyword_args [("hk", Some(hk |> Np.Obj.to_pyobject))])

                  let ricker ~points ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ricker"
                       [||]
                       (Wrap_utils.keyword_args [("points", Some(points |> Py.Int.of_int)); ("a", Some(a |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
module Windows = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.signal.windows"

let get_py name = Py.Module.get __wrap_namespace name
module Windows = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.signal.windows.windows"

let get_py name = Py.Module.get __wrap_namespace name
module Sp_fft = struct
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


end
let barthann ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "barthann"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let bartlett ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "bartlett"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let blackman ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "blackman"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let blackmanharris ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "blackmanharris"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let bohman ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "bohman"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let boxcar ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "boxcar"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let chebwin ?sym ~m ~at () =
   Py.Module.get_function_with_keywords __wrap_namespace "chebwin"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int)); ("at", Some(at |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let cosine ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "cosine"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let dpss ?kmax ?sym ?norm ?return_ratios ~m ~nw () =
                     Py.Module.get_function_with_keywords __wrap_namespace "dpss"
                       [||]
                       (Wrap_utils.keyword_args [("Kmax", Wrap_utils.Option.map kmax Py.Int.of_int); ("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("norm", Wrap_utils.Option.map norm (function
| `T_2 x -> Wrap_utils.id x
| `Optional x -> (function
| `T_subsample_ x -> Wrap_utils.id x
| `None -> Py.none
) x
| `Approximate -> Py.String.of_string "approximate"
)); ("return_ratios", Wrap_utils.Option.map return_ratios Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int)); ("NW", Some(nw |> Py.Float.of_float))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let exponential ?center ?tau ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "exponential"
     [||]
     (Wrap_utils.keyword_args [("center", Wrap_utils.Option.map center Py.Float.of_float); ("tau", Wrap_utils.Option.map tau Py.Float.of_float); ("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let flattop ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "flattop"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let gaussian ?sym ~m ~std () =
   Py.Module.get_function_with_keywords __wrap_namespace "gaussian"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int)); ("std", Some(std |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let general_cosine ?sym ~m ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "general_cosine"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int)); ("a", Some(a |> Np.Obj.to_pyobject))])

let general_gaussian ?sym ~m ~p ~sig_ () =
   Py.Module.get_function_with_keywords __wrap_namespace "general_gaussian"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int)); ("p", Some(p |> Py.Float.of_float)); ("sig", Some(sig_ |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let general_hamming ?sym ~m ~alpha () =
   Py.Module.get_function_with_keywords __wrap_namespace "general_hamming"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int)); ("alpha", Some(alpha |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let get_window ?fftbins ~window ~nx () =
                     Py.Module.get_function_with_keywords __wrap_namespace "get_window"
                       [||]
                       (Wrap_utils.keyword_args [("fftbins", Wrap_utils.Option.map fftbins Py.Bool.of_bool); ("window", Some(window |> (function
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
))); ("Nx", Some(nx |> Py.Int.of_int))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let hamming ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "hamming"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let hann ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "hann"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let hanning ?kwds args =
   Py.Module.get_function_with_keywords __wrap_namespace "hanning"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwds with None -> [] | Some x -> x)

let kaiser ?sym ~m ~beta () =
   Py.Module.get_function_with_keywords __wrap_namespace "kaiser"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int)); ("beta", Some(beta |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let nuttall ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "nuttall"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let parzen ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "parzen"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let slepian ?sym ~m ~width () =
   Py.Module.get_function_with_keywords __wrap_namespace "slepian"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int)); ("width", Some(width |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let sp_fft ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "sp_fft"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

let triang ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "triang"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let tukey ?alpha ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "tukey"
     [||]
     (Wrap_utils.keyword_args [("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
let barthann ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "barthann"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let bartlett ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "bartlett"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let blackman ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "blackman"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let blackmanharris ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "blackmanharris"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let bohman ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "bohman"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let boxcar ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "boxcar"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let chebwin ?sym ~m ~at () =
   Py.Module.get_function_with_keywords __wrap_namespace "chebwin"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int)); ("at", Some(at |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let cosine ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "cosine"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let dpss ?kmax ?sym ?norm ?return_ratios ~m ~nw () =
                     Py.Module.get_function_with_keywords __wrap_namespace "dpss"
                       [||]
                       (Wrap_utils.keyword_args [("Kmax", Wrap_utils.Option.map kmax Py.Int.of_int); ("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("norm", Wrap_utils.Option.map norm (function
| `T_2 x -> Wrap_utils.id x
| `Optional x -> (function
| `T_subsample_ x -> Wrap_utils.id x
| `None -> Py.none
) x
| `Approximate -> Py.String.of_string "approximate"
)); ("return_ratios", Wrap_utils.Option.map return_ratios Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int)); ("NW", Some(nw |> Py.Float.of_float))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let exponential ?center ?tau ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "exponential"
     [||]
     (Wrap_utils.keyword_args [("center", Wrap_utils.Option.map center Py.Float.of_float); ("tau", Wrap_utils.Option.map tau Py.Float.of_float); ("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let flattop ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "flattop"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let gaussian ?sym ~m ~std () =
   Py.Module.get_function_with_keywords __wrap_namespace "gaussian"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int)); ("std", Some(std |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let general_cosine ?sym ~m ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "general_cosine"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int)); ("a", Some(a |> Np.Obj.to_pyobject))])

let general_gaussian ?sym ~m ~p ~sig_ () =
   Py.Module.get_function_with_keywords __wrap_namespace "general_gaussian"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int)); ("p", Some(p |> Py.Float.of_float)); ("sig", Some(sig_ |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let general_hamming ?sym ~m ~alpha () =
   Py.Module.get_function_with_keywords __wrap_namespace "general_hamming"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int)); ("alpha", Some(alpha |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let get_window ?fftbins ~window ~nx () =
                     Py.Module.get_function_with_keywords __wrap_namespace "get_window"
                       [||]
                       (Wrap_utils.keyword_args [("fftbins", Wrap_utils.Option.map fftbins Py.Bool.of_bool); ("window", Some(window |> (function
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
))); ("Nx", Some(nx |> Py.Int.of_int))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let hamming ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "hamming"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let hann ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "hann"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let hanning ?kwds args =
   Py.Module.get_function_with_keywords __wrap_namespace "hanning"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwds with None -> [] | Some x -> x)

let kaiser ?sym ~m ~beta () =
   Py.Module.get_function_with_keywords __wrap_namespace "kaiser"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int)); ("beta", Some(beta |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let nuttall ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "nuttall"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let parzen ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "parzen"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let slepian ?sym ~m ~width () =
   Py.Module.get_function_with_keywords __wrap_namespace "slepian"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int)); ("width", Some(width |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let triang ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "triang"
     [||]
     (Wrap_utils.keyword_args [("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let tukey ?alpha ?sym ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "tukey"
     [||]
     (Wrap_utils.keyword_args [("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("sym", Wrap_utils.Option.map sym Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
let abcd_normalize ?a ?b ?c ?d () =
   Py.Module.get_function_with_keywords __wrap_namespace "abcd_normalize"
     [||]
     (Wrap_utils.keyword_args [("A", a); ("B", b); ("C", c); ("D", d)])

let argrelextrema ?axis ?order ?mode ~data ~comparator () =
   Py.Module.get_function_with_keywords __wrap_namespace "argrelextrema"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("order", Wrap_utils.Option.map order Py.Int.of_int); ("mode", Wrap_utils.Option.map mode Py.String.of_string); ("data", Some(data |> Np.Obj.to_pyobject)); ("comparator", Some(comparator ))])

let argrelmax ?axis ?order ?mode ~data () =
   Py.Module.get_function_with_keywords __wrap_namespace "argrelmax"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("order", Wrap_utils.Option.map order Py.Int.of_int); ("mode", Wrap_utils.Option.map mode Py.String.of_string); ("data", Some(data |> Np.Obj.to_pyobject))])

let argrelmin ?axis ?order ?mode ~data () =
   Py.Module.get_function_with_keywords __wrap_namespace "argrelmin"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("order", Wrap_utils.Option.map order Py.Int.of_int); ("mode", Wrap_utils.Option.map mode Py.String.of_string); ("data", Some(data |> Np.Obj.to_pyobject))])

                  let band_stop_obj ~wp ~ind ~passb ~stopb ~gpass ~gstop ~type_ () =
                     Py.Module.get_function_with_keywords __wrap_namespace "band_stop_obj"
                       [||]
                       (Wrap_utils.keyword_args [("wp", Some(wp |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
))); ("ind", Some(ind |> (function
| `I x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
))); ("passb", Some(passb |> Np.Obj.to_pyobject)); ("stopb", Some(stopb |> Np.Obj.to_pyobject)); ("gpass", Some(gpass |> Py.Float.of_float)); ("gstop", Some(gstop |> Py.Float.of_float)); ("type", Some(type_ |> (function
| `Butter -> Py.String.of_string "butter"
| `Cheby -> Py.String.of_string "cheby"
| `Ellip -> Py.String.of_string "ellip"
)))])

let barthann ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "barthann"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let bartlett ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "bartlett"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let bessel ?btype ?analog ?output ?norm ?fs ~n ~wn () =
                     Py.Module.get_function_with_keywords __wrap_namespace "bessel"
                       [||]
                       (Wrap_utils.keyword_args [("btype", Wrap_utils.Option.map btype (function
| `Lowpass -> Py.String.of_string "lowpass"
| `Highpass -> Py.String.of_string "highpass"
| `Bandpass -> Py.String.of_string "bandpass"
| `Bandstop -> Py.String.of_string "bandstop"
)); ("analog", Wrap_utils.Option.map analog Py.Bool.of_bool); ("output", Wrap_utils.Option.map output (function
| `Ba -> Py.String.of_string "ba"
| `Zpk -> Py.String.of_string "zpk"
| `Sos -> Py.String.of_string "sos"
)); ("norm", Wrap_utils.Option.map norm (function
| `Phase -> Py.String.of_string "phase"
| `Delay -> Py.String.of_string "delay"
| `Mag -> Py.String.of_string "mag"
)); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("N", Some(n |> Py.Int.of_int)); ("Wn", Some(wn |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let besselap ?norm ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "besselap"
                       [||]
                       (Wrap_utils.keyword_args [("norm", Wrap_utils.Option.map norm (function
| `Phase -> Py.String.of_string "phase"
| `Delay -> Py.String.of_string "delay"
| `Mag -> Py.String.of_string "mag"
)); ("N", Some(n |> Py.Int.of_int))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
let bilinear ?fs ~b ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "bilinear"
     [||]
     (Wrap_utils.keyword_args [("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let bilinear_zpk ~z ~p ~k ~fs () =
   Py.Module.get_function_with_keywords __wrap_namespace "bilinear_zpk"
     [||]
     (Wrap_utils.keyword_args [("z", Some(z |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject)); ("k", Some(k |> Py.Float.of_float)); ("fs", Some(fs |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let blackman ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "blackman"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let blackmanharris ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "blackmanharris"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let bode ?w ?n ~system () =
   Py.Module.get_function_with_keywords __wrap_namespace "bode"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("system", Some(system ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
let bohman ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "bohman"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let boxcar ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "boxcar"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let bspline ~x ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "bspline"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x )); ("n", Some(n ))])

let buttap n =
   Py.Module.get_function_with_keywords __wrap_namespace "buttap"
     [||]
     (Wrap_utils.keyword_args [("N", Some(n ))])

                  let butter ?btype ?analog ?output ?fs ~n ~wn () =
                     Py.Module.get_function_with_keywords __wrap_namespace "butter"
                       [||]
                       (Wrap_utils.keyword_args [("btype", Wrap_utils.Option.map btype (function
| `Lowpass -> Py.String.of_string "lowpass"
| `Highpass -> Py.String.of_string "highpass"
| `Bandpass -> Py.String.of_string "bandpass"
| `Bandstop -> Py.String.of_string "bandstop"
)); ("analog", Wrap_utils.Option.map analog Py.Bool.of_bool); ("output", Wrap_utils.Option.map output (function
| `Ba -> Py.String.of_string "ba"
| `Zpk -> Py.String.of_string "zpk"
| `Sos -> Py.String.of_string "sos"
)); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("N", Some(n |> Py.Int.of_int)); ("Wn", Some(wn |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let buttord ?analog ?fs ~wp ~ws ~gpass ~gstop () =
   Py.Module.get_function_with_keywords __wrap_namespace "buttord"
     [||]
     (Wrap_utils.keyword_args [("analog", Wrap_utils.Option.map analog Py.Bool.of_bool); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("wp", Some(wp )); ("ws", Some(ws )); ("gpass", Some(gpass |> Py.Float.of_float)); ("gstop", Some(gstop |> Py.Float.of_float))])
     |> (fun x -> ((Py.Int.to_int (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let cascade ?j ~hk () =
   Py.Module.get_function_with_keywords __wrap_namespace "cascade"
     [||]
     (Wrap_utils.keyword_args [("J", Wrap_utils.Option.map j Py.Int.of_int); ("hk", Some(hk |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
let cheb1ap ~n ~rp () =
   Py.Module.get_function_with_keywords __wrap_namespace "cheb1ap"
     [||]
     (Wrap_utils.keyword_args [("N", Some(n )); ("rp", Some(rp ))])

let cheb1ord ?analog ?fs ~wp ~ws ~gpass ~gstop () =
   Py.Module.get_function_with_keywords __wrap_namespace "cheb1ord"
     [||]
     (Wrap_utils.keyword_args [("analog", Wrap_utils.Option.map analog Py.Bool.of_bool); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("wp", Some(wp )); ("ws", Some(ws )); ("gpass", Some(gpass |> Py.Float.of_float)); ("gstop", Some(gstop |> Py.Float.of_float))])
     |> (fun x -> ((Py.Int.to_int (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let cheb2ap ~n ~rs () =
   Py.Module.get_function_with_keywords __wrap_namespace "cheb2ap"
     [||]
     (Wrap_utils.keyword_args [("N", Some(n )); ("rs", Some(rs ))])

let cheb2ord ?analog ?fs ~wp ~ws ~gpass ~gstop () =
   Py.Module.get_function_with_keywords __wrap_namespace "cheb2ord"
     [||]
     (Wrap_utils.keyword_args [("analog", Wrap_utils.Option.map analog Py.Bool.of_bool); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("wp", Some(wp )); ("ws", Some(ws )); ("gpass", Some(gpass |> Py.Float.of_float)); ("gstop", Some(gstop |> Py.Float.of_float))])
     |> (fun x -> ((Py.Int.to_int (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let chebwin ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "chebwin"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let cheby1 ?btype ?analog ?output ?fs ~n ~rp ~wn () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cheby1"
                       [||]
                       (Wrap_utils.keyword_args [("btype", Wrap_utils.Option.map btype (function
| `Lowpass -> Py.String.of_string "lowpass"
| `Highpass -> Py.String.of_string "highpass"
| `Bandpass -> Py.String.of_string "bandpass"
| `Bandstop -> Py.String.of_string "bandstop"
)); ("analog", Wrap_utils.Option.map analog Py.Bool.of_bool); ("output", Wrap_utils.Option.map output (function
| `Ba -> Py.String.of_string "ba"
| `Zpk -> Py.String.of_string "zpk"
| `Sos -> Py.String.of_string "sos"
)); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("N", Some(n |> Py.Int.of_int)); ("rp", Some(rp |> Py.Float.of_float)); ("Wn", Some(wn |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let cheby2 ?btype ?analog ?output ?fs ~n ~rs ~wn () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cheby2"
                       [||]
                       (Wrap_utils.keyword_args [("btype", Wrap_utils.Option.map btype (function
| `Lowpass -> Py.String.of_string "lowpass"
| `Highpass -> Py.String.of_string "highpass"
| `Bandpass -> Py.String.of_string "bandpass"
| `Bandstop -> Py.String.of_string "bandstop"
)); ("analog", Wrap_utils.Option.map analog Py.Bool.of_bool); ("output", Wrap_utils.Option.map output (function
| `Ba -> Py.String.of_string "ba"
| `Zpk -> Py.String.of_string "zpk"
| `Sos -> Py.String.of_string "sos"
)); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("N", Some(n |> Py.Int.of_int)); ("rs", Some(rs |> Py.Float.of_float)); ("Wn", Some(wn |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let check_COLA ?tol ~window ~nperseg ~noverlap () =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_COLA"
                       [||]
                       (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("window", Some(window |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
))); ("nperseg", Some(nperseg |> Py.Int.of_int)); ("noverlap", Some(noverlap |> Py.Int.of_int))])
                       |> Py.Bool.to_bool
                  let check_NOLA ?tol ~window ~nperseg ~noverlap () =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_NOLA"
                       [||]
                       (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("window", Some(window |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
))); ("nperseg", Some(nperseg |> Py.Int.of_int)); ("noverlap", Some(noverlap |> Py.Int.of_int))])
                       |> Py.Bool.to_bool
                  let chirp ?method_ ?phi ?vertex_zero ~t ~f0 ~t1 ~f1 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "chirp"
                       [||]
                       (Wrap_utils.keyword_args [("method", Wrap_utils.Option.map method_ (function
| `Linear -> Py.String.of_string "linear"
| `Quadratic -> Py.String.of_string "quadratic"
| `Logarithmic -> Py.String.of_string "logarithmic"
| `Hyperbolic -> Py.String.of_string "hyperbolic"
)); ("phi", Wrap_utils.Option.map phi Py.Float.of_float); ("vertex_zero", Wrap_utils.Option.map vertex_zero Py.Bool.of_bool); ("t", Some(t |> Np.Obj.to_pyobject)); ("f0", Some(f0 |> Py.Float.of_float)); ("t1", Some(t1 |> Py.Float.of_float)); ("f1", Some(f1 |> Py.Float.of_float))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let choose_conv_method ?mode ?measure ~in1 ~in2 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "choose_conv_method"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Full -> Py.String.of_string "full"
| `Valid -> Py.String.of_string "valid"
| `Same -> Py.String.of_string "same"
)); ("measure", Wrap_utils.Option.map measure Py.Bool.of_bool); ("in1", Some(in1 |> Np.Obj.to_pyobject)); ("in2", Some(in2 |> Np.Obj.to_pyobject))])
                       |> (fun x -> ((Py.String.to_string (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let cmplx_sort p =
   Py.Module.get_function_with_keywords __wrap_namespace "cmplx_sort"
     [||]
     (Wrap_utils.keyword_args [("p", Some(p |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let coherence ?fs ?window ?nperseg ?noverlap ?nfft ?detrend ?axis ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "coherence"
                       [||]
                       (Wrap_utils.keyword_args [("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("window", Wrap_utils.Option.map window (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("nperseg", Wrap_utils.Option.map nperseg Py.Int.of_int); ("noverlap", Wrap_utils.Option.map noverlap Py.Int.of_int); ("nfft", Wrap_utils.Option.map nfft Py.Int.of_int); ("detrend", Wrap_utils.Option.map detrend (function
| `T_False_ x -> Wrap_utils.id x
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let cont2discrete ?method_ ?alpha ~system ~dt () =
   Py.Module.get_function_with_keywords __wrap_namespace "cont2discrete"
     [||]
     (Wrap_utils.keyword_args [("method", Wrap_utils.Option.map method_ Py.String.of_string); ("alpha", alpha); ("system", Some(system )); ("dt", Some(dt |> Py.Float.of_float))])

                  let convolve ?mode ?method_ ~in1 ~in2 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "convolve"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Full -> Py.String.of_string "full"
| `Valid -> Py.String.of_string "valid"
| `Same -> Py.String.of_string "same"
)); ("method", Wrap_utils.Option.map method_ (function
| `Auto -> Py.String.of_string "auto"
| `Direct -> Py.String.of_string "direct"
| `Fft -> Py.String.of_string "fft"
)); ("in1", Some(in1 |> Np.Obj.to_pyobject)); ("in2", Some(in2 |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let convolve2d ?mode ?boundary ?fillvalue ~in1 ~in2 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "convolve2d"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Full -> Py.String.of_string "full"
| `Valid -> Py.String.of_string "valid"
| `Same -> Py.String.of_string "same"
)); ("boundary", Wrap_utils.Option.map boundary (function
| `Fill -> Py.String.of_string "fill"
| `Wrap -> Py.String.of_string "wrap"
| `Symm -> Py.String.of_string "symm"
)); ("fillvalue", Wrap_utils.Option.map fillvalue (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("in1", Some(in1 |> Np.Obj.to_pyobject)); ("in2", Some(in2 |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let correlate ?mode ?method_ ~in1 ~in2 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "correlate"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Full -> Py.String.of_string "full"
| `Valid -> Py.String.of_string "valid"
| `Same -> Py.String.of_string "same"
)); ("method", Wrap_utils.Option.map method_ (function
| `Auto -> Py.String.of_string "auto"
| `Direct -> Py.String.of_string "direct"
| `Fft -> Py.String.of_string "fft"
)); ("in1", Some(in1 |> Np.Obj.to_pyobject)); ("in2", Some(in2 |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let correlate2d ?mode ?boundary ?fillvalue ~in1 ~in2 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "correlate2d"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Full -> Py.String.of_string "full"
| `Valid -> Py.String.of_string "valid"
| `Same -> Py.String.of_string "same"
)); ("boundary", Wrap_utils.Option.map boundary (function
| `Fill -> Py.String.of_string "fill"
| `Wrap -> Py.String.of_string "wrap"
| `Symm -> Py.String.of_string "symm"
)); ("fillvalue", Wrap_utils.Option.map fillvalue (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("in1", Some(in1 |> Np.Obj.to_pyobject)); ("in2", Some(in2 |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let cosine ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "cosine"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let csd ?fs ?window ?nperseg ?noverlap ?nfft ?detrend ?return_onesided ?scaling ?axis ?average ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "csd"
                       [||]
                       (Wrap_utils.keyword_args [("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("window", Wrap_utils.Option.map window (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("nperseg", Wrap_utils.Option.map nperseg Py.Int.of_int); ("noverlap", Wrap_utils.Option.map noverlap Py.Int.of_int); ("nfft", Wrap_utils.Option.map nfft Py.Int.of_int); ("detrend", Wrap_utils.Option.map detrend (function
| `T_False_ x -> Wrap_utils.id x
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("return_onesided", Wrap_utils.Option.map return_onesided Py.Bool.of_bool); ("scaling", Wrap_utils.Option.map scaling (function
| `Density -> Py.String.of_string "density"
| `Spectrum -> Py.String.of_string "spectrum"
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("average", Wrap_utils.Option.map average (function
| `Mean -> Py.String.of_string "mean"
| `Median -> Py.String.of_string "median"
)); ("x", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let cspline1d ?lamb ~signal () =
   Py.Module.get_function_with_keywords __wrap_namespace "cspline1d"
     [||]
     (Wrap_utils.keyword_args [("lamb", Wrap_utils.Option.map lamb Py.Float.of_float); ("signal", Some(signal |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let cspline1d_eval ?dx ?x0 ~cj ~newx () =
   Py.Module.get_function_with_keywords __wrap_namespace "cspline1d_eval"
     [||]
     (Wrap_utils.keyword_args [("dx", dx); ("x0", x0); ("cj", Some(cj )); ("newx", Some(newx ))])

let cubic x =
   Py.Module.get_function_with_keywords __wrap_namespace "cubic"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let cwt ?dtype ?kwargs ~data ~wavelet ~widths () =
   Py.Module.get_function_with_keywords __wrap_namespace "cwt"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("data", Some(data |> Np.Obj.to_pyobject)); ("wavelet", Some(wavelet )); ("widths", Some(widths ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let daub p =
   Py.Module.get_function_with_keywords __wrap_namespace "daub"
     [||]
     (Wrap_utils.keyword_args [("p", Some(p |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let dbode ?w ?n ~system () =
   Py.Module.get_function_with_keywords __wrap_namespace "dbode"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("system", Some(system ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
                  let decimate ?n ?ftype ?axis ?zero_phase ~x ~q () =
                     Py.Module.get_function_with_keywords __wrap_namespace "decimate"
                       [||]
                       (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("ftype", Wrap_utils.Option.map ftype (function
| `Fir -> Py.String.of_string "fir"
| `Iir -> Py.String.of_string "iir"
| `T_dlti_instance x -> Wrap_utils.id x
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("zero_phase", Wrap_utils.Option.map zero_phase Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject)); ("q", Some(q |> Py.Int.of_int))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let deconvolve ~signal ~divisor () =
   Py.Module.get_function_with_keywords __wrap_namespace "deconvolve"
     [||]
     (Wrap_utils.keyword_args [("signal", Some(signal |> Np.Obj.to_pyobject)); ("divisor", Some(divisor |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let detrend ?axis ?type_ ?bp ?overwrite_data ~data () =
                     Py.Module.get_function_with_keywords __wrap_namespace "detrend"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("type", Wrap_utils.Option.map type_ (function
| `Linear -> Py.String.of_string "linear"
| `Constant -> Py.String.of_string "constant"
)); ("bp", bp); ("overwrite_data", Wrap_utils.Option.map overwrite_data Py.Bool.of_bool); ("data", Some(data |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let dfreqresp ?w ?n ?whole ~system () =
   Py.Module.get_function_with_keywords __wrap_namespace "dfreqresp"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("whole", Wrap_utils.Option.map whole Py.Bool.of_bool); ("system", Some(system ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let dimpulse ?x0 ?t ?n ~system () =
   Py.Module.get_function_with_keywords __wrap_namespace "dimpulse"
     [||]
     (Wrap_utils.keyword_args [("x0", Wrap_utils.Option.map x0 Np.Obj.to_pyobject); ("t", Wrap_utils.Option.map t Np.Obj.to_pyobject); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("system", Some(system ))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let dlsim ?t ?x0 ~system ~u () =
   Py.Module.get_function_with_keywords __wrap_namespace "dlsim"
     [||]
     (Wrap_utils.keyword_args [("t", Wrap_utils.Option.map t Np.Obj.to_pyobject); ("x0", Wrap_utils.Option.map x0 Np.Obj.to_pyobject); ("system", Some(system )); ("u", Some(u |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
let dstep ?x0 ?t ?n ~system () =
   Py.Module.get_function_with_keywords __wrap_namespace "dstep"
     [||]
     (Wrap_utils.keyword_args [("x0", Wrap_utils.Option.map x0 Np.Obj.to_pyobject); ("t", Wrap_utils.Option.map t Np.Obj.to_pyobject); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("system", Some(system ))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let ellip ?btype ?analog ?output ?fs ~n ~rp ~rs ~wn () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ellip"
                       [||]
                       (Wrap_utils.keyword_args [("btype", Wrap_utils.Option.map btype (function
| `Lowpass -> Py.String.of_string "lowpass"
| `Highpass -> Py.String.of_string "highpass"
| `Bandpass -> Py.String.of_string "bandpass"
| `Bandstop -> Py.String.of_string "bandstop"
)); ("analog", Wrap_utils.Option.map analog Py.Bool.of_bool); ("output", Wrap_utils.Option.map output (function
| `Ba -> Py.String.of_string "ba"
| `Zpk -> Py.String.of_string "zpk"
| `Sos -> Py.String.of_string "sos"
)); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("N", Some(n |> Py.Int.of_int)); ("rp", Some(rp |> Py.Float.of_float)); ("rs", Some(rs |> Py.Float.of_float)); ("Wn", Some(wn |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let ellipap ~n ~rp ~rs () =
   Py.Module.get_function_with_keywords __wrap_namespace "ellipap"
     [||]
     (Wrap_utils.keyword_args [("N", Some(n )); ("rp", Some(rp )); ("rs", Some(rs ))])

let ellipord ?analog ?fs ~wp ~ws ~gpass ~gstop () =
   Py.Module.get_function_with_keywords __wrap_namespace "ellipord"
     [||]
     (Wrap_utils.keyword_args [("analog", Wrap_utils.Option.map analog Py.Bool.of_bool); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("wp", Some(wp )); ("ws", Some(ws )); ("gpass", Some(gpass |> Py.Float.of_float)); ("gstop", Some(gstop |> Py.Float.of_float))])
     |> (fun x -> ((Py.Int.to_int (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let exponential ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "exponential"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let fftconvolve ?mode ?axes ~in1 ~in2 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fftconvolve"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Full -> Py.String.of_string "full"
| `Valid -> Py.String.of_string "valid"
| `Same -> Py.String.of_string "same"
)); ("axes", Wrap_utils.Option.map axes (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("in1", Some(in1 |> Np.Obj.to_pyobject)); ("in2", Some(in2 |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let filtfilt ?axis ?padtype ?padlen ?method_ ?irlen ~b ~a ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "filtfilt"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("padtype", Wrap_utils.Option.map padtype (function
| `S x -> Py.String.of_string x
| `None -> Py.none
)); ("padlen", Wrap_utils.Option.map padlen Py.Int.of_int); ("method", Wrap_utils.Option.map method_ Py.String.of_string); ("irlen", Wrap_utils.Option.map irlen Py.Int.of_int); ("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject)); ("x", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let find_peaks ?height ?threshold ?distance ?prominence ?width ?wlen ?rel_height ?plateau_size ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "find_peaks"
                       [||]
                       (Wrap_utils.keyword_args [("height", Wrap_utils.Option.map height (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("threshold", Wrap_utils.Option.map threshold (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("distance", Wrap_utils.Option.map distance (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("prominence", Wrap_utils.Option.map prominence (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("width", Wrap_utils.Option.map width (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("wlen", Wrap_utils.Option.map wlen Py.Int.of_int); ("rel_height", Wrap_utils.Option.map rel_height Py.Float.of_float); ("plateau_size", Wrap_utils.Option.map plateau_size (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("x", Some(x ))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let find_peaks_cwt ?wavelet ?max_distances ?gap_thresh ?min_length ?min_snr ?noise_perc ~vector ~widths () =
   Py.Module.get_function_with_keywords __wrap_namespace "find_peaks_cwt"
     [||]
     (Wrap_utils.keyword_args [("wavelet", wavelet); ("max_distances", Wrap_utils.Option.map max_distances Np.Obj.to_pyobject); ("gap_thresh", Wrap_utils.Option.map gap_thresh Py.Float.of_float); ("min_length", Wrap_utils.Option.map min_length Py.Int.of_int); ("min_snr", Wrap_utils.Option.map min_snr Py.Float.of_float); ("noise_perc", Wrap_utils.Option.map noise_perc Py.Float.of_float); ("vector", Some(vector |> Np.Obj.to_pyobject)); ("widths", Some(widths ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let findfreqs ?kind ~num ~den ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "findfreqs"
                       [||]
                       (Wrap_utils.keyword_args [("kind", Wrap_utils.Option.map kind (function
| `Ba -> Py.String.of_string "ba"
| `Zp -> Py.String.of_string "zp"
)); ("num", Some(num )); ("den", Some(den )); ("N", Some(n |> Py.Int.of_int))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let firls ?weight ?nyq ?fs ~numtaps ~bands ~desired () =
   Py.Module.get_function_with_keywords __wrap_namespace "firls"
     [||]
     (Wrap_utils.keyword_args [("weight", Wrap_utils.Option.map weight Np.Obj.to_pyobject); ("nyq", Wrap_utils.Option.map nyq Py.Float.of_float); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("numtaps", Some(numtaps |> Py.Int.of_int)); ("bands", Some(bands |> Np.Obj.to_pyobject)); ("desired", Some(desired |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let firwin ?width ?window ?pass_zero ?scale ?nyq ?fs ~numtaps ~cutoff () =
                     Py.Module.get_function_with_keywords __wrap_namespace "firwin"
                       [||]
                       (Wrap_utils.keyword_args [("width", Wrap_utils.Option.map width Py.Float.of_float); ("window", Wrap_utils.Option.map window (function
| `Tuple_of_string_and_parameter_values x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("pass_zero", Wrap_utils.Option.map pass_zero (function
| `Highpass -> Py.String.of_string "highpass"
| `Lowpass -> Py.String.of_string "lowpass"
| `Bandstop -> Py.String.of_string "bandstop"
| `Bool x -> Py.Bool.of_bool x
| `Bandpass -> Py.String.of_string "bandpass"
)); ("scale", Wrap_utils.Option.map scale Py.Float.of_float); ("nyq", Wrap_utils.Option.map nyq Py.Float.of_float); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("numtaps", Some(numtaps |> Py.Int.of_int)); ("cutoff", Some(cutoff |> (function
| `F x -> Py.Float.of_float x
| `T1D_array_like x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let firwin2 ?nfreqs ?window ?nyq ?antisymmetric ?fs ~numtaps ~freq ~gain () =
                     Py.Module.get_function_with_keywords __wrap_namespace "firwin2"
                       [||]
                       (Wrap_utils.keyword_args [("nfreqs", Wrap_utils.Option.map nfreqs Py.Int.of_int); ("window", Wrap_utils.Option.map window (function
| `T_string_float_ x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `None -> Py.none
)); ("nyq", Wrap_utils.Option.map nyq Py.Float.of_float); ("antisymmetric", Wrap_utils.Option.map antisymmetric Py.Bool.of_bool); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("numtaps", Some(numtaps |> Py.Int.of_int)); ("freq", Some(freq |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `T1D x -> Wrap_utils.id x
))); ("gain", Some(gain |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let flattop ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "flattop"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let freqresp ?w ?n ~system () =
   Py.Module.get_function_with_keywords __wrap_namespace "freqresp"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Np.Obj.to_pyobject); ("n", Wrap_utils.Option.map n Py.Int.of_int); ("system", Some(system ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let freqs ?worN ?plot ~b ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "freqs"
                       [||]
                       (Wrap_utils.keyword_args [("worN", Wrap_utils.Option.map worN (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("plot", plot); ("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let freqs_zpk ?worN ~z ~p ~k () =
                     Py.Module.get_function_with_keywords __wrap_namespace "freqs_zpk"
                       [||]
                       (Wrap_utils.keyword_args [("worN", Wrap_utils.Option.map worN (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("z", Some(z |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject)); ("k", Some(k |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let freqz ?a ?worN ?whole ?plot ?fs ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "freqz"
                       [||]
                       (Wrap_utils.keyword_args [("a", Wrap_utils.Option.map a Np.Obj.to_pyobject); ("worN", Wrap_utils.Option.map worN (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("whole", Wrap_utils.Option.map whole Py.Bool.of_bool); ("plot", plot); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let freqz_zpk ?worN ?whole ?fs ~z ~p ~k () =
                     Py.Module.get_function_with_keywords __wrap_namespace "freqz_zpk"
                       [||]
                       (Wrap_utils.keyword_args [("worN", Wrap_utils.Option.map worN (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("whole", Wrap_utils.Option.map whole Py.Bool.of_bool); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("z", Some(z |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject)); ("k", Some(k |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let gauss_spline ~x ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "gauss_spline"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x )); ("n", Some(n |> Py.Int.of_int))])

let gaussian ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "gaussian"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let gausspulse ?fc ?bw ?bwr ?tpr ?retquad ?retenv ~t () =
                     Py.Module.get_function_with_keywords __wrap_namespace "gausspulse"
                       [||]
                       (Wrap_utils.keyword_args [("fc", Wrap_utils.Option.map fc Py.Int.of_int); ("bw", Wrap_utils.Option.map bw Py.Float.of_float); ("bwr", Wrap_utils.Option.map bwr Py.Float.of_float); ("tpr", Wrap_utils.Option.map tpr Py.Float.of_float); ("retquad", Wrap_utils.Option.map retquad Py.Bool.of_bool); ("retenv", Wrap_utils.Option.map retenv Py.Bool.of_bool); ("t", Some(t |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `The_string_cutoff_ x -> Wrap_utils.id x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
let general_gaussian ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "general_gaussian"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let get_window ?fftbins ~window ~nx () =
                     Py.Module.get_function_with_keywords __wrap_namespace "get_window"
                       [||]
                       (Wrap_utils.keyword_args [("fftbins", Wrap_utils.Option.map fftbins Py.Bool.of_bool); ("window", Some(window |> (function
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
))); ("Nx", Some(nx |> Py.Int.of_int))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let group_delay ?w ?whole ?fs ~system () =
                     Py.Module.get_function_with_keywords __wrap_namespace "group_delay"
                       [||]
                       (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("whole", Wrap_utils.Option.map whole Py.Bool.of_bool); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("system", Some(system ))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let hamming ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "hamming"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let hann ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "hann"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let hanning ?kwds args =
   Py.Module.get_function_with_keywords __wrap_namespace "hanning"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwds with None -> [] | Some x -> x)

let hilbert ?n ?axis ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "hilbert"
     [||]
     (Wrap_utils.keyword_args [("N", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let hilbert2 ?n ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "hilbert2"
                       [||]
                       (Wrap_utils.keyword_args [("N", Wrap_utils.Option.map n (function
| `Tuple_of_two_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("x", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let iirdesign ?analog ?ftype ?output ?fs ~wp ~ws ~gpass ~gstop () =
                     Py.Module.get_function_with_keywords __wrap_namespace "iirdesign"
                       [||]
                       (Wrap_utils.keyword_args [("analog", Wrap_utils.Option.map analog Py.Bool.of_bool); ("ftype", Wrap_utils.Option.map ftype Py.String.of_string); ("output", Wrap_utils.Option.map output (function
| `Ba -> Py.String.of_string "ba"
| `Zpk -> Py.String.of_string "zpk"
| `Sos -> Py.String.of_string "sos"
)); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("wp", Some(wp )); ("ws", Some(ws )); ("gpass", Some(gpass |> Py.Float.of_float)); ("gstop", Some(gstop |> Py.Float.of_float))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let iirfilter ?rp ?rs ?btype ?analog ?ftype ?output ?fs ~n ~wn () =
                     Py.Module.get_function_with_keywords __wrap_namespace "iirfilter"
                       [||]
                       (Wrap_utils.keyword_args [("rp", Wrap_utils.Option.map rp Py.Float.of_float); ("rs", Wrap_utils.Option.map rs Py.Float.of_float); ("btype", Wrap_utils.Option.map btype (function
| `Bandpass -> Py.String.of_string "bandpass"
| `Lowpass -> Py.String.of_string "lowpass"
| `Highpass -> Py.String.of_string "highpass"
| `Bandstop -> Py.String.of_string "bandstop"
)); ("analog", Wrap_utils.Option.map analog Py.Bool.of_bool); ("ftype", Wrap_utils.Option.map ftype Py.String.of_string); ("output", Wrap_utils.Option.map output (function
| `Ba -> Py.String.of_string "ba"
| `Zpk -> Py.String.of_string "zpk"
| `Sos -> Py.String.of_string "sos"
)); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("N", Some(n |> Py.Int.of_int)); ("Wn", Some(wn |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let iirnotch ?fs ~w0 ~q () =
   Py.Module.get_function_with_keywords __wrap_namespace "iirnotch"
     [||]
     (Wrap_utils.keyword_args [("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("w0", Some(w0 |> Py.Float.of_float)); ("Q", Some(q |> Py.Float.of_float))])

let iirpeak ?fs ~w0 ~q () =
   Py.Module.get_function_with_keywords __wrap_namespace "iirpeak"
     [||]
     (Wrap_utils.keyword_args [("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("w0", Some(w0 |> Py.Float.of_float)); ("Q", Some(q |> Py.Float.of_float))])

let impulse ?x0 ?t ?n ~system () =
   Py.Module.get_function_with_keywords __wrap_namespace "impulse"
     [||]
     (Wrap_utils.keyword_args [("X0", Wrap_utils.Option.map x0 Np.Obj.to_pyobject); ("T", Wrap_utils.Option.map t Np.Obj.to_pyobject); ("N", Wrap_utils.Option.map n Py.Int.of_int); ("system", Some(system ))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let impulse2 ?x0 ?t ?n ?kwargs ~system () =
   Py.Module.get_function_with_keywords __wrap_namespace "impulse2"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("X0", x0); ("T", t); ("N", Wrap_utils.Option.map n Py.Int.of_int); ("system", Some(system ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let invres ?tol ?rtype ~r ~p ~k () =
                     Py.Module.get_function_with_keywords __wrap_namespace "invres"
                       [||]
                       (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("rtype", Wrap_utils.Option.map rtype (function
| `Avg -> Py.String.of_string "avg"
| `Min -> Py.String.of_string "min"
| `Max -> Py.String.of_string "max"
)); ("r", Some(r |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject)); ("k", Some(k |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let invresz ?tol ?rtype ~r ~p ~k () =
                     Py.Module.get_function_with_keywords __wrap_namespace "invresz"
                       [||]
                       (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("rtype", Wrap_utils.Option.map rtype (function
| `Avg -> Py.String.of_string "avg"
| `Min -> Py.String.of_string "min"
| `Max -> Py.String.of_string "max"
)); ("r", Some(r |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject)); ("k", Some(k |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let istft ?fs ?window ?nperseg ?noverlap ?nfft ?input_onesided ?boundary ?time_axis ?freq_axis ~zxx () =
                     Py.Module.get_function_with_keywords __wrap_namespace "istft"
                       [||]
                       (Wrap_utils.keyword_args [("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("window", Wrap_utils.Option.map window (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("nperseg", Wrap_utils.Option.map nperseg Py.Int.of_int); ("noverlap", Wrap_utils.Option.map noverlap Py.Int.of_int); ("nfft", Wrap_utils.Option.map nfft Py.Int.of_int); ("input_onesided", Wrap_utils.Option.map input_onesided Py.Bool.of_bool); ("boundary", Wrap_utils.Option.map boundary Py.Bool.of_bool); ("time_axis", Wrap_utils.Option.map time_axis Py.Int.of_int); ("freq_axis", Wrap_utils.Option.map freq_axis Py.Int.of_int); ("Zxx", Some(zxx |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let kaiser ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "kaiser"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let kaiser_atten ~numtaps ~width () =
   Py.Module.get_function_with_keywords __wrap_namespace "kaiser_atten"
     [||]
     (Wrap_utils.keyword_args [("numtaps", Some(numtaps |> Py.Int.of_int)); ("width", Some(width |> Py.Float.of_float))])
     |> Py.Float.to_float
let kaiser_beta a =
   Py.Module.get_function_with_keywords __wrap_namespace "kaiser_beta"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Py.Float.of_float))])
     |> Py.Float.to_float
let kaiserord ~ripple ~width () =
   Py.Module.get_function_with_keywords __wrap_namespace "kaiserord"
     [||]
     (Wrap_utils.keyword_args [("ripple", Some(ripple |> Py.Float.of_float)); ("width", Some(width |> Py.Float.of_float))])
     |> (fun x -> ((Py.Int.to_int (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))
let lfilter ?axis ?zi ~b ~a ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "lfilter"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("zi", Wrap_utils.Option.map zi Np.Obj.to_pyobject); ("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject)); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let lfilter_zi ~b ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "lfilter_zi"
     [||]
     (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))])

let lfiltic ?x ~b ~a ~y () =
   Py.Module.get_function_with_keywords __wrap_namespace "lfiltic"
     [||]
     (Wrap_utils.keyword_args [("x", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let lombscargle ?precenter ?normalize ~x ~y ~freqs () =
   Py.Module.get_function_with_keywords __wrap_namespace "lombscargle"
     [||]
     (Wrap_utils.keyword_args [("precenter", Wrap_utils.Option.map precenter Py.Bool.of_bool); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject)); ("freqs", Some(freqs |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let lp2bp ?wo ?bw ~b ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "lp2bp"
     [||]
     (Wrap_utils.keyword_args [("wo", Wrap_utils.Option.map wo Py.Float.of_float); ("bw", Wrap_utils.Option.map bw Py.Float.of_float); ("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let lp2bp_zpk ?wo ?bw ~z ~p ~k () =
   Py.Module.get_function_with_keywords __wrap_namespace "lp2bp_zpk"
     [||]
     (Wrap_utils.keyword_args [("wo", Wrap_utils.Option.map wo Py.Float.of_float); ("bw", Wrap_utils.Option.map bw Py.Float.of_float); ("z", Some(z |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject)); ("k", Some(k |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let lp2bs ?wo ?bw ~b ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "lp2bs"
     [||]
     (Wrap_utils.keyword_args [("wo", Wrap_utils.Option.map wo Py.Float.of_float); ("bw", Wrap_utils.Option.map bw Py.Float.of_float); ("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let lp2bs_zpk ?wo ?bw ~z ~p ~k () =
   Py.Module.get_function_with_keywords __wrap_namespace "lp2bs_zpk"
     [||]
     (Wrap_utils.keyword_args [("wo", Wrap_utils.Option.map wo Py.Float.of_float); ("bw", Wrap_utils.Option.map bw Py.Float.of_float); ("z", Some(z |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject)); ("k", Some(k |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let lp2hp ?wo ~b ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "lp2hp"
     [||]
     (Wrap_utils.keyword_args [("wo", Wrap_utils.Option.map wo Py.Float.of_float); ("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let lp2hp_zpk ?wo ~z ~p ~k () =
   Py.Module.get_function_with_keywords __wrap_namespace "lp2hp_zpk"
     [||]
     (Wrap_utils.keyword_args [("wo", Wrap_utils.Option.map wo Py.Float.of_float); ("z", Some(z |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject)); ("k", Some(k |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let lp2lp ?wo ~b ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "lp2lp"
     [||]
     (Wrap_utils.keyword_args [("wo", Wrap_utils.Option.map wo Py.Float.of_float); ("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let lp2lp_zpk ?wo ~z ~p ~k () =
   Py.Module.get_function_with_keywords __wrap_namespace "lp2lp_zpk"
     [||]
     (Wrap_utils.keyword_args [("wo", Wrap_utils.Option.map wo Py.Float.of_float); ("z", Some(z |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject)); ("k", Some(k |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let lsim ?x0 ?interp ~system ~u ~t () =
   Py.Module.get_function_with_keywords __wrap_namespace "lsim"
     [||]
     (Wrap_utils.keyword_args [("X0", Wrap_utils.Option.map x0 Np.Obj.to_pyobject); ("interp", Wrap_utils.Option.map interp Py.Bool.of_bool); ("system", Some(system )); ("U", Some(u |> Np.Obj.to_pyobject)); ("T", Some(t |> Np.Obj.to_pyobject))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
let lsim2 ?u ?t ?x0 ?kwargs ~system () =
   Py.Module.get_function_with_keywords __wrap_namespace "lsim2"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("U", Wrap_utils.Option.map u Np.Obj.to_pyobject); ("T", Wrap_utils.Option.map t Np.Obj.to_pyobject); ("X0", x0); ("system", Some(system ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
let max_len_seq ?state ?length ?taps ~nbits () =
   Py.Module.get_function_with_keywords __wrap_namespace "max_len_seq"
     [||]
     (Wrap_utils.keyword_args [("state", Wrap_utils.Option.map state Np.Obj.to_pyobject); ("length", Wrap_utils.Option.map length Py.Int.of_int); ("taps", Wrap_utils.Option.map taps Np.Obj.to_pyobject); ("nbits", Some(nbits |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let medfilt ?kernel_size ~volume () =
   Py.Module.get_function_with_keywords __wrap_namespace "medfilt"
     [||]
     (Wrap_utils.keyword_args [("kernel_size", Wrap_utils.Option.map kernel_size Np.Obj.to_pyobject); ("volume", Some(volume |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let medfilt2d ?kernel_size ~input () =
   Py.Module.get_function_with_keywords __wrap_namespace "medfilt2d"
     [||]
     (Wrap_utils.keyword_args [("kernel_size", Wrap_utils.Option.map kernel_size Np.Obj.to_pyobject); ("input", Some(input |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let minimum_phase ?method_ ?n_fft ~h () =
                     Py.Module.get_function_with_keywords __wrap_namespace "minimum_phase"
                       [||]
                       (Wrap_utils.keyword_args [("method", Wrap_utils.Option.map method_ (function
| `Hilbert -> Py.String.of_string "hilbert"
| `Homomorphic -> Py.String.of_string "homomorphic"
)); ("n_fft", Wrap_utils.Option.map n_fft Py.Int.of_int); ("h", Some(h |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let morlet ?w ?s ?complete ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "morlet"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Py.Float.of_float); ("s", Wrap_utils.Option.map s Py.Float.of_float); ("complete", Wrap_utils.Option.map complete Py.Bool.of_bool); ("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let morlet2 ?w ~m ~s () =
   Py.Module.get_function_with_keywords __wrap_namespace "morlet2"
     [||]
     (Wrap_utils.keyword_args [("w", Wrap_utils.Option.map w Py.Float.of_float); ("M", Some(m |> Py.Int.of_int)); ("s", Some(s |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let normalize ~b ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "normalize"
     [||]
     (Wrap_utils.keyword_args [("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let nuttall ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "nuttall"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let oaconvolve ?mode ?axes ~in1 ~in2 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "oaconvolve"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Full -> Py.String.of_string "full"
| `Valid -> Py.String.of_string "valid"
| `Same -> Py.String.of_string "same"
)); ("axes", Wrap_utils.Option.map axes (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)); ("in1", Some(in1 |> Np.Obj.to_pyobject)); ("in2", Some(in2 |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let order_filter ~a ~domain ~rank () =
   Py.Module.get_function_with_keywords __wrap_namespace "order_filter"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Np.Obj.to_pyobject)); ("domain", Some(domain |> Np.Obj.to_pyobject)); ("rank", Some(rank |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let parzen ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "parzen"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let peak_prominences ?wlen ~x ~peaks () =
   Py.Module.get_function_with_keywords __wrap_namespace "peak_prominences"
     [||]
     (Wrap_utils.keyword_args [("wlen", Wrap_utils.Option.map wlen Py.Int.of_int); ("x", Some(x )); ("peaks", Some(peaks ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let peak_widths ?rel_height ?prominence_data ?wlen ~x ~peaks () =
   Py.Module.get_function_with_keywords __wrap_namespace "peak_widths"
     [||]
     (Wrap_utils.keyword_args [("rel_height", Wrap_utils.Option.map rel_height Py.Float.of_float); ("prominence_data", prominence_data); ("wlen", Wrap_utils.Option.map wlen Py.Int.of_int); ("x", Some(x )); ("peaks", Some(peaks ))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let periodogram ?fs ?window ?nfft ?detrend ?return_onesided ?scaling ?axis ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "periodogram"
                       [||]
                       (Wrap_utils.keyword_args [("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("window", Wrap_utils.Option.map window (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("nfft", Wrap_utils.Option.map nfft Py.Int.of_int); ("detrend", Wrap_utils.Option.map detrend (function
| `T_False_ x -> Wrap_utils.id x
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("return_onesided", Wrap_utils.Option.map return_onesided Py.Bool.of_bool); ("scaling", Wrap_utils.Option.map scaling (function
| `Density -> Py.String.of_string "density"
| `Spectrum -> Py.String.of_string "spectrum"
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let place_poles ?method_ ?rtol ?maxiter ~a ~b ~poles () =
                     Py.Module.get_function_with_keywords __wrap_namespace "place_poles"
                       [||]
                       (Wrap_utils.keyword_args [("method", Wrap_utils.Option.map method_ (function
| `YT -> Py.String.of_string "YT"
| `KNV0 -> Py.String.of_string "KNV0"
)); ("rtol", Wrap_utils.Option.map rtol Py.Float.of_float); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("A", Some(a )); ("B", Some(b )); ("poles", Some(poles |> Np.Obj.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3)), (Py.Float.to_float (Py.Tuple.get x 4)), (Py.Int.to_int (Py.Tuple.get x 5))))
let qmf hk =
   Py.Module.get_function_with_keywords __wrap_namespace "qmf"
     [||]
     (Wrap_utils.keyword_args [("hk", Some(hk |> Np.Obj.to_pyobject))])

let qspline1d ?lamb ~signal () =
   Py.Module.get_function_with_keywords __wrap_namespace "qspline1d"
     [||]
     (Wrap_utils.keyword_args [("lamb", Wrap_utils.Option.map lamb Py.Float.of_float); ("signal", Some(signal |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let qspline1d_eval ?dx ?x0 ~cj ~newx () =
   Py.Module.get_function_with_keywords __wrap_namespace "qspline1d_eval"
     [||]
     (Wrap_utils.keyword_args [("dx", dx); ("x0", x0); ("cj", Some(cj )); ("newx", Some(newx ))])

let quadratic x =
   Py.Module.get_function_with_keywords __wrap_namespace "quadratic"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

                  let remez ?weight ?hz ?type_ ?maxiter ?grid_density ?fs ~numtaps ~bands ~desired () =
                     Py.Module.get_function_with_keywords __wrap_namespace "remez"
                       [||]
                       (Wrap_utils.keyword_args [("weight", Wrap_utils.Option.map weight Np.Obj.to_pyobject); ("Hz", Wrap_utils.Option.map hz (function
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("type", Wrap_utils.Option.map type_ (function
| `Bandpass -> Py.String.of_string "bandpass"
| `Differentiator -> Py.String.of_string "differentiator"
| `Hilbert -> Py.String.of_string "hilbert"
)); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("grid_density", Wrap_utils.Option.map grid_density Py.Int.of_int); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("numtaps", Some(numtaps |> Py.Int.of_int)); ("bands", Some(bands |> Np.Obj.to_pyobject)); ("desired", Some(desired |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let resample ?t ?axis ?window ~x ~num () =
                     Py.Module.get_function_with_keywords __wrap_namespace "resample"
                       [||]
                       (Wrap_utils.keyword_args [("t", Wrap_utils.Option.map t Np.Obj.to_pyobject); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("window", Wrap_utils.Option.map window (function
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Np.Obj.to_pyobject x
| `Callable x -> Wrap_utils.id x
)); ("x", Some(x |> Np.Obj.to_pyobject)); ("num", Some(num |> Py.Int.of_int))])

                  let resample_poly ?axis ?window ?padtype ?cval ~x ~up ~down () =
                     Py.Module.get_function_with_keywords __wrap_namespace "resample_poly"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("window", Wrap_utils.Option.map window (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("padtype", Wrap_utils.Option.map padtype Py.String.of_string); ("cval", Wrap_utils.Option.map cval Py.Float.of_float); ("x", Some(x |> Np.Obj.to_pyobject)); ("up", Some(up |> Py.Int.of_int)); ("down", Some(down |> Py.Int.of_int))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let residue ?tol ?rtype ~b ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "residue"
                       [||]
                       (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("rtype", Wrap_utils.Option.map rtype (function
| `Avg -> Py.String.of_string "avg"
| `Min -> Py.String.of_string "min"
| `Max -> Py.String.of_string "max"
)); ("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
                  let residuez ?tol ?rtype ~b ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "residuez"
                       [||]
                       (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("rtype", Wrap_utils.Option.map rtype (function
| `Avg -> Py.String.of_string "avg"
| `Min -> Py.String.of_string "min"
| `Max -> Py.String.of_string "max"
)); ("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
                  let ricker ~points ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ricker"
                       [||]
                       (Wrap_utils.keyword_args [("points", Some(points |> Py.Int.of_int)); ("a", Some(a |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let savgol_coeffs ?deriv ?delta ?pos ?use ~window_length ~polyorder () =
   Py.Module.get_function_with_keywords __wrap_namespace "savgol_coeffs"
     [||]
     (Wrap_utils.keyword_args [("deriv", Wrap_utils.Option.map deriv Py.Int.of_int); ("delta", Wrap_utils.Option.map delta Py.Float.of_float); ("pos", Wrap_utils.Option.map pos Py.Int.of_int); ("use", Wrap_utils.Option.map use Py.String.of_string); ("window_length", Some(window_length |> Py.Int.of_int)); ("polyorder", Some(polyorder |> Py.Int.of_int))])

                  let savgol_filter ?deriv ?delta ?axis ?mode ?cval ~x ~window_length ~polyorder () =
                     Py.Module.get_function_with_keywords __wrap_namespace "savgol_filter"
                       [||]
                       (Wrap_utils.keyword_args [("deriv", Wrap_utils.Option.map deriv Py.Int.of_int); ("delta", Wrap_utils.Option.map delta Py.Float.of_float); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("mode", Wrap_utils.Option.map mode Py.String.of_string); ("cval", Wrap_utils.Option.map cval (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("x", Some(x |> Np.Obj.to_pyobject)); ("window_length", Some(window_length |> Py.Int.of_int)); ("polyorder", Some(polyorder |> Py.Int.of_int))])

let sawtooth ?width ~t () =
   Py.Module.get_function_with_keywords __wrap_namespace "sawtooth"
     [||]
     (Wrap_utils.keyword_args [("width", Wrap_utils.Option.map width Np.Obj.to_pyobject); ("t", Some(t |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let sepfir2d ~input ~hrow ~hcol () =
   Py.Module.get_function_with_keywords __wrap_namespace "sepfir2d"
     [||]
     (Wrap_utils.keyword_args [("input", Some(input )); ("hrow", Some(hrow )); ("hcol", Some(hcol ))])

let slepian ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "slepian"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let sos2tf sos =
   Py.Module.get_function_with_keywords __wrap_namespace "sos2tf"
     [||]
     (Wrap_utils.keyword_args [("sos", Some(sos |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let sos2zpk sos =
   Py.Module.get_function_with_keywords __wrap_namespace "sos2zpk"
     [||]
     (Wrap_utils.keyword_args [("sos", Some(sos |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let sosfilt ?axis ?zi ~sos ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "sosfilt"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("zi", Wrap_utils.Option.map zi Np.Obj.to_pyobject); ("sos", Some(sos |> Np.Obj.to_pyobject)); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let sosfilt_zi sos =
   Py.Module.get_function_with_keywords __wrap_namespace "sosfilt_zi"
     [||]
     (Wrap_utils.keyword_args [("sos", Some(sos |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let sosfiltfilt ?axis ?padtype ?padlen ~sos ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "sosfiltfilt"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("padtype", Wrap_utils.Option.map padtype (function
| `S x -> Py.String.of_string x
| `None -> Py.none
)); ("padlen", Wrap_utils.Option.map padlen Py.Int.of_int); ("sos", Some(sos |> Np.Obj.to_pyobject)); ("x", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let sosfreqz ?worN ?whole ?fs ~sos () =
                     Py.Module.get_function_with_keywords __wrap_namespace "sosfreqz"
                       [||]
                       (Wrap_utils.keyword_args [("worN", Wrap_utils.Option.map worN (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("whole", Wrap_utils.Option.map whole Py.Bool.of_bool); ("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("sos", Some(sos |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let spectrogram ?fs ?window ?nperseg ?noverlap ?nfft ?detrend ?return_onesided ?scaling ?axis ?mode ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "spectrogram"
                       [||]
                       (Wrap_utils.keyword_args [("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("window", Wrap_utils.Option.map window (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("nperseg", Wrap_utils.Option.map nperseg Py.Int.of_int); ("noverlap", Wrap_utils.Option.map noverlap Py.Int.of_int); ("nfft", Wrap_utils.Option.map nfft Py.Int.of_int); ("detrend", Wrap_utils.Option.map detrend (function
| `T_False_ x -> Wrap_utils.id x
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("return_onesided", Wrap_utils.Option.map return_onesided Py.Bool.of_bool); ("scaling", Wrap_utils.Option.map scaling (function
| `Density -> Py.String.of_string "density"
| `Spectrum -> Py.String.of_string "spectrum"
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("mode", Wrap_utils.Option.map mode Py.String.of_string); ("x", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
let spline_filter ?lmbda ~iin () =
   Py.Module.get_function_with_keywords __wrap_namespace "spline_filter"
     [||]
     (Wrap_utils.keyword_args [("lmbda", lmbda); ("Iin", Some(iin ))])

let square ?duty ~t () =
   Py.Module.get_function_with_keywords __wrap_namespace "square"
     [||]
     (Wrap_utils.keyword_args [("duty", Wrap_utils.Option.map duty Np.Obj.to_pyobject); ("t", Some(t |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let ss2tf ?input ~a ~b ~c ~d () =
   Py.Module.get_function_with_keywords __wrap_namespace "ss2tf"
     [||]
     (Wrap_utils.keyword_args [("input", Wrap_utils.Option.map input Py.Int.of_int); ("A", Some(a |> Np.Obj.to_pyobject)); ("B", Some(b |> Np.Obj.to_pyobject)); ("C", Some(c |> Np.Obj.to_pyobject)); ("D", Some(d |> Np.Obj.to_pyobject))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let ss2zpk ?input ~a ~b ~c ~d () =
   Py.Module.get_function_with_keywords __wrap_namespace "ss2zpk"
     [||]
     (Wrap_utils.keyword_args [("input", Wrap_utils.Option.map input Py.Int.of_int); ("A", Some(a |> Np.Obj.to_pyobject)); ("B", Some(b |> Np.Obj.to_pyobject)); ("C", Some(c |> Np.Obj.to_pyobject)); ("D", Some(d |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let step ?x0 ?t ?n ~system () =
   Py.Module.get_function_with_keywords __wrap_namespace "step"
     [||]
     (Wrap_utils.keyword_args [("X0", Wrap_utils.Option.map x0 Np.Obj.to_pyobject); ("T", Wrap_utils.Option.map t Np.Obj.to_pyobject); ("N", Wrap_utils.Option.map n Py.Int.of_int); ("system", Some(system ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let step2 ?x0 ?t ?n ?kwargs ~system () =
   Py.Module.get_function_with_keywords __wrap_namespace "step2"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("X0", Wrap_utils.Option.map x0 Np.Obj.to_pyobject); ("T", Wrap_utils.Option.map t Np.Obj.to_pyobject); ("N", Wrap_utils.Option.map n Py.Int.of_int); ("system", Some(system ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let stft ?fs ?window ?nperseg ?noverlap ?nfft ?detrend ?return_onesided ?boundary ?padded ?axis ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "stft"
                       [||]
                       (Wrap_utils.keyword_args [("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("window", Wrap_utils.Option.map window (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("nperseg", Wrap_utils.Option.map nperseg Py.Int.of_int); ("noverlap", Wrap_utils.Option.map noverlap Py.Int.of_int); ("nfft", Wrap_utils.Option.map nfft Py.Int.of_int); ("detrend", Wrap_utils.Option.map detrend (function
| `T_False_ x -> Wrap_utils.id x
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("return_onesided", Wrap_utils.Option.map return_onesided Py.Bool.of_bool); ("boundary", Wrap_utils.Option.map boundary (function
| `S x -> Py.String.of_string x
| `None -> Py.none
)); ("padded", Wrap_utils.Option.map padded Py.Bool.of_bool); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("x", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
let sweep_poly ?phi ~t ~poly () =
   Py.Module.get_function_with_keywords __wrap_namespace "sweep_poly"
     [||]
     (Wrap_utils.keyword_args [("phi", Wrap_utils.Option.map phi Py.Float.of_float); ("t", Some(t |> Np.Obj.to_pyobject)); ("poly", Some(poly ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let tf2sos ?pairing ~b ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "tf2sos"
                       [||]
                       (Wrap_utils.keyword_args [("pairing", Wrap_utils.Option.map pairing (function
| `Nearest -> Py.String.of_string "nearest"
| `Keep_odd -> Py.String.of_string "keep_odd"
)); ("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let tf2ss ~num ~den () =
   Py.Module.get_function_with_keywords __wrap_namespace "tf2ss"
     [||]
     (Wrap_utils.keyword_args [("num", Some(num )); ("den", Some(den ))])

let tf2zpk ~b ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "tf2zpk"
     [||]
     (Wrap_utils.keyword_args [("b", Some(b |> Np.Obj.to_pyobject)); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let triang ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "triang"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let tukey ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "tukey"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let unique_roots ?tol ?rtype ~p () =
                     Py.Module.get_function_with_keywords __wrap_namespace "unique_roots"
                       [||]
                       (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("rtype", Wrap_utils.Option.map rtype (function
| `Max -> Py.String.of_string "max"
| `Maximum -> Py.String.of_string "maximum"
| `Min -> Py.String.of_string "min"
| `Minimum -> Py.String.of_string "minimum"
| `Avg -> Py.String.of_string "avg"
| `Mean -> Py.String.of_string "mean"
)); ("p", Some(p |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let unit_impulse ?idx ?dtype ~shape () =
                     Py.Module.get_function_with_keywords __wrap_namespace "unit_impulse"
                       [||]
                       (Wrap_utils.keyword_args [("idx", Wrap_utils.Option.map idx (function
| `I x -> Py.Int.of_int x
| `Tuple_of_int x -> Wrap_utils.id x
| `Mid -> Py.String.of_string "mid"
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("shape", Some(shape |> (function
| `I x -> Py.Int.of_int x
| `Tuple_of_int x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let upfirdn ?up ?down ?axis ?mode ?cval ~h ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "upfirdn"
     [||]
     (Wrap_utils.keyword_args [("up", Wrap_utils.Option.map up Py.Int.of_int); ("down", Wrap_utils.Option.map down Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("mode", Wrap_utils.Option.map mode Py.String.of_string); ("cval", Wrap_utils.Option.map cval Py.Float.of_float); ("h", Some(h |> Np.Obj.to_pyobject)); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let vectorstrength ~events ~period () =
                     Py.Module.get_function_with_keywords __wrap_namespace "vectorstrength"
                       [||]
                       (Wrap_utils.keyword_args [("events", Some(events )); ("period", Some(period |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let welch ?fs ?window ?nperseg ?noverlap ?nfft ?detrend ?return_onesided ?scaling ?axis ?average ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "welch"
                       [||]
                       (Wrap_utils.keyword_args [("fs", Wrap_utils.Option.map fs Py.Float.of_float); ("window", Wrap_utils.Option.map window (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("nperseg", Wrap_utils.Option.map nperseg Py.Int.of_int); ("noverlap", Wrap_utils.Option.map noverlap Py.Int.of_int); ("nfft", Wrap_utils.Option.map nfft Py.Int.of_int); ("detrend", Wrap_utils.Option.map detrend (function
| `T_False_ x -> Wrap_utils.id x
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("return_onesided", Wrap_utils.Option.map return_onesided Py.Bool.of_bool); ("scaling", Wrap_utils.Option.map scaling (function
| `Density -> Py.String.of_string "density"
| `Spectrum -> Py.String.of_string "spectrum"
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("average", Wrap_utils.Option.map average (function
| `Mean -> Py.String.of_string "mean"
| `Median -> Py.String.of_string "median"
)); ("x", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let wiener ?mysize ?noise ~im () =
                     Py.Module.get_function_with_keywords __wrap_namespace "wiener"
                       [||]
                       (Wrap_utils.keyword_args [("mysize", Wrap_utils.Option.map mysize (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("noise", Wrap_utils.Option.map noise Py.Float.of_float); ("im", Some(im |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let zpk2sos ?pairing ~z ~p ~k () =
                     Py.Module.get_function_with_keywords __wrap_namespace "zpk2sos"
                       [||]
                       (Wrap_utils.keyword_args [("pairing", Wrap_utils.Option.map pairing (function
| `Nearest -> Py.String.of_string "nearest"
| `Keep_odd -> Py.String.of_string "keep_odd"
)); ("z", Some(z |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject)); ("k", Some(k |> Py.Float.of_float))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let zpk2ss ~z ~p ~k () =
   Py.Module.get_function_with_keywords __wrap_namespace "zpk2ss"
     [||]
     (Wrap_utils.keyword_args [("z", Some(z )); ("p", Some(p )); ("k", Some(k |> Py.Float.of_float))])

let zpk2tf ~z ~p ~k () =
   Py.Module.get_function_with_keywords __wrap_namespace "zpk2tf"
     [||]
     (Wrap_utils.keyword_args [("z", Some(z |> Np.Obj.to_pyobject)); ("p", Some(p |> Np.Obj.to_pyobject)); ("k", Some(k |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
