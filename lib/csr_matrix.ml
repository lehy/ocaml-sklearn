let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.metrics.pairwise"

type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?shape ?dtype ?copy ~arg1 () =
   Py.Module.get_function_with_keywords ns "csr_matrix"
     [||]
     (Wrap_utils.keyword_args [("shape", Wrap_utils.Option.map shape (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("dtype", dtype); ("copy", copy); ("arg1", Some(arg1 ))])

let get_item ~key self =
   Py.Module.get_function_with_keywords self "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key ))])

let arcsin self =
   Py.Module.get_function_with_keywords self "arcsin"
     [||]
     []

let arcsinh self =
   Py.Module.get_function_with_keywords self "arcsinh"
     [||]
     []

let arctan self =
   Py.Module.get_function_with_keywords self "arctan"
     [||]
     []

let arctanh self =
   Py.Module.get_function_with_keywords self "arctanh"
     [||]
     []

                  let argmax ?axis ?out self =
                     Py.Module.get_function_with_keywords self "argmax"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

                  let argmin ?axis ?out self =
                     Py.Module.get_function_with_keywords self "argmin"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

let asformat ?copy ~format self =
   Py.Module.get_function_with_keywords self "asformat"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("format", Some(format |> Py.String.of_string))])

let asfptype self =
   Py.Module.get_function_with_keywords self "asfptype"
     [||]
     []

                  let astype ?casting ?copy ~dtype self =
                     Py.Module.get_function_with_keywords self "astype"
                       [||]
                       (Wrap_utils.keyword_args [("casting", casting); ("copy", copy); ("dtype", Some(dtype |> (function
| `String x -> Py.String.of_string x
| `PyObject x -> Wrap_utils.id x
)))])

let ceil self =
   Py.Module.get_function_with_keywords self "ceil"
     [||]
     []

let check_format ?full_check self =
   Py.Module.get_function_with_keywords self "check_format"
     [||]
     (Wrap_utils.keyword_args [("full_check", Wrap_utils.Option.map full_check Py.Bool.of_bool)])

let conj ?copy self =
   Py.Module.get_function_with_keywords self "conj"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let conjugate ?copy self =
   Py.Module.get_function_with_keywords self "conjugate"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let copy self =
   Py.Module.get_function_with_keywords self "copy"
     [||]
     []

let count_nonzero self =
   Py.Module.get_function_with_keywords self "count_nonzero"
     [||]
     []

let deg2rad self =
   Py.Module.get_function_with_keywords self "deg2rad"
     [||]
     []

let diagonal ?k self =
   Py.Module.get_function_with_keywords self "diagonal"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int)])

let dot ~other self =
   Py.Module.get_function_with_keywords self "dot"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let eliminate_zeros self =
   Py.Module.get_function_with_keywords self "eliminate_zeros"
     [||]
     []

let expm1 self =
   Py.Module.get_function_with_keywords self "expm1"
     [||]
     []

let floor self =
   Py.Module.get_function_with_keywords self "floor"
     [||]
     []

let getH self =
   Py.Module.get_function_with_keywords self "getH"
     [||]
     []

let get_shape self =
   Py.Module.get_function_with_keywords self "get_shape"
     [||]
     []

let getcol ~i self =
   Py.Module.get_function_with_keywords self "getcol"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i ))])

let getformat self =
   Py.Module.get_function_with_keywords self "getformat"
     [||]
     []

let getmaxprint self =
   Py.Module.get_function_with_keywords self "getmaxprint"
     [||]
     []

                  let getnnz ?axis self =
                     Py.Module.get_function_with_keywords self "getnnz"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
))])

let getrow ~i self =
   Py.Module.get_function_with_keywords self "getrow"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i ))])

let log1p self =
   Py.Module.get_function_with_keywords self "log1p"
     [||]
     []

                  let max ?axis ?out self =
                     Py.Module.get_function_with_keywords self "max"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

let maximum ~other self =
   Py.Module.get_function_with_keywords self "maximum"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

                  let mean ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords self "mean"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
)); ("dtype", dtype); ("out", out)])

                  let min ?axis ?out self =
                     Py.Module.get_function_with_keywords self "min"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

let minimum ~other self =
   Py.Module.get_function_with_keywords self "minimum"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let multiply ~other self =
   Py.Module.get_function_with_keywords self "multiply"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let nonzero self =
   Py.Module.get_function_with_keywords self "nonzero"
     [||]
     []

let power ?dtype ~n self =
   Py.Module.get_function_with_keywords self "power"
     [||]
     (Wrap_utils.keyword_args [("dtype", dtype); ("n", Some(n ))])

let prune self =
   Py.Module.get_function_with_keywords self "prune"
     [||]
     []

let rad2deg self =
   Py.Module.get_function_with_keywords self "rad2deg"
     [||]
     []

let reshape ?kwargs args self =
   Py.Module.get_function_with_keywords self "reshape"
     (Wrap_utils.pos_arg Wrap_utils.id args)
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let resize shape self =
   Py.Module.get_function_with_keywords self "resize"
     (Wrap_utils.pos_arg Py.Int.of_int shape)
     []

let rint self =
   Py.Module.get_function_with_keywords self "rint"
     [||]
     []

let set_shape ~shape self =
   Py.Module.get_function_with_keywords self "set_shape"
     [||]
     (Wrap_utils.keyword_args [("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])

let setdiag ?k ~values self =
   Py.Module.get_function_with_keywords self "setdiag"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("values", Some(values |> Ndarray.to_pyobject))])

let sign self =
   Py.Module.get_function_with_keywords self "sign"
     [||]
     []

let sin self =
   Py.Module.get_function_with_keywords self "sin"
     [||]
     []

let sinh self =
   Py.Module.get_function_with_keywords self "sinh"
     [||]
     []

let sort_indices self =
   Py.Module.get_function_with_keywords self "sort_indices"
     [||]
     []

let sorted_indices self =
   Py.Module.get_function_with_keywords self "sorted_indices"
     [||]
     []

let sqrt self =
   Py.Module.get_function_with_keywords self "sqrt"
     [||]
     []

                  let sum ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords self "sum"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
)); ("dtype", dtype); ("out", out)])

let sum_duplicates self =
   Py.Module.get_function_with_keywords self "sum_duplicates"
     [||]
     []

let tan self =
   Py.Module.get_function_with_keywords self "tan"
     [||]
     []

let tanh self =
   Py.Module.get_function_with_keywords self "tanh"
     [||]
     []

                  let toarray ?order ?out self =
                     Py.Module.get_function_with_keywords self "toarray"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("out", Wrap_utils.Option.map out Ndarray.to_pyobject)])
                       |> Ndarray.of_pyobject
let tobsr ?blocksize ?copy self =
   Py.Module.get_function_with_keywords self "tobsr"
     [||]
     (Wrap_utils.keyword_args [("blocksize", blocksize); ("copy", copy)])

let tocoo ?copy self =
   Py.Module.get_function_with_keywords self "tocoo"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tocsc ?copy self =
   Py.Module.get_function_with_keywords self "tocsc"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tocsr ?copy self =
   Py.Module.get_function_with_keywords self "tocsr"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

                  let todense ?order ?out self =
                     Py.Module.get_function_with_keywords self "todense"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("out", Wrap_utils.Option.map out Ndarray.to_pyobject)])

let todia ?copy self =
   Py.Module.get_function_with_keywords self "todia"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let todok ?copy self =
   Py.Module.get_function_with_keywords self "todok"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tolil ?copy self =
   Py.Module.get_function_with_keywords self "tolil"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let transpose ?axes ?copy self =
   Py.Module.get_function_with_keywords self "transpose"
     [||]
     (Wrap_utils.keyword_args [("axes", axes); ("copy", copy)])

let trunc self =
   Py.Module.get_function_with_keywords self "trunc"
     [||]
     []

let dtype self =
  match Py.Object.get_attr_string self "dtype" with
| None -> raise (Wrap_utils.Attribute_not_found "dtype")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)
