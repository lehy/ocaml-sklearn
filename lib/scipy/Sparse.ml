let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse"

let get_py name = Py.Module.get __wrap_namespace name
module SparseEfficiencyWarning = struct
type tag = [`SparseEfficiencyWarning]
type t = [`BaseException | `Object | `SparseEfficiencyWarning] Obj.t
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
module SparseWarning = struct
type tag = [`SparseWarning]
type t = [`BaseException | `Object | `SparseWarning] Obj.t
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
module Bsr_matrix = struct
type tag = [`Bsr_matrix]
type t = [`ArrayLike | `Bsr_matrix | `IndexMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_index x = (x :> [`IndexMixin] Obj.t)
let create ?shape ?dtype ?copy ?blocksize ~arg1 () =
   Py.Module.get_function_with_keywords __wrap_namespace "bsr_matrix"
     [||]
     (Wrap_utils.keyword_args [("shape", shape); ("dtype", dtype); ("copy", copy); ("blocksize", blocksize); ("arg1", Some(arg1 ))])
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key ))])

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let __setitem__ ~key ~val_ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key )); ("val", Some(val_ ))])

let arcsin self =
   Py.Module.get_function_with_keywords (to_pyobject self) "arcsin"
     [||]
     []

let arcsinh self =
   Py.Module.get_function_with_keywords (to_pyobject self) "arcsinh"
     [||]
     []

let arctan self =
   Py.Module.get_function_with_keywords (to_pyobject self) "arctan"
     [||]
     []

let arctanh self =
   Py.Module.get_function_with_keywords (to_pyobject self) "arctanh"
     [||]
     []

                  let argmax ?axis ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "argmax"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

                  let argmin ?axis ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "argmin"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

                  let asformat ?copy ~format self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "asformat"
                       [||]
                       (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("format", Some(format |> (function
| `S x -> Py.String.of_string x
| `None -> Py.none
)))])

let asfptype self =
   Py.Module.get_function_with_keywords (to_pyobject self) "asfptype"
     [||]
     []

                  let astype ?casting ?copy ~dtype self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "astype"
                       [||]
                       (Wrap_utils.keyword_args [("casting", Wrap_utils.Option.map casting (function
| `No -> Py.String.of_string "no"
| `Equiv -> Py.String.of_string "equiv"
| `Safe -> Py.String.of_string "safe"
| `Same_kind -> Py.String.of_string "same_kind"
| `Unsafe -> Py.String.of_string "unsafe"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("dtype", Some(dtype |> (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)))])

let ceil self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ceil"
     [||]
     []

let check_format ?full_check self =
   Py.Module.get_function_with_keywords (to_pyobject self) "check_format"
     [||]
     (Wrap_utils.keyword_args [("full_check", full_check)])

let conj ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conj"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let conjugate ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conjugate"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "copy"
     [||]
     []

let count_nonzero self =
   Py.Module.get_function_with_keywords (to_pyobject self) "count_nonzero"
     [||]
     []

let deg2rad self =
   Py.Module.get_function_with_keywords (to_pyobject self) "deg2rad"
     [||]
     []

let diagonal ?k self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diagonal"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int)])

let dot ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let eliminate_zeros self =
   Py.Module.get_function_with_keywords (to_pyobject self) "eliminate_zeros"
     [||]
     []

let expm1 self =
   Py.Module.get_function_with_keywords (to_pyobject self) "expm1"
     [||]
     []

let floor self =
   Py.Module.get_function_with_keywords (to_pyobject self) "floor"
     [||]
     []

let getH self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getH"
     [||]
     []

let get_shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_shape"
     [||]
     []

let getcol ~j self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getcol"
     [||]
     (Wrap_utils.keyword_args [("j", Some(j ))])

let getformat self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getformat"
     [||]
     []

let getmaxprint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getmaxprint"
     [||]
     []

                  let getnnz ?axis self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "getnnz"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
))])

let getrow ~i self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getrow"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i ))])

let log1p self =
   Py.Module.get_function_with_keywords (to_pyobject self) "log1p"
     [||]
     []

                  let max ?axis ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "max"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

let maximum ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "maximum"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

                  let mean ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "mean"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let min ?axis ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "min"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

let minimum ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "minimum"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let multiply ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "multiply"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let nonzero self =
   Py.Module.get_function_with_keywords (to_pyobject self) "nonzero"
     [||]
     []

let power ?dtype ~n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "power"
     [||]
     (Wrap_utils.keyword_args [("dtype", dtype); ("n", Some(n ))])

let prune self =
   Py.Module.get_function_with_keywords (to_pyobject self) "prune"
     [||]
     []

let rad2deg self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rad2deg"
     [||]
     []

let reshape ?kwargs args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "reshape"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t))
let resize shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "resize"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id shape)])
     []

let rint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rint"
     [||]
     []

let set_shape ~shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_shape"
     [||]
     (Wrap_utils.keyword_args [("shape", Some(shape ))])

let setdiag ?k ~values self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setdiag"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("values", Some(values |> Np.Obj.to_pyobject))])

let sign self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sign"
     [||]
     []

let sin self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sin"
     [||]
     []

let sinh self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sinh"
     [||]
     []

let sort_indices self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sort_indices"
     [||]
     []

let sorted_indices self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sorted_indices"
     [||]
     []

let sqrt self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sqrt"
     [||]
     []

                  let sum ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "sum"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let sum_duplicates self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sum_duplicates"
     [||]
     []

let tan self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tan"
     [||]
     []

let tanh self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tanh"
     [||]
     []

                  let toarray ?order ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "toarray"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let tobsr ?blocksize ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tobsr"
     [||]
     (Wrap_utils.keyword_args [("blocksize", blocksize); ("copy", copy)])

let tocoo ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocoo"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tocsc ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocsc"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tocsr ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocsr"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

                  let todense ?order ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "todense"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let todia ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "todia"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let todok ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "todok"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tolil ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tolil"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let transpose ?axes ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     [||]
     (Wrap_utils.keyword_args [("axes", axes); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let trunc self =
   Py.Module.get_function_with_keywords (to_pyobject self) "trunc"
     [||]
     []


let dtype_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dtype" with
  | None -> failwith "attribute dtype not found"
  | Some x -> if Py.is_none x then None else Some (Np.Dtype.of_pyobject x)

let dtype self = match dtype_opt self with
  | None -> raise Not_found
  | Some x -> x

let shape_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "shape" with
  | None -> failwith "attribute shape not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let shape self = match shape_opt self with
  | None -> raise Not_found
  | Some x -> x

let ndim_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "ndim" with
  | None -> failwith "attribute ndim not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let ndim self = match ndim_opt self with
  | None -> raise Not_found
  | Some x -> x

let nnz_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nnz" with
  | None -> failwith "attribute nnz not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let nnz self = match nnz_opt self with
  | None -> raise Not_found
  | Some x -> x

let data_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "data" with
  | None -> failwith "attribute data not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let data self = match data_opt self with
  | None -> raise Not_found
  | Some x -> x

let indices_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "indices" with
  | None -> failwith "attribute indices not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let indices self = match indices_opt self with
  | None -> raise Not_found
  | Some x -> x

let indptr_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "indptr" with
  | None -> failwith "attribute indptr not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let indptr self = match indptr_opt self with
  | None -> raise Not_found
  | Some x -> x

let blocksize_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "blocksize" with
  | None -> failwith "attribute blocksize not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let blocksize self = match blocksize_opt self with
  | None -> raise Not_found
  | Some x -> x

let has_sorted_indices_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "has_sorted_indices" with
  | None -> failwith "attribute has_sorted_indices not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let has_sorted_indices self = match has_sorted_indices_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Coo_matrix = struct
type tag = [`Coo_matrix]
type t = [`ArrayLike | `Coo_matrix | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?shape ?dtype ?copy ~arg1 () =
   Py.Module.get_function_with_keywords __wrap_namespace "coo_matrix"
     [||]
     (Wrap_utils.keyword_args [("shape", shape); ("dtype", dtype); ("copy", copy); ("arg1", Some(arg1 ))])
     |> of_pyobject
let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let arcsin self =
   Py.Module.get_function_with_keywords (to_pyobject self) "arcsin"
     [||]
     []

let arcsinh self =
   Py.Module.get_function_with_keywords (to_pyobject self) "arcsinh"
     [||]
     []

let arctan self =
   Py.Module.get_function_with_keywords (to_pyobject self) "arctan"
     [||]
     []

let arctanh self =
   Py.Module.get_function_with_keywords (to_pyobject self) "arctanh"
     [||]
     []

                  let argmax ?axis ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "argmax"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

                  let argmin ?axis ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "argmin"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

                  let asformat ?copy ~format self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "asformat"
                       [||]
                       (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("format", Some(format |> (function
| `S x -> Py.String.of_string x
| `None -> Py.none
)))])

let asfptype self =
   Py.Module.get_function_with_keywords (to_pyobject self) "asfptype"
     [||]
     []

                  let astype ?casting ?copy ~dtype self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "astype"
                       [||]
                       (Wrap_utils.keyword_args [("casting", Wrap_utils.Option.map casting (function
| `No -> Py.String.of_string "no"
| `Equiv -> Py.String.of_string "equiv"
| `Safe -> Py.String.of_string "safe"
| `Same_kind -> Py.String.of_string "same_kind"
| `Unsafe -> Py.String.of_string "unsafe"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("dtype", Some(dtype |> (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)))])

let ceil self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ceil"
     [||]
     []

let conj ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conj"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let conjugate ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conjugate"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "copy"
     [||]
     []

let count_nonzero self =
   Py.Module.get_function_with_keywords (to_pyobject self) "count_nonzero"
     [||]
     []

let deg2rad self =
   Py.Module.get_function_with_keywords (to_pyobject self) "deg2rad"
     [||]
     []

let diagonal ?k self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diagonal"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int)])

let dot ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let eliminate_zeros self =
   Py.Module.get_function_with_keywords (to_pyobject self) "eliminate_zeros"
     [||]
     []

let expm1 self =
   Py.Module.get_function_with_keywords (to_pyobject self) "expm1"
     [||]
     []

let floor self =
   Py.Module.get_function_with_keywords (to_pyobject self) "floor"
     [||]
     []

let getH self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getH"
     [||]
     []

let get_shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_shape"
     [||]
     []

let getcol ~j self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getcol"
     [||]
     (Wrap_utils.keyword_args [("j", Some(j ))])

let getformat self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getformat"
     [||]
     []

let getmaxprint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getmaxprint"
     [||]
     []

                  let getnnz ?axis self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "getnnz"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
))])

let getrow ~i self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getrow"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i ))])

let log1p self =
   Py.Module.get_function_with_keywords (to_pyobject self) "log1p"
     [||]
     []

                  let max ?axis ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "max"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

let maximum ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "maximum"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

                  let mean ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "mean"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let min ?axis ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "min"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

let minimum ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "minimum"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let multiply ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "multiply"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let nonzero self =
   Py.Module.get_function_with_keywords (to_pyobject self) "nonzero"
     [||]
     []

let power ?dtype ~n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "power"
     [||]
     (Wrap_utils.keyword_args [("dtype", dtype); ("n", Some(n ))])

let rad2deg self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rad2deg"
     [||]
     []

let reshape ?kwargs args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "reshape"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t))
let resize shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "resize"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id shape)])
     []

let rint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rint"
     [||]
     []

let set_shape ~shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_shape"
     [||]
     (Wrap_utils.keyword_args [("shape", Some(shape ))])

let setdiag ?k ~values self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setdiag"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("values", Some(values |> Np.Obj.to_pyobject))])

let sign self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sign"
     [||]
     []

let sin self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sin"
     [||]
     []

let sinh self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sinh"
     [||]
     []

let sqrt self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sqrt"
     [||]
     []

                  let sum ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "sum"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let sum_duplicates self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sum_duplicates"
     [||]
     []

let tan self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tan"
     [||]
     []

let tanh self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tanh"
     [||]
     []

let toarray ?order ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "toarray"
     [||]
     (Wrap_utils.keyword_args [("order", order); ("out", out)])

let tobsr ?blocksize ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tobsr"
     [||]
     (Wrap_utils.keyword_args [("blocksize", blocksize); ("copy", copy)])

let tocoo ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocoo"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tocsc ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocsc"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tocsr ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocsr"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

                  let todense ?order ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "todense"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let todia ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "todia"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let todok ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "todok"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tolil ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tolil"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let transpose ?axes ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     [||]
     (Wrap_utils.keyword_args [("axes", axes); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let trunc self =
   Py.Module.get_function_with_keywords (to_pyobject self) "trunc"
     [||]
     []


let dtype_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dtype" with
  | None -> failwith "attribute dtype not found"
  | Some x -> if Py.is_none x then None else Some (Np.Dtype.of_pyobject x)

let dtype self = match dtype_opt self with
  | None -> raise Not_found
  | Some x -> x

let shape_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "shape" with
  | None -> failwith "attribute shape not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let shape self = match shape_opt self with
  | None -> raise Not_found
  | Some x -> x

let ndim_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "ndim" with
  | None -> failwith "attribute ndim not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let ndim self = match ndim_opt self with
  | None -> raise Not_found
  | Some x -> x

let nnz_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nnz" with
  | None -> failwith "attribute nnz not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let nnz self = match nnz_opt self with
  | None -> raise Not_found
  | Some x -> x

let data_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "data" with
  | None -> failwith "attribute data not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let data self = match data_opt self with
  | None -> raise Not_found
  | Some x -> x

let row_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "row" with
  | None -> failwith "attribute row not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let row self = match row_opt self with
  | None -> raise Not_found
  | Some x -> x

let col_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "col" with
  | None -> failwith "attribute col not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let col self = match col_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Csc_matrix = struct
type tag = [`Csc_matrix]
type t = [`ArrayLike | `Csc_matrix | `IndexMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_index x = (x :> [`IndexMixin] Obj.t)
let create ?shape ?dtype ?copy ~arg1 () =
   Py.Module.get_function_with_keywords __wrap_namespace "csc_matrix"
     [||]
     (Wrap_utils.keyword_args [("shape", shape); ("dtype", dtype); ("copy", copy); ("arg1", Some(arg1 ))])
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key ))])

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let __setitem__ ~key ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key )); ("x", Some(x ))])

let arcsin self =
   Py.Module.get_function_with_keywords (to_pyobject self) "arcsin"
     [||]
     []

let arcsinh self =
   Py.Module.get_function_with_keywords (to_pyobject self) "arcsinh"
     [||]
     []

let arctan self =
   Py.Module.get_function_with_keywords (to_pyobject self) "arctan"
     [||]
     []

let arctanh self =
   Py.Module.get_function_with_keywords (to_pyobject self) "arctanh"
     [||]
     []

                  let argmax ?axis ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "argmax"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

                  let argmin ?axis ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "argmin"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

                  let asformat ?copy ~format self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "asformat"
                       [||]
                       (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("format", Some(format |> (function
| `S x -> Py.String.of_string x
| `None -> Py.none
)))])

let asfptype self =
   Py.Module.get_function_with_keywords (to_pyobject self) "asfptype"
     [||]
     []

                  let astype ?casting ?copy ~dtype self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "astype"
                       [||]
                       (Wrap_utils.keyword_args [("casting", Wrap_utils.Option.map casting (function
| `No -> Py.String.of_string "no"
| `Equiv -> Py.String.of_string "equiv"
| `Safe -> Py.String.of_string "safe"
| `Same_kind -> Py.String.of_string "same_kind"
| `Unsafe -> Py.String.of_string "unsafe"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("dtype", Some(dtype |> (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)))])

let ceil self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ceil"
     [||]
     []

let check_format ?full_check self =
   Py.Module.get_function_with_keywords (to_pyobject self) "check_format"
     [||]
     (Wrap_utils.keyword_args [("full_check", Wrap_utils.Option.map full_check Py.Bool.of_bool)])

let conj ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conj"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let conjugate ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conjugate"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "copy"
     [||]
     []

let count_nonzero self =
   Py.Module.get_function_with_keywords (to_pyobject self) "count_nonzero"
     [||]
     []

let deg2rad self =
   Py.Module.get_function_with_keywords (to_pyobject self) "deg2rad"
     [||]
     []

let diagonal ?k self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diagonal"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int)])

let dot ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let eliminate_zeros self =
   Py.Module.get_function_with_keywords (to_pyobject self) "eliminate_zeros"
     [||]
     []

let expm1 self =
   Py.Module.get_function_with_keywords (to_pyobject self) "expm1"
     [||]
     []

let floor self =
   Py.Module.get_function_with_keywords (to_pyobject self) "floor"
     [||]
     []

let getH self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getH"
     [||]
     []

let get_shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_shape"
     [||]
     []

let getcol ~i self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getcol"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i ))])

let getformat self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getformat"
     [||]
     []

let getmaxprint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getmaxprint"
     [||]
     []

                  let getnnz ?axis self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "getnnz"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
))])

let getrow ~i self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getrow"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i ))])

let log1p self =
   Py.Module.get_function_with_keywords (to_pyobject self) "log1p"
     [||]
     []

                  let max ?axis ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "max"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

let maximum ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "maximum"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

                  let mean ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "mean"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let min ?axis ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "min"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

let minimum ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "minimum"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let multiply ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "multiply"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let nonzero self =
   Py.Module.get_function_with_keywords (to_pyobject self) "nonzero"
     [||]
     []

let power ?dtype ~n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "power"
     [||]
     (Wrap_utils.keyword_args [("dtype", dtype); ("n", Some(n ))])

let prune self =
   Py.Module.get_function_with_keywords (to_pyobject self) "prune"
     [||]
     []

let rad2deg self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rad2deg"
     [||]
     []

let reshape ?kwargs args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "reshape"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t))
let resize shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "resize"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id shape)])
     []

let rint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rint"
     [||]
     []

let set_shape ~shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_shape"
     [||]
     (Wrap_utils.keyword_args [("shape", Some(shape ))])

let setdiag ?k ~values self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setdiag"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("values", Some(values |> Np.Obj.to_pyobject))])

let sign self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sign"
     [||]
     []

let sin self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sin"
     [||]
     []

let sinh self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sinh"
     [||]
     []

let sort_indices self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sort_indices"
     [||]
     []

let sorted_indices self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sorted_indices"
     [||]
     []

let sqrt self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sqrt"
     [||]
     []

                  let sum ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "sum"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let sum_duplicates self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sum_duplicates"
     [||]
     []

let tan self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tan"
     [||]
     []

let tanh self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tanh"
     [||]
     []

                  let toarray ?order ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "toarray"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let tobsr ?blocksize ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tobsr"
     [||]
     (Wrap_utils.keyword_args [("blocksize", blocksize); ("copy", copy)])

let tocoo ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocoo"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tocsc ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocsc"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tocsr ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocsr"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

                  let todense ?order ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "todense"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let todia ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "todia"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let todok ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "todok"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tolil ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tolil"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let transpose ?axes ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     [||]
     (Wrap_utils.keyword_args [("axes", axes); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let trunc self =
   Py.Module.get_function_with_keywords (to_pyobject self) "trunc"
     [||]
     []


let dtype_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dtype" with
  | None -> failwith "attribute dtype not found"
  | Some x -> if Py.is_none x then None else Some (Np.Dtype.of_pyobject x)

let dtype self = match dtype_opt self with
  | None -> raise Not_found
  | Some x -> x

let shape_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "shape" with
  | None -> failwith "attribute shape not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let shape self = match shape_opt self with
  | None -> raise Not_found
  | Some x -> x

let ndim_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "ndim" with
  | None -> failwith "attribute ndim not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let ndim self = match ndim_opt self with
  | None -> raise Not_found
  | Some x -> x

let nnz_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nnz" with
  | None -> failwith "attribute nnz not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let nnz self = match nnz_opt self with
  | None -> raise Not_found
  | Some x -> x

let data_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "data" with
  | None -> failwith "attribute data not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let data self = match data_opt self with
  | None -> raise Not_found
  | Some x -> x

let indices_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "indices" with
  | None -> failwith "attribute indices not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let indices self = match indices_opt self with
  | None -> raise Not_found
  | Some x -> x

let indptr_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "indptr" with
  | None -> failwith "attribute indptr not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let indptr self = match indptr_opt self with
  | None -> raise Not_found
  | Some x -> x

let has_sorted_indices_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "has_sorted_indices" with
  | None -> failwith "attribute has_sorted_indices not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let has_sorted_indices self = match has_sorted_indices_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Csr_matrix = struct
type tag = [`Csr_matrix]
type t = [`ArrayLike | `Csr_matrix | `IndexMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_index x = (x :> [`IndexMixin] Obj.t)
let create ?shape ?dtype ?copy ~arg1 () =
   Py.Module.get_function_with_keywords __wrap_namespace "csr_matrix"
     [||]
     (Wrap_utils.keyword_args [("shape", shape); ("dtype", dtype); ("copy", copy); ("arg1", Some(arg1 ))])
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key ))])

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let __setitem__ ~key ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key )); ("x", Some(x ))])

let arcsin self =
   Py.Module.get_function_with_keywords (to_pyobject self) "arcsin"
     [||]
     []

let arcsinh self =
   Py.Module.get_function_with_keywords (to_pyobject self) "arcsinh"
     [||]
     []

let arctan self =
   Py.Module.get_function_with_keywords (to_pyobject self) "arctan"
     [||]
     []

let arctanh self =
   Py.Module.get_function_with_keywords (to_pyobject self) "arctanh"
     [||]
     []

                  let argmax ?axis ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "argmax"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

                  let argmin ?axis ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "argmin"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

                  let asformat ?copy ~format self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "asformat"
                       [||]
                       (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("format", Some(format |> (function
| `S x -> Py.String.of_string x
| `None -> Py.none
)))])

let asfptype self =
   Py.Module.get_function_with_keywords (to_pyobject self) "asfptype"
     [||]
     []

                  let astype ?casting ?copy ~dtype self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "astype"
                       [||]
                       (Wrap_utils.keyword_args [("casting", Wrap_utils.Option.map casting (function
| `No -> Py.String.of_string "no"
| `Equiv -> Py.String.of_string "equiv"
| `Safe -> Py.String.of_string "safe"
| `Same_kind -> Py.String.of_string "same_kind"
| `Unsafe -> Py.String.of_string "unsafe"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("dtype", Some(dtype |> (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)))])

let ceil self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ceil"
     [||]
     []

let check_format ?full_check self =
   Py.Module.get_function_with_keywords (to_pyobject self) "check_format"
     [||]
     (Wrap_utils.keyword_args [("full_check", Wrap_utils.Option.map full_check Py.Bool.of_bool)])

let conj ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conj"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let conjugate ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conjugate"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "copy"
     [||]
     []

let count_nonzero self =
   Py.Module.get_function_with_keywords (to_pyobject self) "count_nonzero"
     [||]
     []

let deg2rad self =
   Py.Module.get_function_with_keywords (to_pyobject self) "deg2rad"
     [||]
     []

let diagonal ?k self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diagonal"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int)])

let dot ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let eliminate_zeros self =
   Py.Module.get_function_with_keywords (to_pyobject self) "eliminate_zeros"
     [||]
     []

let expm1 self =
   Py.Module.get_function_with_keywords (to_pyobject self) "expm1"
     [||]
     []

let floor self =
   Py.Module.get_function_with_keywords (to_pyobject self) "floor"
     [||]
     []

let getH self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getH"
     [||]
     []

let get_shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_shape"
     [||]
     []

let getcol ~i self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getcol"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i ))])

let getformat self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getformat"
     [||]
     []

let getmaxprint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getmaxprint"
     [||]
     []

                  let getnnz ?axis self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "getnnz"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
))])

let getrow ~i self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getrow"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i ))])

let log1p self =
   Py.Module.get_function_with_keywords (to_pyobject self) "log1p"
     [||]
     []

                  let max ?axis ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "max"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

let maximum ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "maximum"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

                  let mean ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "mean"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let min ?axis ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "min"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

let minimum ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "minimum"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let multiply ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "multiply"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let nonzero self =
   Py.Module.get_function_with_keywords (to_pyobject self) "nonzero"
     [||]
     []

let power ?dtype ~n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "power"
     [||]
     (Wrap_utils.keyword_args [("dtype", dtype); ("n", Some(n ))])

let prune self =
   Py.Module.get_function_with_keywords (to_pyobject self) "prune"
     [||]
     []

let rad2deg self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rad2deg"
     [||]
     []

let reshape ?kwargs args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "reshape"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t))
let resize shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "resize"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id shape)])
     []

let rint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rint"
     [||]
     []

let set_shape ~shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_shape"
     [||]
     (Wrap_utils.keyword_args [("shape", Some(shape ))])

let setdiag ?k ~values self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setdiag"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("values", Some(values |> Np.Obj.to_pyobject))])

let sign self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sign"
     [||]
     []

let sin self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sin"
     [||]
     []

let sinh self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sinh"
     [||]
     []

let sort_indices self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sort_indices"
     [||]
     []

let sorted_indices self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sorted_indices"
     [||]
     []

let sqrt self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sqrt"
     [||]
     []

                  let sum ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "sum"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let sum_duplicates self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sum_duplicates"
     [||]
     []

let tan self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tan"
     [||]
     []

let tanh self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tanh"
     [||]
     []

                  let toarray ?order ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "toarray"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let tobsr ?blocksize ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tobsr"
     [||]
     (Wrap_utils.keyword_args [("blocksize", blocksize); ("copy", copy)])

let tocoo ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocoo"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tocsc ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocsc"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tocsr ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocsr"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

                  let todense ?order ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "todense"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let todia ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "todia"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let todok ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "todok"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tolil ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tolil"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let transpose ?axes ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     [||]
     (Wrap_utils.keyword_args [("axes", axes); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let trunc self =
   Py.Module.get_function_with_keywords (to_pyobject self) "trunc"
     [||]
     []


let dtype_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dtype" with
  | None -> failwith "attribute dtype not found"
  | Some x -> if Py.is_none x then None else Some (Np.Dtype.of_pyobject x)

let dtype self = match dtype_opt self with
  | None -> raise Not_found
  | Some x -> x

let shape_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "shape" with
  | None -> failwith "attribute shape not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let shape self = match shape_opt self with
  | None -> raise Not_found
  | Some x -> x

let ndim_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "ndim" with
  | None -> failwith "attribute ndim not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let ndim self = match ndim_opt self with
  | None -> raise Not_found
  | Some x -> x

let nnz_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nnz" with
  | None -> failwith "attribute nnz not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let nnz self = match nnz_opt self with
  | None -> raise Not_found
  | Some x -> x

let data_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "data" with
  | None -> failwith "attribute data not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let data self = match data_opt self with
  | None -> raise Not_found
  | Some x -> x

let indices_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "indices" with
  | None -> failwith "attribute indices not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let indices self = match indices_opt self with
  | None -> raise Not_found
  | Some x -> x

let indptr_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "indptr" with
  | None -> failwith "attribute indptr not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let indptr self = match indptr_opt self with
  | None -> raise Not_found
  | Some x -> x

let has_sorted_indices_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "has_sorted_indices" with
  | None -> failwith "attribute has_sorted_indices not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let has_sorted_indices self = match has_sorted_indices_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Dia_matrix = struct
type tag = [`Dia_matrix]
type t = [`ArrayLike | `Dia_matrix | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?shape ?dtype ?copy ~arg1 () =
   Py.Module.get_function_with_keywords __wrap_namespace "dia_matrix"
     [||]
     (Wrap_utils.keyword_args [("shape", shape); ("dtype", dtype); ("copy", copy); ("arg1", Some(arg1 ))])
     |> of_pyobject
let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let arcsin self =
   Py.Module.get_function_with_keywords (to_pyobject self) "arcsin"
     [||]
     []

let arcsinh self =
   Py.Module.get_function_with_keywords (to_pyobject self) "arcsinh"
     [||]
     []

let arctan self =
   Py.Module.get_function_with_keywords (to_pyobject self) "arctan"
     [||]
     []

let arctanh self =
   Py.Module.get_function_with_keywords (to_pyobject self) "arctanh"
     [||]
     []

                  let asformat ?copy ~format self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "asformat"
                       [||]
                       (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("format", Some(format |> (function
| `S x -> Py.String.of_string x
| `None -> Py.none
)))])

let asfptype self =
   Py.Module.get_function_with_keywords (to_pyobject self) "asfptype"
     [||]
     []

                  let astype ?casting ?copy ~dtype self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "astype"
                       [||]
                       (Wrap_utils.keyword_args [("casting", Wrap_utils.Option.map casting (function
| `No -> Py.String.of_string "no"
| `Equiv -> Py.String.of_string "equiv"
| `Safe -> Py.String.of_string "safe"
| `Same_kind -> Py.String.of_string "same_kind"
| `Unsafe -> Py.String.of_string "unsafe"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("dtype", Some(dtype |> (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)))])

let ceil self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ceil"
     [||]
     []

let conj ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conj"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let conjugate ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conjugate"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "copy"
     [||]
     []

let count_nonzero self =
   Py.Module.get_function_with_keywords (to_pyobject self) "count_nonzero"
     [||]
     []

let deg2rad self =
   Py.Module.get_function_with_keywords (to_pyobject self) "deg2rad"
     [||]
     []

let diagonal ?k self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diagonal"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int)])

let dot ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let expm1 self =
   Py.Module.get_function_with_keywords (to_pyobject self) "expm1"
     [||]
     []

let floor self =
   Py.Module.get_function_with_keywords (to_pyobject self) "floor"
     [||]
     []

let getH self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getH"
     [||]
     []

let get_shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_shape"
     [||]
     []

let getcol ~j self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getcol"
     [||]
     (Wrap_utils.keyword_args [("j", Some(j ))])

let getformat self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getformat"
     [||]
     []

let getmaxprint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getmaxprint"
     [||]
     []

                  let getnnz ?axis self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "getnnz"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
))])

let getrow ~i self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getrow"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i ))])

let log1p self =
   Py.Module.get_function_with_keywords (to_pyobject self) "log1p"
     [||]
     []

let maximum ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "maximum"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

                  let mean ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "mean"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let minimum ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "minimum"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let multiply ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "multiply"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let nonzero self =
   Py.Module.get_function_with_keywords (to_pyobject self) "nonzero"
     [||]
     []

let power ?dtype ~n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "power"
     [||]
     (Wrap_utils.keyword_args [("dtype", dtype); ("n", Some(n ))])

let rad2deg self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rad2deg"
     [||]
     []

let reshape ?kwargs args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "reshape"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t))
let resize shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "resize"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id shape)])
     []

let rint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rint"
     [||]
     []

let set_shape ~shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_shape"
     [||]
     (Wrap_utils.keyword_args [("shape", Some(shape ))])

let setdiag ?k ~values self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setdiag"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("values", Some(values |> Np.Obj.to_pyobject))])

let sign self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sign"
     [||]
     []

let sin self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sin"
     [||]
     []

let sinh self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sinh"
     [||]
     []

let sqrt self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sqrt"
     [||]
     []

                  let sum ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "sum"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let tan self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tan"
     [||]
     []

let tanh self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tanh"
     [||]
     []

                  let toarray ?order ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "toarray"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let tobsr ?blocksize ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tobsr"
     [||]
     (Wrap_utils.keyword_args [("blocksize", blocksize); ("copy", copy)])

let tocoo ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocoo"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tocsc ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocsc"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tocsr ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocsr"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

                  let todense ?order ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "todense"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let todia ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "todia"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let todok ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "todok"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tolil ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tolil"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let transpose ?axes ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     [||]
     (Wrap_utils.keyword_args [("axes", axes); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let trunc self =
   Py.Module.get_function_with_keywords (to_pyobject self) "trunc"
     [||]
     []


let dtype_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dtype" with
  | None -> failwith "attribute dtype not found"
  | Some x -> if Py.is_none x then None else Some (Np.Dtype.of_pyobject x)

let dtype self = match dtype_opt self with
  | None -> raise Not_found
  | Some x -> x

let shape_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "shape" with
  | None -> failwith "attribute shape not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let shape self = match shape_opt self with
  | None -> raise Not_found
  | Some x -> x

let ndim_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "ndim" with
  | None -> failwith "attribute ndim not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let ndim self = match ndim_opt self with
  | None -> raise Not_found
  | Some x -> x

let nnz_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nnz" with
  | None -> failwith "attribute nnz not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let nnz self = match nnz_opt self with
  | None -> raise Not_found
  | Some x -> x

let data_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "data" with
  | None -> failwith "attribute data not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let data self = match data_opt self with
  | None -> raise Not_found
  | Some x -> x

let offsets_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "offsets" with
  | None -> failwith "attribute offsets not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let offsets self = match offsets_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Dok_matrix = struct
type tag = [`Dok_matrix]
type t = [`ArrayLike | `Dok_matrix | `IndexMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_index x = (x :> [`IndexMixin] Obj.t)
let create ?shape ?dtype ?copy ~arg1 () =
   Py.Module.get_function_with_keywords __wrap_namespace "dok_matrix"
     [||]
     (Wrap_utils.keyword_args [("shape", shape); ("dtype", dtype); ("copy", copy); ("arg1", Some(arg1 ))])
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key ))])

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let __setitem__ ~key ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key )); ("x", Some(x ))])

                  let asformat ?copy ~format self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "asformat"
                       [||]
                       (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("format", Some(format |> (function
| `S x -> Py.String.of_string x
| `None -> Py.none
)))])

let asfptype self =
   Py.Module.get_function_with_keywords (to_pyobject self) "asfptype"
     [||]
     []

                  let astype ?casting ?copy ~dtype self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "astype"
                       [||]
                       (Wrap_utils.keyword_args [("casting", Wrap_utils.Option.map casting (function
| `No -> Py.String.of_string "no"
| `Equiv -> Py.String.of_string "equiv"
| `Safe -> Py.String.of_string "safe"
| `Same_kind -> Py.String.of_string "same_kind"
| `Unsafe -> Py.String.of_string "unsafe"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("dtype", Some(dtype |> (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)))])

let clear self =
   Py.Module.get_function_with_keywords (to_pyobject self) "clear"
     [||]
     []

let conj ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conj"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let conjtransp self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conjtransp"
     [||]
     []

let conjugate ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conjugate"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "copy"
     [||]
     []

let count_nonzero self =
   Py.Module.get_function_with_keywords (to_pyobject self) "count_nonzero"
     [||]
     []

let diagonal ?k self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diagonal"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int)])

let dot ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let fromkeys ?value ~iterable self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fromkeys"
     (Array.of_list @@ List.concat [(match value with None -> [] | Some x -> [x ]);[iterable ]])
     []

let get ?default ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get"
     [||]
     (Wrap_utils.keyword_args [("default", default); ("key", Some(key ))])

let getH self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getH"
     [||]
     []

let get_shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_shape"
     [||]
     []

let getcol ~j self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getcol"
     [||]
     (Wrap_utils.keyword_args [("j", Some(j ))])

let getformat self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getformat"
     [||]
     []

let getmaxprint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getmaxprint"
     [||]
     []

                  let getnnz ?axis self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "getnnz"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
))])

let getrow ~i self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getrow"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i ))])

let items self =
   Py.Module.get_function_with_keywords (to_pyobject self) "items"
     [||]
     []

let keys self =
   Py.Module.get_function_with_keywords (to_pyobject self) "keys"
     [||]
     []

let maximum ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "maximum"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

                  let mean ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "mean"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let minimum ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "minimum"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let multiply ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "multiply"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let nonzero self =
   Py.Module.get_function_with_keywords (to_pyobject self) "nonzero"
     [||]
     []

let pop ?d ~k self =
   Py.Module.get_function_with_keywords (to_pyobject self) "pop"
     [||]
     (Wrap_utils.keyword_args [("d", d); ("k", Some(k ))])

let popitem self =
   Py.Module.get_function_with_keywords (to_pyobject self) "popitem"
     [||]
     []

let power ?dtype ~n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "power"
     [||]
     (Wrap_utils.keyword_args [("dtype", dtype); ("n", Some(n ))])

let reshape ?kwargs args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "reshape"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t))
let resize shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "resize"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id shape)])
     []

let set_shape ~shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_shape"
     [||]
     (Wrap_utils.keyword_args [("shape", Some(shape ))])

let setdefault ?default ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setdefault"
     (Array.of_list @@ List.concat [(match default with None -> [] | Some x -> [x ]);[key ]])
     []

let setdiag ?k ~values self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setdiag"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("values", Some(values |> Np.Obj.to_pyobject))])

                  let sum ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "sum"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let toarray ?order ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "toarray"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let tobsr ?blocksize ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tobsr"
     [||]
     (Wrap_utils.keyword_args [("blocksize", blocksize); ("copy", copy)])

let tocoo ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocoo"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tocsc ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocsc"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tocsr ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocsr"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

                  let todense ?order ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "todense"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let todia ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "todia"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let todok ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "todok"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tolil ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tolil"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let transpose ?axes ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     [||]
     (Wrap_utils.keyword_args [("axes", axes); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let update ~val_ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "update"
     [||]
     (Wrap_utils.keyword_args [("val", Some(val_ ))])

let values self =
   Py.Module.get_function_with_keywords (to_pyobject self) "values"
     [||]
     []


let dtype_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dtype" with
  | None -> failwith "attribute dtype not found"
  | Some x -> if Py.is_none x then None else Some (Np.Dtype.of_pyobject x)

let dtype self = match dtype_opt self with
  | None -> raise Not_found
  | Some x -> x

let shape_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "shape" with
  | None -> failwith "attribute shape not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let shape self = match shape_opt self with
  | None -> raise Not_found
  | Some x -> x

let ndim_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "ndim" with
  | None -> failwith "attribute ndim not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let ndim self = match ndim_opt self with
  | None -> raise Not_found
  | Some x -> x

let nnz_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nnz" with
  | None -> failwith "attribute nnz not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let nnz self = match nnz_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Lil_matrix = struct
type tag = [`Lil_matrix]
type t = [`ArrayLike | `IndexMixin | `Lil_matrix | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_index x = (x :> [`IndexMixin] Obj.t)
let create ?shape ?dtype ?copy ~arg1 () =
   Py.Module.get_function_with_keywords __wrap_namespace "lil_matrix"
     [||]
     (Wrap_utils.keyword_args [("shape", shape); ("dtype", dtype); ("copy", copy); ("arg1", Some(arg1 ))])
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key ))])

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let __setitem__ ~key ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key )); ("x", Some(x ))])

                  let asformat ?copy ~format self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "asformat"
                       [||]
                       (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("format", Some(format |> (function
| `S x -> Py.String.of_string x
| `None -> Py.none
)))])

let asfptype self =
   Py.Module.get_function_with_keywords (to_pyobject self) "asfptype"
     [||]
     []

                  let astype ?casting ?copy ~dtype self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "astype"
                       [||]
                       (Wrap_utils.keyword_args [("casting", Wrap_utils.Option.map casting (function
| `No -> Py.String.of_string "no"
| `Equiv -> Py.String.of_string "equiv"
| `Safe -> Py.String.of_string "safe"
| `Same_kind -> Py.String.of_string "same_kind"
| `Unsafe -> Py.String.of_string "unsafe"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("dtype", Some(dtype |> (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)))])

let conj ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conj"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let conjugate ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conjugate"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "copy"
     [||]
     []

let count_nonzero self =
   Py.Module.get_function_with_keywords (to_pyobject self) "count_nonzero"
     [||]
     []

let diagonal ?k self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diagonal"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int)])

let dot ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let getH self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getH"
     [||]
     []

let get_shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_shape"
     [||]
     []

let getcol ~j self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getcol"
     [||]
     (Wrap_utils.keyword_args [("j", Some(j ))])

let getformat self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getformat"
     [||]
     []

let getmaxprint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getmaxprint"
     [||]
     []

                  let getnnz ?axis self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "getnnz"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
))])

let getrow ~i self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getrow"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i ))])

let getrowview ~i self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getrowview"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i ))])

let maximum ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "maximum"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

                  let mean ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "mean"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let minimum ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "minimum"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let multiply ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "multiply"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let nonzero self =
   Py.Module.get_function_with_keywords (to_pyobject self) "nonzero"
     [||]
     []

let power ?dtype ~n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "power"
     [||]
     (Wrap_utils.keyword_args [("dtype", dtype); ("n", Some(n ))])

let reshape ?kwargs args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "reshape"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t))
let resize shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "resize"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id shape)])
     []

let set_shape ~shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_shape"
     [||]
     (Wrap_utils.keyword_args [("shape", Some(shape ))])

let setdiag ?k ~values self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setdiag"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("values", Some(values |> Np.Obj.to_pyobject))])

                  let sum ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "sum"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let toarray ?order ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "toarray"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let tobsr ?blocksize ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tobsr"
     [||]
     (Wrap_utils.keyword_args [("blocksize", blocksize); ("copy", copy)])

let tocoo ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocoo"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tocsc ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocsc"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tocsr ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocsr"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

                  let todense ?order ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "todense"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let todia ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "todia"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let todok ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "todok"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tolil ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tolil"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let transpose ?axes ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     [||]
     (Wrap_utils.keyword_args [("axes", axes); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])


let dtype_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dtype" with
  | None -> failwith "attribute dtype not found"
  | Some x -> if Py.is_none x then None else Some (Np.Dtype.of_pyobject x)

let dtype self = match dtype_opt self with
  | None -> raise Not_found
  | Some x -> x

let shape_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "shape" with
  | None -> failwith "attribute shape not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let shape self = match shape_opt self with
  | None -> raise Not_found
  | Some x -> x

let ndim_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "ndim" with
  | None -> failwith "attribute ndim not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let ndim self = match ndim_opt self with
  | None -> raise Not_found
  | Some x -> x

let nnz_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nnz" with
  | None -> failwith "attribute nnz not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let nnz self = match nnz_opt self with
  | None -> raise Not_found
  | Some x -> x

let data_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "data" with
  | None -> failwith "attribute data not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let data self = match data_opt self with
  | None -> raise Not_found
  | Some x -> x

let rows_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "rows" with
  | None -> failwith "attribute rows not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let rows self = match rows_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Spmatrix = struct
type tag = [`Spmatrix]
type t = [`ArrayLike | `Object | `Spmatrix] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?maxprint () =
   Py.Module.get_function_with_keywords __wrap_namespace "spmatrix"
     [||]
     (Wrap_utils.keyword_args [("maxprint", maxprint)])
     |> of_pyobject
let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

                  let asformat ?copy ~format self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "asformat"
                       [||]
                       (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("format", Some(format |> (function
| `S x -> Py.String.of_string x
| `None -> Py.none
)))])

let asfptype self =
   Py.Module.get_function_with_keywords (to_pyobject self) "asfptype"
     [||]
     []

                  let astype ?casting ?copy ~dtype self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "astype"
                       [||]
                       (Wrap_utils.keyword_args [("casting", Wrap_utils.Option.map casting (function
| `No -> Py.String.of_string "no"
| `Equiv -> Py.String.of_string "equiv"
| `Safe -> Py.String.of_string "safe"
| `Same_kind -> Py.String.of_string "same_kind"
| `Unsafe -> Py.String.of_string "unsafe"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("dtype", Some(dtype |> (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)))])

let conj ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conj"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let conjugate ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conjugate"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "copy"
     [||]
     []

let count_nonzero self =
   Py.Module.get_function_with_keywords (to_pyobject self) "count_nonzero"
     [||]
     []

let diagonal ?k self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diagonal"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int)])

let dot ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let getH self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getH"
     [||]
     []

let get_shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_shape"
     [||]
     []

let getcol ~j self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getcol"
     [||]
     (Wrap_utils.keyword_args [("j", Some(j ))])

let getformat self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getformat"
     [||]
     []

let getmaxprint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getmaxprint"
     [||]
     []

                  let getnnz ?axis self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "getnnz"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
))])

let getrow ~i self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getrow"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i ))])

let maximum ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "maximum"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

                  let mean ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "mean"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let minimum ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "minimum"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let multiply ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "multiply"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let nonzero self =
   Py.Module.get_function_with_keywords (to_pyobject self) "nonzero"
     [||]
     []

let power ?dtype ~n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "power"
     [||]
     (Wrap_utils.keyword_args [("dtype", dtype); ("n", Some(n ))])

let reshape ?kwargs args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "reshape"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t))
let resize ~shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "resize"
     [||]
     (Wrap_utils.keyword_args [("shape", Some(shape |> (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.Int.of_int ml_0); (Py.Int.of_int ml_1)])))])

let set_shape ~shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_shape"
     [||]
     (Wrap_utils.keyword_args [("shape", Some(shape ))])

let setdiag ?k ~values self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setdiag"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("values", Some(values |> Np.Obj.to_pyobject))])

                  let sum ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "sum"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let toarray ?order ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "toarray"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let tobsr ?blocksize ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tobsr"
     [||]
     (Wrap_utils.keyword_args [("blocksize", blocksize); ("copy", copy)])

let tocoo ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocoo"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tocsc ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocsc"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tocsr ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocsr"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

                  let todense ?order ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "todense"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let todia ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "todia"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let todok ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "todok"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tolil ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tolil"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let transpose ?axes ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     [||]
     (Wrap_utils.keyword_args [("axes", axes); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Base = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.base"

let get_py name = Py.Module.get __wrap_namespace name
module SparseFormatWarning = struct
type tag = [`SparseFormatWarning]
type t = [`BaseException | `Object | `SparseFormatWarning] Obj.t
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
let asmatrix ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "asmatrix"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

let broadcast_to ?subok ~array ~shape () =
   Py.Module.get_function_with_keywords __wrap_namespace "broadcast_to"
     [||]
     (Wrap_utils.keyword_args [("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("array", Some(array |> Np.Obj.to_pyobject)); ("shape", Some(shape ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let check_reshape_kwargs kwargs =
   Py.Module.get_function_with_keywords __wrap_namespace "check_reshape_kwargs"
     [||]
     (Wrap_utils.keyword_args [("kwargs", Some(kwargs ))])

let check_shape ?current_shape ~args () =
   Py.Module.get_function_with_keywords __wrap_namespace "check_shape"
     [||]
     (Wrap_utils.keyword_args [("current_shape", current_shape); ("args", Some(args ))])

let get_sum_dtype dtype =
   Py.Module.get_function_with_keywords __wrap_namespace "get_sum_dtype"
     [||]
     (Wrap_utils.keyword_args [("dtype", Some(dtype ))])

let isdense x =
   Py.Module.get_function_with_keywords __wrap_namespace "isdense"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isintlike x =
   Py.Module.get_function_with_keywords __wrap_namespace "isintlike"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isscalarlike x =
   Py.Module.get_function_with_keywords __wrap_namespace "isscalarlike"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let issparse x =
   Py.Module.get_function_with_keywords __wrap_namespace "issparse"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isspmatrix x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let validateaxis axis =
   Py.Module.get_function_with_keywords __wrap_namespace "validateaxis"
     [||]
     (Wrap_utils.keyword_args [("axis", Some(axis ))])


end
module Bsr = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.bsr"

let get_py name = Py.Module.get __wrap_namespace name
let check_shape ?current_shape ~args () =
   Py.Module.get_function_with_keywords __wrap_namespace "check_shape"
     [||]
     (Wrap_utils.keyword_args [("current_shape", current_shape); ("args", Some(args ))])

let get_index_dtype ?arrays ?maxval ?check_contents () =
   Py.Module.get_function_with_keywords __wrap_namespace "get_index_dtype"
     [||]
     (Wrap_utils.keyword_args [("arrays", arrays); ("maxval", Wrap_utils.Option.map maxval Py.Float.of_float); ("check_contents", Wrap_utils.Option.map check_contents Py.Bool.of_bool)])
     |> Np.Dtype.of_pyobject
let getdtype ?a ?default ~dtype () =
   Py.Module.get_function_with_keywords __wrap_namespace "getdtype"
     [||]
     (Wrap_utils.keyword_args [("a", a); ("default", default); ("dtype", Some(dtype ))])

let isshape ?nonneg ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "isshape"
     [||]
     (Wrap_utils.keyword_args [("nonneg", nonneg); ("x", Some(x ))])

let isspmatrix x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isspmatrix_bsr x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix_bsr"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let to_native a =
   Py.Module.get_function_with_keywords __wrap_namespace "to_native"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a ))])

let upcast args =
   Py.Module.get_function_with_keywords __wrap_namespace "upcast"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

let warn ?category ?stacklevel ?source ~message () =
   Py.Module.get_function_with_keywords __wrap_namespace "warn"
     [||]
     (Wrap_utils.keyword_args [("category", category); ("stacklevel", stacklevel); ("source", source); ("message", Some(message ))])


end
module Compressed = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.compressed"

let get_py name = Py.Module.get __wrap_namespace name
module IndexMixin = struct
type tag = [`IndexMixin]
type t = [`IndexMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "IndexMixin"
     [||]
     []
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key ))])

let __setitem__ ~key ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key )); ("x", Some(x ))])

let getcol ~i self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getcol"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i ))])

let getrow ~i self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getrow"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let asmatrix ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "asmatrix"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

let check_shape ?current_shape ~args () =
   Py.Module.get_function_with_keywords __wrap_namespace "check_shape"
     [||]
     (Wrap_utils.keyword_args [("current_shape", current_shape); ("args", Some(args ))])

let downcast_intp_index arr =
   Py.Module.get_function_with_keywords __wrap_namespace "downcast_intp_index"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr ))])

let get_index_dtype ?arrays ?maxval ?check_contents () =
   Py.Module.get_function_with_keywords __wrap_namespace "get_index_dtype"
     [||]
     (Wrap_utils.keyword_args [("arrays", arrays); ("maxval", Wrap_utils.Option.map maxval Py.Float.of_float); ("check_contents", Wrap_utils.Option.map check_contents Py.Bool.of_bool)])
     |> Np.Dtype.of_pyobject
let get_sum_dtype dtype =
   Py.Module.get_function_with_keywords __wrap_namespace "get_sum_dtype"
     [||]
     (Wrap_utils.keyword_args [("dtype", Some(dtype ))])

let getdtype ?a ?default ~dtype () =
   Py.Module.get_function_with_keywords __wrap_namespace "getdtype"
     [||]
     (Wrap_utils.keyword_args [("a", a); ("default", default); ("dtype", Some(dtype ))])

let isdense x =
   Py.Module.get_function_with_keywords __wrap_namespace "isdense"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isintlike x =
   Py.Module.get_function_with_keywords __wrap_namespace "isintlike"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isscalarlike x =
   Py.Module.get_function_with_keywords __wrap_namespace "isscalarlike"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isshape ?nonneg ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "isshape"
     [||]
     (Wrap_utils.keyword_args [("nonneg", nonneg); ("x", Some(x ))])

let isspmatrix x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let matrix ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "matrix"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

let to_native a =
   Py.Module.get_function_with_keywords __wrap_namespace "to_native"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a ))])

let upcast args =
   Py.Module.get_function_with_keywords __wrap_namespace "upcast"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

let upcast_char args =
   Py.Module.get_function_with_keywords __wrap_namespace "upcast_char"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

let warn ?category ?stacklevel ?source ~message () =
   Py.Module.get_function_with_keywords __wrap_namespace "warn"
     [||]
     (Wrap_utils.keyword_args [("category", category); ("stacklevel", stacklevel); ("source", source); ("message", Some(message ))])


end
module Construct = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.construct"

let get_py name = Py.Module.get __wrap_namespace name
let block_diag ?format ?dtype ~mats () =
   Py.Module.get_function_with_keywords __wrap_namespace "block_diag"
     [||]
     (Wrap_utils.keyword_args [("format", Wrap_utils.Option.map format Py.String.of_string); ("dtype", dtype); ("mats", Some(mats ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t))
                  let bmat ?format ?dtype ~blocks () =
                     Py.Module.get_function_with_keywords __wrap_namespace "bmat"
                       [||]
                       (Wrap_utils.keyword_args [("format", Wrap_utils.Option.map format (function
| `Lil -> Py.String.of_string "lil"
| `Bsr -> Py.String.of_string "bsr"
| `Csr -> Py.String.of_string "csr"
| `Csc -> Py.String.of_string "csc"
| `Coo -> Py.String.of_string "coo"
| `Dia -> Py.String.of_string "dia"
| `Dok -> Py.String.of_string "dok"
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("blocks", Some(blocks |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t))
                  let diags ?offsets ?shape ?format ?dtype ~diagonals () =
                     Py.Module.get_function_with_keywords __wrap_namespace "diags"
                       [||]
                       (Wrap_utils.keyword_args [("offsets", offsets); ("shape", shape); ("format", Wrap_utils.Option.map format (function
| `Lil -> Py.String.of_string "lil"
| `Csr -> Py.String.of_string "csr"
| `Csc -> Py.String.of_string "csc"
| `Dia -> Py.String.of_string "dia"
| `T x -> Wrap_utils.id x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("diagonals", Some(diagonals ))])

let eye ?n ?k ?dtype ?format ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "eye"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("k", Wrap_utils.Option.map k Py.Int.of_int); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("format", Wrap_utils.Option.map format Py.String.of_string); ("m", Some(m |> Py.Int.of_int))])

let get_index_dtype ?arrays ?maxval ?check_contents () =
   Py.Module.get_function_with_keywords __wrap_namespace "get_index_dtype"
     [||]
     (Wrap_utils.keyword_args [("arrays", arrays); ("maxval", Wrap_utils.Option.map maxval Py.Float.of_float); ("check_contents", Wrap_utils.Option.map check_contents Py.Bool.of_bool)])
     |> Np.Dtype.of_pyobject
let get_randint random_state =
   Py.Module.get_function_with_keywords __wrap_namespace "get_randint"
     [||]
     (Wrap_utils.keyword_args [("random_state", Some(random_state ))])

let hstack ?format ?dtype ~blocks () =
   Py.Module.get_function_with_keywords __wrap_namespace "hstack"
     [||]
     (Wrap_utils.keyword_args [("format", Wrap_utils.Option.map format Py.String.of_string); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("blocks", Some(blocks ))])

let identity ?dtype ?format ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "identity"
     [||]
     (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("format", Wrap_utils.Option.map format Py.String.of_string); ("n", Some(n |> Py.Int.of_int))])

let isscalarlike x =
   Py.Module.get_function_with_keywords __wrap_namespace "isscalarlike"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let issparse x =
   Py.Module.get_function_with_keywords __wrap_namespace "issparse"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let kron ?format ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "kron"
     [||]
     (Wrap_utils.keyword_args [("format", Wrap_utils.Option.map format Py.String.of_string); ("A", Some(a )); ("B", Some(b ))])

let kronsum ?format ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "kronsum"
     [||]
     (Wrap_utils.keyword_args [("format", Wrap_utils.Option.map format Py.String.of_string); ("A", Some(a )); ("B", Some(b ))])

                  let rand ?density ?format ?dtype ?random_state ~m ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "rand"
                       [||]
                       (Wrap_utils.keyword_args [("density", density); ("format", Wrap_utils.Option.map format Py.String.of_string); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("random_state", Wrap_utils.Option.map random_state (function
| `I x -> Py.Int.of_int x
| `Numpy_random_RandomState x -> Wrap_utils.id x
)); ("m", Some(m )); ("n", Some(n ))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t))
                  let random ?density ?format ?dtype ?random_state ?data_rvs ~m ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "random"
                       [||]
                       (Wrap_utils.keyword_args [("density", density); ("format", Wrap_utils.Option.map format Py.String.of_string); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("random_state", Wrap_utils.Option.map random_state (function
| `I x -> Py.Int.of_int x
| `Numpy_random_RandomState x -> Wrap_utils.id x
)); ("data_rvs", data_rvs); ("m", Some(m )); ("n", Some(n ))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t))
let spdiags ?format ~data ~diags ~m ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "spdiags"
     [||]
     (Wrap_utils.keyword_args [("format", Wrap_utils.Option.map format Py.String.of_string); ("data", Some(data |> Np.Obj.to_pyobject)); ("diags", Some(diags )); ("m", Some(m )); ("n", Some(n ))])

let upcast args =
   Py.Module.get_function_with_keywords __wrap_namespace "upcast"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

let vstack ?format ?dtype ~blocks () =
   Py.Module.get_function_with_keywords __wrap_namespace "vstack"
     [||]
     (Wrap_utils.keyword_args [("format", Wrap_utils.Option.map format Py.String.of_string); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("blocks", Some(blocks ))])


end
module Coo = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.coo"

let get_py name = Py.Module.get __wrap_namespace name
module Izip = struct
type tag = [`Zip]
type t = [`Object | `Zip] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create iterables =
   Py.Module.get_function_with_keywords __wrap_namespace "izip"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id iterables)])
     []
     |> of_pyobject
let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let check_reshape_kwargs kwargs =
   Py.Module.get_function_with_keywords __wrap_namespace "check_reshape_kwargs"
     [||]
     (Wrap_utils.keyword_args [("kwargs", Some(kwargs ))])

let check_shape ?current_shape ~args () =
   Py.Module.get_function_with_keywords __wrap_namespace "check_shape"
     [||]
     (Wrap_utils.keyword_args [("current_shape", current_shape); ("args", Some(args ))])

let downcast_intp_index arr =
   Py.Module.get_function_with_keywords __wrap_namespace "downcast_intp_index"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr ))])

let get_index_dtype ?arrays ?maxval ?check_contents () =
   Py.Module.get_function_with_keywords __wrap_namespace "get_index_dtype"
     [||]
     (Wrap_utils.keyword_args [("arrays", arrays); ("maxval", Wrap_utils.Option.map maxval Py.Float.of_float); ("check_contents", Wrap_utils.Option.map check_contents Py.Bool.of_bool)])
     |> Np.Dtype.of_pyobject
let getdtype ?a ?default ~dtype () =
   Py.Module.get_function_with_keywords __wrap_namespace "getdtype"
     [||]
     (Wrap_utils.keyword_args [("a", a); ("default", default); ("dtype", Some(dtype ))])

let isshape ?nonneg ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "isshape"
     [||]
     (Wrap_utils.keyword_args [("nonneg", nonneg); ("x", Some(x ))])

let isspmatrix x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isspmatrix_coo x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix_coo"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let matrix ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "matrix"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

let to_native a =
   Py.Module.get_function_with_keywords __wrap_namespace "to_native"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a ))])

let upcast args =
   Py.Module.get_function_with_keywords __wrap_namespace "upcast"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

let upcast_char args =
   Py.Module.get_function_with_keywords __wrap_namespace "upcast_char"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

let warn ?category ?stacklevel ?source ~message () =
   Py.Module.get_function_with_keywords __wrap_namespace "warn"
     [||]
     (Wrap_utils.keyword_args [("category", category); ("stacklevel", stacklevel); ("source", source); ("message", Some(message ))])


end
module Csc = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.csc"

let get_py name = Py.Module.get __wrap_namespace name
let get_index_dtype ?arrays ?maxval ?check_contents () =
   Py.Module.get_function_with_keywords __wrap_namespace "get_index_dtype"
     [||]
     (Wrap_utils.keyword_args [("arrays", arrays); ("maxval", Wrap_utils.Option.map maxval Py.Float.of_float); ("check_contents", Wrap_utils.Option.map check_contents Py.Bool.of_bool)])
     |> Np.Dtype.of_pyobject
let isspmatrix_csc x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix_csc"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let upcast args =
   Py.Module.get_function_with_keywords __wrap_namespace "upcast"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []


end
module Csgraph = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.csgraph"

let get_py name = Py.Module.get __wrap_namespace name
module NegativeCycleError = struct
type tag = [`NegativeCycleError]
type t = [`BaseException | `NegativeCycleError | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_exception x = (x :> [`BaseException] Obj.t)
let create ?message () =
   Py.Module.get_function_with_keywords __wrap_namespace "NegativeCycleError"
     [||]
     (Wrap_utils.keyword_args [("message", message)])
     |> of_pyobject
let with_traceback ~tb self =
   Py.Module.get_function_with_keywords (to_pyobject self) "with_traceback"
     [||]
     (Wrap_utils.keyword_args [("tb", Some(tb ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let bellman_ford ?directed ?indices ?return_predecessors ?unweighted ~csgraph () =
                     Py.Module.get_function_with_keywords __wrap_namespace "bellman_ford"
                       [||]
                       (Wrap_utils.keyword_args [("directed", Wrap_utils.Option.map directed Py.Bool.of_bool); ("indices", Wrap_utils.Option.map indices (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("return_predecessors", Wrap_utils.Option.map return_predecessors Py.Bool.of_bool); ("unweighted", Wrap_utils.Option.map unweighted Py.Bool.of_bool); ("csgraph", Some(csgraph ))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let breadth_first_order ?directed ?return_predecessors ~csgraph ~i_start () =
   Py.Module.get_function_with_keywords __wrap_namespace "breadth_first_order"
     [||]
     (Wrap_utils.keyword_args [("directed", Wrap_utils.Option.map directed Py.Bool.of_bool); ("return_predecessors", Wrap_utils.Option.map return_predecessors Py.Bool.of_bool); ("csgraph", Some(csgraph |> Np.Obj.to_pyobject)); ("i_start", Some(i_start |> Py.Int.of_int))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let breadth_first_tree ?directed ~csgraph ~i_start () =
   Py.Module.get_function_with_keywords __wrap_namespace "breadth_first_tree"
     [||]
     (Wrap_utils.keyword_args [("directed", Wrap_utils.Option.map directed Py.Bool.of_bool); ("csgraph", Some(csgraph |> Np.Obj.to_pyobject)); ("i_start", Some(i_start |> Py.Int.of_int))])

let connected_components ?directed ?connection ?return_labels ~csgraph () =
   Py.Module.get_function_with_keywords __wrap_namespace "connected_components"
     [||]
     (Wrap_utils.keyword_args [("directed", Wrap_utils.Option.map directed Py.Bool.of_bool); ("connection", Wrap_utils.Option.map connection Py.String.of_string); ("return_labels", Wrap_utils.Option.map return_labels Py.Bool.of_bool); ("csgraph", Some(csgraph |> Np.Obj.to_pyobject))])
     |> (fun x -> ((Py.Int.to_int (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let csgraph_from_dense ?null_value ?nan_null ?infinity_null ~graph () =
                     Py.Module.get_function_with_keywords __wrap_namespace "csgraph_from_dense"
                       [||]
                       (Wrap_utils.keyword_args [("null_value", Wrap_utils.Option.map null_value (function
| `F x -> Py.Float.of_float x
| `None -> Py.none
)); ("nan_null", Wrap_utils.Option.map nan_null Py.Bool.of_bool); ("infinity_null", Wrap_utils.Option.map infinity_null Py.Bool.of_bool); ("graph", Some(graph |> Np.Obj.to_pyobject))])

let csgraph_from_masked graph =
   Py.Module.get_function_with_keywords __wrap_namespace "csgraph_from_masked"
     [||]
     (Wrap_utils.keyword_args [("graph", Some(graph ))])

                  let csgraph_masked_from_dense ?null_value ?nan_null ?infinity_null ?copy ~graph () =
                     Py.Module.get_function_with_keywords __wrap_namespace "csgraph_masked_from_dense"
                       [||]
                       (Wrap_utils.keyword_args [("null_value", Wrap_utils.Option.map null_value (function
| `F x -> Py.Float.of_float x
| `None -> Py.none
)); ("nan_null", Wrap_utils.Option.map nan_null Py.Bool.of_bool); ("infinity_null", Wrap_utils.Option.map infinity_null Py.Bool.of_bool); ("copy", copy); ("graph", Some(graph |> Np.Obj.to_pyobject))])

let csgraph_to_dense ?null_value ~csgraph () =
   Py.Module.get_function_with_keywords __wrap_namespace "csgraph_to_dense"
     [||]
     (Wrap_utils.keyword_args [("null_value", Wrap_utils.Option.map null_value Py.Float.of_float); ("csgraph", Some(csgraph ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let csgraph_to_masked csgraph =
   Py.Module.get_function_with_keywords __wrap_namespace "csgraph_to_masked"
     [||]
     (Wrap_utils.keyword_args [("csgraph", Some(csgraph ))])

let depth_first_order ?directed ?return_predecessors ~csgraph ~i_start () =
   Py.Module.get_function_with_keywords __wrap_namespace "depth_first_order"
     [||]
     (Wrap_utils.keyword_args [("directed", Wrap_utils.Option.map directed Py.Bool.of_bool); ("return_predecessors", Wrap_utils.Option.map return_predecessors Py.Bool.of_bool); ("csgraph", Some(csgraph |> Np.Obj.to_pyobject)); ("i_start", Some(i_start |> Py.Int.of_int))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let depth_first_tree ?directed ~csgraph ~i_start () =
   Py.Module.get_function_with_keywords __wrap_namespace "depth_first_tree"
     [||]
     (Wrap_utils.keyword_args [("directed", Wrap_utils.Option.map directed Py.Bool.of_bool); ("csgraph", Some(csgraph |> Np.Obj.to_pyobject)); ("i_start", Some(i_start |> Py.Int.of_int))])

let floyd_warshall ?directed ?return_predecessors ?unweighted ?overwrite ~csgraph () =
   Py.Module.get_function_with_keywords __wrap_namespace "floyd_warshall"
     [||]
     (Wrap_utils.keyword_args [("directed", Wrap_utils.Option.map directed Py.Bool.of_bool); ("return_predecessors", Wrap_utils.Option.map return_predecessors Py.Bool.of_bool); ("unweighted", Wrap_utils.Option.map unweighted Py.Bool.of_bool); ("overwrite", Wrap_utils.Option.map overwrite Py.Bool.of_bool); ("csgraph", Some(csgraph ))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let johnson ?directed ?indices ?return_predecessors ?unweighted ~csgraph () =
                     Py.Module.get_function_with_keywords __wrap_namespace "johnson"
                       [||]
                       (Wrap_utils.keyword_args [("directed", Wrap_utils.Option.map directed Py.Bool.of_bool); ("indices", Wrap_utils.Option.map indices (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("return_predecessors", Wrap_utils.Option.map return_predecessors Py.Bool.of_bool); ("unweighted", Wrap_utils.Option.map unweighted Py.Bool.of_bool); ("csgraph", Some(csgraph ))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let laplacian ?normed ?return_diag ?use_out_degree ~csgraph () =
   Py.Module.get_function_with_keywords __wrap_namespace "laplacian"
     [||]
     (Wrap_utils.keyword_args [("normed", Wrap_utils.Option.map normed Py.Bool.of_bool); ("return_diag", Wrap_utils.Option.map return_diag Py.Bool.of_bool); ("use_out_degree", Wrap_utils.Option.map use_out_degree Py.Bool.of_bool); ("csgraph", Some(csgraph ))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let maximum_bipartite_matching ?perm_type ~graph () =
                     Py.Module.get_function_with_keywords __wrap_namespace "maximum_bipartite_matching"
                       [||]
                       (Wrap_utils.keyword_args [("perm_type", Wrap_utils.Option.map perm_type (function
| `Row -> Py.String.of_string "row"
| `Column -> Py.String.of_string "column"
)); ("graph", Some(graph |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let maximum_flow ~csgraph ~source ~sink () =
   Py.Module.get_function_with_keywords __wrap_namespace "maximum_flow"
     [||]
     (Wrap_utils.keyword_args [("csgraph", Some(csgraph )); ("source", Some(source |> Py.Int.of_int)); ("sink", Some(sink |> Py.Int.of_int))])

let minimum_spanning_tree ?overwrite ~csgraph () =
   Py.Module.get_function_with_keywords __wrap_namespace "minimum_spanning_tree"
     [||]
     (Wrap_utils.keyword_args [("overwrite", Wrap_utils.Option.map overwrite Py.Bool.of_bool); ("csgraph", Some(csgraph ))])

                  let reconstruct_path ?directed ~csgraph ~predecessors () =
                     Py.Module.get_function_with_keywords __wrap_namespace "reconstruct_path"
                       [||]
                       (Wrap_utils.keyword_args [("directed", Wrap_utils.Option.map directed Py.Bool.of_bool); ("csgraph", Some(csgraph |> Np.Obj.to_pyobject)); ("predecessors", Some(predecessors |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `One_dimension x -> Wrap_utils.id x
)))])

let reverse_cuthill_mckee ?symmetric_mode ~graph () =
   Py.Module.get_function_with_keywords __wrap_namespace "reverse_cuthill_mckee"
     [||]
     (Wrap_utils.keyword_args [("symmetric_mode", Wrap_utils.Option.map symmetric_mode Py.Bool.of_bool); ("graph", Some(graph |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let shortest_path ?method_ ?directed ?return_predecessors ?unweighted ?overwrite ?indices ~csgraph () =
                     Py.Module.get_function_with_keywords __wrap_namespace "shortest_path"
                       [||]
                       (Wrap_utils.keyword_args [("method", Wrap_utils.Option.map method_ (function
| `Auto -> Py.String.of_string "auto"
| `FW -> Py.String.of_string "FW"
| `D -> Py.String.of_string "D"
)); ("directed", Wrap_utils.Option.map directed Py.Bool.of_bool); ("return_predecessors", Wrap_utils.Option.map return_predecessors Py.Bool.of_bool); ("unweighted", Wrap_utils.Option.map unweighted Py.Bool.of_bool); ("overwrite", Wrap_utils.Option.map overwrite Py.Bool.of_bool); ("indices", Wrap_utils.Option.map indices (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("csgraph", Some(csgraph ))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let structural_rank graph =
   Py.Module.get_function_with_keywords __wrap_namespace "structural_rank"
     [||]
     (Wrap_utils.keyword_args [("graph", Some(graph |> Np.Obj.to_pyobject))])
     |> Py.Int.to_int

end
module Csr = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.csr"

let get_py name = Py.Module.get __wrap_namespace name
module Xrange = struct
type tag = [`Range]
type t = [`Object | `Range] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create stop =
   Py.Module.get_function_with_keywords __wrap_namespace "xrange"
     [||]
     (Wrap_utils.keyword_args [("stop", Some(stop ))])
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let count ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "count"
     [||]
     (Wrap_utils.keyword_args [("value", Some(value ))])

let index ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "index"
     [||]
     (Wrap_utils.keyword_args [("value", Some(value ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let get_index_dtype ?arrays ?maxval ?check_contents () =
   Py.Module.get_function_with_keywords __wrap_namespace "get_index_dtype"
     [||]
     (Wrap_utils.keyword_args [("arrays", arrays); ("maxval", Wrap_utils.Option.map maxval Py.Float.of_float); ("check_contents", Wrap_utils.Option.map check_contents Py.Bool.of_bool)])
     |> Np.Dtype.of_pyobject
let isspmatrix_csr x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix_csr"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let upcast args =
   Py.Module.get_function_with_keywords __wrap_namespace "upcast"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []


end
module Data = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.data"

let get_py name = Py.Module.get __wrap_namespace name
let isscalarlike x =
   Py.Module.get_function_with_keywords __wrap_namespace "isscalarlike"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let matrix ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "matrix"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

                  let npfunc ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "npfunc"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let validateaxis axis =
   Py.Module.get_function_with_keywords __wrap_namespace "validateaxis"
     [||]
     (Wrap_utils.keyword_args [("axis", Some(axis ))])


end
module Dia = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.dia"

let get_py name = Py.Module.get __wrap_namespace name
let check_shape ?current_shape ~args () =
   Py.Module.get_function_with_keywords __wrap_namespace "check_shape"
     [||]
     (Wrap_utils.keyword_args [("current_shape", current_shape); ("args", Some(args ))])

let get_index_dtype ?arrays ?maxval ?check_contents () =
   Py.Module.get_function_with_keywords __wrap_namespace "get_index_dtype"
     [||]
     (Wrap_utils.keyword_args [("arrays", arrays); ("maxval", Wrap_utils.Option.map maxval Py.Float.of_float); ("check_contents", Wrap_utils.Option.map check_contents Py.Bool.of_bool)])
     |> Np.Dtype.of_pyobject
let get_sum_dtype dtype =
   Py.Module.get_function_with_keywords __wrap_namespace "get_sum_dtype"
     [||]
     (Wrap_utils.keyword_args [("dtype", Some(dtype ))])

let getdtype ?a ?default ~dtype () =
   Py.Module.get_function_with_keywords __wrap_namespace "getdtype"
     [||]
     (Wrap_utils.keyword_args [("a", a); ("default", default); ("dtype", Some(dtype ))])

let isshape ?nonneg ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "isshape"
     [||]
     (Wrap_utils.keyword_args [("nonneg", nonneg); ("x", Some(x ))])

let isspmatrix x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isspmatrix_dia x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix_dia"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let matrix ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "matrix"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

let upcast_char args =
   Py.Module.get_function_with_keywords __wrap_namespace "upcast_char"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

let validateaxis axis =
   Py.Module.get_function_with_keywords __wrap_namespace "validateaxis"
     [||]
     (Wrap_utils.keyword_args [("axis", Some(axis ))])


end
module Dok = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.dok"

let get_py name = Py.Module.get __wrap_namespace name
module IndexMixin = struct
type tag = [`IndexMixin]
type t = [`IndexMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "IndexMixin"
     [||]
     []
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key ))])

let __setitem__ ~key ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key )); ("x", Some(x ))])

let getcol ~i self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getcol"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i ))])

let getrow ~i self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getrow"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Izip = struct
type tag = [`Zip]
type t = [`Object | `Zip] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create iterables =
   Py.Module.get_function_with_keywords __wrap_namespace "izip"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id iterables)])
     []
     |> of_pyobject
let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Xrange = struct
type tag = [`Range]
type t = [`Object | `Range] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create stop =
   Py.Module.get_function_with_keywords __wrap_namespace "xrange"
     [||]
     (Wrap_utils.keyword_args [("stop", Some(stop ))])
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let count ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "count"
     [||]
     (Wrap_utils.keyword_args [("value", Some(value ))])

let index ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "index"
     [||]
     (Wrap_utils.keyword_args [("value", Some(value ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let check_shape ?current_shape ~args () =
   Py.Module.get_function_with_keywords __wrap_namespace "check_shape"
     [||]
     (Wrap_utils.keyword_args [("current_shape", current_shape); ("args", Some(args ))])

let get_index_dtype ?arrays ?maxval ?check_contents () =
   Py.Module.get_function_with_keywords __wrap_namespace "get_index_dtype"
     [||]
     (Wrap_utils.keyword_args [("arrays", arrays); ("maxval", Wrap_utils.Option.map maxval Py.Float.of_float); ("check_contents", Wrap_utils.Option.map check_contents Py.Bool.of_bool)])
     |> Np.Dtype.of_pyobject
let getdtype ?a ?default ~dtype () =
   Py.Module.get_function_with_keywords __wrap_namespace "getdtype"
     [||]
     (Wrap_utils.keyword_args [("a", a); ("default", default); ("dtype", Some(dtype ))])

let isdense x =
   Py.Module.get_function_with_keywords __wrap_namespace "isdense"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isintlike x =
   Py.Module.get_function_with_keywords __wrap_namespace "isintlike"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isscalarlike x =
   Py.Module.get_function_with_keywords __wrap_namespace "isscalarlike"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isshape ?nonneg ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "isshape"
     [||]
     (Wrap_utils.keyword_args [("nonneg", nonneg); ("x", Some(x ))])

let isspmatrix x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isspmatrix_dok x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix_dok"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

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

let upcast args =
   Py.Module.get_function_with_keywords __wrap_namespace "upcast"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

let upcast_scalar ~dtype ~scalar () =
   Py.Module.get_function_with_keywords __wrap_namespace "upcast_scalar"
     [||]
     (Wrap_utils.keyword_args [("dtype", Some(dtype )); ("scalar", Some(scalar ))])


end
module Extract = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.extract"

let get_py name = Py.Module.get __wrap_namespace name
                  let find a =
                     Py.Module.get_function_with_keywords __wrap_namespace "find"
                       [||]
                       (Wrap_utils.keyword_args [("A", Some(a |> (function
| `Dense x -> Wrap_utils.id x
| `Spmatrix x -> Np.Obj.to_pyobject x
)))])

                  let tril ?k ?format ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "tril"
                       [||]
                       (Wrap_utils.keyword_args [("k", k); ("format", Wrap_utils.Option.map format Py.String.of_string); ("A", Some(a |> (function
| `Dense x -> Wrap_utils.id x
| `Spmatrix x -> Np.Obj.to_pyobject x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t))
                  let triu ?k ?format ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "triu"
                       [||]
                       (Wrap_utils.keyword_args [("k", k); ("format", Wrap_utils.Option.map format Py.String.of_string); ("A", Some(a |> (function
| `Dense x -> Wrap_utils.id x
| `Spmatrix x -> Np.Obj.to_pyobject x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t))

end
module Lil = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.lil"

let get_py name = Py.Module.get __wrap_namespace name
module IndexMixin = struct
type tag = [`IndexMixin]
type t = [`IndexMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "IndexMixin"
     [||]
     []
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key ))])

let __setitem__ ~key ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key )); ("x", Some(x ))])

let getcol ~i self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getcol"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i ))])

let getrow ~i self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getrow"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Xrange = struct
type tag = [`Range]
type t = [`Object | `Range] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create stop =
   Py.Module.get_function_with_keywords __wrap_namespace "xrange"
     [||]
     (Wrap_utils.keyword_args [("stop", Some(stop ))])
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let count ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "count"
     [||]
     (Wrap_utils.keyword_args [("value", Some(value ))])

let index ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "index"
     [||]
     (Wrap_utils.keyword_args [("value", Some(value ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Zip = struct
type tag = [`Zip]
type t = [`Object | `Zip] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create iterables =
   Py.Module.get_function_with_keywords __wrap_namespace "zip"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id iterables)])
     []
     |> of_pyobject
let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let asmatrix ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "asmatrix"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

let check_reshape_kwargs kwargs =
   Py.Module.get_function_with_keywords __wrap_namespace "check_reshape_kwargs"
     [||]
     (Wrap_utils.keyword_args [("kwargs", Some(kwargs ))])

let check_shape ?current_shape ~args () =
   Py.Module.get_function_with_keywords __wrap_namespace "check_shape"
     [||]
     (Wrap_utils.keyword_args [("current_shape", current_shape); ("args", Some(args ))])

let get_index_dtype ?arrays ?maxval ?check_contents () =
   Py.Module.get_function_with_keywords __wrap_namespace "get_index_dtype"
     [||]
     (Wrap_utils.keyword_args [("arrays", arrays); ("maxval", Wrap_utils.Option.map maxval Py.Float.of_float); ("check_contents", Wrap_utils.Option.map check_contents Py.Bool.of_bool)])
     |> Np.Dtype.of_pyobject
let getdtype ?a ?default ~dtype () =
   Py.Module.get_function_with_keywords __wrap_namespace "getdtype"
     [||]
     (Wrap_utils.keyword_args [("a", a); ("default", default); ("dtype", Some(dtype ))])

let isscalarlike x =
   Py.Module.get_function_with_keywords __wrap_namespace "isscalarlike"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isshape ?nonneg ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "isshape"
     [||]
     (Wrap_utils.keyword_args [("nonneg", nonneg); ("x", Some(x ))])

let isspmatrix x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isspmatrix_lil x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix_lil"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let upcast_scalar ~dtype ~scalar () =
   Py.Module.get_function_with_keywords __wrap_namespace "upcast_scalar"
     [||]
     (Wrap_utils.keyword_args [("dtype", Some(dtype )); ("scalar", Some(scalar ))])


end
module Linalg = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.linalg"

let get_py name = Py.Module.get __wrap_namespace name
module ArpackError = struct
type tag = [`ArpackError]
type t = [`ArpackError | `BaseException | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_exception x = (x :> [`BaseException] Obj.t)
let create ?infodict ~info () =
   Py.Module.get_function_with_keywords __wrap_namespace "ArpackError"
     [||]
     (Wrap_utils.keyword_args [("infodict", infodict); ("info", Some(info ))])
     |> of_pyobject
let with_traceback ~tb self =
   Py.Module.get_function_with_keywords (to_pyobject self) "with_traceback"
     [||]
     (Wrap_utils.keyword_args [("tb", Some(tb ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ArpackNoConvergence = struct
type tag = [`ArpackNoConvergence]
type t = [`ArpackNoConvergence | `BaseException | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_exception x = (x :> [`BaseException] Obj.t)
let create ~msg ~eigenvalues ~eigenvectors () =
   Py.Module.get_function_with_keywords __wrap_namespace "ArpackNoConvergence"
     [||]
     (Wrap_utils.keyword_args [("msg", Some(msg )); ("eigenvalues", Some(eigenvalues )); ("eigenvectors", Some(eigenvectors ))])
     |> of_pyobject
let with_traceback ~tb self =
   Py.Module.get_function_with_keywords (to_pyobject self) "with_traceback"
     [||]
     (Wrap_utils.keyword_args [("tb", Some(tb ))])


let eigenvalues_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "eigenvalues" with
  | None -> failwith "attribute eigenvalues not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let eigenvalues self = match eigenvalues_opt self with
  | None -> raise Not_found
  | Some x -> x

let eigenvectors_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "eigenvectors" with
  | None -> failwith "attribute eigenvectors not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let eigenvectors self = match eigenvectors_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LinearOperator = struct
type tag = [`LinearOperator]
type t = [`LinearOperator | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "LinearOperator"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let adjoint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "adjoint"
     [||]
     []

let dot ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let matmat ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matmat"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])

let matvec ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matvec"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let rmatmat ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatmat"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])

let rmatvec ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatvec"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let transpose self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     [||]
     []


let args_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "args" with
  | None -> failwith "attribute args not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let args self = match args_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MatrixRankWarning = struct
type tag = [`MatrixRankWarning]
type t = [`BaseException | `MatrixRankWarning | `Object] Obj.t
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
module SuperLU = struct
type tag = [`SuperLU]
type t = [`Object | `SuperLU] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "SuperLU"
     [||]
     []
     |> of_pyobject

let shape_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "shape" with
  | None -> failwith "attribute shape not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let shape self = match shape_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Arpack = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.linalg.eigen.arpack.arpack"

let get_py name = Py.Module.get __wrap_namespace name
module IterInv = struct
type tag = [`IterInv]
type t = [`IterInv | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "IterInv"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let adjoint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "adjoint"
     [||]
     []

let dot ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let matmat ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matmat"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])

let matvec ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matvec"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let rmatmat ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatmat"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])

let rmatvec ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatvec"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let transpose self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module IterOpInv = struct
type tag = [`IterOpInv]
type t = [`IterOpInv | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "IterOpInv"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let adjoint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "adjoint"
     [||]
     []

let dot ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let matmat ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matmat"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])

let matvec ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matvec"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let rmatmat ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatmat"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])

let rmatvec ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatvec"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let transpose self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LuInv = struct
type tag = [`LuInv]
type t = [`LuInv | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "LuInv"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let adjoint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "adjoint"
     [||]
     []

let dot ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let matmat ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matmat"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])

let matvec ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matvec"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let rmatmat ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatmat"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])

let rmatvec ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatvec"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let transpose self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ReentrancyLock = struct
type tag = [`ReentrancyLock]
type t = [`Object | `ReentrancyLock] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create err_msg =
   Py.Module.get_function_with_keywords __wrap_namespace "ReentrancyLock"
     [||]
     (Wrap_utils.keyword_args [("err_msg", Some(err_msg ))])
     |> of_pyobject
let decorate ~func self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decorate"
     [||]
     (Wrap_utils.keyword_args [("func", Some(func ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SpLuInv = struct
type tag = [`SpLuInv]
type t = [`Object | `SpLuInv] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "SpLuInv"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let adjoint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "adjoint"
     [||]
     []

let dot ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let matmat ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matmat"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])

let matvec ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matvec"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let rmatmat ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatmat"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])

let rmatvec ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatvec"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let transpose self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let aslinearoperator a =
   Py.Module.get_function_with_keywords __wrap_namespace "aslinearoperator"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a ))])

let choose_ncv k =
   Py.Module.get_function_with_keywords __wrap_namespace "choose_ncv"
     [||]
     (Wrap_utils.keyword_args [("k", Some(k ))])

let eig ?b ?left ?right ?overwrite_a ?overwrite_b ?check_finite ?homogeneous_eigvals ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "eig"
     [||]
     (Wrap_utils.keyword_args [("b", Wrap_utils.Option.map b Np.Obj.to_pyobject); ("left", Wrap_utils.Option.map left Py.Bool.of_bool); ("right", Wrap_utils.Option.map right Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("homogeneous_eigvals", Wrap_utils.Option.map homogeneous_eigvals Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
let eigh ?b ?lower ?eigvals_only ?overwrite_a ?overwrite_b ?turbo ?eigvals ?type_ ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "eigh"
     [||]
     (Wrap_utils.keyword_args [("b", Wrap_utils.Option.map b Np.Obj.to_pyobject); ("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("eigvals_only", Wrap_utils.Option.map eigvals_only Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("turbo", Wrap_utils.Option.map turbo Py.Bool.of_bool); ("eigvals", eigvals); ("type", Wrap_utils.Option.map type_ Py.Int.of_int); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let eigs ?k ?m ?sigma ?which ?v0 ?ncv ?maxiter ?tol ?return_eigenvectors ?minv ?oPinv ?oPpart ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "eigs"
                       [||]
                       (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("M", Wrap_utils.Option.map m (function
| `Arr x -> Np.Obj.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
)); ("sigma", sigma); ("which", Wrap_utils.Option.map which (function
| `LM -> Py.String.of_string "LM"
| `SM -> Py.String.of_string "SM"
| `LR -> Py.String.of_string "LR"
| `SR -> Py.String.of_string "SR"
| `LI -> Py.String.of_string "LI"
| `SI -> Py.String.of_string "SI"
)); ("v0", Wrap_utils.Option.map v0 Np.Obj.to_pyobject); ("ncv", Wrap_utils.Option.map ncv Py.Int.of_int); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("return_eigenvectors", Wrap_utils.Option.map return_eigenvectors Py.Bool.of_bool); ("Minv", Wrap_utils.Option.map minv (function
| `Arr x -> Np.Obj.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
)); ("OPinv", Wrap_utils.Option.map oPinv (function
| `Arr x -> Np.Obj.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
)); ("OPpart", oPpart); ("A", Some(a |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let eigsh ?k ?m ?sigma ?which ?v0 ?ncv ?maxiter ?tol ?return_eigenvectors ?minv ?oPinv ?mode ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "eigsh"
                       [||]
                       (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("M", m); ("sigma", sigma); ("which", which); ("v0", v0); ("ncv", ncv); ("maxiter", maxiter); ("tol", tol); ("return_eigenvectors", return_eigenvectors); ("Minv", minv); ("OPinv", oPinv); ("mode", mode); ("A", Some(a |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let eye ?n ?k ?dtype ?format ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "eye"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("k", Wrap_utils.Option.map k Py.Int.of_int); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("format", Wrap_utils.Option.map format Py.String.of_string); ("m", Some(m |> Py.Int.of_int))])

let get_OPinv_matvec ?hermitian ?tol ~a ~m ~sigma () =
   Py.Module.get_function_with_keywords __wrap_namespace "get_OPinv_matvec"
     [||]
     (Wrap_utils.keyword_args [("hermitian", hermitian); ("tol", tol); ("A", Some(a )); ("M", Some(m )); ("sigma", Some(sigma ))])

let get_inv_matvec ?hermitian ?tol ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "get_inv_matvec"
     [||]
     (Wrap_utils.keyword_args [("hermitian", hermitian); ("tol", tol); ("M", Some(m ))])

                  let gmres ?x0 ?tol ?restart ?maxiter ?m ?callback ?restrt ?atol ?callback_type ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "gmres"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("tol", tol); ("restart", restart); ("maxiter", maxiter); ("M", m); ("callback", callback); ("restrt", restrt); ("atol", atol); ("callback_type", callback_type); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
let gmres_loose ~a ~b ~tol () =
   Py.Module.get_function_with_keywords __wrap_namespace "gmres_loose"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a )); ("b", Some(b )); ("tol", Some(tol ))])

let is_pydata_spmatrix m =
   Py.Module.get_function_with_keywords __wrap_namespace "is_pydata_spmatrix"
     [||]
     (Wrap_utils.keyword_args [("m", Some(m ))])

let isdense x =
   Py.Module.get_function_with_keywords __wrap_namespace "isdense"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let issparse x =
   Py.Module.get_function_with_keywords __wrap_namespace "issparse"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isspmatrix x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isspmatrix_csr x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix_csr"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

                  let lobpcg ?b ?m ?y ?tol ?maxiter ?largest ?verbosityLevel ?retLambdaHistory ?retResidualNormsHistory ~a ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "lobpcg"
                       [||]
                       (Wrap_utils.keyword_args [("B", Wrap_utils.Option.map b (function
| `PyObject x -> Wrap_utils.id x
| `Spmatrix x -> Np.Obj.to_pyobject x
)); ("M", Wrap_utils.Option.map m (function
| `PyObject x -> Wrap_utils.id x
| `Spmatrix x -> Np.Obj.to_pyobject x
)); ("Y", Wrap_utils.Option.map y (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("tol", Wrap_utils.Option.map tol (function
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("largest", Wrap_utils.Option.map largest Py.Bool.of_bool); ("verbosityLevel", Wrap_utils.Option.map verbosityLevel Py.Int.of_int); ("retLambdaHistory", Wrap_utils.Option.map retLambdaHistory Py.Bool.of_bool); ("retResidualNormsHistory", Wrap_utils.Option.map retResidualNormsHistory Py.Bool.of_bool); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("X", Some(x |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3))))
let lu_factor ?overwrite_a ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "lu_factor"
     [||]
     (Wrap_utils.keyword_args [("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let lu_solve ?trans ?overwrite_b ?check_finite ~lu_and_piv ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "lu_solve"
                       [||]
                       (Wrap_utils.keyword_args [("trans", Wrap_utils.Option.map trans (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `Two -> Py.Int.of_int 2
)); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("lu_and_piv", Some(lu_and_piv )); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let splu ?permc_spec ?diag_pivot_thresh ?relax ?panel_size ?options ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "splu"
     [||]
     (Wrap_utils.keyword_args [("permc_spec", Wrap_utils.Option.map permc_spec Py.String.of_string); ("diag_pivot_thresh", Wrap_utils.Option.map diag_pivot_thresh Py.Float.of_float); ("relax", Wrap_utils.Option.map relax Py.Int.of_int); ("panel_size", Wrap_utils.Option.map panel_size Py.Int.of_int); ("options", options); ("A", Some(a |> Np.Obj.to_pyobject))])

                  let svds ?k ?ncv ?tol ?which ?v0 ?maxiter ?return_singular_vectors ?solver ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "svds"
                       [||]
                       (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("ncv", Wrap_utils.Option.map ncv Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("which", Wrap_utils.Option.map which (function
| `LM -> Py.String.of_string "LM"
| `SM -> Py.String.of_string "SM"
)); ("v0", Wrap_utils.Option.map v0 Np.Obj.to_pyobject); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("return_singular_vectors", Wrap_utils.Option.map return_singular_vectors (function
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("solver", Wrap_utils.Option.map solver Py.String.of_string); ("A", Some(a |> (function
| `LinearOperator x -> Wrap_utils.id x
| `Spmatrix x -> Np.Obj.to_pyobject x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))

end
module Dsolve = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.linalg.dsolve"

let get_py name = Py.Module.get __wrap_namespace name
module Linsolve = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.linalg.dsolve.linsolve"

let get_py name = Py.Module.get __wrap_namespace name
                  let asarray ?dtype ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let factorized a =
   Py.Module.get_function_with_keywords __wrap_namespace "factorized"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject))])

let is_pydata_spmatrix m =
   Py.Module.get_function_with_keywords __wrap_namespace "is_pydata_spmatrix"
     [||]
     (Wrap_utils.keyword_args [("m", Some(m ))])

let isspmatrix x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isspmatrix_csc x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix_csc"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isspmatrix_csr x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix_csr"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let spilu ?drop_tol ?fill_factor ?drop_rule ?permc_spec ?diag_pivot_thresh ?relax ?panel_size ?options ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "spilu"
     [||]
     (Wrap_utils.keyword_args [("drop_tol", Wrap_utils.Option.map drop_tol Py.Float.of_float); ("fill_factor", Wrap_utils.Option.map fill_factor Py.Float.of_float); ("drop_rule", Wrap_utils.Option.map drop_rule Py.String.of_string); ("permc_spec", permc_spec); ("diag_pivot_thresh", diag_pivot_thresh); ("relax", relax); ("panel_size", panel_size); ("options", options); ("A", Some(a |> Np.Obj.to_pyobject))])

let splu ?permc_spec ?diag_pivot_thresh ?relax ?panel_size ?options ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "splu"
     [||]
     (Wrap_utils.keyword_args [("permc_spec", Wrap_utils.Option.map permc_spec Py.String.of_string); ("diag_pivot_thresh", Wrap_utils.Option.map diag_pivot_thresh Py.Float.of_float); ("relax", Wrap_utils.Option.map relax Py.Int.of_int); ("panel_size", Wrap_utils.Option.map panel_size Py.Int.of_int); ("options", options); ("A", Some(a |> Np.Obj.to_pyobject))])

let spsolve ?permc_spec ?use_umfpack ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "spsolve"
     [||]
     (Wrap_utils.keyword_args [("permc_spec", Wrap_utils.Option.map permc_spec Py.String.of_string); ("use_umfpack", Wrap_utils.Option.map use_umfpack Py.Bool.of_bool); ("A", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let spsolve_triangular ?lower ?overwrite_A ?overwrite_b ?unit_diagonal ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "spsolve_triangular"
     [||]
     (Wrap_utils.keyword_args [("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("overwrite_A", Wrap_utils.Option.map overwrite_A Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("unit_diagonal", Wrap_utils.Option.map unit_diagonal Py.Bool.of_bool); ("A", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b ))])

let use_solver ?kwargs () =
   Py.Module.get_function_with_keywords __wrap_namespace "use_solver"
     [||]
     (match kwargs with None -> [] | Some x -> x)

let warn ?category ?stacklevel ?source ~message () =
   Py.Module.get_function_with_keywords __wrap_namespace "warn"
     [||]
     (Wrap_utils.keyword_args [("category", category); ("stacklevel", stacklevel); ("source", source); ("message", Some(message ))])


end
let factorized a =
   Py.Module.get_function_with_keywords __wrap_namespace "factorized"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject))])

let spilu ?drop_tol ?fill_factor ?drop_rule ?permc_spec ?diag_pivot_thresh ?relax ?panel_size ?options ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "spilu"
     [||]
     (Wrap_utils.keyword_args [("drop_tol", Wrap_utils.Option.map drop_tol Py.Float.of_float); ("fill_factor", Wrap_utils.Option.map fill_factor Py.Float.of_float); ("drop_rule", Wrap_utils.Option.map drop_rule Py.String.of_string); ("permc_spec", permc_spec); ("diag_pivot_thresh", diag_pivot_thresh); ("relax", relax); ("panel_size", panel_size); ("options", options); ("A", Some(a |> Np.Obj.to_pyobject))])

let splu ?permc_spec ?diag_pivot_thresh ?relax ?panel_size ?options ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "splu"
     [||]
     (Wrap_utils.keyword_args [("permc_spec", Wrap_utils.Option.map permc_spec Py.String.of_string); ("diag_pivot_thresh", Wrap_utils.Option.map diag_pivot_thresh Py.Float.of_float); ("relax", Wrap_utils.Option.map relax Py.Int.of_int); ("panel_size", Wrap_utils.Option.map panel_size Py.Int.of_int); ("options", options); ("A", Some(a |> Np.Obj.to_pyobject))])

let spsolve ?permc_spec ?use_umfpack ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "spsolve"
     [||]
     (Wrap_utils.keyword_args [("permc_spec", Wrap_utils.Option.map permc_spec Py.String.of_string); ("use_umfpack", Wrap_utils.Option.map use_umfpack Py.Bool.of_bool); ("A", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let spsolve_triangular ?lower ?overwrite_A ?overwrite_b ?unit_diagonal ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "spsolve_triangular"
     [||]
     (Wrap_utils.keyword_args [("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("overwrite_A", Wrap_utils.Option.map overwrite_A Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("unit_diagonal", Wrap_utils.Option.map unit_diagonal Py.Bool.of_bool); ("A", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b ))])

let use_solver ?kwargs () =
   Py.Module.get_function_with_keywords __wrap_namespace "use_solver"
     [||]
     (match kwargs with None -> [] | Some x -> x)


end
module Eigen = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.linalg.eigen"

let get_py name = Py.Module.get __wrap_namespace name
module Arpack = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.linalg.eigen.arpack.arpack"

let get_py name = Py.Module.get __wrap_namespace name
let aslinearoperator a =
   Py.Module.get_function_with_keywords __wrap_namespace "aslinearoperator"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a ))])

let choose_ncv k =
   Py.Module.get_function_with_keywords __wrap_namespace "choose_ncv"
     [||]
     (Wrap_utils.keyword_args [("k", Some(k ))])

let eig ?b ?left ?right ?overwrite_a ?overwrite_b ?check_finite ?homogeneous_eigvals ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "eig"
     [||]
     (Wrap_utils.keyword_args [("b", Wrap_utils.Option.map b Np.Obj.to_pyobject); ("left", Wrap_utils.Option.map left Py.Bool.of_bool); ("right", Wrap_utils.Option.map right Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("homogeneous_eigvals", Wrap_utils.Option.map homogeneous_eigvals Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
let eigh ?b ?lower ?eigvals_only ?overwrite_a ?overwrite_b ?turbo ?eigvals ?type_ ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "eigh"
     [||]
     (Wrap_utils.keyword_args [("b", Wrap_utils.Option.map b Np.Obj.to_pyobject); ("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("eigvals_only", Wrap_utils.Option.map eigvals_only Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("turbo", Wrap_utils.Option.map turbo Py.Bool.of_bool); ("eigvals", eigvals); ("type", Wrap_utils.Option.map type_ Py.Int.of_int); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let eigs ?k ?m ?sigma ?which ?v0 ?ncv ?maxiter ?tol ?return_eigenvectors ?minv ?oPinv ?oPpart ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "eigs"
                       [||]
                       (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("M", Wrap_utils.Option.map m (function
| `Arr x -> Np.Obj.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
)); ("sigma", sigma); ("which", Wrap_utils.Option.map which (function
| `LM -> Py.String.of_string "LM"
| `SM -> Py.String.of_string "SM"
| `LR -> Py.String.of_string "LR"
| `SR -> Py.String.of_string "SR"
| `LI -> Py.String.of_string "LI"
| `SI -> Py.String.of_string "SI"
)); ("v0", Wrap_utils.Option.map v0 Np.Obj.to_pyobject); ("ncv", Wrap_utils.Option.map ncv Py.Int.of_int); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("return_eigenvectors", Wrap_utils.Option.map return_eigenvectors Py.Bool.of_bool); ("Minv", Wrap_utils.Option.map minv (function
| `Arr x -> Np.Obj.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
)); ("OPinv", Wrap_utils.Option.map oPinv (function
| `Arr x -> Np.Obj.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
)); ("OPpart", oPpart); ("A", Some(a |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let eigsh ?k ?m ?sigma ?which ?v0 ?ncv ?maxiter ?tol ?return_eigenvectors ?minv ?oPinv ?mode ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "eigsh"
                       [||]
                       (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("M", m); ("sigma", sigma); ("which", which); ("v0", v0); ("ncv", ncv); ("maxiter", maxiter); ("tol", tol); ("return_eigenvectors", return_eigenvectors); ("Minv", minv); ("OPinv", oPinv); ("mode", mode); ("A", Some(a |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let eye ?n ?k ?dtype ?format ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "eye"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("k", Wrap_utils.Option.map k Py.Int.of_int); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("format", Wrap_utils.Option.map format Py.String.of_string); ("m", Some(m |> Py.Int.of_int))])

let get_OPinv_matvec ?hermitian ?tol ~a ~m ~sigma () =
   Py.Module.get_function_with_keywords __wrap_namespace "get_OPinv_matvec"
     [||]
     (Wrap_utils.keyword_args [("hermitian", hermitian); ("tol", tol); ("A", Some(a )); ("M", Some(m )); ("sigma", Some(sigma ))])

let get_inv_matvec ?hermitian ?tol ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "get_inv_matvec"
     [||]
     (Wrap_utils.keyword_args [("hermitian", hermitian); ("tol", tol); ("M", Some(m ))])

                  let gmres ?x0 ?tol ?restart ?maxiter ?m ?callback ?restrt ?atol ?callback_type ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "gmres"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("tol", tol); ("restart", restart); ("maxiter", maxiter); ("M", m); ("callback", callback); ("restrt", restrt); ("atol", atol); ("callback_type", callback_type); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
let gmres_loose ~a ~b ~tol () =
   Py.Module.get_function_with_keywords __wrap_namespace "gmres_loose"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a )); ("b", Some(b )); ("tol", Some(tol ))])

let is_pydata_spmatrix m =
   Py.Module.get_function_with_keywords __wrap_namespace "is_pydata_spmatrix"
     [||]
     (Wrap_utils.keyword_args [("m", Some(m ))])

let isdense x =
   Py.Module.get_function_with_keywords __wrap_namespace "isdense"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let issparse x =
   Py.Module.get_function_with_keywords __wrap_namespace "issparse"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isspmatrix x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isspmatrix_csr x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix_csr"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

                  let lobpcg ?b ?m ?y ?tol ?maxiter ?largest ?verbosityLevel ?retLambdaHistory ?retResidualNormsHistory ~a ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "lobpcg"
                       [||]
                       (Wrap_utils.keyword_args [("B", Wrap_utils.Option.map b (function
| `PyObject x -> Wrap_utils.id x
| `Spmatrix x -> Np.Obj.to_pyobject x
)); ("M", Wrap_utils.Option.map m (function
| `PyObject x -> Wrap_utils.id x
| `Spmatrix x -> Np.Obj.to_pyobject x
)); ("Y", Wrap_utils.Option.map y (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("tol", Wrap_utils.Option.map tol (function
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("largest", Wrap_utils.Option.map largest Py.Bool.of_bool); ("verbosityLevel", Wrap_utils.Option.map verbosityLevel Py.Int.of_int); ("retLambdaHistory", Wrap_utils.Option.map retLambdaHistory Py.Bool.of_bool); ("retResidualNormsHistory", Wrap_utils.Option.map retResidualNormsHistory Py.Bool.of_bool); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("X", Some(x |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3))))
let lu_factor ?overwrite_a ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "lu_factor"
     [||]
     (Wrap_utils.keyword_args [("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let lu_solve ?trans ?overwrite_b ?check_finite ~lu_and_piv ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "lu_solve"
                       [||]
                       (Wrap_utils.keyword_args [("trans", Wrap_utils.Option.map trans (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `Two -> Py.Int.of_int 2
)); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("lu_and_piv", Some(lu_and_piv )); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let splu ?permc_spec ?diag_pivot_thresh ?relax ?panel_size ?options ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "splu"
     [||]
     (Wrap_utils.keyword_args [("permc_spec", Wrap_utils.Option.map permc_spec Py.String.of_string); ("diag_pivot_thresh", Wrap_utils.Option.map diag_pivot_thresh Py.Float.of_float); ("relax", Wrap_utils.Option.map relax Py.Int.of_int); ("panel_size", Wrap_utils.Option.map panel_size Py.Int.of_int); ("options", options); ("A", Some(a |> Np.Obj.to_pyobject))])

                  let svds ?k ?ncv ?tol ?which ?v0 ?maxiter ?return_singular_vectors ?solver ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "svds"
                       [||]
                       (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("ncv", Wrap_utils.Option.map ncv Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("which", Wrap_utils.Option.map which (function
| `LM -> Py.String.of_string "LM"
| `SM -> Py.String.of_string "SM"
)); ("v0", Wrap_utils.Option.map v0 Np.Obj.to_pyobject); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("return_singular_vectors", Wrap_utils.Option.map return_singular_vectors (function
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("solver", Wrap_utils.Option.map solver Py.String.of_string); ("A", Some(a |> (function
| `LinearOperator x -> Wrap_utils.id x
| `Spmatrix x -> Np.Obj.to_pyobject x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))

end
                  let eigs ?k ?m ?sigma ?which ?v0 ?ncv ?maxiter ?tol ?return_eigenvectors ?minv ?oPinv ?oPpart ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "eigs"
                       [||]
                       (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("M", Wrap_utils.Option.map m (function
| `Arr x -> Np.Obj.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
)); ("sigma", sigma); ("which", Wrap_utils.Option.map which (function
| `LM -> Py.String.of_string "LM"
| `SM -> Py.String.of_string "SM"
| `LR -> Py.String.of_string "LR"
| `SR -> Py.String.of_string "SR"
| `LI -> Py.String.of_string "LI"
| `SI -> Py.String.of_string "SI"
)); ("v0", Wrap_utils.Option.map v0 Np.Obj.to_pyobject); ("ncv", Wrap_utils.Option.map ncv Py.Int.of_int); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("return_eigenvectors", Wrap_utils.Option.map return_eigenvectors Py.Bool.of_bool); ("Minv", Wrap_utils.Option.map minv (function
| `Arr x -> Np.Obj.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
)); ("OPinv", Wrap_utils.Option.map oPinv (function
| `Arr x -> Np.Obj.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
)); ("OPpart", oPpart); ("A", Some(a |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let eigsh ?k ?m ?sigma ?which ?v0 ?ncv ?maxiter ?tol ?return_eigenvectors ?minv ?oPinv ?mode ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "eigsh"
                       [||]
                       (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("M", m); ("sigma", sigma); ("which", which); ("v0", v0); ("ncv", ncv); ("maxiter", maxiter); ("tol", tol); ("return_eigenvectors", return_eigenvectors); ("Minv", minv); ("OPinv", oPinv); ("mode", mode); ("A", Some(a |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let lobpcg ?b ?m ?y ?tol ?maxiter ?largest ?verbosityLevel ?retLambdaHistory ?retResidualNormsHistory ~a ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "lobpcg"
                       [||]
                       (Wrap_utils.keyword_args [("B", Wrap_utils.Option.map b (function
| `PyObject x -> Wrap_utils.id x
| `Spmatrix x -> Np.Obj.to_pyobject x
)); ("M", Wrap_utils.Option.map m (function
| `PyObject x -> Wrap_utils.id x
| `Spmatrix x -> Np.Obj.to_pyobject x
)); ("Y", Wrap_utils.Option.map y (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("tol", Wrap_utils.Option.map tol (function
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("largest", Wrap_utils.Option.map largest Py.Bool.of_bool); ("verbosityLevel", Wrap_utils.Option.map verbosityLevel Py.Int.of_int); ("retLambdaHistory", Wrap_utils.Option.map retLambdaHistory Py.Bool.of_bool); ("retResidualNormsHistory", Wrap_utils.Option.map retResidualNormsHistory Py.Bool.of_bool); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("X", Some(x |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3))))
                  let svds ?k ?ncv ?tol ?which ?v0 ?maxiter ?return_singular_vectors ?solver ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "svds"
                       [||]
                       (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("ncv", Wrap_utils.Option.map ncv Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("which", Wrap_utils.Option.map which (function
| `LM -> Py.String.of_string "LM"
| `SM -> Py.String.of_string "SM"
)); ("v0", Wrap_utils.Option.map v0 Np.Obj.to_pyobject); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("return_singular_vectors", Wrap_utils.Option.map return_singular_vectors (function
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("solver", Wrap_utils.Option.map solver Py.String.of_string); ("A", Some(a |> (function
| `LinearOperator x -> Wrap_utils.id x
| `Spmatrix x -> Np.Obj.to_pyobject x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))

end
module Interface = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.linalg.interface"

let get_py name = Py.Module.get __wrap_namespace name
module IdentityOperator = struct
type tag = [`IdentityOperator]
type t = [`IdentityOperator | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "IdentityOperator"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let adjoint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "adjoint"
     [||]
     []

let dot ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let matmat ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matmat"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])

let matvec ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matvec"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let rmatmat ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatmat"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])

let rmatvec ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatvec"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let transpose self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MatrixLinearOperator = struct
type tag = [`MatrixLinearOperator]
type t = [`MatrixLinearOperator | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "MatrixLinearOperator"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let adjoint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "adjoint"
     [||]
     []

let dot ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let matmat ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matmat"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])

let matvec ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matvec"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let rmatmat ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatmat"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])

let rmatvec ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatvec"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let transpose self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let aslinearoperator a =
   Py.Module.get_function_with_keywords __wrap_namespace "aslinearoperator"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a ))])

let asmatrix ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "asmatrix"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

let is_pydata_spmatrix m =
   Py.Module.get_function_with_keywords __wrap_namespace "is_pydata_spmatrix"
     [||]
     (Wrap_utils.keyword_args [("m", Some(m ))])

let isintlike x =
   Py.Module.get_function_with_keywords __wrap_namespace "isintlike"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isshape ?nonneg ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "isshape"
     [||]
     (Wrap_utils.keyword_args [("nonneg", nonneg); ("x", Some(x ))])

let isspmatrix x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])


end
module Isolve = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.linalg.isolve"

let get_py name = Py.Module.get __wrap_namespace name
module Iterative = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.linalg.isolve.iterative"

let get_py name = Py.Module.get __wrap_namespace name
                  let bicg ?x0 ?tol ?maxiter ?m ?callback ?atol ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "bicg"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("tol", tol); ("maxiter", maxiter); ("M", m); ("callback", callback); ("atol", atol); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let bicgstab ?x0 ?tol ?maxiter ?m ?callback ?atol ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "bicgstab"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("tol", tol); ("maxiter", maxiter); ("M", m); ("callback", callback); ("atol", atol); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let cg ?x0 ?tol ?maxiter ?m ?callback ?atol ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cg"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("tol", tol); ("maxiter", maxiter); ("M", m); ("callback", callback); ("atol", atol); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let cgs ?x0 ?tol ?maxiter ?m ?callback ?atol ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cgs"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("tol", tol); ("maxiter", maxiter); ("M", m); ("callback", callback); ("atol", atol); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let gmres ?x0 ?tol ?restart ?maxiter ?m ?callback ?restrt ?atol ?callback_type ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "gmres"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("tol", tol); ("restart", restart); ("maxiter", maxiter); ("M", m); ("callback", callback); ("restrt", restrt); ("atol", atol); ("callback_type", callback_type); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let make_system ~a ~m ~x0 ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "make_system"
                       [||]
                       (Wrap_utils.keyword_args [("A", Some(a )); ("M", Some(m )); ("x0", Some(x0 |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `None -> Py.none
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3))))
let non_reentrant ?err_msg () =
   Py.Module.get_function_with_keywords __wrap_namespace "non_reentrant"
     [||]
     (Wrap_utils.keyword_args [("err_msg", err_msg)])

                  let qmr ?x0 ?tol ?maxiter ?m1 ?m2 ?callback ?atol ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "qmr"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("tol", tol); ("maxiter", maxiter); ("M1", m1); ("M2", m2); ("callback", callback); ("atol", atol); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
let set_docstring ?footer ?atol_default ~header ~ainfo () =
   Py.Module.get_function_with_keywords __wrap_namespace "set_docstring"
     [||]
     (Wrap_utils.keyword_args [("footer", footer); ("atol_default", atol_default); ("header", Some(header )); ("Ainfo", Some(ainfo ))])


end
module Utils = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.linalg.isolve.utils"

let get_py name = Py.Module.get __wrap_namespace name
module Matrix = struct
type tag = [`Matrix]
type t = [`ArrayLike | `Matrix | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?dtype ?copy ~data () =
                     Py.Module.get_function_with_keywords __wrap_namespace "matrix"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("data", Some(data |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `S x -> Py.String.of_string x
)))])
                       |> of_pyobject
let __getitem__ ~index self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let __setitem__ ~key ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     (Array.of_list @@ List.concat [[key ];[value ]])
     []

let all ?axis ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "all"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out)])

let any ?axis ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "any"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])

let argmax ?axis ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "argmax"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out)])

let argmin ?axis ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "argmin"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out)])

let argpartition ?axis ?kind ?order ~kth self =
   Py.Module.get_function_with_keywords (to_pyobject self) "argpartition"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("kind", kind); ("order", order); ("kth", Some(kth ))])

let argsort ?axis ?kind ?order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "argsort"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("kind", kind); ("order", order)])

                  let astype ?order ?casting ?subok ?copy ~dtype self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "astype"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
| `A -> Py.String.of_string "A"
| `K -> Py.String.of_string "K"
)); ("casting", Wrap_utils.Option.map casting (function
| `No -> Py.String.of_string "no"
| `Equiv -> Py.String.of_string "equiv"
| `Safe -> Py.String.of_string "safe"
| `Same_kind -> Py.String.of_string "same_kind"
| `Unsafe -> Py.String.of_string "unsafe"
)); ("subok", subok); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("dtype", Some(dtype |> (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let byteswap ?inplace self =
   Py.Module.get_function_with_keywords (to_pyobject self) "byteswap"
     [||]
     (Wrap_utils.keyword_args [("inplace", Wrap_utils.Option.map inplace Py.Bool.of_bool)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let choose ?out ?mode ~choices self =
   Py.Module.get_function_with_keywords (to_pyobject self) "choose"
     [||]
     (Wrap_utils.keyword_args [("out", out); ("mode", mode); ("choices", Some(choices ))])

let clip ?min ?max ?out ?kwargs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "clip"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("min", min); ("max", max); ("out", out)]) (match kwargs with None -> [] | Some x -> x))

let compress ?axis ?out ~condition self =
   Py.Module.get_function_with_keywords (to_pyobject self) "compress"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("condition", Some(condition ))])

let conj self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conj"
     [||]
     []

let conjugate self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conjugate"
     [||]
     []

                  let copy ?order self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "copy"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
| `A -> Py.String.of_string "A"
| `K -> Py.String.of_string "K"
))])

let cumprod ?axis ?dtype ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cumprod"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out)])

let cumsum ?axis ?dtype ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cumsum"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out)])

let diagonal ?offset ?axis1 ?axis2 self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diagonal"
     [||]
     (Wrap_utils.keyword_args [("offset", offset); ("axis1", axis1); ("axis2", axis2)])

let dot ?out ~b self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("out", out); ("b", Some(b ))])

                  let dump ~file self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "dump"
                       [||]
                       (Wrap_utils.keyword_args [("file", Some(file |> (function
| `S x -> Py.String.of_string x
| `Path x -> Wrap_utils.id x
)))])

let dumps self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dumps"
     [||]
     []

                  let fill ~value self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fill"
                       [||]
                       (Wrap_utils.keyword_args [("value", Some(value |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])

                  let flatten ?order self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "flatten"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
| `A -> Py.String.of_string "A"
| `K -> Py.String.of_string "K"
))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let getA self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getA"
     [||]
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let getA1 self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getA1"
     [||]
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let getH self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getH"
     [||]
     []

let getI self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getI"
     [||]
     []

let getT self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getT"
     [||]
     []

                  let getfield ?offset ~dtype self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "getfield"
                       [||]
                       (Wrap_utils.keyword_args [("offset", Wrap_utils.Option.map offset Py.Int.of_int); ("dtype", Some(dtype |> (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)))])

let item args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "item"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

let itemset args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "itemset"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

let max ?axis ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "max"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out)])

let mean ?axis ?dtype ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "mean"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out)])

let min ?axis ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "min"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out)])

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let nonzero self =
   Py.Module.get_function_with_keywords (to_pyobject self) "nonzero"
     [||]
     []

                  let partition ?axis ?kind ?order ~kth self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "partition"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("kind", Wrap_utils.Option.map kind (function
| `Introselect -> Py.String.of_string "introselect"
)); ("order", Wrap_utils.Option.map order (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `S x -> Py.String.of_string x
)); ("kth", Some(kth |> (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)))])

let prod ?axis ?dtype ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "prod"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out)])

let ptp ?axis ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ptp"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out)])

let put ?mode ~indices ~values self =
   Py.Module.get_function_with_keywords (to_pyobject self) "put"
     [||]
     (Wrap_utils.keyword_args [("mode", mode); ("indices", Some(indices )); ("values", Some(values ))])

                  let ravel ?order self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "ravel"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
| `A -> Py.String.of_string "A"
| `K -> Py.String.of_string "K"
))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let repeat ?axis ~repeats self =
   Py.Module.get_function_with_keywords (to_pyobject self) "repeat"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("repeats", Some(repeats ))])

let reshape ?order ~shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "reshape"
     [||]
     (Wrap_utils.keyword_args [("order", order); ("shape", Some(shape ))])

                  let resize ?refcheck ~new_shape self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "resize"
                       [||]
                       (Wrap_utils.keyword_args [("refcheck", Wrap_utils.Option.map refcheck Py.Bool.of_bool); ("new_shape", Some(new_shape |> (function
| `TupleOfInts x -> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml) x
| `T_n_ints x -> Wrap_utils.id x
)))])

let round ?decimals ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "round"
     [||]
     (Wrap_utils.keyword_args [("decimals", decimals); ("out", out)])

let searchsorted ?side ?sorter ~v self =
   Py.Module.get_function_with_keywords (to_pyobject self) "searchsorted"
     [||]
     (Wrap_utils.keyword_args [("side", side); ("sorter", sorter); ("v", Some(v ))])

let setfield ?offset ~val_ ~dtype self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setfield"
     [||]
     (Wrap_utils.keyword_args [("offset", Wrap_utils.Option.map offset Py.Int.of_int); ("val", Some(val_ )); ("dtype", Some(dtype ))])

let setflags ?write ?align ?uic self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setflags"
     [||]
     (Wrap_utils.keyword_args [("write", Wrap_utils.Option.map write Py.Bool.of_bool); ("align", Wrap_utils.Option.map align Py.Bool.of_bool); ("uic", Wrap_utils.Option.map uic Py.Bool.of_bool)])

                  let sort ?axis ?kind ?order self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "sort"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("kind", Wrap_utils.Option.map kind (function
| `Stable -> Py.String.of_string "stable"
| `Quicksort -> Py.String.of_string "quicksort"
| `Heapsort -> Py.String.of_string "heapsort"
| `Mergesort -> Py.String.of_string "mergesort"
)); ("order", Wrap_utils.Option.map order (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `S x -> Py.String.of_string x
))])

let squeeze ?axis self =
   Py.Module.get_function_with_keywords (to_pyobject self) "squeeze"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let std ?axis ?dtype ?out ?ddof self =
   Py.Module.get_function_with_keywords (to_pyobject self) "std"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("ddof", ddof)])

let sum ?axis ?dtype ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sum"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out)])

let swapaxes ~axis1 ~axis2 self =
   Py.Module.get_function_with_keywords (to_pyobject self) "swapaxes"
     [||]
     (Wrap_utils.keyword_args [("axis1", Some(axis1 )); ("axis2", Some(axis2 ))])

let take ?axis ?out ?mode ~indices self =
   Py.Module.get_function_with_keywords (to_pyobject self) "take"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("mode", mode); ("indices", Some(indices ))])

                  let tobytes ?order self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "tobytes"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
| `None -> Py.none
))])

                  let tofile ?sep ?format ~fid self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "tofile"
                       [||]
                       (Wrap_utils.keyword_args [("sep", Wrap_utils.Option.map sep Py.String.of_string); ("format", Wrap_utils.Option.map format Py.String.of_string); ("fid", Some(fid |> (function
| `S x -> Py.String.of_string x
| `PyObject x -> Wrap_utils.id x
)))])

let tolist self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tolist"
     [||]
     []

                  let tostring ?order self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "tostring"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
| `None -> Py.none
))])

let trace ?offset ?axis1 ?axis2 ?dtype ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "trace"
     [||]
     (Wrap_utils.keyword_args [("offset", offset); ("axis1", axis1); ("axis2", axis2); ("dtype", dtype); ("out", out)])

let transpose axes self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id axes)])
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let var ?axis ?dtype ?out ?ddof self =
   Py.Module.get_function_with_keywords (to_pyobject self) "var"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("ddof", ddof)])

                  let view ?dtype ?type_ self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "view"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype (function
| `Ndarray_sub_class x -> Wrap_utils.id x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("type", type_)])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
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
                  let asanyarray ?dtype ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asanyarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("a", Some(a |> Np.Obj.to_pyobject))])

                  let asarray ?dtype ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let aslinearoperator a =
   Py.Module.get_function_with_keywords __wrap_namespace "aslinearoperator"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a ))])

let asmatrix ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "asmatrix"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

let coerce ~x ~y () =
   Py.Module.get_function_with_keywords __wrap_namespace "coerce"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x )); ("y", Some(y ))])

let id x =
   Py.Module.get_function_with_keywords __wrap_namespace "id"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

                  let make_system ~a ~m ~x0 ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "make_system"
                       [||]
                       (Wrap_utils.keyword_args [("A", Some(a )); ("M", Some(m )); ("x0", Some(x0 |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `None -> Py.none
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3))))
                  let zeros ?dtype ?order ~shape () =
                     Py.Module.get_function_with_keywords __wrap_namespace "zeros"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
                  let bicg ?x0 ?tol ?maxiter ?m ?callback ?atol ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "bicg"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("tol", tol); ("maxiter", maxiter); ("M", m); ("callback", callback); ("atol", atol); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let bicgstab ?x0 ?tol ?maxiter ?m ?callback ?atol ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "bicgstab"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("tol", tol); ("maxiter", maxiter); ("M", m); ("callback", callback); ("atol", atol); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let cg ?x0 ?tol ?maxiter ?m ?callback ?atol ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cg"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("tol", tol); ("maxiter", maxiter); ("M", m); ("callback", callback); ("atol", atol); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let cgs ?x0 ?tol ?maxiter ?m ?callback ?atol ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cgs"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("tol", tol); ("maxiter", maxiter); ("M", m); ("callback", callback); ("atol", atol); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let gcrotmk ?x0 ?tol ?maxiter ?m ?callback ?m' ?k ?cu ?discard_C ?truncate ?atol ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "gcrotmk"
                       [||]
                       (Wrap_utils.keyword_args [("x0", Wrap_utils.Option.map x0 Np.Obj.to_pyobject); ("tol", tol); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("M", Wrap_utils.Option.map m (function
| `PyObject x -> Wrap_utils.id x
| `Spmatrix x -> Np.Obj.to_pyobject x
)); ("callback", callback); ("m", Wrap_utils.Option.map m' Py.Int.of_int); ("k", Wrap_utils.Option.map k Py.Int.of_int); ("CU", cu); ("discard_C", Wrap_utils.Option.map discard_C Py.Bool.of_bool); ("truncate", Wrap_utils.Option.map truncate (function
| `Oldest -> Py.String.of_string "oldest"
| `Smallest -> Py.String.of_string "smallest"
)); ("atol", atol); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let gmres ?x0 ?tol ?restart ?maxiter ?m ?callback ?restrt ?atol ?callback_type ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "gmres"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("tol", tol); ("restart", restart); ("maxiter", maxiter); ("M", m); ("callback", callback); ("restrt", restrt); ("atol", atol); ("callback_type", callback_type); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let lgmres ?x0 ?tol ?maxiter ?m ?callback ?inner_m ?outer_k ?outer_v ?store_outer_Av ?prepend_outer_v ?atol ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "lgmres"
                       [||]
                       (Wrap_utils.keyword_args [("x0", Wrap_utils.Option.map x0 Np.Obj.to_pyobject); ("tol", tol); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("M", Wrap_utils.Option.map m (function
| `PyObject x -> Wrap_utils.id x
| `Spmatrix x -> Np.Obj.to_pyobject x
)); ("callback", callback); ("inner_m", Wrap_utils.Option.map inner_m Py.Int.of_int); ("outer_k", Wrap_utils.Option.map outer_k Py.Int.of_int); ("outer_v", outer_v); ("store_outer_Av", Wrap_utils.Option.map store_outer_Av Py.Bool.of_bool); ("prepend_outer_v", Wrap_utils.Option.map prepend_outer_v Py.Bool.of_bool); ("atol", atol); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let lsmr ?damp ?atol ?btol ?conlim ?maxiter ?show ?x0 ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "lsmr"
                       [||]
                       (Wrap_utils.keyword_args [("damp", Wrap_utils.Option.map damp Py.Float.of_float); ("atol", atol); ("btol", btol); ("conlim", Wrap_utils.Option.map conlim Py.Float.of_float); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("show", Wrap_utils.Option.map show Py.Bool.of_bool); ("x0", Wrap_utils.Option.map x0 Np.Obj.to_pyobject); ("A", Some(a |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Py.Float.to_float (Py.Tuple.get x 3)), (Py.Float.to_float (Py.Tuple.get x 4)), (Py.Float.to_float (Py.Tuple.get x 5)), (Py.Float.to_float (Py.Tuple.get x 6)), (Py.Float.to_float (Py.Tuple.get x 7))))
                  let lsqr ?damp ?atol ?btol ?conlim ?iter_lim ?show ?calc_var ?x0 ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "lsqr"
                       [||]
                       (Wrap_utils.keyword_args [("damp", Wrap_utils.Option.map damp Py.Float.of_float); ("atol", atol); ("btol", btol); ("conlim", Wrap_utils.Option.map conlim Py.Float.of_float); ("iter_lim", Wrap_utils.Option.map iter_lim Py.Int.of_int); ("show", Wrap_utils.Option.map show Py.Bool.of_bool); ("calc_var", Wrap_utils.Option.map calc_var Py.Bool.of_bool); ("x0", Wrap_utils.Option.map x0 Np.Obj.to_pyobject); ("A", Some(a |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Py.Float.to_float (Py.Tuple.get x 3)), (Py.Float.to_float (Py.Tuple.get x 4)), (Py.Float.to_float (Py.Tuple.get x 5)), (Py.Float.to_float (Py.Tuple.get x 6)), (Py.Float.to_float (Py.Tuple.get x 7)), (Py.Float.to_float (Py.Tuple.get x 8)), (Wrap_utils.id (Py.Tuple.get x 9))))
                  let minres ?x0 ?shift ?tol ?maxiter ?m ?callback ?show ?check ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "minres"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("shift", shift); ("tol", tol); ("maxiter", maxiter); ("M", m); ("callback", callback); ("show", show); ("check", check); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let qmr ?x0 ?tol ?maxiter ?m1 ?m2 ?callback ?atol ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "qmr"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("tol", tol); ("maxiter", maxiter); ("M1", m1); ("M2", m2); ("callback", callback); ("atol", atol); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))

end
module Iterative = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.linalg.isolve.iterative"

let get_py name = Py.Module.get __wrap_namespace name
                  let bicg ?x0 ?tol ?maxiter ?m ?callback ?atol ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "bicg"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("tol", tol); ("maxiter", maxiter); ("M", m); ("callback", callback); ("atol", atol); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let bicgstab ?x0 ?tol ?maxiter ?m ?callback ?atol ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "bicgstab"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("tol", tol); ("maxiter", maxiter); ("M", m); ("callback", callback); ("atol", atol); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let cg ?x0 ?tol ?maxiter ?m ?callback ?atol ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cg"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("tol", tol); ("maxiter", maxiter); ("M", m); ("callback", callback); ("atol", atol); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let cgs ?x0 ?tol ?maxiter ?m ?callback ?atol ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cgs"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("tol", tol); ("maxiter", maxiter); ("M", m); ("callback", callback); ("atol", atol); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let gmres ?x0 ?tol ?restart ?maxiter ?m ?callback ?restrt ?atol ?callback_type ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "gmres"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("tol", tol); ("restart", restart); ("maxiter", maxiter); ("M", m); ("callback", callback); ("restrt", restrt); ("atol", atol); ("callback_type", callback_type); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let make_system ~a ~m ~x0 ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "make_system"
                       [||]
                       (Wrap_utils.keyword_args [("A", Some(a )); ("M", Some(m )); ("x0", Some(x0 |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `None -> Py.none
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3))))
let non_reentrant ?err_msg () =
   Py.Module.get_function_with_keywords __wrap_namespace "non_reentrant"
     [||]
     (Wrap_utils.keyword_args [("err_msg", err_msg)])

                  let qmr ?x0 ?tol ?maxiter ?m1 ?m2 ?callback ?atol ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "qmr"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("tol", tol); ("maxiter", maxiter); ("M1", m1); ("M2", m2); ("callback", callback); ("atol", atol); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
let set_docstring ?footer ?atol_default ~header ~ainfo () =
   Py.Module.get_function_with_keywords __wrap_namespace "set_docstring"
     [||]
     (Wrap_utils.keyword_args [("footer", footer); ("atol_default", atol_default); ("header", Some(header )); ("Ainfo", Some(ainfo ))])


end
module Linsolve = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.linalg.dsolve.linsolve"

let get_py name = Py.Module.get __wrap_namespace name
                  let asarray ?dtype ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let factorized a =
   Py.Module.get_function_with_keywords __wrap_namespace "factorized"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject))])

let is_pydata_spmatrix m =
   Py.Module.get_function_with_keywords __wrap_namespace "is_pydata_spmatrix"
     [||]
     (Wrap_utils.keyword_args [("m", Some(m ))])

let isspmatrix x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isspmatrix_csc x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix_csc"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isspmatrix_csr x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix_csr"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let spilu ?drop_tol ?fill_factor ?drop_rule ?permc_spec ?diag_pivot_thresh ?relax ?panel_size ?options ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "spilu"
     [||]
     (Wrap_utils.keyword_args [("drop_tol", Wrap_utils.Option.map drop_tol Py.Float.of_float); ("fill_factor", Wrap_utils.Option.map fill_factor Py.Float.of_float); ("drop_rule", Wrap_utils.Option.map drop_rule Py.String.of_string); ("permc_spec", permc_spec); ("diag_pivot_thresh", diag_pivot_thresh); ("relax", relax); ("panel_size", panel_size); ("options", options); ("A", Some(a |> Np.Obj.to_pyobject))])

let splu ?permc_spec ?diag_pivot_thresh ?relax ?panel_size ?options ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "splu"
     [||]
     (Wrap_utils.keyword_args [("permc_spec", Wrap_utils.Option.map permc_spec Py.String.of_string); ("diag_pivot_thresh", Wrap_utils.Option.map diag_pivot_thresh Py.Float.of_float); ("relax", Wrap_utils.Option.map relax Py.Int.of_int); ("panel_size", Wrap_utils.Option.map panel_size Py.Int.of_int); ("options", options); ("A", Some(a |> Np.Obj.to_pyobject))])

let spsolve ?permc_spec ?use_umfpack ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "spsolve"
     [||]
     (Wrap_utils.keyword_args [("permc_spec", Wrap_utils.Option.map permc_spec Py.String.of_string); ("use_umfpack", Wrap_utils.Option.map use_umfpack Py.Bool.of_bool); ("A", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let spsolve_triangular ?lower ?overwrite_A ?overwrite_b ?unit_diagonal ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "spsolve_triangular"
     [||]
     (Wrap_utils.keyword_args [("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("overwrite_A", Wrap_utils.Option.map overwrite_A Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("unit_diagonal", Wrap_utils.Option.map unit_diagonal Py.Bool.of_bool); ("A", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b ))])

let use_solver ?kwargs () =
   Py.Module.get_function_with_keywords __wrap_namespace "use_solver"
     [||]
     (match kwargs with None -> [] | Some x -> x)

let warn ?category ?stacklevel ?source ~message () =
   Py.Module.get_function_with_keywords __wrap_namespace "warn"
     [||]
     (Wrap_utils.keyword_args [("category", category); ("stacklevel", stacklevel); ("source", source); ("message", Some(message ))])


end
module Matfuncs = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.linalg.matfuncs"

let get_py name = Py.Module.get __wrap_namespace name
module MatrixPowerOperator = struct
type tag = [`MatrixPowerOperator]
type t = [`MatrixPowerOperator | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "MatrixPowerOperator"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let adjoint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "adjoint"
     [||]
     []

let dot ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let matmat ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matmat"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])

let matvec ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matvec"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let rmatmat ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatmat"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])

let rmatvec ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatvec"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let transpose self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ProductOperator = struct
type tag = [`ProductOperator]
type t = [`Object | `ProductOperator] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "ProductOperator"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let adjoint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "adjoint"
     [||]
     []

let dot ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let matmat ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matmat"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])

let matvec ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matvec"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let rmatmat ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatmat"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])

let rmatvec ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatvec"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let transpose self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let expm a =
   Py.Module.get_function_with_keywords __wrap_namespace "expm"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let inv a =
   Py.Module.get_function_with_keywords __wrap_namespace "inv"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let is_pydata_spmatrix m =
   Py.Module.get_function_with_keywords __wrap_namespace "is_pydata_spmatrix"
     [||]
     (Wrap_utils.keyword_args [("m", Some(m ))])

let isspmatrix x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let solve ?sym_pos ?lower ?overwrite_a ?overwrite_b ?debug ?check_finite ?assume_a ?transposed ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "solve"
     [||]
     (Wrap_utils.keyword_args [("sym_pos", Wrap_utils.Option.map sym_pos Py.Bool.of_bool); ("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("debug", debug); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("assume_a", Wrap_utils.Option.map assume_a Py.String.of_string); ("transposed", Wrap_utils.Option.map transposed Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let solve_triangular ?trans ?lower ?unit_diagonal ?overwrite_b ?debug ?check_finite ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "solve_triangular"
                       [||]
                       (Wrap_utils.keyword_args [("trans", Wrap_utils.Option.map trans (function
| `Zero -> Py.Int.of_int 0
| `N -> Py.String.of_string "N"
| `C -> Py.String.of_string "C"
| `One -> Py.Int.of_int 1
| `Two -> Py.Int.of_int 2
| `T -> Py.String.of_string "T"
)); ("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("unit_diagonal", Wrap_utils.Option.map unit_diagonal Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("debug", debug); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b ))])

let speye ?n ?k ?dtype ?format ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "speye"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("k", Wrap_utils.Option.map k Py.Int.of_int); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("format", Wrap_utils.Option.map format Py.String.of_string); ("m", Some(m |> Py.Int.of_int))])

let spsolve ?permc_spec ?use_umfpack ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "spsolve"
     [||]
     (Wrap_utils.keyword_args [("permc_spec", Wrap_utils.Option.map permc_spec Py.String.of_string); ("use_umfpack", Wrap_utils.Option.map use_umfpack Py.Bool.of_bool); ("A", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

end
module Utils = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.linalg.isolve.utils"

let get_py name = Py.Module.get __wrap_namespace name
module IdentityOperator = struct
type tag = [`IdentityOperator]
type t = [`IdentityOperator | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "IdentityOperator"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let adjoint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "adjoint"
     [||]
     []

let dot ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let matmat ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matmat"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])

let matvec ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "matvec"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let rmatmat ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatmat"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])

let rmatvec ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmatvec"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let transpose self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Matrix = struct
type tag = [`Matrix]
type t = [`ArrayLike | `Matrix | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?dtype ?copy ~data () =
                     Py.Module.get_function_with_keywords __wrap_namespace "matrix"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("data", Some(data |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `S x -> Py.String.of_string x
)))])
                       |> of_pyobject
let __getitem__ ~index self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let __setitem__ ~key ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     (Array.of_list @@ List.concat [[key ];[value ]])
     []

let all ?axis ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "all"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out)])

let any ?axis ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "any"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])

let argmax ?axis ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "argmax"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out)])

let argmin ?axis ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "argmin"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out)])

let argpartition ?axis ?kind ?order ~kth self =
   Py.Module.get_function_with_keywords (to_pyobject self) "argpartition"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("kind", kind); ("order", order); ("kth", Some(kth ))])

let argsort ?axis ?kind ?order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "argsort"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("kind", kind); ("order", order)])

                  let astype ?order ?casting ?subok ?copy ~dtype self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "astype"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
| `A -> Py.String.of_string "A"
| `K -> Py.String.of_string "K"
)); ("casting", Wrap_utils.Option.map casting (function
| `No -> Py.String.of_string "no"
| `Equiv -> Py.String.of_string "equiv"
| `Safe -> Py.String.of_string "safe"
| `Same_kind -> Py.String.of_string "same_kind"
| `Unsafe -> Py.String.of_string "unsafe"
)); ("subok", subok); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("dtype", Some(dtype |> (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let byteswap ?inplace self =
   Py.Module.get_function_with_keywords (to_pyobject self) "byteswap"
     [||]
     (Wrap_utils.keyword_args [("inplace", Wrap_utils.Option.map inplace Py.Bool.of_bool)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let choose ?out ?mode ~choices self =
   Py.Module.get_function_with_keywords (to_pyobject self) "choose"
     [||]
     (Wrap_utils.keyword_args [("out", out); ("mode", mode); ("choices", Some(choices ))])

let clip ?min ?max ?out ?kwargs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "clip"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("min", min); ("max", max); ("out", out)]) (match kwargs with None -> [] | Some x -> x))

let compress ?axis ?out ~condition self =
   Py.Module.get_function_with_keywords (to_pyobject self) "compress"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("condition", Some(condition ))])

let conj self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conj"
     [||]
     []

let conjugate self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conjugate"
     [||]
     []

                  let copy ?order self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "copy"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
| `A -> Py.String.of_string "A"
| `K -> Py.String.of_string "K"
))])

let cumprod ?axis ?dtype ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cumprod"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out)])

let cumsum ?axis ?dtype ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cumsum"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out)])

let diagonal ?offset ?axis1 ?axis2 self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diagonal"
     [||]
     (Wrap_utils.keyword_args [("offset", offset); ("axis1", axis1); ("axis2", axis2)])

let dot ?out ~b self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("out", out); ("b", Some(b ))])

                  let dump ~file self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "dump"
                       [||]
                       (Wrap_utils.keyword_args [("file", Some(file |> (function
| `S x -> Py.String.of_string x
| `Path x -> Wrap_utils.id x
)))])

let dumps self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dumps"
     [||]
     []

                  let fill ~value self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fill"
                       [||]
                       (Wrap_utils.keyword_args [("value", Some(value |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])

                  let flatten ?order self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "flatten"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
| `A -> Py.String.of_string "A"
| `K -> Py.String.of_string "K"
))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let getA self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getA"
     [||]
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let getA1 self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getA1"
     [||]
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let getH self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getH"
     [||]
     []

let getI self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getI"
     [||]
     []

let getT self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getT"
     [||]
     []

                  let getfield ?offset ~dtype self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "getfield"
                       [||]
                       (Wrap_utils.keyword_args [("offset", Wrap_utils.Option.map offset Py.Int.of_int); ("dtype", Some(dtype |> (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)))])

let item args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "item"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

let itemset args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "itemset"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

let max ?axis ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "max"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out)])

let mean ?axis ?dtype ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "mean"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out)])

let min ?axis ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "min"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out)])

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let nonzero self =
   Py.Module.get_function_with_keywords (to_pyobject self) "nonzero"
     [||]
     []

                  let partition ?axis ?kind ?order ~kth self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "partition"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("kind", Wrap_utils.Option.map kind (function
| `Introselect -> Py.String.of_string "introselect"
)); ("order", Wrap_utils.Option.map order (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `S x -> Py.String.of_string x
)); ("kth", Some(kth |> (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)))])

let prod ?axis ?dtype ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "prod"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out)])

let ptp ?axis ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ptp"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out)])

let put ?mode ~indices ~values self =
   Py.Module.get_function_with_keywords (to_pyobject self) "put"
     [||]
     (Wrap_utils.keyword_args [("mode", mode); ("indices", Some(indices )); ("values", Some(values ))])

                  let ravel ?order self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "ravel"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
| `A -> Py.String.of_string "A"
| `K -> Py.String.of_string "K"
))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let repeat ?axis ~repeats self =
   Py.Module.get_function_with_keywords (to_pyobject self) "repeat"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("repeats", Some(repeats ))])

let reshape ?order ~shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "reshape"
     [||]
     (Wrap_utils.keyword_args [("order", order); ("shape", Some(shape ))])

                  let resize ?refcheck ~new_shape self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "resize"
                       [||]
                       (Wrap_utils.keyword_args [("refcheck", Wrap_utils.Option.map refcheck Py.Bool.of_bool); ("new_shape", Some(new_shape |> (function
| `TupleOfInts x -> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml) x
| `T_n_ints x -> Wrap_utils.id x
)))])

let round ?decimals ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "round"
     [||]
     (Wrap_utils.keyword_args [("decimals", decimals); ("out", out)])

let searchsorted ?side ?sorter ~v self =
   Py.Module.get_function_with_keywords (to_pyobject self) "searchsorted"
     [||]
     (Wrap_utils.keyword_args [("side", side); ("sorter", sorter); ("v", Some(v ))])

let setfield ?offset ~val_ ~dtype self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setfield"
     [||]
     (Wrap_utils.keyword_args [("offset", Wrap_utils.Option.map offset Py.Int.of_int); ("val", Some(val_ )); ("dtype", Some(dtype ))])

let setflags ?write ?align ?uic self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setflags"
     [||]
     (Wrap_utils.keyword_args [("write", Wrap_utils.Option.map write Py.Bool.of_bool); ("align", Wrap_utils.Option.map align Py.Bool.of_bool); ("uic", Wrap_utils.Option.map uic Py.Bool.of_bool)])

                  let sort ?axis ?kind ?order self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "sort"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("kind", Wrap_utils.Option.map kind (function
| `Stable -> Py.String.of_string "stable"
| `Quicksort -> Py.String.of_string "quicksort"
| `Heapsort -> Py.String.of_string "heapsort"
| `Mergesort -> Py.String.of_string "mergesort"
)); ("order", Wrap_utils.Option.map order (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `S x -> Py.String.of_string x
))])

let squeeze ?axis self =
   Py.Module.get_function_with_keywords (to_pyobject self) "squeeze"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let std ?axis ?dtype ?out ?ddof self =
   Py.Module.get_function_with_keywords (to_pyobject self) "std"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("ddof", ddof)])

let sum ?axis ?dtype ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sum"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out)])

let swapaxes ~axis1 ~axis2 self =
   Py.Module.get_function_with_keywords (to_pyobject self) "swapaxes"
     [||]
     (Wrap_utils.keyword_args [("axis1", Some(axis1 )); ("axis2", Some(axis2 ))])

let take ?axis ?out ?mode ~indices self =
   Py.Module.get_function_with_keywords (to_pyobject self) "take"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("mode", mode); ("indices", Some(indices ))])

                  let tobytes ?order self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "tobytes"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
| `None -> Py.none
))])

                  let tofile ?sep ?format ~fid self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "tofile"
                       [||]
                       (Wrap_utils.keyword_args [("sep", Wrap_utils.Option.map sep Py.String.of_string); ("format", Wrap_utils.Option.map format Py.String.of_string); ("fid", Some(fid |> (function
| `S x -> Py.String.of_string x
| `PyObject x -> Wrap_utils.id x
)))])

let tolist self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tolist"
     [||]
     []

                  let tostring ?order self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "tostring"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
| `None -> Py.none
))])

let trace ?offset ?axis1 ?axis2 ?dtype ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "trace"
     [||]
     (Wrap_utils.keyword_args [("offset", offset); ("axis1", axis1); ("axis2", axis2); ("dtype", dtype); ("out", out)])

let transpose axes self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id axes)])
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let var ?axis ?dtype ?out ?ddof self =
   Py.Module.get_function_with_keywords (to_pyobject self) "var"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("ddof", ddof)])

                  let view ?dtype ?type_ self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "view"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype (function
| `Ndarray_sub_class x -> Wrap_utils.id x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("type", type_)])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
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
                  let asanyarray ?dtype ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asanyarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("a", Some(a |> Np.Obj.to_pyobject))])

                  let asarray ?dtype ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let aslinearoperator a =
   Py.Module.get_function_with_keywords __wrap_namespace "aslinearoperator"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a ))])

let asmatrix ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "asmatrix"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

let coerce ~x ~y () =
   Py.Module.get_function_with_keywords __wrap_namespace "coerce"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x )); ("y", Some(y ))])

let id x =
   Py.Module.get_function_with_keywords __wrap_namespace "id"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

                  let make_system ~a ~m ~x0 ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "make_system"
                       [||]
                       (Wrap_utils.keyword_args [("A", Some(a )); ("M", Some(m )); ("x0", Some(x0 |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `None -> Py.none
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3))))
                  let zeros ?dtype ?order ~shape () =
                     Py.Module.get_function_with_keywords __wrap_namespace "zeros"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
let aslinearoperator a =
   Py.Module.get_function_with_keywords __wrap_namespace "aslinearoperator"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a ))])

                  let bicg ?x0 ?tol ?maxiter ?m ?callback ?atol ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "bicg"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("tol", tol); ("maxiter", maxiter); ("M", m); ("callback", callback); ("atol", atol); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let bicgstab ?x0 ?tol ?maxiter ?m ?callback ?atol ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "bicgstab"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("tol", tol); ("maxiter", maxiter); ("M", m); ("callback", callback); ("atol", atol); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let cg ?x0 ?tol ?maxiter ?m ?callback ?atol ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cg"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("tol", tol); ("maxiter", maxiter); ("M", m); ("callback", callback); ("atol", atol); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let cgs ?x0 ?tol ?maxiter ?m ?callback ?atol ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cgs"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("tol", tol); ("maxiter", maxiter); ("M", m); ("callback", callback); ("atol", atol); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let eigs ?k ?m ?sigma ?which ?v0 ?ncv ?maxiter ?tol ?return_eigenvectors ?minv ?oPinv ?oPpart ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "eigs"
                       [||]
                       (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("M", Wrap_utils.Option.map m (function
| `Arr x -> Np.Obj.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
)); ("sigma", sigma); ("which", Wrap_utils.Option.map which (function
| `LM -> Py.String.of_string "LM"
| `SM -> Py.String.of_string "SM"
| `LR -> Py.String.of_string "LR"
| `SR -> Py.String.of_string "SR"
| `LI -> Py.String.of_string "LI"
| `SI -> Py.String.of_string "SI"
)); ("v0", Wrap_utils.Option.map v0 Np.Obj.to_pyobject); ("ncv", Wrap_utils.Option.map ncv Py.Int.of_int); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("return_eigenvectors", Wrap_utils.Option.map return_eigenvectors Py.Bool.of_bool); ("Minv", Wrap_utils.Option.map minv (function
| `Arr x -> Np.Obj.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
)); ("OPinv", Wrap_utils.Option.map oPinv (function
| `Arr x -> Np.Obj.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
)); ("OPpart", oPpart); ("A", Some(a |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let eigsh ?k ?m ?sigma ?which ?v0 ?ncv ?maxiter ?tol ?return_eigenvectors ?minv ?oPinv ?mode ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "eigsh"
                       [||]
                       (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("M", m); ("sigma", sigma); ("which", which); ("v0", v0); ("ncv", ncv); ("maxiter", maxiter); ("tol", tol); ("return_eigenvectors", return_eigenvectors); ("Minv", minv); ("OPinv", oPinv); ("mode", mode); ("A", Some(a |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let expm a =
   Py.Module.get_function_with_keywords __wrap_namespace "expm"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let expm_multiply ?start ?stop ?num ?endpoint ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "expm_multiply"
                       [||]
                       (Wrap_utils.keyword_args [("start", Wrap_utils.Option.map start (function
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("stop", Wrap_utils.Option.map stop (function
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("num", Wrap_utils.Option.map num Py.Int.of_int); ("endpoint", Wrap_utils.Option.map endpoint Py.Bool.of_bool); ("A", Some(a )); ("B", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let factorized a =
   Py.Module.get_function_with_keywords __wrap_namespace "factorized"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject))])

                  let gcrotmk ?x0 ?tol ?maxiter ?m ?callback ?m' ?k ?cu ?discard_C ?truncate ?atol ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "gcrotmk"
                       [||]
                       (Wrap_utils.keyword_args [("x0", Wrap_utils.Option.map x0 Np.Obj.to_pyobject); ("tol", tol); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("M", Wrap_utils.Option.map m (function
| `PyObject x -> Wrap_utils.id x
| `Spmatrix x -> Np.Obj.to_pyobject x
)); ("callback", callback); ("m", Wrap_utils.Option.map m' Py.Int.of_int); ("k", Wrap_utils.Option.map k Py.Int.of_int); ("CU", cu); ("discard_C", Wrap_utils.Option.map discard_C Py.Bool.of_bool); ("truncate", Wrap_utils.Option.map truncate (function
| `Oldest -> Py.String.of_string "oldest"
| `Smallest -> Py.String.of_string "smallest"
)); ("atol", atol); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let gmres ?x0 ?tol ?restart ?maxiter ?m ?callback ?restrt ?atol ?callback_type ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "gmres"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("tol", tol); ("restart", restart); ("maxiter", maxiter); ("M", m); ("callback", callback); ("restrt", restrt); ("atol", atol); ("callback_type", callback_type); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
let inv a =
   Py.Module.get_function_with_keywords __wrap_namespace "inv"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let lgmres ?x0 ?tol ?maxiter ?m ?callback ?inner_m ?outer_k ?outer_v ?store_outer_Av ?prepend_outer_v ?atol ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "lgmres"
                       [||]
                       (Wrap_utils.keyword_args [("x0", Wrap_utils.Option.map x0 Np.Obj.to_pyobject); ("tol", tol); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("M", Wrap_utils.Option.map m (function
| `PyObject x -> Wrap_utils.id x
| `Spmatrix x -> Np.Obj.to_pyobject x
)); ("callback", callback); ("inner_m", Wrap_utils.Option.map inner_m Py.Int.of_int); ("outer_k", Wrap_utils.Option.map outer_k Py.Int.of_int); ("outer_v", outer_v); ("store_outer_Av", Wrap_utils.Option.map store_outer_Av Py.Bool.of_bool); ("prepend_outer_v", Wrap_utils.Option.map prepend_outer_v Py.Bool.of_bool); ("atol", atol); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let lobpcg ?b ?m ?y ?tol ?maxiter ?largest ?verbosityLevel ?retLambdaHistory ?retResidualNormsHistory ~a ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "lobpcg"
                       [||]
                       (Wrap_utils.keyword_args [("B", Wrap_utils.Option.map b (function
| `PyObject x -> Wrap_utils.id x
| `Spmatrix x -> Np.Obj.to_pyobject x
)); ("M", Wrap_utils.Option.map m (function
| `PyObject x -> Wrap_utils.id x
| `Spmatrix x -> Np.Obj.to_pyobject x
)); ("Y", Wrap_utils.Option.map y (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("tol", Wrap_utils.Option.map tol (function
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("largest", Wrap_utils.Option.map largest Py.Bool.of_bool); ("verbosityLevel", Wrap_utils.Option.map verbosityLevel Py.Int.of_int); ("retLambdaHistory", Wrap_utils.Option.map retLambdaHistory Py.Bool.of_bool); ("retResidualNormsHistory", Wrap_utils.Option.map retResidualNormsHistory Py.Bool.of_bool); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("X", Some(x |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3))))
                  let lsmr ?damp ?atol ?btol ?conlim ?maxiter ?show ?x0 ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "lsmr"
                       [||]
                       (Wrap_utils.keyword_args [("damp", Wrap_utils.Option.map damp Py.Float.of_float); ("atol", atol); ("btol", btol); ("conlim", Wrap_utils.Option.map conlim Py.Float.of_float); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("show", Wrap_utils.Option.map show Py.Bool.of_bool); ("x0", Wrap_utils.Option.map x0 Np.Obj.to_pyobject); ("A", Some(a |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Py.Float.to_float (Py.Tuple.get x 3)), (Py.Float.to_float (Py.Tuple.get x 4)), (Py.Float.to_float (Py.Tuple.get x 5)), (Py.Float.to_float (Py.Tuple.get x 6)), (Py.Float.to_float (Py.Tuple.get x 7))))
                  let lsqr ?damp ?atol ?btol ?conlim ?iter_lim ?show ?calc_var ?x0 ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "lsqr"
                       [||]
                       (Wrap_utils.keyword_args [("damp", Wrap_utils.Option.map damp Py.Float.of_float); ("atol", atol); ("btol", btol); ("conlim", Wrap_utils.Option.map conlim Py.Float.of_float); ("iter_lim", Wrap_utils.Option.map iter_lim Py.Int.of_int); ("show", Wrap_utils.Option.map show Py.Bool.of_bool); ("calc_var", Wrap_utils.Option.map calc_var Py.Bool.of_bool); ("x0", Wrap_utils.Option.map x0 Np.Obj.to_pyobject); ("A", Some(a |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Py.Float.to_float (Py.Tuple.get x 3)), (Py.Float.to_float (Py.Tuple.get x 4)), (Py.Float.to_float (Py.Tuple.get x 5)), (Py.Float.to_float (Py.Tuple.get x 6)), (Py.Float.to_float (Py.Tuple.get x 7)), (Py.Float.to_float (Py.Tuple.get x 8)), (Wrap_utils.id (Py.Tuple.get x 9))))
                  let minres ?x0 ?shift ?tol ?maxiter ?m ?callback ?show ?check ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "minres"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("shift", shift); ("tol", tol); ("maxiter", maxiter); ("M", m); ("callback", callback); ("show", show); ("check", check); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let norm ?ord ?axis ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "norm"
                       [||]
                       (Wrap_utils.keyword_args [("ord", Wrap_utils.Option.map ord (function
| `Fro -> Py.String.of_string "fro"
| `PyObject x -> Wrap_utils.id x
)); ("axis", Wrap_utils.Option.map axis (function
| `T2_tuple_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("x", Some(x ))])

                  let onenormest ?t ?itmax ?compute_v ?compute_w ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "onenormest"
                       [||]
                       (Wrap_utils.keyword_args [("t", Wrap_utils.Option.map t Py.Int.of_int); ("itmax", Wrap_utils.Option.map itmax Py.Int.of_int); ("compute_v", Wrap_utils.Option.map compute_v Py.Bool.of_bool); ("compute_w", Wrap_utils.Option.map compute_w Py.Bool.of_bool); ("A", Some(a |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Other_linear_operator x -> Wrap_utils.id x
)))])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
                  let qmr ?x0 ?tol ?maxiter ?m1 ?m2 ?callback ?atol ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "qmr"
                       [||]
                       (Wrap_utils.keyword_args [("x0", x0); ("tol", tol); ("maxiter", maxiter); ("M1", m1); ("M2", m2); ("callback", callback); ("atol", atol); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
let spilu ?drop_tol ?fill_factor ?drop_rule ?permc_spec ?diag_pivot_thresh ?relax ?panel_size ?options ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "spilu"
     [||]
     (Wrap_utils.keyword_args [("drop_tol", Wrap_utils.Option.map drop_tol Py.Float.of_float); ("fill_factor", Wrap_utils.Option.map fill_factor Py.Float.of_float); ("drop_rule", Wrap_utils.Option.map drop_rule Py.String.of_string); ("permc_spec", permc_spec); ("diag_pivot_thresh", diag_pivot_thresh); ("relax", relax); ("panel_size", panel_size); ("options", options); ("A", Some(a |> Np.Obj.to_pyobject))])

let splu ?permc_spec ?diag_pivot_thresh ?relax ?panel_size ?options ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "splu"
     [||]
     (Wrap_utils.keyword_args [("permc_spec", Wrap_utils.Option.map permc_spec Py.String.of_string); ("diag_pivot_thresh", Wrap_utils.Option.map diag_pivot_thresh Py.Float.of_float); ("relax", Wrap_utils.Option.map relax Py.Int.of_int); ("panel_size", Wrap_utils.Option.map panel_size Py.Int.of_int); ("options", options); ("A", Some(a |> Np.Obj.to_pyobject))])

let spsolve ?permc_spec ?use_umfpack ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "spsolve"
     [||]
     (Wrap_utils.keyword_args [("permc_spec", Wrap_utils.Option.map permc_spec Py.String.of_string); ("use_umfpack", Wrap_utils.Option.map use_umfpack Py.Bool.of_bool); ("A", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let spsolve_triangular ?lower ?overwrite_A ?overwrite_b ?unit_diagonal ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "spsolve_triangular"
     [||]
     (Wrap_utils.keyword_args [("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("overwrite_A", Wrap_utils.Option.map overwrite_A Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("unit_diagonal", Wrap_utils.Option.map unit_diagonal Py.Bool.of_bool); ("A", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b ))])

                  let svds ?k ?ncv ?tol ?which ?v0 ?maxiter ?return_singular_vectors ?solver ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "svds"
                       [||]
                       (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("ncv", Wrap_utils.Option.map ncv Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("which", Wrap_utils.Option.map which (function
| `LM -> Py.String.of_string "LM"
| `SM -> Py.String.of_string "SM"
)); ("v0", Wrap_utils.Option.map v0 Np.Obj.to_pyobject); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("return_singular_vectors", Wrap_utils.Option.map return_singular_vectors (function
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("solver", Wrap_utils.Option.map solver Py.String.of_string); ("A", Some(a |> (function
| `LinearOperator x -> Wrap_utils.id x
| `Spmatrix x -> Np.Obj.to_pyobject x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
let use_solver ?kwargs () =
   Py.Module.get_function_with_keywords __wrap_namespace "use_solver"
     [||]
     (match kwargs with None -> [] | Some x -> x)


end
module Sputils = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.sparse.sputils"

let get_py name = Py.Module.get __wrap_namespace name
let asmatrix ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "asmatrix"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

let bmat ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "bmat"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

let check_reshape_kwargs kwargs =
   Py.Module.get_function_with_keywords __wrap_namespace "check_reshape_kwargs"
     [||]
     (Wrap_utils.keyword_args [("kwargs", Some(kwargs ))])

let check_shape ?current_shape ~args () =
   Py.Module.get_function_with_keywords __wrap_namespace "check_shape"
     [||]
     (Wrap_utils.keyword_args [("current_shape", current_shape); ("args", Some(args ))])

let downcast_intp_index arr =
   Py.Module.get_function_with_keywords __wrap_namespace "downcast_intp_index"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr ))])

let get_index_dtype ?arrays ?maxval ?check_contents () =
   Py.Module.get_function_with_keywords __wrap_namespace "get_index_dtype"
     [||]
     (Wrap_utils.keyword_args [("arrays", arrays); ("maxval", Wrap_utils.Option.map maxval Py.Float.of_float); ("check_contents", Wrap_utils.Option.map check_contents Py.Bool.of_bool)])
     |> Np.Dtype.of_pyobject
let get_sum_dtype dtype =
   Py.Module.get_function_with_keywords __wrap_namespace "get_sum_dtype"
     [||]
     (Wrap_utils.keyword_args [("dtype", Some(dtype ))])

let getdtype ?a ?default ~dtype () =
   Py.Module.get_function_with_keywords __wrap_namespace "getdtype"
     [||]
     (Wrap_utils.keyword_args [("a", a); ("default", default); ("dtype", Some(dtype ))])

let is_pydata_spmatrix m =
   Py.Module.get_function_with_keywords __wrap_namespace "is_pydata_spmatrix"
     [||]
     (Wrap_utils.keyword_args [("m", Some(m ))])

let isdense x =
   Py.Module.get_function_with_keywords __wrap_namespace "isdense"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isintlike x =
   Py.Module.get_function_with_keywords __wrap_namespace "isintlike"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let ismatrix t =
   Py.Module.get_function_with_keywords __wrap_namespace "ismatrix"
     [||]
     (Wrap_utils.keyword_args [("t", Some(t ))])

let isscalarlike x =
   Py.Module.get_function_with_keywords __wrap_namespace "isscalarlike"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let issequence t =
   Py.Module.get_function_with_keywords __wrap_namespace "issequence"
     [||]
     (Wrap_utils.keyword_args [("t", Some(t ))])

let isshape ?nonneg ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "isshape"
     [||]
     (Wrap_utils.keyword_args [("nonneg", nonneg); ("x", Some(x ))])

let matrix ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "matrix"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

let to_native a =
   Py.Module.get_function_with_keywords __wrap_namespace "to_native"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a ))])

let upcast args =
   Py.Module.get_function_with_keywords __wrap_namespace "upcast"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

let upcast_char args =
   Py.Module.get_function_with_keywords __wrap_namespace "upcast_char"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

let upcast_scalar ~dtype ~scalar () =
   Py.Module.get_function_with_keywords __wrap_namespace "upcast_scalar"
     [||]
     (Wrap_utils.keyword_args [("dtype", Some(dtype )); ("scalar", Some(scalar ))])

let validateaxis axis =
   Py.Module.get_function_with_keywords __wrap_namespace "validateaxis"
     [||]
     (Wrap_utils.keyword_args [("axis", Some(axis ))])


end
let block_diag ?format ?dtype ~mats () =
   Py.Module.get_function_with_keywords __wrap_namespace "block_diag"
     [||]
     (Wrap_utils.keyword_args [("format", Wrap_utils.Option.map format Py.String.of_string); ("dtype", dtype); ("mats", Some(mats ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t))
                  let bmat ?format ?dtype ~blocks () =
                     Py.Module.get_function_with_keywords __wrap_namespace "bmat"
                       [||]
                       (Wrap_utils.keyword_args [("format", Wrap_utils.Option.map format (function
| `Lil -> Py.String.of_string "lil"
| `Bsr -> Py.String.of_string "bsr"
| `Csr -> Py.String.of_string "csr"
| `Csc -> Py.String.of_string "csc"
| `Coo -> Py.String.of_string "coo"
| `Dia -> Py.String.of_string "dia"
| `Dok -> Py.String.of_string "dok"
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("blocks", Some(blocks |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t))
                  let diags ?offsets ?shape ?format ?dtype ~diagonals () =
                     Py.Module.get_function_with_keywords __wrap_namespace "diags"
                       [||]
                       (Wrap_utils.keyword_args [("offsets", offsets); ("shape", shape); ("format", Wrap_utils.Option.map format (function
| `Lil -> Py.String.of_string "lil"
| `Csr -> Py.String.of_string "csr"
| `Csc -> Py.String.of_string "csc"
| `Dia -> Py.String.of_string "dia"
| `T x -> Wrap_utils.id x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("diagonals", Some(diagonals ))])

let eye ?n ?k ?dtype ?format ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "eye"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("k", Wrap_utils.Option.map k Py.Int.of_int); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("format", Wrap_utils.Option.map format Py.String.of_string); ("m", Some(m |> Py.Int.of_int))])

                  let find a =
                     Py.Module.get_function_with_keywords __wrap_namespace "find"
                       [||]
                       (Wrap_utils.keyword_args [("A", Some(a |> (function
| `Dense x -> Wrap_utils.id x
| `Spmatrix x -> Np.Obj.to_pyobject x
)))])

let hstack ?format ?dtype ~blocks () =
   Py.Module.get_function_with_keywords __wrap_namespace "hstack"
     [||]
     (Wrap_utils.keyword_args [("format", Wrap_utils.Option.map format Py.String.of_string); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("blocks", Some(blocks ))])

let identity ?dtype ?format ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "identity"
     [||]
     (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("format", Wrap_utils.Option.map format Py.String.of_string); ("n", Some(n |> Py.Int.of_int))])

let issparse x =
   Py.Module.get_function_with_keywords __wrap_namespace "issparse"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isspmatrix x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isspmatrix_bsr x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix_bsr"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isspmatrix_coo x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix_coo"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isspmatrix_csc x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix_csc"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isspmatrix_csr x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix_csr"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isspmatrix_dia x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix_dia"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isspmatrix_dok x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix_dok"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isspmatrix_lil x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix_lil"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let kron ?format ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "kron"
     [||]
     (Wrap_utils.keyword_args [("format", Wrap_utils.Option.map format Py.String.of_string); ("A", Some(a )); ("B", Some(b ))])

let kronsum ?format ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "kronsum"
     [||]
     (Wrap_utils.keyword_args [("format", Wrap_utils.Option.map format Py.String.of_string); ("A", Some(a )); ("B", Some(b ))])

                  let load_npz file =
                     Py.Module.get_function_with_keywords __wrap_namespace "load_npz"
                       [||]
                       (Wrap_utils.keyword_args [("file", Some(file |> (function
| `File_like_object x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))])

                  let rand ?density ?format ?dtype ?random_state ~m ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "rand"
                       [||]
                       (Wrap_utils.keyword_args [("density", density); ("format", Wrap_utils.Option.map format Py.String.of_string); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("random_state", Wrap_utils.Option.map random_state (function
| `I x -> Py.Int.of_int x
| `Numpy_random_RandomState x -> Wrap_utils.id x
)); ("m", Some(m )); ("n", Some(n ))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t))
                  let random ?density ?format ?dtype ?random_state ?data_rvs ~m ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "random"
                       [||]
                       (Wrap_utils.keyword_args [("density", density); ("format", Wrap_utils.Option.map format Py.String.of_string); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("random_state", Wrap_utils.Option.map random_state (function
| `I x -> Py.Int.of_int x
| `Numpy_random_RandomState x -> Wrap_utils.id x
)); ("data_rvs", data_rvs); ("m", Some(m )); ("n", Some(n ))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t))
                  let save_npz ?compressed ~file ~matrix () =
                     Py.Module.get_function_with_keywords __wrap_namespace "save_npz"
                       [||]
                       (Wrap_utils.keyword_args [("compressed", Wrap_utils.Option.map compressed Py.Bool.of_bool); ("file", Some(file |> (function
| `File_like_object x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
))); ("matrix", Some(matrix ))])

let spdiags ?format ~data ~diags ~m ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "spdiags"
     [||]
     (Wrap_utils.keyword_args [("format", Wrap_utils.Option.map format Py.String.of_string); ("data", Some(data |> Np.Obj.to_pyobject)); ("diags", Some(diags )); ("m", Some(m )); ("n", Some(n ))])

                  let tril ?k ?format ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "tril"
                       [||]
                       (Wrap_utils.keyword_args [("k", k); ("format", Wrap_utils.Option.map format Py.String.of_string); ("A", Some(a |> (function
| `Dense x -> Wrap_utils.id x
| `Spmatrix x -> Np.Obj.to_pyobject x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t))
                  let triu ?k ?format ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "triu"
                       [||]
                       (Wrap_utils.keyword_args [("k", k); ("format", Wrap_utils.Option.map format Py.String.of_string); ("A", Some(a |> (function
| `Dense x -> Wrap_utils.id x
| `Spmatrix x -> Np.Obj.to_pyobject x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t))
let vstack ?format ?dtype ~blocks () =
   Py.Module.get_function_with_keywords __wrap_namespace "vstack"
     [||]
     (Wrap_utils.keyword_args [("format", Wrap_utils.Option.map format Py.String.of_string); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("blocks", Some(blocks ))])

