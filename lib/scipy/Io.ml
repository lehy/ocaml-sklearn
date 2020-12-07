let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.io"

let get_py name = Py.Module.get __wrap_namespace name
module FortranEOFError = struct
type tag = [`FortranEOFError]
type t = [`BaseException | `FortranEOFError | `Object] Obj.t
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
module FortranFile = struct
type tag = [`FortranFile]
type t = [`FortranFile | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?mode ?header_dtype ~filename () =
                     Py.Module.get_function_with_keywords __wrap_namespace "FortranFile"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `R -> Py.String.of_string "r"
| `W -> Py.String.of_string "w"
)); ("header_dtype", Wrap_utils.Option.map header_dtype Np.Dtype.to_pyobject); ("filename", Some(filename |> (function
| `File x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))])
                       |> of_pyobject
let close self =
   Py.Module.get_function_with_keywords (to_pyobject self) "close"
     [||]
     []

let read_ints ?dtype self =
   Py.Module.get_function_with_keywords (to_pyobject self) "read_ints"
     [||]
     (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let read_reals ?dtype self =
   Py.Module.get_function_with_keywords (to_pyobject self) "read_reals"
     [||]
     (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let read_record ?kwargs dtypes self =
   Py.Module.get_function_with_keywords (to_pyobject self) "read_record"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id dtypes)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let write_record items self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write_record"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id items)])
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module FortranFormattingError = struct
type tag = [`FortranFormattingError]
type t = [`BaseException | `FortranFormattingError | `Object] Obj.t
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
module Netcdf_file = struct
type tag = [`Netcdf_file]
type t = [`Netcdf_file | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?mode ?mmap ?version ?maskandscale ~filename () =
                     Py.Module.get_function_with_keywords __wrap_namespace "netcdf_file"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `R -> Py.String.of_string "r"
| `W -> Py.String.of_string "w"
| `A -> Py.String.of_string "a"
)); ("mmap", Wrap_utils.Option.map mmap Py.Bool.of_bool); ("version", Wrap_utils.Option.map version (function
| `Two -> Py.Int.of_int 2
| `One -> Py.Int.of_int 1
)); ("maskandscale", Wrap_utils.Option.map maskandscale Py.Bool.of_bool); ("filename", Some(filename |> (function
| `File_like x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))])
                       |> of_pyobject
let close self =
   Py.Module.get_function_with_keywords (to_pyobject self) "close"
     [||]
     []

let createDimension ~name ~length self =
   Py.Module.get_function_with_keywords (to_pyobject self) "createDimension"
     [||]
     (Wrap_utils.keyword_args [("name", Some(name |> Py.String.of_string)); ("length", Some(length |> Py.Int.of_int))])

                  let createVariable ~name ~type_ ~dimensions self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "createVariable"
                       [||]
                       (Wrap_utils.keyword_args [("name", Some(name |> Py.String.of_string)); ("type", Some(type_ |> (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
))); ("dimensions", Some(dimensions ))])

let flush self =
   Py.Module.get_function_with_keywords (to_pyobject self) "flush"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Netcdf_variable = struct
type tag = [`Netcdf_variable]
type t = [`Netcdf_variable | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?attributes ?maskandscale ~data ~typecode ~size ~shape ~dimensions () =
   Py.Module.get_function_with_keywords __wrap_namespace "netcdf_variable"
     [||]
     (Wrap_utils.keyword_args [("attributes", attributes); ("maskandscale", Wrap_utils.Option.map maskandscale Py.Bool.of_bool); ("data", Some(data |> Np.Obj.to_pyobject)); ("typecode", Some(typecode )); ("size", Some(size |> Py.Int.of_int)); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml))); ("dimensions", Some(dimensions ))])
     |> of_pyobject
let __getitem__ ~index self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let __setitem__ ~index ~data self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index )); ("data", Some(data ))])

                  let assignValue ~value self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "assignValue"
                       [||]
                       (Wrap_utils.keyword_args [("value", Some(value |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])

let getValue self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getValue"
     [||]
     []

let itemsize self =
   Py.Module.get_function_with_keywords (to_pyobject self) "itemsize"
     [||]
     []
     |> Py.Int.to_int
let typecode self =
   Py.Module.get_function_with_keywords (to_pyobject self) "typecode"
     [||]
     []


let dimensions_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dimensions" with
  | None -> failwith "attribute dimensions not found"
  | Some x -> if Py.is_none x then None else Some ((Py.List.to_list_map Py.String.to_string) x)

let dimensions self = match dimensions_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Harwell_boeing = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.io.harwell_boeing"

let get_py name = Py.Module.get __wrap_namespace name
module HBFile = struct
type tag = [`HBFile]
type t = [`HBFile | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?hb_info ~file () =
   Py.Module.get_function_with_keywords __wrap_namespace "HBFile"
     [||]
     (Wrap_utils.keyword_args [("hb_info", hb_info); ("file", Some(file ))])
     |> of_pyobject
let read_matrix self =
   Py.Module.get_function_with_keywords (to_pyobject self) "read_matrix"
     [||]
     []

let write_matrix ~m self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write_matrix"
     [||]
     (Wrap_utils.keyword_args [("m", Some(m ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module HBInfo = struct
type tag = [`HBInfo]
type t = [`HBInfo | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?right_hand_sides_nlines ?nelementals ~title ~key ~total_nlines ~pointer_nlines ~indices_nlines ~values_nlines ~mxtype ~nrows ~ncols ~nnon_zeros ~pointer_format_str ~indices_format_str ~values_format_str () =
   Py.Module.get_function_with_keywords __wrap_namespace "HBInfo"
     [||]
     (Wrap_utils.keyword_args [("right_hand_sides_nlines", right_hand_sides_nlines); ("nelementals", nelementals); ("title", Some(title )); ("key", Some(key )); ("total_nlines", Some(total_nlines )); ("pointer_nlines", Some(pointer_nlines )); ("indices_nlines", Some(indices_nlines )); ("values_nlines", Some(values_nlines )); ("mxtype", Some(mxtype )); ("nrows", Some(nrows )); ("ncols", Some(ncols )); ("nnon_zeros", Some(nnon_zeros )); ("pointer_format_str", Some(pointer_format_str )); ("indices_format_str", Some(indices_format_str )); ("values_format_str", Some(values_format_str ))])
     |> of_pyobject
let dump self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dump"
     [||]
     []

let from_data ?title ?key ?mxtype ?fmt ~m self =
   Py.Module.get_function_with_keywords (to_pyobject self) "from_data"
     [||]
     (Wrap_utils.keyword_args [("title", Wrap_utils.Option.map title Py.String.of_string); ("key", Wrap_utils.Option.map key Py.String.of_string); ("mxtype", mxtype); ("fmt", fmt); ("m", Some(m |> Np.Obj.to_pyobject))])

let from_file ~fid self =
   Py.Module.get_function_with_keywords (to_pyobject self) "from_file"
     [||]
     (Wrap_utils.keyword_args [("fid", Some(fid ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module HBMatrixType = struct
type tag = [`HBMatrixType]
type t = [`HBMatrixType | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?storage ~value_type ~structure () =
   Py.Module.get_function_with_keywords __wrap_namespace "HBMatrixType"
     [||]
     (Wrap_utils.keyword_args [("storage", storage); ("value_type", Some(value_type )); ("structure", Some(structure ))])
     |> of_pyobject
let from_fortran ~fmt self =
   Py.Module.get_function_with_keywords (to_pyobject self) "from_fortran"
     [||]
     (Wrap_utils.keyword_args [("fmt", Some(fmt ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MalformedHeader = struct
type tag = [`MalformedHeader]
type t = [`BaseException | `MalformedHeader | `Object] Obj.t
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
module Hb = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.io.harwell_boeing.hb"

let get_py name = Py.Module.get __wrap_namespace name
module ExpFormat = struct
type tag = [`ExpFormat]
type t = [`ExpFormat | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?min ?repeat ~width ~significand () =
   Py.Module.get_function_with_keywords __wrap_namespace "ExpFormat"
     [||]
     (Wrap_utils.keyword_args [("min", min); ("repeat", repeat); ("width", Some(width )); ("significand", Some(significand ))])
     |> of_pyobject
let from_number ?min ~n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "from_number"
     [||]
     (Wrap_utils.keyword_args [("min", Wrap_utils.Option.map min Py.Int.of_int); ("n", Some(n |> Py.Float.of_float))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module FortranFormatParser = struct
type tag = [`FortranFormatParser]
type t = [`FortranFormatParser | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "FortranFormatParser"
     [||]
     []
     |> of_pyobject
let parse ~s self =
   Py.Module.get_function_with_keywords (to_pyobject self) "parse"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module IntFormat = struct
type tag = [`IntFormat]
type t = [`IntFormat | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?min ?repeat ~width () =
   Py.Module.get_function_with_keywords __wrap_namespace "IntFormat"
     [||]
     (Wrap_utils.keyword_args [("min", min); ("repeat", repeat); ("width", Some(width ))])
     |> of_pyobject
let from_number ?min ~n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "from_number"
     [||]
     (Wrap_utils.keyword_args [("min", Wrap_utils.Option.map min Py.Int.of_int); ("n", Some(n |> Py.Int.of_int))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LineOverflow = struct
type tag = [`LineOverflow]
type t = [`BaseException | `LineOverflow | `Object] Obj.t
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
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

                  let argmin ?axis ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "argmin"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
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
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
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
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
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
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
| `PyObject x -> Wrap_utils.id x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let min ?axis ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "min"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
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
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
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
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `T2_D x -> Wrap_utils.id x
))])

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
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `T2_D x -> Wrap_utils.id x
))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
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
let hb_read path_or_open_file =
   Py.Module.get_function_with_keywords __wrap_namespace "hb_read"
     [||]
     (Wrap_utils.keyword_args [("path_or_open_file", Some(path_or_open_file ))])

let hb_write ?hb_info ~path_or_open_file ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "hb_write"
     [||]
     (Wrap_utils.keyword_args [("hb_info", hb_info); ("path_or_open_file", Some(path_or_open_file )); ("m", Some(m |> Np.Obj.to_pyobject))])


end
let hb_read path_or_open_file =
   Py.Module.get_function_with_keywords __wrap_namespace "hb_read"
     [||]
     (Wrap_utils.keyword_args [("path_or_open_file", Some(path_or_open_file ))])

let hb_write ?hb_info ~path_or_open_file ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "hb_write"
     [||]
     (Wrap_utils.keyword_args [("hb_info", hb_info); ("path_or_open_file", Some(path_or_open_file )); ("m", Some(m |> Np.Obj.to_pyobject))])


end
module Idl = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.io.idl"

let get_py name = Py.Module.get __wrap_namespace name
module AttrDict = struct
type tag = [`AttrDict]
type t = [`AttrDict | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?init () =
   Py.Module.get_function_with_keywords __wrap_namespace "AttrDict"
     [||]
     (Wrap_utils.keyword_args [("init", init)])
     |> of_pyobject
let __getitem__ ~name self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("name", Some(name ))])

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let __setitem__ ~key ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key )); ("value", Some(value ))])

let fromkeys ?value ~iterable self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fromkeys"
     (Array.of_list @@ List.concat [(match value with None -> [] | Some x -> [x ]);[iterable ]])
     []

let get ?default ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get"
     (Array.of_list @@ List.concat [(match default with None -> [] | Some x -> [x ]);[key ]])
     []

let pop ?d ~k self =
   Py.Module.get_function_with_keywords (to_pyobject self) "pop"
     [||]
     (Wrap_utils.keyword_args [("d", d); ("k", Some(k ))])

let popitem self =
   Py.Module.get_function_with_keywords (to_pyobject self) "popitem"
     [||]
     []

let setdefault ?default ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setdefault"
     (Array.of_list @@ List.concat [(match default with None -> [] | Some x -> [x ]);[key ]])
     []

let update ?e ?f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "update"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("E", e)]) (match f with None -> [] | Some x -> x))

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ObjectPointer = struct
type tag = [`ObjectPointer]
type t = [`Object | `ObjectPointer] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create index =
   Py.Module.get_function_with_keywords __wrap_namespace "ObjectPointer"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Pointer = struct
type tag = [`Pointer]
type t = [`Object | `Pointer] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create index =
   Py.Module.get_function_with_keywords __wrap_namespace "Pointer"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let asstr s =
   Py.Module.get_function_with_keywords __wrap_namespace "asstr"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s ))])

let readsav ?idict ?python_dict ?uncompressed_file_name ?verbose ~file_name () =
   Py.Module.get_function_with_keywords __wrap_namespace "readsav"
     [||]
     (Wrap_utils.keyword_args [("idict", idict); ("python_dict", Wrap_utils.Option.map python_dict Py.Bool.of_bool); ("uncompressed_file_name", Wrap_utils.Option.map uncompressed_file_name Py.String.of_string); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("file_name", Some(file_name |> Py.String.of_string))])


end
module Matlab = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.io.matlab"

let get_py name = Py.Module.get __wrap_namespace name
module Mio = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.io.matlab.mio"

let get_py name = Py.Module.get __wrap_namespace name
module MatFile4Reader = struct
type tag = [`MatFile4Reader]
type t = [`MatFile4Reader | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs ~mat_stream args =
   Py.Module.get_function_with_keywords __wrap_namespace "MatFile4Reader"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("mat_stream", Some(mat_stream ))]) (match kwargs with None -> [] | Some x -> x))
     |> of_pyobject
let end_of_stream self =
   Py.Module.get_function_with_keywords (to_pyobject self) "end_of_stream"
     [||]
     []

                  let get_variables ?variable_names self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "get_variables"
                       [||]
                       (Wrap_utils.keyword_args [("variable_names", Wrap_utils.Option.map variable_names (function
| `Sequence_of_str x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
))])

let guess_byte_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "guess_byte_order"
     [||]
     []

let initialize_read self =
   Py.Module.get_function_with_keywords (to_pyobject self) "initialize_read"
     [||]
     []

let list_variables self =
   Py.Module.get_function_with_keywords (to_pyobject self) "list_variables"
     [||]
     []

let read_var_array ?process ~header self =
   Py.Module.get_function_with_keywords (to_pyobject self) "read_var_array"
     [||]
     (Wrap_utils.keyword_args [("process", Wrap_utils.Option.map process Py.Bool.of_bool); ("header", Some(header ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let read_var_header self =
   Py.Module.get_function_with_keywords (to_pyobject self) "read_var_header"
     [||]
     []
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
let set_matlab_compatible self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_matlab_compatible"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MatFile4Writer = struct
type tag = [`MatFile4Writer]
type t = [`MatFile4Writer | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?oned_as ~file_stream () =
   Py.Module.get_function_with_keywords __wrap_namespace "MatFile4Writer"
     [||]
     (Wrap_utils.keyword_args [("oned_as", oned_as); ("file_stream", Some(file_stream ))])
     |> of_pyobject
let put_variables ?write_header ~mdict self =
   Py.Module.get_function_with_keywords (to_pyobject self) "put_variables"
     [||]
     (Wrap_utils.keyword_args [("write_header", Wrap_utils.Option.map write_header Py.Bool.of_bool); ("mdict", Some(mdict ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MatFile5Reader = struct
type tag = [`MatFile5Reader]
type t = [`MatFile5Reader | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?byte_order ?mat_dtype ?squeeze_me ?chars_as_strings ?matlab_compatible ?struct_as_record ?verify_compressed_data_integrity ?uint16_codec ?simplify_cells ~mat_stream () =
   Py.Module.get_function_with_keywords __wrap_namespace "MatFile5Reader"
     [||]
     (Wrap_utils.keyword_args [("byte_order", byte_order); ("mat_dtype", mat_dtype); ("squeeze_me", squeeze_me); ("chars_as_strings", chars_as_strings); ("matlab_compatible", matlab_compatible); ("struct_as_record", struct_as_record); ("verify_compressed_data_integrity", verify_compressed_data_integrity); ("uint16_codec", uint16_codec); ("simplify_cells", simplify_cells); ("mat_stream", Some(mat_stream ))])
     |> of_pyobject
let end_of_stream self =
   Py.Module.get_function_with_keywords (to_pyobject self) "end_of_stream"
     [||]
     []

let get_variables ?variable_names self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_variables"
     [||]
     (Wrap_utils.keyword_args [("variable_names", variable_names)])

let guess_byte_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "guess_byte_order"
     [||]
     []

let initialize_read self =
   Py.Module.get_function_with_keywords (to_pyobject self) "initialize_read"
     [||]
     []

let list_variables self =
   Py.Module.get_function_with_keywords (to_pyobject self) "list_variables"
     [||]
     []

let read_file_header self =
   Py.Module.get_function_with_keywords (to_pyobject self) "read_file_header"
     [||]
     []

let read_var_array ?process ~header self =
   Py.Module.get_function_with_keywords (to_pyobject self) "read_var_array"
     [||]
     (Wrap_utils.keyword_args [("process", process); ("header", Some(header ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let read_var_header self =
   Py.Module.get_function_with_keywords (to_pyobject self) "read_var_header"
     [||]
     []
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
let set_matlab_compatible self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_matlab_compatible"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MatFile5Writer = struct
type tag = [`MatFile5Writer]
type t = [`MatFile5Writer | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?do_compression ?unicode_strings ?global_vars ?long_field_names ?oned_as ~file_stream () =
   Py.Module.get_function_with_keywords __wrap_namespace "MatFile5Writer"
     [||]
     (Wrap_utils.keyword_args [("do_compression", do_compression); ("unicode_strings", unicode_strings); ("global_vars", global_vars); ("long_field_names", long_field_names); ("oned_as", oned_as); ("file_stream", Some(file_stream ))])
     |> of_pyobject
let put_variables ?write_header ~mdict self =
   Py.Module.get_function_with_keywords (to_pyobject self) "put_variables"
     [||]
     (Wrap_utils.keyword_args [("write_header", Wrap_utils.Option.map write_header Py.Bool.of_bool); ("mdict", Some(mdict ))])

let write_file_header self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write_file_header"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let contextmanager func =
   Py.Module.get_function_with_keywords __wrap_namespace "contextmanager"
     [||]
     (Wrap_utils.keyword_args [("func", Some(func ))])

let docfiller f =
   Py.Module.get_function_with_keywords __wrap_namespace "docfiller"
     [||]
     (Wrap_utils.keyword_args [("f", Some(f ))])

let get_matfile_version fileobj =
   Py.Module.get_function_with_keywords __wrap_namespace "get_matfile_version"
     [||]
     (Wrap_utils.keyword_args [("fileobj", Some(fileobj ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
let loadmat ?mdict ?appendmat ?kwargs ~file_name () =
   Py.Module.get_function_with_keywords __wrap_namespace "loadmat"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("mdict", mdict); ("appendmat", Wrap_utils.Option.map appendmat Py.Bool.of_bool); ("file_name", Some(file_name |> Py.String.of_string))]) (match kwargs with None -> [] | Some x -> x))

let mat_reader_factory ?appendmat ?kwargs ~file_name () =
   Py.Module.get_function_with_keywords __wrap_namespace "mat_reader_factory"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("appendmat", Wrap_utils.Option.map appendmat Py.Bool.of_bool); ("file_name", Some(file_name |> Py.String.of_string))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Py.Bool.to_bool (Py.Tuple.get x 1))))
                  let savemat ?appendmat ?format ?long_field_names ?do_compression ?oned_as ~file_name ~mdict () =
                     Py.Module.get_function_with_keywords __wrap_namespace "savemat"
                       [||]
                       (Wrap_utils.keyword_args [("appendmat", Wrap_utils.Option.map appendmat Py.Bool.of_bool); ("format", Wrap_utils.Option.map format (function
| `T4 -> Py.String.of_string "4"
| `T5 -> Py.String.of_string "5"
)); ("long_field_names", Wrap_utils.Option.map long_field_names Py.Bool.of_bool); ("do_compression", Wrap_utils.Option.map do_compression Py.Bool.of_bool); ("oned_as", Wrap_utils.Option.map oned_as (function
| `Row -> Py.String.of_string "row"
| `Column -> Py.String.of_string "column"
)); ("file_name", Some(file_name |> (function
| `S x -> Py.String.of_string x
| `File_like_object x -> Wrap_utils.id x
))); ("mdict", Some(mdict ))])

let whosmat ?appendmat ?kwargs ~file_name () =
   Py.Module.get_function_with_keywords __wrap_namespace "whosmat"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("appendmat", Wrap_utils.Option.map appendmat Py.Bool.of_bool); ("file_name", Some(file_name |> Py.String.of_string))]) (match kwargs with None -> [] | Some x -> x))


end
module Mio4 = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.io.matlab.mio4"

let get_py name = Py.Module.get __wrap_namespace name
module MatFileReader = struct
type tag = [`MatFileReader]
type t = [`MatFileReader | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?byte_order ?mat_dtype ?squeeze_me ?chars_as_strings ?matlab_compatible ?struct_as_record ?verify_compressed_data_integrity ?simplify_cells ~mat_stream () =
   Py.Module.get_function_with_keywords __wrap_namespace "MatFileReader"
     [||]
     (Wrap_utils.keyword_args [("byte_order", byte_order); ("mat_dtype", mat_dtype); ("squeeze_me", squeeze_me); ("chars_as_strings", chars_as_strings); ("matlab_compatible", matlab_compatible); ("struct_as_record", struct_as_record); ("verify_compressed_data_integrity", verify_compressed_data_integrity); ("simplify_cells", simplify_cells); ("mat_stream", Some(mat_stream ))])
     |> of_pyobject
let end_of_stream self =
   Py.Module.get_function_with_keywords (to_pyobject self) "end_of_stream"
     [||]
     []

let guess_byte_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "guess_byte_order"
     [||]
     []

let set_matlab_compatible self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_matlab_compatible"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module VarHeader4 = struct
type tag = [`VarHeader4]
type t = [`Object | `VarHeader4] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ~name ~dtype ~mclass ~dims ~is_complex () =
   Py.Module.get_function_with_keywords __wrap_namespace "VarHeader4"
     [||]
     (Wrap_utils.keyword_args [("name", Some(name )); ("dtype", Some(dtype )); ("mclass", Some(mclass )); ("dims", Some(dims )); ("is_complex", Some(is_complex ))])
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module VarReader4 = struct
type tag = [`VarReader4]
type t = [`Object | `VarReader4] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create file_reader =
   Py.Module.get_function_with_keywords __wrap_namespace "VarReader4"
     [||]
     (Wrap_utils.keyword_args [("file_reader", Some(file_reader ))])
     |> of_pyobject
let array_from_header ?process ~hdr self =
   Py.Module.get_function_with_keywords (to_pyobject self) "array_from_header"
     [||]
     (Wrap_utils.keyword_args [("process", process); ("hdr", Some(hdr ))])

let read_char_array ~hdr self =
   Py.Module.get_function_with_keywords (to_pyobject self) "read_char_array"
     [||]
     (Wrap_utils.keyword_args [("hdr", Some(hdr ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let read_full_array ~hdr self =
   Py.Module.get_function_with_keywords (to_pyobject self) "read_full_array"
     [||]
     (Wrap_utils.keyword_args [("hdr", Some(hdr ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let read_header self =
   Py.Module.get_function_with_keywords (to_pyobject self) "read_header"
     [||]
     []

let read_sparse_array ~hdr self =
   Py.Module.get_function_with_keywords (to_pyobject self) "read_sparse_array"
     [||]
     (Wrap_utils.keyword_args [("hdr", Some(hdr ))])

let read_sub_array ?copy ~hdr self =
   Py.Module.get_function_with_keywords (to_pyobject self) "read_sub_array"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("hdr", Some(hdr ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let shape_from_header ~hdr self =
   Py.Module.get_function_with_keywords (to_pyobject self) "shape_from_header"
     [||]
     (Wrap_utils.keyword_args [("hdr", Some(hdr ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module VarWriter4 = struct
type tag = [`VarWriter4]
type t = [`Object | `VarWriter4] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create file_writer =
   Py.Module.get_function_with_keywords __wrap_namespace "VarWriter4"
     [||]
     (Wrap_utils.keyword_args [("file_writer", Some(file_writer ))])
     |> of_pyobject
let write ~arr ~name self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr |> Np.Obj.to_pyobject)); ("name", Some(name |> Py.String.of_string))])

let write_bytes ~arr self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write_bytes"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr ))])

let write_char ~arr ~name self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write_char"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr )); ("name", Some(name ))])

let write_header ?p ?t ?imagf ~name ~shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write_header"
     [||]
     (Wrap_utils.keyword_args [("P", Wrap_utils.Option.map p Py.Int.of_int); ("T", Wrap_utils.Option.map t Py.Int.of_int); ("imagf", Wrap_utils.Option.map imagf Py.Int.of_int); ("name", Some(name |> Py.String.of_string)); ("shape", Some(shape ))])

let write_numeric ~arr ~name self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write_numeric"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr )); ("name", Some(name ))])

let write_sparse ~arr ~name self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write_sparse"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr )); ("name", Some(name ))])

let write_string ~s self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write_string"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let arr_dtype_number ~arr ~num () =
   Py.Module.get_function_with_keywords __wrap_namespace "arr_dtype_number"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr )); ("num", Some(num ))])

let arr_to_2d ?oned_as ~arr () =
   Py.Module.get_function_with_keywords __wrap_namespace "arr_to_2d"
     [||]
     (Wrap_utils.keyword_args [("oned_as", oned_as); ("arr", Some(arr |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let arr_to_chars arr =
   Py.Module.get_function_with_keywords __wrap_namespace "arr_to_chars"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr ))])

let asbytes s =
   Py.Module.get_function_with_keywords __wrap_namespace "asbytes"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s ))])

let asstr s =
   Py.Module.get_function_with_keywords __wrap_namespace "asstr"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s ))])

let convert_dtypes ~dtype_template ~order_code () =
   Py.Module.get_function_with_keywords __wrap_namespace "convert_dtypes"
     [||]
     (Wrap_utils.keyword_args [("dtype_template", Some(dtype_template )); ("order_code", Some(order_code |> Py.String.of_string))])

let docfiller f =
   Py.Module.get_function_with_keywords __wrap_namespace "docfiller"
     [||]
     (Wrap_utils.keyword_args [("f", Some(f ))])

                  let matdims ?oned_as ~arr () =
                     Py.Module.get_function_with_keywords __wrap_namespace "matdims"
                       [||]
                       (Wrap_utils.keyword_args [("oned_as", Wrap_utils.Option.map oned_as (function
| `Column -> Py.String.of_string "column"
| `Row -> Py.String.of_string "row"
)); ("arr", Some(arr |> Np.Obj.to_pyobject))])

let read_dtype ~mat_stream ~a_dtype () =
   Py.Module.get_function_with_keywords __wrap_namespace "read_dtype"
     [||]
     (Wrap_utils.keyword_args [("mat_stream", Some(mat_stream )); ("a_dtype", Some(a_dtype |> Np.Dtype.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let reduce ?initial ~function_ ~sequence () =
   Py.Module.get_function_with_keywords __wrap_namespace "reduce"
     [||]
     (Wrap_utils.keyword_args [("initial", initial); ("function", Some(function_ )); ("sequence", Some(sequence ))])


end
module Mio5 = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.io.matlab.mio5"

let get_py name = Py.Module.get __wrap_namespace name
module BytesIO = struct
type tag = [`BytesIO]
type t = [`BytesIO | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?initial_bytes () =
   Py.Module.get_function_with_keywords __wrap_namespace "BytesIO"
     [||]
     (Wrap_utils.keyword_args [("initial_bytes", initial_bytes)])
     |> of_pyobject
let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let close self =
   Py.Module.get_function_with_keywords (to_pyobject self) "close"
     [||]
     []

let detach self =
   Py.Module.get_function_with_keywords (to_pyobject self) "detach"
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

let getbuffer self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getbuffer"
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

let read1 ?size self =
   Py.Module.get_function_with_keywords (to_pyobject self) "read1"
     (Array.of_list @@ List.concat [(match size with None -> [] | Some x -> [x ])])
     []

let readable self =
   Py.Module.get_function_with_keywords (to_pyobject self) "readable"
     [||]
     []

let readinto ~buffer self =
   Py.Module.get_function_with_keywords (to_pyobject self) "readinto"
     (Array.of_list @@ List.concat [[buffer ]])
     []

let readinto1 ~buffer self =
   Py.Module.get_function_with_keywords (to_pyobject self) "readinto1"
     (Array.of_list @@ List.concat [[buffer ]])
     []

let readline ?size self =
   Py.Module.get_function_with_keywords (to_pyobject self) "readline"
     (Array.of_list @@ List.concat [(match size with None -> [] | Some x -> [x ])])
     []

let readlines ?size self =
   Py.Module.get_function_with_keywords (to_pyobject self) "readlines"
     (Array.of_list @@ List.concat [(match size with None -> [] | Some x -> [x ])])
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

let truncate ?size self =
   Py.Module.get_function_with_keywords (to_pyobject self) "truncate"
     (Array.of_list @@ List.concat [(match size with None -> [] | Some x -> [x ])])
     []

let writable self =
   Py.Module.get_function_with_keywords (to_pyobject self) "writable"
     [||]
     []

let write ~b self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write"
     (Array.of_list @@ List.concat [[b ]])
     []

let writelines ~lines self =
   Py.Module.get_function_with_keywords (to_pyobject self) "writelines"
     (Array.of_list @@ List.concat [[lines ]])
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module EmptyStructMarker = struct
type tag = [`EmptyStructMarker]
type t = [`EmptyStructMarker | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "EmptyStructMarker"
     [||]
     []
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MatFileReader = struct
type tag = [`MatFileReader]
type t = [`MatFileReader | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?byte_order ?mat_dtype ?squeeze_me ?chars_as_strings ?matlab_compatible ?struct_as_record ?verify_compressed_data_integrity ?simplify_cells ~mat_stream () =
   Py.Module.get_function_with_keywords __wrap_namespace "MatFileReader"
     [||]
     (Wrap_utils.keyword_args [("byte_order", byte_order); ("mat_dtype", mat_dtype); ("squeeze_me", squeeze_me); ("chars_as_strings", chars_as_strings); ("matlab_compatible", matlab_compatible); ("struct_as_record", struct_as_record); ("verify_compressed_data_integrity", verify_compressed_data_integrity); ("simplify_cells", simplify_cells); ("mat_stream", Some(mat_stream ))])
     |> of_pyobject
let end_of_stream self =
   Py.Module.get_function_with_keywords (to_pyobject self) "end_of_stream"
     [||]
     []

let guess_byte_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "guess_byte_order"
     [||]
     []

let set_matlab_compatible self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_matlab_compatible"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MatReadError = struct
type tag = [`MatReadError]
type t = [`BaseException | `MatReadError | `Object] Obj.t
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
module MatReadWarning = struct
type tag = [`MatReadWarning]
type t = [`BaseException | `MatReadWarning | `Object] Obj.t
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
module MatWriteError = struct
type tag = [`MatWriteError]
type t = [`BaseException | `MatWriteError | `Object] Obj.t
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
module MatlabFunction = struct
type tag = [`MatlabFunction]
type t = [`ArrayLike | `MatlabFunction | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create input_array =
   Py.Module.get_function_with_keywords __wrap_namespace "MatlabFunction"
     [||]
     (Wrap_utils.keyword_args [("input_array", Some(input_array ))])
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let __setitem__ ~key ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     (Array.of_list @@ List.concat [[key ];[value ]])
     []

let all ?axis ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "all"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims)])

let any ?axis ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "any"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims)])

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

let max ?axis ?out ?keepdims ?initial ?where self =
   Py.Module.get_function_with_keywords (to_pyobject self) "max"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims); ("initial", initial); ("where", where)])

let mean ?axis ?dtype ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "mean"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("keepdims", keepdims)])

let min ?axis ?out ?keepdims ?initial ?where self =
   Py.Module.get_function_with_keywords (to_pyobject self) "min"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims); ("initial", initial); ("where", where)])

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
| `S x -> Py.String.of_string x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
)); ("kth", Some(kth |> (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)))])

let prod ?axis ?dtype ?out ?keepdims ?initial ?where self =
   Py.Module.get_function_with_keywords (to_pyobject self) "prod"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("keepdims", keepdims); ("initial", initial); ("where", where)])

let ptp ?axis ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ptp"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims)])

let put ?mode ~indices ~values self =
   Py.Module.get_function_with_keywords (to_pyobject self) "put"
     [||]
     (Wrap_utils.keyword_args [("mode", mode); ("indices", Some(indices )); ("values", Some(values ))])

let ravel ?order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ravel"
     [||]
     (Wrap_utils.keyword_args [("order", order)])

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
| `T_n_ints x -> Wrap_utils.id x
| `TupleOfInts x -> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml) x
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
| `Quicksort -> Py.String.of_string "quicksort"
| `Heapsort -> Py.String.of_string "heapsort"
| `Stable -> Py.String.of_string "stable"
| `Mergesort -> Py.String.of_string "mergesort"
)); ("order", Wrap_utils.Option.map order (function
| `S x -> Py.String.of_string x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
))])

let squeeze ?axis self =
   Py.Module.get_function_with_keywords (to_pyobject self) "squeeze"
     [||]
     (Wrap_utils.keyword_args [("axis", axis)])

let std ?axis ?dtype ?out ?ddof ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "std"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("ddof", ddof); ("keepdims", keepdims)])

let sum ?axis ?dtype ?out ?keepdims ?initial ?where self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sum"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("keepdims", keepdims); ("initial", initial); ("where", where)])

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
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
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
     (Wrap_utils.keyword_args [("order", order)])

let trace ?offset ?axis1 ?axis2 ?dtype ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "trace"
     [||]
     (Wrap_utils.keyword_args [("offset", offset); ("axis1", axis1); ("axis2", axis2); ("dtype", dtype); ("out", out)])

let transpose axes self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id axes)])
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let var ?axis ?dtype ?out ?ddof ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "var"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("ddof", ddof); ("keepdims", keepdims)])

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
module MatlabObject = struct
type tag = [`MatlabObject]
type t = [`ArrayLike | `MatlabObject | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?classname ~input_array () =
   Py.Module.get_function_with_keywords __wrap_namespace "MatlabObject"
     [||]
     (Wrap_utils.keyword_args [("classname", classname); ("input_array", Some(input_array ))])
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let __setitem__ ~key ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     (Array.of_list @@ List.concat [[key ];[value ]])
     []

let all ?axis ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "all"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims)])

let any ?axis ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "any"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims)])

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

let max ?axis ?out ?keepdims ?initial ?where self =
   Py.Module.get_function_with_keywords (to_pyobject self) "max"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims); ("initial", initial); ("where", where)])

let mean ?axis ?dtype ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "mean"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("keepdims", keepdims)])

let min ?axis ?out ?keepdims ?initial ?where self =
   Py.Module.get_function_with_keywords (to_pyobject self) "min"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims); ("initial", initial); ("where", where)])

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
| `S x -> Py.String.of_string x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
)); ("kth", Some(kth |> (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)))])

let prod ?axis ?dtype ?out ?keepdims ?initial ?where self =
   Py.Module.get_function_with_keywords (to_pyobject self) "prod"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("keepdims", keepdims); ("initial", initial); ("where", where)])

let ptp ?axis ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ptp"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims)])

let put ?mode ~indices ~values self =
   Py.Module.get_function_with_keywords (to_pyobject self) "put"
     [||]
     (Wrap_utils.keyword_args [("mode", mode); ("indices", Some(indices )); ("values", Some(values ))])

let ravel ?order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ravel"
     [||]
     (Wrap_utils.keyword_args [("order", order)])

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
| `T_n_ints x -> Wrap_utils.id x
| `TupleOfInts x -> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml) x
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
| `Quicksort -> Py.String.of_string "quicksort"
| `Heapsort -> Py.String.of_string "heapsort"
| `Stable -> Py.String.of_string "stable"
| `Mergesort -> Py.String.of_string "mergesort"
)); ("order", Wrap_utils.Option.map order (function
| `S x -> Py.String.of_string x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
))])

let squeeze ?axis self =
   Py.Module.get_function_with_keywords (to_pyobject self) "squeeze"
     [||]
     (Wrap_utils.keyword_args [("axis", axis)])

let std ?axis ?dtype ?out ?ddof ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "std"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("ddof", ddof); ("keepdims", keepdims)])

let sum ?axis ?dtype ?out ?keepdims ?initial ?where self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sum"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("keepdims", keepdims); ("initial", initial); ("where", where)])

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
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
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
     (Wrap_utils.keyword_args [("order", order)])

let trace ?offset ?axis1 ?axis2 ?dtype ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "trace"
     [||]
     (Wrap_utils.keyword_args [("offset", offset); ("axis1", axis1); ("axis2", axis2); ("dtype", dtype); ("out", out)])

let transpose axes self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id axes)])
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let var ?axis ?dtype ?out ?ddof ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "var"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("ddof", ddof); ("keepdims", keepdims)])

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
module VarReader5 = struct
type tag = [`VarReader5]
type t = [`Object | `VarReader5] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module VarWriter5 = struct
type tag = [`VarWriter5]
type t = [`Object | `VarWriter5] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create file_writer =
   Py.Module.get_function_with_keywords __wrap_namespace "VarWriter5"
     [||]
     (Wrap_utils.keyword_args [("file_writer", Some(file_writer ))])
     |> of_pyobject
let update_matrix_tag ~start_pos self =
   Py.Module.get_function_with_keywords (to_pyobject self) "update_matrix_tag"
     [||]
     (Wrap_utils.keyword_args [("start_pos", Some(start_pos ))])

let write ~arr self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr |> Np.Obj.to_pyobject))])

let write_bytes ~arr self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write_bytes"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr ))])

let write_cells ~arr self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write_cells"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr ))])

let write_char ?codec ~arr self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write_char"
     [||]
     (Wrap_utils.keyword_args [("codec", codec); ("arr", Some(arr ))])

let write_element ?mdtype ~arr self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write_element"
     [||]
     (Wrap_utils.keyword_args [("mdtype", mdtype); ("arr", Some(arr ))])

let write_empty_struct self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write_empty_struct"
     [||]
     []

let write_header ?is_complex ?is_logical ?nzmax ~shape ~mclass self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write_header"
     [||]
     (Wrap_utils.keyword_args [("is_complex", is_complex); ("is_logical", is_logical); ("nzmax", nzmax); ("shape", Some(shape )); ("mclass", Some(mclass ))])

let write_numeric ~arr self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write_numeric"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr ))])

let write_object ~arr self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write_object"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr ))])

let write_regular_element ~arr ~mdtype ~byte_count self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write_regular_element"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr )); ("mdtype", Some(mdtype )); ("byte_count", Some(byte_count ))])

let write_smalldata_element ~arr ~mdtype ~byte_count self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write_smalldata_element"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr )); ("mdtype", Some(mdtype )); ("byte_count", Some(byte_count ))])

let write_sparse ~arr self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write_sparse"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr ))])

let write_string ~s self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write_string"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s ))])

let write_struct ~arr self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write_struct"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr ))])

let write_top ~arr ~name ~is_global self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write_top"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr |> Np.Obj.to_pyobject)); ("name", Some(name |> Py.String.of_string)); ("is_global", Some(is_global |> Py.Bool.of_bool))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ZlibInputStream = struct
type tag = [`ZlibInputStream]
type t = [`Object | `ZlibInputStream] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Mat_struct = struct
type tag = [`Mat_struct]
type t = [`Mat_struct | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "mat_struct"
     [||]
     []
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let arr_dtype_number ~arr ~num () =
   Py.Module.get_function_with_keywords __wrap_namespace "arr_dtype_number"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr )); ("num", Some(num ))])

let arr_to_chars arr =
   Py.Module.get_function_with_keywords __wrap_namespace "arr_to_chars"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr ))])

let asbytes s =
   Py.Module.get_function_with_keywords __wrap_namespace "asbytes"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s ))])

let asstr s =
   Py.Module.get_function_with_keywords __wrap_namespace "asstr"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s ))])

let docfiller f =
   Py.Module.get_function_with_keywords __wrap_namespace "docfiller"
     [||]
     (Wrap_utils.keyword_args [("f", Some(f ))])

                  let matdims ?oned_as ~arr () =
                     Py.Module.get_function_with_keywords __wrap_namespace "matdims"
                       [||]
                       (Wrap_utils.keyword_args [("oned_as", Wrap_utils.Option.map oned_as (function
| `Column -> Py.String.of_string "column"
| `Row -> Py.String.of_string "row"
)); ("arr", Some(arr |> Np.Obj.to_pyobject))])

let read_dtype ~mat_stream ~a_dtype () =
   Py.Module.get_function_with_keywords __wrap_namespace "read_dtype"
     [||]
     (Wrap_utils.keyword_args [("mat_stream", Some(mat_stream )); ("a_dtype", Some(a_dtype |> Np.Dtype.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let to_writeable source =
   Py.Module.get_function_with_keywords __wrap_namespace "to_writeable"
     [||]
     (Wrap_utils.keyword_args [("source", Some(source ))])
     |> (fun py -> if Py.is_none py then None else Some (Wrap_utils.id py))
let varmats_from_mat file_obj =
   Py.Module.get_function_with_keywords __wrap_namespace "varmats_from_mat"
     [||]
     (Wrap_utils.keyword_args [("file_obj", Some(file_obj ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
module Mio5_params = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.io.matlab.mio5_params"

let get_py name = Py.Module.get __wrap_namespace name
module MatlabOpaque = struct
type tag = [`MatlabOpaque]
type t = [`ArrayLike | `MatlabOpaque | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create input_array =
   Py.Module.get_function_with_keywords __wrap_namespace "MatlabOpaque"
     [||]
     (Wrap_utils.keyword_args [("input_array", Some(input_array ))])
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let __setitem__ ~key ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     (Array.of_list @@ List.concat [[key ];[value ]])
     []

let all ?axis ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "all"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims)])

let any ?axis ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "any"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims)])

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

let max ?axis ?out ?keepdims ?initial ?where self =
   Py.Module.get_function_with_keywords (to_pyobject self) "max"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims); ("initial", initial); ("where", where)])

let mean ?axis ?dtype ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "mean"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("keepdims", keepdims)])

let min ?axis ?out ?keepdims ?initial ?where self =
   Py.Module.get_function_with_keywords (to_pyobject self) "min"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims); ("initial", initial); ("where", where)])

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
| `S x -> Py.String.of_string x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
)); ("kth", Some(kth |> (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)))])

let prod ?axis ?dtype ?out ?keepdims ?initial ?where self =
   Py.Module.get_function_with_keywords (to_pyobject self) "prod"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("keepdims", keepdims); ("initial", initial); ("where", where)])

let ptp ?axis ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ptp"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims)])

let put ?mode ~indices ~values self =
   Py.Module.get_function_with_keywords (to_pyobject self) "put"
     [||]
     (Wrap_utils.keyword_args [("mode", mode); ("indices", Some(indices )); ("values", Some(values ))])

let ravel ?order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ravel"
     [||]
     (Wrap_utils.keyword_args [("order", order)])

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
| `T_n_ints x -> Wrap_utils.id x
| `TupleOfInts x -> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml) x
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
| `Quicksort -> Py.String.of_string "quicksort"
| `Heapsort -> Py.String.of_string "heapsort"
| `Stable -> Py.String.of_string "stable"
| `Mergesort -> Py.String.of_string "mergesort"
)); ("order", Wrap_utils.Option.map order (function
| `S x -> Py.String.of_string x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
))])

let squeeze ?axis self =
   Py.Module.get_function_with_keywords (to_pyobject self) "squeeze"
     [||]
     (Wrap_utils.keyword_args [("axis", axis)])

let std ?axis ?dtype ?out ?ddof ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "std"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("ddof", ddof); ("keepdims", keepdims)])

let sum ?axis ?dtype ?out ?keepdims ?initial ?where self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sum"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("keepdims", keepdims); ("initial", initial); ("where", where)])

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
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
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
     (Wrap_utils.keyword_args [("order", order)])

let trace ?offset ?axis1 ?axis2 ?dtype ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "trace"
     [||]
     (Wrap_utils.keyword_args [("offset", offset); ("axis1", axis1); ("axis2", axis2); ("dtype", dtype); ("out", out)])

let transpose axes self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id axes)])
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let var ?axis ?dtype ?out ?ddof ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "var"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("ddof", ddof); ("keepdims", keepdims)])

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
let convert_dtypes ~dtype_template ~order_code () =
   Py.Module.get_function_with_keywords __wrap_namespace "convert_dtypes"
     [||]
     (Wrap_utils.keyword_args [("dtype_template", Some(dtype_template )); ("order_code", Some(order_code |> Py.String.of_string))])


end
module Mio5_utils = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.io.matlab.mio5_utils"

let get_py name = Py.Module.get __wrap_namespace name
module VarHeader5 = struct
type tag = [`VarHeader5]
type t = [`Object | `VarHeader5] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
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
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

                  let argmin ?axis ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "argmin"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
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
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
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
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
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
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
| `PyObject x -> Wrap_utils.id x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let min ?axis ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "min"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
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
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
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
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `T2_D x -> Wrap_utils.id x
))])

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
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `T2_D x -> Wrap_utils.id x
))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
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
let asbytes s =
   Py.Module.get_function_with_keywords __wrap_namespace "asbytes"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s ))])

let asstr s =
   Py.Module.get_function_with_keywords __wrap_namespace "asstr"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s ))])

let pycopy x =
   Py.Module.get_function_with_keywords __wrap_namespace "pycopy"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])


end
module Mio_utils = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.io.matlab.mio_utils"

let get_py name = Py.Module.get __wrap_namespace name

end
module Miobase = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.io.matlab.miobase"

let get_py name = Py.Module.get __wrap_namespace name
module MatVarReader = struct
type tag = [`MatVarReader]
type t = [`MatVarReader | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create file_reader =
   Py.Module.get_function_with_keywords __wrap_namespace "MatVarReader"
     [||]
     (Wrap_utils.keyword_args [("file_reader", Some(file_reader ))])
     |> of_pyobject
let array_from_header ~header self =
   Py.Module.get_function_with_keywords (to_pyobject self) "array_from_header"
     [||]
     (Wrap_utils.keyword_args [("header", Some(header ))])

let read_header self =
   Py.Module.get_function_with_keywords (to_pyobject self) "read_header"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let arr_dtype_number ~arr ~num () =
   Py.Module.get_function_with_keywords __wrap_namespace "arr_dtype_number"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr )); ("num", Some(num ))])

let arr_to_chars arr =
   Py.Module.get_function_with_keywords __wrap_namespace "arr_to_chars"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr ))])

let convert_dtypes ~dtype_template ~order_code () =
   Py.Module.get_function_with_keywords __wrap_namespace "convert_dtypes"
     [||]
     (Wrap_utils.keyword_args [("dtype_template", Some(dtype_template )); ("order_code", Some(order_code |> Py.String.of_string))])

let docfiller f =
   Py.Module.get_function_with_keywords __wrap_namespace "docfiller"
     [||]
     (Wrap_utils.keyword_args [("f", Some(f ))])

let get_matfile_version fileobj =
   Py.Module.get_function_with_keywords __wrap_namespace "get_matfile_version"
     [||]
     (Wrap_utils.keyword_args [("fileobj", Some(fileobj ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let matdims ?oned_as ~arr () =
                     Py.Module.get_function_with_keywords __wrap_namespace "matdims"
                       [||]
                       (Wrap_utils.keyword_args [("oned_as", Wrap_utils.Option.map oned_as (function
| `Column -> Py.String.of_string "column"
| `Row -> Py.String.of_string "row"
)); ("arr", Some(arr |> Np.Obj.to_pyobject))])

let read_dtype ~mat_stream ~a_dtype () =
   Py.Module.get_function_with_keywords __wrap_namespace "read_dtype"
     [||]
     (Wrap_utils.keyword_args [("mat_stream", Some(mat_stream )); ("a_dtype", Some(a_dtype |> Np.Dtype.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
module Streams = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.io.matlab.streams"

let get_py name = Py.Module.get __wrap_namespace name
module GenericStream = struct
type tag = [`GenericStream]
type t = [`GenericStream | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end

end
let loadmat ?mdict ?appendmat ?kwargs ~file_name () =
   Py.Module.get_function_with_keywords __wrap_namespace "loadmat"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("mdict", mdict); ("appendmat", Wrap_utils.Option.map appendmat Py.Bool.of_bool); ("file_name", Some(file_name |> Py.String.of_string))]) (match kwargs with None -> [] | Some x -> x))

                  let savemat ?appendmat ?format ?long_field_names ?do_compression ?oned_as ~file_name ~mdict () =
                     Py.Module.get_function_with_keywords __wrap_namespace "savemat"
                       [||]
                       (Wrap_utils.keyword_args [("appendmat", Wrap_utils.Option.map appendmat Py.Bool.of_bool); ("format", Wrap_utils.Option.map format (function
| `T4 -> Py.String.of_string "4"
| `T5 -> Py.String.of_string "5"
)); ("long_field_names", Wrap_utils.Option.map long_field_names Py.Bool.of_bool); ("do_compression", Wrap_utils.Option.map do_compression Py.Bool.of_bool); ("oned_as", Wrap_utils.Option.map oned_as (function
| `Row -> Py.String.of_string "row"
| `Column -> Py.String.of_string "column"
)); ("file_name", Some(file_name |> (function
| `S x -> Py.String.of_string x
| `File_like_object x -> Wrap_utils.id x
))); ("mdict", Some(mdict ))])

let whosmat ?appendmat ?kwargs ~file_name () =
   Py.Module.get_function_with_keywords __wrap_namespace "whosmat"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("appendmat", Wrap_utils.Option.map appendmat Py.Bool.of_bool); ("file_name", Some(file_name |> Py.String.of_string))]) (match kwargs with None -> [] | Some x -> x))


end
module Mmio = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.io.mmio"

let get_py name = Py.Module.get __wrap_namespace name
module MMFile = struct
type tag = [`MMFile]
type t = [`MMFile | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs () =
   Py.Module.get_function_with_keywords __wrap_namespace "MMFile"
     [||]
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
                  let info ~source self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "info"
                       [||]
                       (Wrap_utils.keyword_args [("source", Some(source |> (function
| `File_like x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))])
                       |> (fun x -> ((Py.Int.to_int (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Py.String.to_string (Py.Tuple.get x 3)), (Py.String.to_string (Py.Tuple.get x 4)), (Py.String.to_string (Py.Tuple.get x 5))))
                  let read ~source self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "read"
                       [||]
                       (Wrap_utils.keyword_args [("source", Some(source |> (function
| `File_like x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))])

                  let write ?comment ?field ?precision ?symmetry ~target ~a self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "write"
                       [||]
                       (Wrap_utils.keyword_args [("comment", Wrap_utils.Option.map comment Py.String.of_string); ("field", Wrap_utils.Option.map field Py.String.of_string); ("precision", Wrap_utils.Option.map precision Py.Int.of_int); ("symmetry", Wrap_utils.Option.map symmetry Py.String.of_string); ("target", Some(target |> (function
| `File_like x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
))); ("a", Some(a |> Np.Obj.to_pyobject))])

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
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

                  let argmin ?axis ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "argmin"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
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
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
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
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
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
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
| `PyObject x -> Wrap_utils.id x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let min ?axis ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "min"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
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
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
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
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `T2_D x -> Wrap_utils.id x
))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
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
module Ndarray = struct
type tag = [`Ndarray]
type t = [`ArrayLike | `Ndarray | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?dtype ?buffer ?offset ?strides ?order ~shape () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ndarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("buffer", buffer); ("offset", Wrap_utils.Option.map offset Py.Int.of_int); ("strides", Wrap_utils.Option.map strides (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)))])
                       |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let __setitem__ ~key ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     (Array.of_list @@ List.concat [[key ];[value ]])
     []

let all ?axis ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "all"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims)])

let any ?axis ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "any"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims)])

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

let max ?axis ?out ?keepdims ?initial ?where self =
   Py.Module.get_function_with_keywords (to_pyobject self) "max"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims); ("initial", initial); ("where", where)])

let mean ?axis ?dtype ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "mean"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("keepdims", keepdims)])

let min ?axis ?out ?keepdims ?initial ?where self =
   Py.Module.get_function_with_keywords (to_pyobject self) "min"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims); ("initial", initial); ("where", where)])

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
| `S x -> Py.String.of_string x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
)); ("kth", Some(kth |> (function
| `I x -> Py.Int.of_int x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
)))])

let prod ?axis ?dtype ?out ?keepdims ?initial ?where self =
   Py.Module.get_function_with_keywords (to_pyobject self) "prod"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("keepdims", keepdims); ("initial", initial); ("where", where)])

let ptp ?axis ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ptp"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims)])

let put ?mode ~indices ~values self =
   Py.Module.get_function_with_keywords (to_pyobject self) "put"
     [||]
     (Wrap_utils.keyword_args [("mode", mode); ("indices", Some(indices )); ("values", Some(values ))])

let ravel ?order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ravel"
     [||]
     (Wrap_utils.keyword_args [("order", order)])

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
| `T_n_ints x -> Wrap_utils.id x
| `TupleOfInts x -> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml) x
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
| `Quicksort -> Py.String.of_string "quicksort"
| `Heapsort -> Py.String.of_string "heapsort"
| `Stable -> Py.String.of_string "stable"
| `Mergesort -> Py.String.of_string "mergesort"
)); ("order", Wrap_utils.Option.map order (function
| `S x -> Py.String.of_string x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
))])

let squeeze ?axis self =
   Py.Module.get_function_with_keywords (to_pyobject self) "squeeze"
     [||]
     (Wrap_utils.keyword_args [("axis", axis)])

let std ?axis ?dtype ?out ?ddof ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "std"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("ddof", ddof); ("keepdims", keepdims)])

let sum ?axis ?dtype ?out ?keepdims ?initial ?where self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sum"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("keepdims", keepdims); ("initial", initial); ("where", where)])

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
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
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
     (Wrap_utils.keyword_args [("order", order)])

let trace ?offset ?axis1 ?axis2 ?dtype ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "trace"
     [||]
     (Wrap_utils.keyword_args [("offset", offset); ("axis1", axis1); ("axis2", axis2); ("dtype", dtype); ("out", out)])

let transpose axes self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id axes)])
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let var ?axis ?dtype ?out ?ddof ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "var"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("ddof", ddof); ("keepdims", keepdims)])

                  let view ?dtype ?type_ self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "view"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype (function
| `Ndarray_sub_class x -> Wrap_utils.id x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("type", type_)])


let t_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "T" with
  | None -> failwith "attribute T not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let t self = match t_opt self with
  | None -> raise Not_found
  | Some x -> x

let data_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "data" with
  | None -> failwith "attribute data not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let data self = match data_opt self with
  | None -> raise Not_found
  | Some x -> x

let dtype_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dtype" with
  | None -> failwith "attribute dtype not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let dtype self = match dtype_opt self with
  | None -> raise Not_found
  | Some x -> x

let flags_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "flags" with
  | None -> failwith "attribute flags not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let flags self = match flags_opt self with
  | None -> raise Not_found
  | Some x -> x

let flat_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "flat" with
  | None -> failwith "attribute flat not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let flat self = match flat_opt self with
  | None -> raise Not_found
  | Some x -> x

let imag_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "imag" with
  | None -> failwith "attribute imag not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let imag self = match imag_opt self with
  | None -> raise Not_found
  | Some x -> x

let real_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "real" with
  | None -> failwith "attribute real not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let real self = match real_opt self with
  | None -> raise Not_found
  | Some x -> x

let size_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "size" with
  | None -> failwith "attribute size not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let size self = match size_opt self with
  | None -> raise Not_found
  | Some x -> x

let itemsize_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "itemsize" with
  | None -> failwith "attribute itemsize not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let itemsize self = match itemsize_opt self with
  | None -> raise Not_found
  | Some x -> x

let nbytes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nbytes" with
  | None -> failwith "attribute nbytes not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let nbytes self = match nbytes_opt self with
  | None -> raise Not_found
  | Some x -> x

let ndim_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "ndim" with
  | None -> failwith "attribute ndim not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let ndim self = match ndim_opt self with
  | None -> raise Not_found
  | Some x -> x

let shape_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "shape" with
  | None -> failwith "attribute shape not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> let len = Py.Sequence.length py in Array.init len
          (fun i -> Py.Int.to_int (Py.Sequence.get_item py i))) x)

let shape self = match shape_opt self with
  | None -> raise Not_found
  | Some x -> x

let strides_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "strides" with
  | None -> failwith "attribute strides not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> let len = Py.Sequence.length py in Array.init len
          (fun i -> Py.Int.to_int (Py.Sequence.get_item py i))) x)

let strides self = match strides_opt self with
  | None -> raise Not_found
  | Some x -> x

let ctypes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "ctypes" with
  | None -> failwith "attribute ctypes not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let ctypes self = match ctypes_opt self with
  | None -> raise Not_found
  | Some x -> x

let base_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "base" with
  | None -> failwith "attribute base not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let base self = match base_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let asarray ?dtype ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let asbytes s =
   Py.Module.get_function_with_keywords __wrap_namespace "asbytes"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s ))])

let asstr s =
   Py.Module.get_function_with_keywords __wrap_namespace "asstr"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s ))])

                  let can_cast ?casting ~from_ ~to_ () =
                     Py.Module.get_function_with_keywords __wrap_namespace "can_cast"
                       [||]
                       (Wrap_utils.keyword_args [("casting", Wrap_utils.Option.map casting (function
| `No -> Py.String.of_string "no"
| `Equiv -> Py.String.of_string "equiv"
| `Safe -> Py.String.of_string "safe"
| `Same_kind -> Py.String.of_string "same_kind"
| `Unsafe -> Py.String.of_string "unsafe"
)); ("from_", Some(from_ |> (function
| `Bool x -> Py.Bool.of_bool x
| `F x -> Py.Float.of_float x
| `Dtype_specifier x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `Dtype x -> Np.Dtype.to_pyobject x
))); ("to", Some(to_ |> (function
| `Dtype_specifier x -> Wrap_utils.id x
| `Dtype x -> Np.Dtype.to_pyobject x
)))])
                       |> Py.Bool.to_bool
let concatenate ?axis ?out ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "concatenate"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("a", Some(a ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let conj ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "conj"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let imag val_ =
   Py.Module.get_function_with_keywords __wrap_namespace "imag"
     [||]
     (Wrap_utils.keyword_args [("val", Some(val_ |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let isspmatrix x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

                  let mminfo source =
                     Py.Module.get_function_with_keywords __wrap_namespace "mminfo"
                       [||]
                       (Wrap_utils.keyword_args [("source", Some(source |> (function
| `File_like x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))])
                       |> (fun x -> ((Py.Int.to_int (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Py.String.to_string (Py.Tuple.get x 3)), (Py.String.to_string (Py.Tuple.get x 4)), (Py.String.to_string (Py.Tuple.get x 5))))
                  let mmread source =
                     Py.Module.get_function_with_keywords __wrap_namespace "mmread"
                       [||]
                       (Wrap_utils.keyword_args [("source", Some(source |> (function
| `File_like x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))])

                  let mmwrite ?comment ?field ?precision ?symmetry ~target ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "mmwrite"
                       [||]
                       (Wrap_utils.keyword_args [("comment", Wrap_utils.Option.map comment Py.String.of_string); ("field", Wrap_utils.Option.map field Py.String.of_string); ("precision", Wrap_utils.Option.map precision Py.Int.of_int); ("symmetry", Wrap_utils.Option.map symmetry Py.String.of_string); ("target", Some(target |> (function
| `File_like x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
))); ("a", Some(a |> Np.Obj.to_pyobject))])

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
let real val_ =
   Py.Module.get_function_with_keywords __wrap_namespace "real"
     [||]
     (Wrap_utils.keyword_args [("val", Some(val_ |> Np.Obj.to_pyobject))])
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
module Netcdf = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.io.netcdf"

let get_py name = Py.Module.get __wrap_namespace name
module OrderedDict = struct
type tag = [`OrderedDict]
type t = [`Object | `OrderedDict] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y ))])

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let __setitem__ ~key ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     (Array.of_list @@ List.concat [[key ];[value ]])
     []

let fromkeys ?value ~iterable self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fromkeys"
     [||]
     (Wrap_utils.keyword_args [("value", value); ("iterable", Some(iterable ))])

let get ?default ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get"
     (Array.of_list @@ List.concat [(match default with None -> [] | Some x -> [x ]);[key ]])
     []

let move_to_end ?last ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "move_to_end"
     [||]
     (Wrap_utils.keyword_args [("last", last); ("key", Some(key ))])

let pop ?d ~k self =
   Py.Module.get_function_with_keywords (to_pyobject self) "pop"
     [||]
     (Wrap_utils.keyword_args [("d", d); ("k", Some(k ))])

let popitem ?last self =
   Py.Module.get_function_with_keywords (to_pyobject self) "popitem"
     [||]
     (Wrap_utils.keyword_args [("last", last)])

let setdefault ?default ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setdefault"
     [||]
     (Wrap_utils.keyword_args [("default", default); ("key", Some(key ))])

let update ?e ?f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "update"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("E", e)]) (match f with None -> [] | Some x -> x))

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Dtype = struct
type tag = [`Dtype]
type t = [`Dtype | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?align ?copy ~obj () =
   Py.Module.get_function_with_keywords __wrap_namespace "dtype"
     [||]
     (Wrap_utils.keyword_args [("align", Wrap_utils.Option.map align Py.Bool.of_bool); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("obj", Some(obj ))])
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

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
                  let asarray ?dtype ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let asbytes s =
   Py.Module.get_function_with_keywords __wrap_namespace "asbytes"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s ))])

let asstr s =
   Py.Module.get_function_with_keywords __wrap_namespace "asstr"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s ))])

                  let empty ?dtype ?order ~shape () =
                     Py.Module.get_function_with_keywords __wrap_namespace "empty"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (function
| `I x -> Py.Int.of_int x
| `Tuple_of_int x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let frombuffer ?dtype ?count ?offset ~buffer () =
   Py.Module.get_function_with_keywords __wrap_namespace "frombuffer"
     [||]
     (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("count", Wrap_utils.Option.map count Py.Int.of_int); ("offset", Wrap_utils.Option.map offset Py.Int.of_int); ("buffer", Some(buffer ))])

let mul ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "mul"
     (Array.of_list @@ List.concat [[a ];[b ]])
     []

let reduce ?initial ~function_ ~sequence () =
   Py.Module.get_function_with_keywords __wrap_namespace "reduce"
     [||]
     (Wrap_utils.keyword_args [("initial", initial); ("function", Some(function_ )); ("sequence", Some(sequence ))])


end
let hb_read path_or_open_file =
   Py.Module.get_function_with_keywords __wrap_namespace "hb_read"
     [||]
     (Wrap_utils.keyword_args [("path_or_open_file", Some(path_or_open_file ))])

let hb_write ?hb_info ~path_or_open_file ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "hb_write"
     [||]
     (Wrap_utils.keyword_args [("hb_info", hb_info); ("path_or_open_file", Some(path_or_open_file )); ("m", Some(m |> Np.Obj.to_pyobject))])

let loadmat ?mdict ?appendmat ?kwargs ~file_name () =
   Py.Module.get_function_with_keywords __wrap_namespace "loadmat"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("mdict", mdict); ("appendmat", Wrap_utils.Option.map appendmat Py.Bool.of_bool); ("file_name", Some(file_name |> Py.String.of_string))]) (match kwargs with None -> [] | Some x -> x))

                  let mminfo source =
                     Py.Module.get_function_with_keywords __wrap_namespace "mminfo"
                       [||]
                       (Wrap_utils.keyword_args [("source", Some(source |> (function
| `File_like x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))])
                       |> (fun x -> ((Py.Int.to_int (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Py.String.to_string (Py.Tuple.get x 3)), (Py.String.to_string (Py.Tuple.get x 4)), (Py.String.to_string (Py.Tuple.get x 5))))
                  let mmread source =
                     Py.Module.get_function_with_keywords __wrap_namespace "mmread"
                       [||]
                       (Wrap_utils.keyword_args [("source", Some(source |> (function
| `File_like x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))])

                  let mmwrite ?comment ?field ?precision ?symmetry ~target ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "mmwrite"
                       [||]
                       (Wrap_utils.keyword_args [("comment", Wrap_utils.Option.map comment Py.String.of_string); ("field", Wrap_utils.Option.map field Py.String.of_string); ("precision", Wrap_utils.Option.map precision Py.Int.of_int); ("symmetry", Wrap_utils.Option.map symmetry Py.String.of_string); ("target", Some(target |> (function
| `File_like x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
))); ("a", Some(a |> Np.Obj.to_pyobject))])

let readsav ?idict ?python_dict ?uncompressed_file_name ?verbose ~file_name () =
   Py.Module.get_function_with_keywords __wrap_namespace "readsav"
     [||]
     (Wrap_utils.keyword_args [("idict", idict); ("python_dict", Wrap_utils.Option.map python_dict Py.Bool.of_bool); ("uncompressed_file_name", Wrap_utils.Option.map uncompressed_file_name Py.String.of_string); ("verbose", Wrap_utils.Option.map verbose Py.Bool.of_bool); ("file_name", Some(file_name |> Py.String.of_string))])

                  let savemat ?appendmat ?format ?long_field_names ?do_compression ?oned_as ~file_name ~mdict () =
                     Py.Module.get_function_with_keywords __wrap_namespace "savemat"
                       [||]
                       (Wrap_utils.keyword_args [("appendmat", Wrap_utils.Option.map appendmat Py.Bool.of_bool); ("format", Wrap_utils.Option.map format (function
| `T4 -> Py.String.of_string "4"
| `T5 -> Py.String.of_string "5"
)); ("long_field_names", Wrap_utils.Option.map long_field_names Py.Bool.of_bool); ("do_compression", Wrap_utils.Option.map do_compression Py.Bool.of_bool); ("oned_as", Wrap_utils.Option.map oned_as (function
| `Row -> Py.String.of_string "row"
| `Column -> Py.String.of_string "column"
)); ("file_name", Some(file_name |> (function
| `S x -> Py.String.of_string x
| `File_like_object x -> Wrap_utils.id x
))); ("mdict", Some(mdict ))])

let whosmat ?appendmat ?kwargs ~file_name () =
   Py.Module.get_function_with_keywords __wrap_namespace "whosmat"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("appendmat", Wrap_utils.Option.map appendmat Py.Bool.of_bool); ("file_name", Some(file_name |> Py.String.of_string))]) (match kwargs with None -> [] | Some x -> x))

