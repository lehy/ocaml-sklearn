let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "numpy"

let get_py name = Py.Module.get __wrap_namespace name
module AxisError = struct
type tag = [`AxisError]
type t = [`AxisError | `BaseException | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_exception x = (x :> [`BaseException] Obj.t)
let create ?ndim ?msg_prefix ~axis () =
   Py.Module.get_function_with_keywords __wrap_namespace "AxisError"
     [||]
     (Wrap_utils.keyword_args [("ndim", ndim); ("msg_prefix", msg_prefix); ("axis", Some(axis ))])
     |> of_pyobject
let with_traceback ~tb self =
   Py.Module.get_function_with_keywords (to_pyobject self) "with_traceback"
     [||]
     (Wrap_utils.keyword_args [("tb", Some(tb ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ComplexWarning = struct
type tag = [`ComplexWarning]
type t = [`BaseException | `ComplexWarning | `Object] Obj.t
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
module DataSource = struct
type tag = [`DataSource]
type t = [`DataSource | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?destpath () =
                     Py.Module.get_function_with_keywords __wrap_namespace "DataSource"
                       [||]
                       (Wrap_utils.keyword_args [("destpath", Wrap_utils.Option.map destpath (function
| `S x -> Py.String.of_string x
| `None -> Py.none
))])
                       |> of_pyobject
let abspath ~path self =
   Py.Module.get_function_with_keywords (to_pyobject self) "abspath"
     [||]
     (Wrap_utils.keyword_args [("path", Some(path |> Py.String.of_string))])
     |> Py.String.to_string
let exists ~path self =
   Py.Module.get_function_with_keywords (to_pyobject self) "exists"
     [||]
     (Wrap_utils.keyword_args [("path", Some(path |> Py.String.of_string))])
     |> Py.Bool.to_bool
                  let open_ ?mode ?encoding ?newline ~path self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "open"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `R -> Py.String.of_string "r"
| `W -> Py.String.of_string "w"
| `A -> Py.String.of_string "a"
)); ("encoding", Wrap_utils.Option.map encoding Py.String.of_string); ("newline", Wrap_utils.Option.map newline Py.String.of_string); ("path", Some(path |> Py.String.of_string))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MachAr = struct
type tag = [`MachAr]
type t = [`MachAr | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?float_conv ?int_conv ?float_to_float ?float_to_str ?title () =
   Py.Module.get_function_with_keywords __wrap_namespace "MachAr"
     [||]
     (Wrap_utils.keyword_args [("float_conv", float_conv); ("int_conv", int_conv); ("float_to_float", float_to_float); ("float_to_str", float_to_str); ("title", Wrap_utils.Option.map title Py.String.of_string)])
     |> of_pyobject

let ibeta_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "ibeta" with
  | None -> failwith "attribute ibeta not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let ibeta self = match ibeta_opt self with
  | None -> raise Not_found
  | Some x -> x

let it_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "it" with
  | None -> failwith "attribute it not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let it self = match it_opt self with
  | None -> raise Not_found
  | Some x -> x

let machep_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "machep" with
  | None -> failwith "attribute machep not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let machep self = match machep_opt self with
  | None -> raise Not_found
  | Some x -> x

let eps_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "eps" with
  | None -> failwith "attribute eps not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let eps self = match eps_opt self with
  | None -> raise Not_found
  | Some x -> x

let negep_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "negep" with
  | None -> failwith "attribute negep not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let negep self = match negep_opt self with
  | None -> raise Not_found
  | Some x -> x

let epsneg_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "epsneg" with
  | None -> failwith "attribute epsneg not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let epsneg self = match epsneg_opt self with
  | None -> raise Not_found
  | Some x -> x

let iexp_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "iexp" with
  | None -> failwith "attribute iexp not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let iexp self = match iexp_opt self with
  | None -> raise Not_found
  | Some x -> x

let minexp_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "minexp" with
  | None -> failwith "attribute minexp not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let minexp self = match minexp_opt self with
  | None -> raise Not_found
  | Some x -> x

let xmin_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "xmin" with
  | None -> failwith "attribute xmin not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let xmin self = match xmin_opt self with
  | None -> raise Not_found
  | Some x -> x

let maxexp_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "maxexp" with
  | None -> failwith "attribute maxexp not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let maxexp self = match maxexp_opt self with
  | None -> raise Not_found
  | Some x -> x

let xmax_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "xmax" with
  | None -> failwith "attribute xmax not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let xmax self = match xmax_opt self with
  | None -> raise Not_found
  | Some x -> x

let irnd_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "irnd" with
  | None -> failwith "attribute irnd not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let irnd self = match irnd_opt self with
  | None -> raise Not_found
  | Some x -> x

let ngrd_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "ngrd" with
  | None -> failwith "attribute ngrd not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let ngrd self = match ngrd_opt self with
  | None -> raise Not_found
  | Some x -> x

let epsilon_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "epsilon" with
  | None -> failwith "attribute epsilon not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let epsilon self = match epsilon_opt self with
  | None -> raise Not_found
  | Some x -> x

let tiny_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "tiny" with
  | None -> failwith "attribute tiny not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let tiny self = match tiny_opt self with
  | None -> raise Not_found
  | Some x -> x

let huge_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "huge" with
  | None -> failwith "attribute huge not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let huge self = match huge_opt self with
  | None -> raise Not_found
  | Some x -> x

let precision_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "precision" with
  | None -> failwith "attribute precision not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let precision self = match precision_opt self with
  | None -> raise Not_found
  | Some x -> x

let resolution_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "resolution" with
  | None -> failwith "attribute resolution not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let resolution self = match resolution_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ModuleDeprecationWarning = struct
type tag = [`ModuleDeprecationWarning]
type t = [`BaseException | `ModuleDeprecationWarning | `Object] Obj.t
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
module RankWarning = struct
type tag = [`RankWarning]
type t = [`BaseException | `Object | `RankWarning] Obj.t
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
module Tester = struct
type tag = [`NoseTester]
type t = [`NoseTester | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?package ?raise_warnings ?depth ?check_fpu_mode () =
                     Py.Module.get_function_with_keywords __wrap_namespace "Tester"
                       [||]
                       (Wrap_utils.keyword_args [("package", Wrap_utils.Option.map package (function
| `Module x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("raise_warnings", Wrap_utils.Option.map raise_warnings (function
| `Sequence_of_warnings x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `None -> Py.none
)); ("depth", Wrap_utils.Option.map depth Py.Int.of_int); ("check_fpu_mode", check_fpu_mode)])
                       |> of_pyobject
                  let bench ?label ?verbose ?extra_argv self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "bench"
                       [||]
                       (Wrap_utils.keyword_args [("label", Wrap_utils.Option.map label (function
| `Full -> Py.String.of_string "full"
| `Fast -> Py.String.of_string "fast"
| `T -> Py.String.of_string ""
| `Attribute_identifier x -> Wrap_utils.id x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("extra_argv", Wrap_utils.Option.map extra_argv Obj.to_pyobject)])
                       |> Py.Bool.to_bool
let prepare_test_args ?label ?verbose ?extra_argv ?doctests ?coverage ?timer self =
   Py.Module.get_function_with_keywords (to_pyobject self) "prepare_test_args"
     [||]
     (Wrap_utils.keyword_args [("label", label); ("verbose", verbose); ("extra_argv", extra_argv); ("doctests", doctests); ("coverage", coverage); ("timer", timer)])

                  let test ?label ?verbose ?extra_argv ?doctests ?coverage ?raise_warnings ?timer self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "test"
                       [||]
                       (Wrap_utils.keyword_args [("label", Wrap_utils.Option.map label (function
| `Full -> Py.String.of_string "full"
| `Fast -> Py.String.of_string "fast"
| `T -> Py.String.of_string ""
| `Attribute_identifier x -> Wrap_utils.id x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("extra_argv", Wrap_utils.Option.map extra_argv Obj.to_pyobject); ("doctests", Wrap_utils.Option.map doctests Py.Bool.of_bool); ("coverage", Wrap_utils.Option.map coverage Py.Bool.of_bool); ("raise_warnings", Wrap_utils.Option.map raise_warnings (function
| `Sequence_of_warnings x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("timer", Wrap_utils.Option.map timer (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
))])


let package_path_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "package_path" with
  | None -> failwith "attribute package_path not found"
  | Some x -> if Py.is_none x then None else Some (Py.String.to_string x)

let package_path self = match package_path_opt self with
  | None -> raise Not_found
  | Some x -> x

let package_name_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "package_name" with
  | None -> failwith "attribute package_name not found"
  | Some x -> if Py.is_none x then None else Some (Py.String.to_string x)

let package_name self = match package_name_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module TooHardError = struct
type tag = [`TooHardError]
type t = [`BaseException | `Object | `TooHardError] Obj.t
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
module VisibleDeprecationWarning = struct
type tag = [`VisibleDeprecationWarning]
type t = [`BaseException | `Object | `VisibleDeprecationWarning] Obj.t
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
module Bool = struct
type tag = [`Bool]
type t = [`Bool | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create x =
   Py.Module.get_function_with_keywords __wrap_namespace "bool"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])
     |> of_pyobject
let as_integer_ratio self =
   Py.Module.get_function_with_keywords (to_pyobject self) "as_integer_ratio"
     [||]
     []

let bit_length self =
   Py.Module.get_function_with_keywords (to_pyobject self) "bit_length"
     [||]
     []

let from_bytes ?signed ~bytes ~byteorder self =
   Py.Module.get_function_with_keywords (to_pyobject self) "from_bytes"
     [||]
     (Wrap_utils.keyword_args [("signed", signed); ("bytes", Some(bytes )); ("byteorder", Some(byteorder ))])

let to_bytes ?signed ~length ~byteorder self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_bytes"
     [||]
     (Wrap_utils.keyword_args [("signed", signed); ("length", Some(length )); ("byteorder", Some(byteorder ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Bool8 = struct
type tag = [`Bool_]
type t = [`Bool_ | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Broadcast = struct
type tag = [`Broadcast]
type t = [`Broadcast | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Busdaycalendar = struct
type tag = [`Busdaycalendar]
type t = [`Busdaycalendar | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?weekmask ?holidays () =
                     Py.Module.get_function_with_keywords __wrap_namespace "busdaycalendar"
                       [||]
                       (Wrap_utils.keyword_args [("weekmask", Wrap_utils.Option.map weekmask (function
| `Array_like_of_bool x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("holidays", holidays)])
                       |> of_pyobject

let note_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "Note" with
  | None -> failwith "attribute Note not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let note self = match note_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Byte = struct
type tag = [`Int8]
type t = [`Int8 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Bytes0 = struct
type tag = [`Bytes_]
type t = [`Bytes_ | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create iterable_of_ints =
   Py.Module.get_function_with_keywords __wrap_namespace "bytes0"
     [||]
     (Wrap_utils.keyword_args [("iterable_of_ints", Some(iterable_of_ints ))])
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let center ?fillchar ~width self =
   Py.Module.get_function_with_keywords (to_pyobject self) "center"
     (Array.of_list @@ List.concat [(match fillchar with None -> [] | Some x -> [x ]);[width ]])
     []

let count ?start ?end_ ~sub self =
   Py.Module.get_function_with_keywords (to_pyobject self) "count"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("sub", Some(sub ))])

let decode ?encoding ?errors self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decode"
     [||]
     (Wrap_utils.keyword_args [("encoding", encoding); ("errors", errors)])

let endswith ?start ?end_ ~suffix self =
   Py.Module.get_function_with_keywords (to_pyobject self) "endswith"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("suffix", Some(suffix ))])

let expandtabs ?tabsize self =
   Py.Module.get_function_with_keywords (to_pyobject self) "expandtabs"
     [||]
     (Wrap_utils.keyword_args [("tabsize", tabsize)])

let find ?start ?end_ ~sub self =
   Py.Module.get_function_with_keywords (to_pyobject self) "find"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("sub", Some(sub ))])

let fromhex ~string self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fromhex"
     (Array.of_list @@ List.concat [[string ]])
     []

let index ?start ?end_ ~sub self =
   Py.Module.get_function_with_keywords (to_pyobject self) "index"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("sub", Some(sub ))])

let join ~iterable_of_bytes self =
   Py.Module.get_function_with_keywords (to_pyobject self) "join"
     (Array.of_list @@ List.concat [[iterable_of_bytes ]])
     []

let ljust ?fillchar ~width self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ljust"
     (Array.of_list @@ List.concat [(match fillchar with None -> [] | Some x -> [x ]);[width ]])
     []

let lstrip ?bytes self =
   Py.Module.get_function_with_keywords (to_pyobject self) "lstrip"
     (Array.of_list @@ List.concat [(match bytes with None -> [] | Some x -> [x ])])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let partition ~sep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "partition"
     (Array.of_list @@ List.concat [[sep ]])
     []

let replace ?count ~old ~new_ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "replace"
     (Array.of_list @@ List.concat [(match count with None -> [] | Some x -> [x ]);[old ];[new_ ]])
     []

let rfind ?start ?end_ ~sub self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rfind"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("sub", Some(sub ))])

let rindex ?start ?end_ ~sub self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rindex"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("sub", Some(sub ))])

let rjust ?fillchar ~width self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rjust"
     (Array.of_list @@ List.concat [(match fillchar with None -> [] | Some x -> [x ]);[width ]])
     []

let rpartition ~sep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rpartition"
     (Array.of_list @@ List.concat [[sep ]])
     []

let rsplit ?sep ?maxsplit self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rsplit"
     [||]
     (Wrap_utils.keyword_args [("sep", sep); ("maxsplit", maxsplit)])

let rstrip ?bytes self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rstrip"
     (Array.of_list @@ List.concat [(match bytes with None -> [] | Some x -> [x ])])
     []

let split ?sep ?maxsplit self =
   Py.Module.get_function_with_keywords (to_pyobject self) "split"
     [||]
     (Wrap_utils.keyword_args [("sep", sep); ("maxsplit", maxsplit)])

let splitlines ?keepends self =
   Py.Module.get_function_with_keywords (to_pyobject self) "splitlines"
     [||]
     (Wrap_utils.keyword_args [("keepends", keepends)])

let startswith ?start ?end_ ~prefix self =
   Py.Module.get_function_with_keywords (to_pyobject self) "startswith"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("prefix", Some(prefix ))])

let strip ?bytes self =
   Py.Module.get_function_with_keywords (to_pyobject self) "strip"
     (Array.of_list @@ List.concat [(match bytes with None -> [] | Some x -> [x ])])
     []

let translate ?delete ~table self =
   Py.Module.get_function_with_keywords (to_pyobject self) "translate"
     (Array.of_list @@ List.concat [[table ]])
     (Wrap_utils.keyword_args [("delete", delete)])

let zfill ~width self =
   Py.Module.get_function_with_keywords (to_pyobject self) "zfill"
     (Array.of_list @@ List.concat [[width ]])
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Cdouble = struct
type tag = [`Complex128]
type t = [`Complex128 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?real ?imag () =
   Py.Module.get_function_with_keywords __wrap_namespace "cdouble"
     [||]
     (Wrap_utils.keyword_args [("real", real); ("imag", imag)])
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Cfloat = struct
type tag = [`Complex128]
type t = [`Complex128 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?real ?imag () =
   Py.Module.get_function_with_keywords __wrap_namespace "cfloat"
     [||]
     (Wrap_utils.keyword_args [("real", real); ("imag", imag)])
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Character = struct
type tag = [`Character]
type t = [`Character | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "character"
     [||]
     []
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Clongdouble = struct
type tag = [`Complex256]
type t = [`Complex256 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Clongfloat = struct
type tag = [`Complex256]
type t = [`Complex256 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Complex = struct
type tag = [`Complex]
type t = [`Complex | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?real ?imag () =
   Py.Module.get_function_with_keywords __wrap_namespace "complex"
     [||]
     (Wrap_utils.keyword_args [("real", real); ("imag", imag)])
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Complex256 = struct
type tag = [`Complex256]
type t = [`Complex256 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Complex64 = struct
type tag = [`Complex64]
type t = [`Complex64 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Complexfloating = struct
type tag = [`Complexfloating]
type t = [`Complexfloating | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "complexfloating"
     [||]
     []
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Csingle = struct
type tag = [`Complex64]
type t = [`Complex64 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Datetime64 = struct
type tag = [`Datetime64]
type t = [`Datetime64 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Double = struct
type tag = [`Float64]
type t = [`Float64 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?x () =
   Py.Module.get_function_with_keywords __wrap_namespace "double"
     (Array.of_list @@ List.concat [(match x with None -> [] | Some x -> [x ])])
     []
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let fromhex ~string self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fromhex"
     (Array.of_list @@ List.concat [[string ]])
     []

let hex self =
   Py.Module.get_function_with_keywords (to_pyobject self) "hex"
     [||]
     []

let is_integer self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_integer"
     [||]
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Errstate = struct
type tag = [`Errstate]
type t = [`Errstate | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?call ?kwargs () =
   Py.Module.get_function_with_keywords __wrap_namespace "errstate"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("call", call)]) (match kwargs with None -> [] | Some x -> x))
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Finfo = struct
type tag = [`Finfo]
type t = [`Finfo | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create dtype =
                     Py.Module.get_function_with_keywords __wrap_namespace "finfo"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Some(dtype |> (function
| `Instance x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
| `Dtype x -> Dtype.to_pyobject x
)))])
                       |> of_pyobject

let bits_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "bits" with
  | None -> failwith "attribute bits not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let bits self = match bits_opt self with
  | None -> raise Not_found
  | Some x -> x

let eps_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "eps" with
  | None -> failwith "attribute eps not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let eps self = match eps_opt self with
  | None -> raise Not_found
  | Some x -> x

let epsneg_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "epsneg" with
  | None -> failwith "attribute epsneg not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let epsneg self = match epsneg_opt self with
  | None -> raise Not_found
  | Some x -> x

let iexp_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "iexp" with
  | None -> failwith "attribute iexp not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let iexp self = match iexp_opt self with
  | None -> raise Not_found
  | Some x -> x

let machar_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "machar" with
  | None -> failwith "attribute machar not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let machar self = match machar_opt self with
  | None -> raise Not_found
  | Some x -> x

let machep_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "machep" with
  | None -> failwith "attribute machep not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let machep self = match machep_opt self with
  | None -> raise Not_found
  | Some x -> x

let max_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "max" with
  | None -> failwith "attribute max not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let max self = match max_opt self with
  | None -> raise Not_found
  | Some x -> x

let maxexp_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "maxexp" with
  | None -> failwith "attribute maxexp not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let maxexp self = match maxexp_opt self with
  | None -> raise Not_found
  | Some x -> x

let min_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "min" with
  | None -> failwith "attribute min not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let min self = match min_opt self with
  | None -> raise Not_found
  | Some x -> x

let minexp_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "minexp" with
  | None -> failwith "attribute minexp not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let minexp self = match minexp_opt self with
  | None -> raise Not_found
  | Some x -> x

let negep_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "negep" with
  | None -> failwith "attribute negep not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let negep self = match negep_opt self with
  | None -> raise Not_found
  | Some x -> x

let nexp_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nexp" with
  | None -> failwith "attribute nexp not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let nexp self = match nexp_opt self with
  | None -> raise Not_found
  | Some x -> x

let nmant_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nmant" with
  | None -> failwith "attribute nmant not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let nmant self = match nmant_opt self with
  | None -> raise Not_found
  | Some x -> x

let precision_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "precision" with
  | None -> failwith "attribute precision not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let precision self = match precision_opt self with
  | None -> raise Not_found
  | Some x -> x

let resolution_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "resolution" with
  | None -> failwith "attribute resolution not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let resolution self = match resolution_opt self with
  | None -> raise Not_found
  | Some x -> x

let tiny_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "tiny" with
  | None -> failwith "attribute tiny not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let tiny self = match tiny_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Flatiter = struct
type tag = [`Flatiter]
type t = [`Flatiter | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "flatiter"
     [||]
     []
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

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Flexible = struct
type tag = [`Flexible]
type t = [`Flexible | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "flexible"
     [||]
     []
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Float = struct
type tag = [`Float]
type t = [`Float | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?x () =
   Py.Module.get_function_with_keywords __wrap_namespace "float"
     (Array.of_list @@ List.concat [(match x with None -> [] | Some x -> [x ])])
     []
     |> of_pyobject
let as_integer_ratio self =
   Py.Module.get_function_with_keywords (to_pyobject self) "as_integer_ratio"
     [||]
     []

let conjugate self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conjugate"
     [||]
     []

let fromhex ~string self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fromhex"
     (Array.of_list @@ List.concat [[string ]])
     []

let hex self =
   Py.Module.get_function_with_keywords (to_pyobject self) "hex"
     [||]
     []

let is_integer self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_integer"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Float128 = struct
type tag = [`Float128]
type t = [`Float128 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Float16 = struct
type tag = [`Float16]
type t = [`Float16 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Float32 = struct
type tag = [`Float32]
type t = [`Float32 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Floating = struct
type tag = [`Floating]
type t = [`Floating | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "floating"
     [||]
     []
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Format_parser = struct
type tag = [`Format_parser]
type t = [`Format_parser | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?aligned ?byteorder ~formats ~names ~titles () =
                     Py.Module.get_function_with_keywords __wrap_namespace "format_parser"
                       [||]
                       (Wrap_utils.keyword_args [("aligned", Wrap_utils.Option.map aligned Py.Bool.of_bool); ("byteorder", Wrap_utils.Option.map byteorder Py.String.of_string); ("formats", Some(formats |> (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `S x -> Py.String.of_string x
))); ("names", Some(names |> (function
| `List_tuple_of_str x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
))); ("titles", Some(titles ))])
                       |> of_pyobject

let dtype_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dtype" with
  | None -> failwith "attribute dtype not found"
  | Some x -> if Py.is_none x then None else Some (Dtype.of_pyobject x)

let dtype self = match dtype_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Generic = struct
type tag = [`Generic]
type t = [`Generic | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "generic"
     [||]
     []
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Half = struct
type tag = [`Float16]
type t = [`Float16 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Iinfo = struct
type tag = [`Iinfo]
type t = [`Iinfo | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create int_type =
                     Py.Module.get_function_with_keywords __wrap_namespace "iinfo"
                       [||]
                       (Wrap_utils.keyword_args [("int_type", Some(int_type |> (function
| `Dtype x -> Dtype.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])
                       |> of_pyobject

let bits_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "bits" with
  | None -> failwith "attribute bits not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let bits self = match bits_opt self with
  | None -> raise Not_found
  | Some x -> x

let min_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "min" with
  | None -> failwith "attribute min not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let min self = match min_opt self with
  | None -> raise Not_found
  | Some x -> x

let max_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "max" with
  | None -> failwith "attribute max not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let max self = match max_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Inexact = struct
type tag = [`Inexact]
type t = [`Inexact | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "inexact"
     [||]
     []
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Int = struct
type tag = [`Int]
type t = [`Int | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?x () =
   Py.Module.get_function_with_keywords __wrap_namespace "int"
     [||]
     (Wrap_utils.keyword_args [("x", x)])
     |> of_pyobject
let as_integer_ratio self =
   Py.Module.get_function_with_keywords (to_pyobject self) "as_integer_ratio"
     [||]
     []

let bit_length self =
   Py.Module.get_function_with_keywords (to_pyobject self) "bit_length"
     [||]
     []

let from_bytes ?signed ~bytes ~byteorder self =
   Py.Module.get_function_with_keywords (to_pyobject self) "from_bytes"
     [||]
     (Wrap_utils.keyword_args [("signed", signed); ("bytes", Some(bytes )); ("byteorder", Some(byteorder ))])

let to_bytes ?signed ~length ~byteorder self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_bytes"
     [||]
     (Wrap_utils.keyword_args [("signed", signed); ("length", Some(length )); ("byteorder", Some(byteorder ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Int0 = struct
type tag = [`Int64]
type t = [`Int64 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Int16 = struct
type tag = [`Int16]
type t = [`Int16 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Int32 = struct
type tag = [`Int32]
type t = [`Int32 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Int8 = struct
type tag = [`Int8]
type t = [`Int8 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Intc = struct
type tag = [`Int32]
type t = [`Int32 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Integer = struct
type tag = [`Integer]
type t = [`Integer | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "integer"
     [||]
     []
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Intp = struct
type tag = [`Int64]
type t = [`Int64 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Longlong = struct
type tag = [`Longlong]
type t = [`Longlong | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
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
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("data", Some(data |> (function
| `Ndarray x -> Obj.to_pyobject x
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
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Obj.to_pyobject)])

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
| `Dtype x -> Dtype.to_pyobject x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let byteswap ?inplace self =
   Py.Module.get_function_with_keywords (to_pyobject self) "byteswap"
     [||]
     (Wrap_utils.keyword_args [("inplace", Wrap_utils.Option.map inplace Py.Bool.of_bool)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
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
| `Path x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
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
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let getA self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getA"
     [||]
     []
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let getA1 self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getA1"
     [||]
     []
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
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
| `Dtype x -> Dtype.to_pyobject x
| `S x -> Py.String.of_string x
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
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
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
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
| `I x -> Py.Int.of_int x
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
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let repeat ?axis ~repeats self =
   Py.Module.get_function_with_keywords (to_pyobject self) "repeat"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("repeats", Some(repeats ))])

let reshape ?order shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "reshape"
     [||]
     (Wrap_utils.keyword_args [("order", order); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])

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
| `Heapsort -> Py.String.of_string "heapsort"
| `Mergesort -> Py.String.of_string "mergesort"
| `Stable -> Py.String.of_string "stable"
| `Quicksort -> Py.String.of_string "quicksort"
)); ("order", Wrap_utils.Option.map order (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `S x -> Py.String.of_string x
))])

let squeeze ?axis self =
   Py.Module.get_function_with_keywords (to_pyobject self) "squeeze"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
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
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let var ?axis ?dtype ?out ?ddof self =
   Py.Module.get_function_with_keywords (to_pyobject self) "var"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("ddof", ddof)])

                  let view ?dtype ?type_ self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "view"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype (function
| `Ndarray_sub_class x -> Wrap_utils.id x
| `Dtype x -> Dtype.to_pyobject x
)); ("type", type_)])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Memmap = struct
type tag = [`Memmap]
type t = [`ArrayLike | `Memmap | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?dtype ?mode ?offset ?shape ?order ~filename () =
                     Py.Module.get_function_with_keywords __wrap_namespace "memmap"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("mode", Wrap_utils.Option.map mode (function
| `R_plus -> Py.String.of_string "r+"
| `R -> Py.String.of_string "r"
| `W_plus -> Py.String.of_string "w+"
| `C -> Py.String.of_string "c"
)); ("offset", Wrap_utils.Option.map offset Py.Int.of_int); ("shape", Wrap_utils.Option.map shape (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("filename", Some(filename |> (function
| `S x -> Py.String.of_string x
| `PyObject x -> Wrap_utils.id x
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
| `Dtype x -> Dtype.to_pyobject x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let byteswap ?inplace self =
   Py.Module.get_function_with_keywords (to_pyobject self) "byteswap"
     [||]
     (Wrap_utils.keyword_args [("inplace", Wrap_utils.Option.map inplace Py.Bool.of_bool)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
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
| `Path x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
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
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let flush self =
   Py.Module.get_function_with_keywords (to_pyobject self) "flush"
     [||]
     []

                  let getfield ?offset ~dtype self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "getfield"
                       [||]
                       (Wrap_utils.keyword_args [("offset", Wrap_utils.Option.map offset Py.Int.of_int); ("dtype", Some(dtype |> (function
| `Dtype x -> Dtype.to_pyobject x
| `S x -> Py.String.of_string x
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
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
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
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
| `I x -> Py.Int.of_int x
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

let reshape ?order shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "reshape"
     [||]
     (Wrap_utils.keyword_args [("order", order); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])

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
| `Heapsort -> Py.String.of_string "heapsort"
| `Mergesort -> Py.String.of_string "mergesort"
| `Stable -> Py.String.of_string "stable"
| `Quicksort -> Py.String.of_string "quicksort"
)); ("order", Wrap_utils.Option.map order (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `S x -> Py.String.of_string x
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
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let var ?axis ?dtype ?out ?ddof ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "var"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("ddof", ddof); ("keepdims", keepdims)])

                  let view ?dtype ?type_ self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "view"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype (function
| `Ndarray_sub_class x -> Wrap_utils.id x
| `Dtype x -> Dtype.to_pyobject x
)); ("type", type_)])


let filename_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "filename" with
  | None -> failwith "attribute filename not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let filename self = match filename_opt self with
  | None -> raise Not_found
  | Some x -> x

let offset_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "offset" with
  | None -> failwith "attribute offset not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let offset self = match offset_opt self with
  | None -> raise Not_found
  | Some x -> x

let mode_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "mode" with
  | None -> failwith "attribute mode not found"
  | Some x -> if Py.is_none x then None else Some (Py.String.to_string x)

let mode self = match mode_opt self with
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
                  let create ?dtype ?buffer ?offset ?strides ?order shape =
                     Py.Module.get_function_with_keywords __wrap_namespace "ndarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("buffer", buffer); ("offset", Wrap_utils.Option.map offset Py.Int.of_int); ("strides", Wrap_utils.Option.map strides (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])
                       |> of_pyobject
let get ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key |> Wrap_utils.Index.to_pyobject]])
     []
     |> of_pyobject
let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let set ~key ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     (Array.of_list @@ List.concat [[key |> Wrap_utils.Index.to_pyobject];[value |> to_pyobject]])
     []
     |> (fun _ -> ())
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
| `Dtype x -> Dtype.to_pyobject x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let byteswap ?inplace self =
   Py.Module.get_function_with_keywords (to_pyobject self) "byteswap"
     [||]
     (Wrap_utils.keyword_args [("inplace", Wrap_utils.Option.map inplace Py.Bool.of_bool)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
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
| `Path x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
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
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let getfield ?offset ~dtype self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "getfield"
                       [||]
                       (Wrap_utils.keyword_args [("offset", Wrap_utils.Option.map offset Py.Int.of_int); ("dtype", Some(dtype |> (function
| `Dtype x -> Dtype.to_pyobject x
| `S x -> Py.String.of_string x
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
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
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
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
| `I x -> Py.Int.of_int x
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

let reshape ?order shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "reshape"
     [||]
     (Wrap_utils.keyword_args [("order", order); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])

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
| `Heapsort -> Py.String.of_string "heapsort"
| `Mergesort -> Py.String.of_string "mergesort"
| `Stable -> Py.String.of_string "stable"
| `Quicksort -> Py.String.of_string "quicksort"
)); ("order", Wrap_utils.Option.map order (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `S x -> Py.String.of_string x
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
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let var ?axis ?dtype ?out ?ddof ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "var"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("ddof", ddof); ("keepdims", keepdims)])

                  let view ?dtype ?type_ self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "view"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype (function
| `Ndarray_sub_class x -> Wrap_utils.id x
| `Dtype x -> Dtype.to_pyobject x
)); ("type", type_)])


let t_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "T" with
  | None -> failwith "attribute T not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) x)

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
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) x)

let imag self = match imag_opt self with
  | None -> raise Not_found
  | Some x -> x

let real_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "real" with
  | None -> failwith "attribute real not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) x)

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
  | Some x -> if Py.is_none x then None else Some ((fun py -> Py.List.to_list_map (Py.Int.to_int) py) x)

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
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) x)

let base self = match base_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Ndenumerate = struct
type tag = [`Ndenumerate]
type t = [`Ndenumerate | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create arr =
   Py.Module.get_function_with_keywords __wrap_namespace "ndenumerate"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr |> Obj.to_pyobject))])
     |> of_pyobject
let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Ndindex = struct
type tag = [`Ndindex]
type t = [`Ndindex | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create shape =
   Py.Module.get_function_with_keywords __wrap_namespace "ndindex"
     (Array.of_list @@ List.concat [(List.map Py.Int.of_int shape)])
     []
     |> of_pyobject
let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let ndincr self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ndincr"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Nditer = struct
type tag = [`Nditer]
type t = [`Nditer | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?flags ?op_flags ?op_dtypes ?order ?casting ?op_axes ?itershape ?buffersize ~op () =
                     Py.Module.get_function_with_keywords __wrap_namespace "nditer"
                       [||]
                       (Wrap_utils.keyword_args [("flags", flags); ("op_flags", op_flags); ("op_dtypes", Wrap_utils.Option.map op_dtypes (function
| `Tuple_of_dtype_s_ x -> Wrap_utils.id x
| `Dtype x -> Dtype.to_pyobject x
)); ("order", Wrap_utils.Option.map order (function
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
)); ("op_axes", op_axes); ("itershape", Wrap_utils.Option.map itershape (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("buffersize", Wrap_utils.Option.map buffersize Py.Int.of_int); ("op", Some(op |> (function
| `Ndarray x -> Obj.to_pyobject x
| `Sequence_of_array_like x -> Wrap_utils.id x
)))])
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


let dtypes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dtypes" with
  | None -> failwith "attribute dtypes not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let dtypes self = match dtypes_opt self with
  | None -> raise Not_found
  | Some x -> x

let finished_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "finished" with
  | None -> failwith "attribute finished not found"
  | Some x -> if Py.is_none x then None else Some (Py.Bool.to_bool x)

let finished self = match finished_opt self with
  | None -> raise Not_found
  | Some x -> x

let has_delayed_bufalloc_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "has_delayed_bufalloc" with
  | None -> failwith "attribute has_delayed_bufalloc not found"
  | Some x -> if Py.is_none x then None else Some (Py.Bool.to_bool x)

let has_delayed_bufalloc self = match has_delayed_bufalloc_opt self with
  | None -> raise Not_found
  | Some x -> x

let has_index_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "has_index" with
  | None -> failwith "attribute has_index not found"
  | Some x -> if Py.is_none x then None else Some (Py.Bool.to_bool x)

let has_index self = match has_index_opt self with
  | None -> raise Not_found
  | Some x -> x

let has_multi_index_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "has_multi_index" with
  | None -> failwith "attribute has_multi_index not found"
  | Some x -> if Py.is_none x then None else Some (Py.Bool.to_bool x)

let has_multi_index self = match has_multi_index_opt self with
  | None -> raise Not_found
  | Some x -> x

let index_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "index" with
  | None -> failwith "attribute index not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let index self = match index_opt self with
  | None -> raise Not_found
  | Some x -> x

let iterationneedsapi_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "iterationneedsapi" with
  | None -> failwith "attribute iterationneedsapi not found"
  | Some x -> if Py.is_none x then None else Some (Py.Bool.to_bool x)

let iterationneedsapi self = match iterationneedsapi_opt self with
  | None -> raise Not_found
  | Some x -> x

let iterindex_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "iterindex" with
  | None -> failwith "attribute iterindex not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let iterindex self = match iterindex_opt self with
  | None -> raise Not_found
  | Some x -> x

let itersize_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "itersize" with
  | None -> failwith "attribute itersize not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let itersize self = match itersize_opt self with
  | None -> raise Not_found
  | Some x -> x

let itviews_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "itviews" with
  | None -> failwith "attribute itviews not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let itviews self = match itviews_opt self with
  | None -> raise Not_found
  | Some x -> x

let multi_index_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "multi_index" with
  | None -> failwith "attribute multi_index not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let multi_index self = match multi_index_opt self with
  | None -> raise Not_found
  | Some x -> x

let ndim_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "ndim" with
  | None -> failwith "attribute ndim not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let ndim self = match ndim_opt self with
  | None -> raise Not_found
  | Some x -> x

let nop_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nop" with
  | None -> failwith "attribute nop not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let nop self = match nop_opt self with
  | None -> raise Not_found
  | Some x -> x

let operands_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "operands" with
  | None -> failwith "attribute operands not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let operands self = match operands_opt self with
  | None -> raise Not_found
  | Some x -> x

let shape_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "shape" with
  | None -> failwith "attribute shape not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> Py.List.to_list_map (Py.Int.to_int) py) x)

let shape self = match shape_opt self with
  | None -> raise Not_found
  | Some x -> x

let value_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "value" with
  | None -> failwith "attribute value not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let value self = match value_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Number = struct
type tag = [`Number]
type t = [`Number | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "number"
     [||]
     []
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Object = struct
type tag = [`Object]
type t = [`Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "object"
     [||]
     []
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Object0 = struct
type tag = [`Object_]
type t = [`Object | `Object_] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Poly1d = struct
type tag = [`Poly1d]
type t = [`Object | `Poly1d] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?r ?variable ~c_or_r () =
   Py.Module.get_function_with_keywords __wrap_namespace "poly1d"
     [||]
     (Wrap_utils.keyword_args [("r", Wrap_utils.Option.map r Py.Bool.of_bool); ("variable", Wrap_utils.Option.map variable Py.String.of_string); ("c_or_r", Some(c_or_r |> Obj.to_pyobject))])
     |> of_pyobject
let __getitem__ ~val_ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("val", Some(val_ ))])

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let __setitem__ ~key ~val_ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key )); ("val", Some(val_ ))])

let deriv ?m self =
   Py.Module.get_function_with_keywords (to_pyobject self) "deriv"
     [||]
     (Wrap_utils.keyword_args [("m", m)])

let integ ?m ?k self =
   Py.Module.get_function_with_keywords (to_pyobject self) "integ"
     [||]
     (Wrap_utils.keyword_args [("m", m); ("k", k)])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Recarray = struct
type tag = [`Recarray]
type t = [`ArrayLike | `Object | `Recarray] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?dtype ?buf ?offset ?strides ?formats ?names ?titles ?byteorder ?aligned ?order shape =
   Py.Module.get_function_with_keywords __wrap_namespace "recarray"
     [||]
     (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("buf", buf); ("offset", offset); ("strides", strides); ("formats", formats); ("names", Wrap_utils.Option.map names (Py.List.of_list_map Py.String.of_string)); ("titles", titles); ("byteorder", byteorder); ("aligned", aligned); ("order", order); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])
     |> of_pyobject
let __getitem__ ~indx self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("indx", Some(indx ))])

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
| `Dtype x -> Dtype.to_pyobject x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let byteswap ?inplace self =
   Py.Module.get_function_with_keywords (to_pyobject self) "byteswap"
     [||]
     (Wrap_utils.keyword_args [("inplace", Wrap_utils.Option.map inplace Py.Bool.of_bool)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
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
| `Path x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))])

let dumps self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dumps"
     [||]
     []

let field ?val_ ~attr self =
   Py.Module.get_function_with_keywords (to_pyobject self) "field"
     [||]
     (Wrap_utils.keyword_args [("val", val_); ("attr", Some(attr ))])

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
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let getfield ?offset ~dtype self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "getfield"
                       [||]
                       (Wrap_utils.keyword_args [("offset", Wrap_utils.Option.map offset Py.Int.of_int); ("dtype", Some(dtype |> (function
| `Dtype x -> Dtype.to_pyobject x
| `S x -> Py.String.of_string x
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
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
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
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
| `I x -> Py.Int.of_int x
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

let reshape ?order shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "reshape"
     [||]
     (Wrap_utils.keyword_args [("order", order); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])

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
| `Heapsort -> Py.String.of_string "heapsort"
| `Mergesort -> Py.String.of_string "mergesort"
| `Stable -> Py.String.of_string "stable"
| `Quicksort -> Py.String.of_string "quicksort"
)); ("order", Wrap_utils.Option.map order (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `S x -> Py.String.of_string x
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
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let var ?axis ?dtype ?out ?ddof ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "var"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("ddof", ddof); ("keepdims", keepdims)])

                  let view ?dtype ?type_ self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "view"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype (function
| `Ndarray_sub_class x -> Wrap_utils.id x
| `Dtype x -> Dtype.to_pyobject x
)); ("type", type_)])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Record = struct
type tag = [`Record]
type t = [`Object | `Record] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~indx self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("indx", Some(indx ))])

let __setitem__ ~key ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     (Array.of_list @@ List.concat [[key ];[value ]])
     []

let pprint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "pprint"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Short = struct
type tag = [`Int16]
type t = [`Int16 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Signedinteger = struct
type tag = [`Signedinteger]
type t = [`Object | `Signedinteger] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "signedinteger"
     [||]
     []
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Single = struct
type tag = [`Float32]
type t = [`Float32 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Str = struct
type tag = [`Str]
type t = [`Object | `Str] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?object_ () =
   Py.Module.get_function_with_keywords __wrap_namespace "str"
     [||]
     (Wrap_utils.keyword_args [("object", object_)])
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let capitalize self =
   Py.Module.get_function_with_keywords (to_pyobject self) "capitalize"
     [||]
     []

let casefold self =
   Py.Module.get_function_with_keywords (to_pyobject self) "casefold"
     [||]
     []

let center ?fillchar ~width self =
   Py.Module.get_function_with_keywords (to_pyobject self) "center"
     (Array.of_list @@ List.concat [(match fillchar with None -> [] | Some x -> [x ]);[width ]])
     []

let count ?start ?end_ ~sub self =
   Py.Module.get_function_with_keywords (to_pyobject self) "count"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("sub", Some(sub ))])

let encode ?encoding ?errors self =
   Py.Module.get_function_with_keywords (to_pyobject self) "encode"
     [||]
     (Wrap_utils.keyword_args [("encoding", encoding); ("errors", errors)])

let endswith ?start ?end_ ~suffix self =
   Py.Module.get_function_with_keywords (to_pyobject self) "endswith"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("suffix", Some(suffix ))])

let expandtabs ?tabsize self =
   Py.Module.get_function_with_keywords (to_pyobject self) "expandtabs"
     [||]
     (Wrap_utils.keyword_args [("tabsize", tabsize)])

let find ?start ?end_ ~sub self =
   Py.Module.get_function_with_keywords (to_pyobject self) "find"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("sub", Some(sub ))])

let format ?kwargs args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "format"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

let format_map ~mapping self =
   Py.Module.get_function_with_keywords (to_pyobject self) "format_map"
     [||]
     (Wrap_utils.keyword_args [("mapping", Some(mapping ))])

let index ?start ?end_ ~sub self =
   Py.Module.get_function_with_keywords (to_pyobject self) "index"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("sub", Some(sub ))])

let isalnum self =
   Py.Module.get_function_with_keywords (to_pyobject self) "isalnum"
     [||]
     []

let isalpha self =
   Py.Module.get_function_with_keywords (to_pyobject self) "isalpha"
     [||]
     []

let isascii self =
   Py.Module.get_function_with_keywords (to_pyobject self) "isascii"
     [||]
     []

let isdecimal self =
   Py.Module.get_function_with_keywords (to_pyobject self) "isdecimal"
     [||]
     []

let isdigit self =
   Py.Module.get_function_with_keywords (to_pyobject self) "isdigit"
     [||]
     []

let isidentifier self =
   Py.Module.get_function_with_keywords (to_pyobject self) "isidentifier"
     [||]
     []

let islower self =
   Py.Module.get_function_with_keywords (to_pyobject self) "islower"
     [||]
     []

let isnumeric self =
   Py.Module.get_function_with_keywords (to_pyobject self) "isnumeric"
     [||]
     []

let isprintable self =
   Py.Module.get_function_with_keywords (to_pyobject self) "isprintable"
     [||]
     []

let isspace self =
   Py.Module.get_function_with_keywords (to_pyobject self) "isspace"
     [||]
     []

let istitle self =
   Py.Module.get_function_with_keywords (to_pyobject self) "istitle"
     [||]
     []

let isupper self =
   Py.Module.get_function_with_keywords (to_pyobject self) "isupper"
     [||]
     []

let join ~iterable self =
   Py.Module.get_function_with_keywords (to_pyobject self) "join"
     (Array.of_list @@ List.concat [[iterable ]])
     []

let ljust ?fillchar ~width self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ljust"
     (Array.of_list @@ List.concat [(match fillchar with None -> [] | Some x -> [x ]);[width ]])
     []

let lower self =
   Py.Module.get_function_with_keywords (to_pyobject self) "lower"
     [||]
     []

let lstrip ?chars self =
   Py.Module.get_function_with_keywords (to_pyobject self) "lstrip"
     (Array.of_list @@ List.concat [(match chars with None -> [] | Some x -> [x ])])
     []

let partition ~sep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "partition"
     (Array.of_list @@ List.concat [[sep ]])
     []

let replace ?count ~old ~new_ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "replace"
     (Array.of_list @@ List.concat [(match count with None -> [] | Some x -> [x ]);[old ];[new_ ]])
     []

let rfind ?start ?end_ ~sub self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rfind"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("sub", Some(sub ))])

let rindex ?start ?end_ ~sub self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rindex"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("sub", Some(sub ))])

let rjust ?fillchar ~width self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rjust"
     (Array.of_list @@ List.concat [(match fillchar with None -> [] | Some x -> [x ]);[width ]])
     []

let rpartition ~sep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rpartition"
     (Array.of_list @@ List.concat [[sep ]])
     []

let rsplit ?sep ?maxsplit self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rsplit"
     [||]
     (Wrap_utils.keyword_args [("sep", sep); ("maxsplit", maxsplit)])

let rstrip ?chars self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rstrip"
     (Array.of_list @@ List.concat [(match chars with None -> [] | Some x -> [x ])])
     []

let split ?sep ?maxsplit self =
   Py.Module.get_function_with_keywords (to_pyobject self) "split"
     [||]
     (Wrap_utils.keyword_args [("sep", sep); ("maxsplit", maxsplit)])

let splitlines ?keepends self =
   Py.Module.get_function_with_keywords (to_pyobject self) "splitlines"
     [||]
     (Wrap_utils.keyword_args [("keepends", keepends)])

let startswith ?start ?end_ ~prefix self =
   Py.Module.get_function_with_keywords (to_pyobject self) "startswith"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("prefix", Some(prefix ))])

let strip ?chars self =
   Py.Module.get_function_with_keywords (to_pyobject self) "strip"
     (Array.of_list @@ List.concat [(match chars with None -> [] | Some x -> [x ])])
     []

let swapcase self =
   Py.Module.get_function_with_keywords (to_pyobject self) "swapcase"
     [||]
     []

let title self =
   Py.Module.get_function_with_keywords (to_pyobject self) "title"
     [||]
     []

let translate ~table self =
   Py.Module.get_function_with_keywords (to_pyobject self) "translate"
     (Array.of_list @@ List.concat [[table ]])
     []

let upper self =
   Py.Module.get_function_with_keywords (to_pyobject self) "upper"
     [||]
     []

let zfill ~width self =
   Py.Module.get_function_with_keywords (to_pyobject self) "zfill"
     (Array.of_list @@ List.concat [[width ]])
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Str0 = struct
type tag = [`Str_]
type t = [`Object | `Str_] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?object_ () =
   Py.Module.get_function_with_keywords __wrap_namespace "str0"
     [||]
     (Wrap_utils.keyword_args [("object", object_)])
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let capitalize self =
   Py.Module.get_function_with_keywords (to_pyobject self) "capitalize"
     [||]
     []

let casefold self =
   Py.Module.get_function_with_keywords (to_pyobject self) "casefold"
     [||]
     []

let center ?fillchar ~width self =
   Py.Module.get_function_with_keywords (to_pyobject self) "center"
     (Array.of_list @@ List.concat [(match fillchar with None -> [] | Some x -> [x ]);[width ]])
     []

let count ?start ?end_ ~sub self =
   Py.Module.get_function_with_keywords (to_pyobject self) "count"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("sub", Some(sub ))])

let encode ?encoding ?errors self =
   Py.Module.get_function_with_keywords (to_pyobject self) "encode"
     [||]
     (Wrap_utils.keyword_args [("encoding", encoding); ("errors", errors)])

let endswith ?start ?end_ ~suffix self =
   Py.Module.get_function_with_keywords (to_pyobject self) "endswith"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("suffix", Some(suffix ))])

let expandtabs ?tabsize self =
   Py.Module.get_function_with_keywords (to_pyobject self) "expandtabs"
     [||]
     (Wrap_utils.keyword_args [("tabsize", tabsize)])

let find ?start ?end_ ~sub self =
   Py.Module.get_function_with_keywords (to_pyobject self) "find"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("sub", Some(sub ))])

let format ?kwargs args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "format"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

let format_map ~mapping self =
   Py.Module.get_function_with_keywords (to_pyobject self) "format_map"
     [||]
     (Wrap_utils.keyword_args [("mapping", Some(mapping ))])

let index ?start ?end_ ~sub self =
   Py.Module.get_function_with_keywords (to_pyobject self) "index"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("sub", Some(sub ))])

let isalnum self =
   Py.Module.get_function_with_keywords (to_pyobject self) "isalnum"
     [||]
     []

let isalpha self =
   Py.Module.get_function_with_keywords (to_pyobject self) "isalpha"
     [||]
     []

let isascii self =
   Py.Module.get_function_with_keywords (to_pyobject self) "isascii"
     [||]
     []

let isdecimal self =
   Py.Module.get_function_with_keywords (to_pyobject self) "isdecimal"
     [||]
     []

let isdigit self =
   Py.Module.get_function_with_keywords (to_pyobject self) "isdigit"
     [||]
     []

let isidentifier self =
   Py.Module.get_function_with_keywords (to_pyobject self) "isidentifier"
     [||]
     []

let islower self =
   Py.Module.get_function_with_keywords (to_pyobject self) "islower"
     [||]
     []

let isnumeric self =
   Py.Module.get_function_with_keywords (to_pyobject self) "isnumeric"
     [||]
     []

let isprintable self =
   Py.Module.get_function_with_keywords (to_pyobject self) "isprintable"
     [||]
     []

let isspace self =
   Py.Module.get_function_with_keywords (to_pyobject self) "isspace"
     [||]
     []

let istitle self =
   Py.Module.get_function_with_keywords (to_pyobject self) "istitle"
     [||]
     []

let isupper self =
   Py.Module.get_function_with_keywords (to_pyobject self) "isupper"
     [||]
     []

let join ~iterable self =
   Py.Module.get_function_with_keywords (to_pyobject self) "join"
     (Array.of_list @@ List.concat [[iterable ]])
     []

let ljust ?fillchar ~width self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ljust"
     (Array.of_list @@ List.concat [(match fillchar with None -> [] | Some x -> [x ]);[width ]])
     []

let lower self =
   Py.Module.get_function_with_keywords (to_pyobject self) "lower"
     [||]
     []

let lstrip ?chars self =
   Py.Module.get_function_with_keywords (to_pyobject self) "lstrip"
     (Array.of_list @@ List.concat [(match chars with None -> [] | Some x -> [x ])])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let partition ~sep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "partition"
     (Array.of_list @@ List.concat [[sep ]])
     []

let replace ?count ~old ~new_ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "replace"
     (Array.of_list @@ List.concat [(match count with None -> [] | Some x -> [x ]);[old ];[new_ ]])
     []

let rfind ?start ?end_ ~sub self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rfind"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("sub", Some(sub ))])

let rindex ?start ?end_ ~sub self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rindex"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("sub", Some(sub ))])

let rjust ?fillchar ~width self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rjust"
     (Array.of_list @@ List.concat [(match fillchar with None -> [] | Some x -> [x ]);[width ]])
     []

let rpartition ~sep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rpartition"
     (Array.of_list @@ List.concat [[sep ]])
     []

let rsplit ?sep ?maxsplit self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rsplit"
     [||]
     (Wrap_utils.keyword_args [("sep", sep); ("maxsplit", maxsplit)])

let rstrip ?chars self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rstrip"
     (Array.of_list @@ List.concat [(match chars with None -> [] | Some x -> [x ])])
     []

let split ?sep ?maxsplit self =
   Py.Module.get_function_with_keywords (to_pyobject self) "split"
     [||]
     (Wrap_utils.keyword_args [("sep", sep); ("maxsplit", maxsplit)])

let splitlines ?keepends self =
   Py.Module.get_function_with_keywords (to_pyobject self) "splitlines"
     [||]
     (Wrap_utils.keyword_args [("keepends", keepends)])

let startswith ?start ?end_ ~prefix self =
   Py.Module.get_function_with_keywords (to_pyobject self) "startswith"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("prefix", Some(prefix ))])

let strip ?chars self =
   Py.Module.get_function_with_keywords (to_pyobject self) "strip"
     (Array.of_list @@ List.concat [(match chars with None -> [] | Some x -> [x ])])
     []

let swapcase self =
   Py.Module.get_function_with_keywords (to_pyobject self) "swapcase"
     [||]
     []

let title self =
   Py.Module.get_function_with_keywords (to_pyobject self) "title"
     [||]
     []

let translate ~table self =
   Py.Module.get_function_with_keywords (to_pyobject self) "translate"
     (Array.of_list @@ List.concat [[table ]])
     []

let upper self =
   Py.Module.get_function_with_keywords (to_pyobject self) "upper"
     [||]
     []

let zfill ~width self =
   Py.Module.get_function_with_keywords (to_pyobject self) "zfill"
     (Array.of_list @@ List.concat [[width ]])
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Timedelta64 = struct
type tag = [`Timedelta64]
type t = [`Object | `Timedelta64] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Ubyte = struct
type tag = [`Uint8]
type t = [`Object | `Uint8] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Ufunc = struct
type tag = [`Ufunc]
type t = [`Object | `Ufunc] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "ufunc"
     [||]
     []
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Uint = struct
type tag = [`Uint64]
type t = [`Object | `Uint64] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Uint16 = struct
type tag = [`Uint16]
type t = [`Object | `Uint16] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Uint32 = struct
type tag = [`Uint32]
type t = [`Object | `Uint32] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Uint8 = struct
type tag = [`Uint8]
type t = [`Object | `Uint8] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Uintc = struct
type tag = [`Uint32]
type t = [`Object | `Uint32] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Ulonglong = struct
type tag = [`Ulonglong]
type t = [`Object | `Ulonglong] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Unsignedinteger = struct
type tag = [`Unsignedinteger]
type t = [`Object | `Unsignedinteger] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "unsignedinteger"
     [||]
     []
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Ushort = struct
type tag = [`Uint16]
type t = [`Object | `Uint16] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Vectorize = struct
type tag = [`Vectorize]
type t = [`Object | `Vectorize] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?otypes ?doc ?excluded ?cache ?signature ~pyfunc () =
                     Py.Module.get_function_with_keywords __wrap_namespace "vectorize"
                       [||]
                       (Wrap_utils.keyword_args [("otypes", Wrap_utils.Option.map otypes (function
| `List_of_dtypes x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("doc", Wrap_utils.Option.map doc Py.String.of_string); ("excluded", excluded); ("cache", Wrap_utils.Option.map cache Py.Bool.of_bool); ("signature", Wrap_utils.Option.map signature Py.String.of_string); ("pyfunc", Some(pyfunc ))])
                       |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Void = struct
type tag = [`Void]
type t = [`Object | `Void] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let __setitem__ ~key ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     (Array.of_list @@ List.concat [[key ];[value ]])
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Emath = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "numpy.lib.scimath"

let get_py name = Py.Module.get __wrap_namespace name
let any ?axis ?out ?keepdims a =
   Py.Module.get_function_with_keywords __wrap_namespace "any"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])

                  let arccos x =
                     Py.Module.get_function_with_keywords __wrap_namespace "arccos"
                       [||]
                       (Wrap_utils.keyword_args [("x", Some(x |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let arcsin x =
                     Py.Module.get_function_with_keywords __wrap_namespace "arcsin"
                       [||]
                       (Wrap_utils.keyword_args [("x", Some(x |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let arctanh x =
   Py.Module.get_function_with_keywords __wrap_namespace "arctanh"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let array_function_dispatch ?module_ ?verify ?docs_from_dispatcher ~dispatcher () =
   Py.Module.get_function_with_keywords __wrap_namespace "array_function_dispatch"
     [||]
     (Wrap_utils.keyword_args [("module", Wrap_utils.Option.map module_ Py.String.of_string); ("verify", Wrap_utils.Option.map verify Py.Bool.of_bool); ("docs_from_dispatcher", Wrap_utils.Option.map docs_from_dispatcher Py.Bool.of_bool); ("dispatcher", Some(dispatcher ))])

                  let asarray ?dtype ?order a =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let isreal x =
   Py.Module.get_function_with_keywords __wrap_namespace "isreal"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let log x =
   Py.Module.get_function_with_keywords __wrap_namespace "log"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let log10 x =
                     Py.Module.get_function_with_keywords __wrap_namespace "log10"
                       [||]
                       (Wrap_utils.keyword_args [("x", Some(x |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let log2 x =
   Py.Module.get_function_with_keywords __wrap_namespace "log2"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let logn ~n x =
   Py.Module.get_function_with_keywords __wrap_namespace "logn"
     [||]
     (Wrap_utils.keyword_args [("n", Some(n |> Obj.to_pyobject)); ("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let power ~p x =
   Py.Module.get_function_with_keywords __wrap_namespace "power"
     [||]
     (Wrap_utils.keyword_args [("p", Some(p )); ("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let sqrt x =
   Py.Module.get_function_with_keywords __wrap_namespace "sqrt"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))

end
module Fft = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "numpy.fft"

let get_py name = Py.Module.get __wrap_namespace name
module Helper = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "numpy.fft.helper"

let get_py name = Py.Module.get __wrap_namespace name
                  let arange ?start ?step ?dtype ~stop () =
                     Py.Module.get_function_with_keywords __wrap_namespace "arange"
                       [||]
                       (Wrap_utils.keyword_args [("start", Wrap_utils.Option.map start (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("step", Wrap_utils.Option.map step (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("stop", Some(stop |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let array_function_dispatch ?module_ ?verify ?docs_from_dispatcher ~dispatcher () =
   Py.Module.get_function_with_keywords __wrap_namespace "array_function_dispatch"
     [||]
     (Wrap_utils.keyword_args [("module", Wrap_utils.Option.map module_ Py.String.of_string); ("verify", Wrap_utils.Option.map verify Py.Bool.of_bool); ("docs_from_dispatcher", Wrap_utils.Option.map docs_from_dispatcher Py.Bool.of_bool); ("dispatcher", Some(dispatcher ))])

                  let asarray ?dtype ?order a =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let empty ?dtype ?order shape =
                     Py.Module.get_function_with_keywords __wrap_namespace "empty"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let fftfreq ?d ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fftfreq"
                       [||]
                       (Wrap_utils.keyword_args [("d", Wrap_utils.Option.map d (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("n", Some(n |> Py.Int.of_int))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let fftshift ?axes x =
                     Py.Module.get_function_with_keywords __wrap_namespace "fftshift"
                       [||]
                       (Wrap_utils.keyword_args [("axes", Wrap_utils.Option.map axes (function
| `I x -> Py.Int.of_int x
| `Shape_tuple x -> Wrap_utils.id x
)); ("x", Some(x |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let ifftshift ?axes x =
                     Py.Module.get_function_with_keywords __wrap_namespace "ifftshift"
                       [||]
                       (Wrap_utils.keyword_args [("axes", Wrap_utils.Option.map axes (function
| `I x -> Py.Int.of_int x
| `Shape_tuple x -> Wrap_utils.id x
)); ("x", Some(x |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let rfftfreq ?d ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "rfftfreq"
                       [||]
                       (Wrap_utils.keyword_args [("d", Wrap_utils.Option.map d (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("n", Some(n |> Py.Int.of_int))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let roll ?axis ~shift a =
   Py.Module.get_function_with_keywords __wrap_namespace "roll"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("shift", Some(shift |> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml))); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let set_module module_ =
   Py.Module.get_function_with_keywords __wrap_namespace "set_module"
     [||]
     (Wrap_utils.keyword_args [("module", Some(module_ ))])


end
let fft ?n ?axis ?norm a =
   Py.Module.get_function_with_keywords __wrap_namespace "fft"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("a", Some(a |> Obj.to_pyobject))])

let fft2 ?s ?axes ?norm a =
   Py.Module.get_function_with_keywords __wrap_namespace "fft2"
     [||]
     (Wrap_utils.keyword_args [("s", Wrap_utils.Option.map s (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("axes", Wrap_utils.Option.map axes (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("a", Some(a |> Obj.to_pyobject))])

                  let fftfreq ?d ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fftfreq"
                       [||]
                       (Wrap_utils.keyword_args [("d", Wrap_utils.Option.map d (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("n", Some(n |> Py.Int.of_int))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let fftn ?s ?axes ?norm a =
   Py.Module.get_function_with_keywords __wrap_namespace "fftn"
     [||]
     (Wrap_utils.keyword_args [("s", Wrap_utils.Option.map s (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("axes", Wrap_utils.Option.map axes (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("a", Some(a |> Obj.to_pyobject))])

                  let fftshift ?axes x =
                     Py.Module.get_function_with_keywords __wrap_namespace "fftshift"
                       [||]
                       (Wrap_utils.keyword_args [("axes", Wrap_utils.Option.map axes (function
| `I x -> Py.Int.of_int x
| `Shape_tuple x -> Wrap_utils.id x
)); ("x", Some(x |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hfft ?n ?axis ?norm a =
   Py.Module.get_function_with_keywords __wrap_namespace "hfft"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let ifft ?n ?axis ?norm a =
   Py.Module.get_function_with_keywords __wrap_namespace "ifft"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("a", Some(a |> Obj.to_pyobject))])

let ifft2 ?s ?axes ?norm a =
   Py.Module.get_function_with_keywords __wrap_namespace "ifft2"
     [||]
     (Wrap_utils.keyword_args [("s", Wrap_utils.Option.map s (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("axes", Wrap_utils.Option.map axes (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("a", Some(a |> Obj.to_pyobject))])

let ifftn ?s ?axes ?norm a =
   Py.Module.get_function_with_keywords __wrap_namespace "ifftn"
     [||]
     (Wrap_utils.keyword_args [("s", Wrap_utils.Option.map s (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("axes", Wrap_utils.Option.map axes (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("a", Some(a |> Obj.to_pyobject))])

                  let ifftshift ?axes x =
                     Py.Module.get_function_with_keywords __wrap_namespace "ifftshift"
                       [||]
                       (Wrap_utils.keyword_args [("axes", Wrap_utils.Option.map axes (function
| `I x -> Py.Int.of_int x
| `Shape_tuple x -> Wrap_utils.id x
)); ("x", Some(x |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let ihfft ?n ?axis ?norm a =
   Py.Module.get_function_with_keywords __wrap_namespace "ihfft"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("a", Some(a |> Obj.to_pyobject))])

let irfft ?n ?axis ?norm a =
   Py.Module.get_function_with_keywords __wrap_namespace "irfft"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let irfft2 ?s ?axes ?norm a =
   Py.Module.get_function_with_keywords __wrap_namespace "irfft2"
     [||]
     (Wrap_utils.keyword_args [("s", Wrap_utils.Option.map s (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("axes", Wrap_utils.Option.map axes (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let irfftn ?s ?axes ?norm a =
   Py.Module.get_function_with_keywords __wrap_namespace "irfftn"
     [||]
     (Wrap_utils.keyword_args [("s", Wrap_utils.Option.map s (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("axes", Wrap_utils.Option.map axes (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let rfft ?n ?axis ?norm a =
   Py.Module.get_function_with_keywords __wrap_namespace "rfft"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("a", Some(a |> Obj.to_pyobject))])

let rfft2 ?s ?axes ?norm a =
   Py.Module.get_function_with_keywords __wrap_namespace "rfft2"
     [||]
     (Wrap_utils.keyword_args [("s", Wrap_utils.Option.map s (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("axes", Wrap_utils.Option.map axes (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let rfftfreq ?d ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "rfftfreq"
                       [||]
                       (Wrap_utils.keyword_args [("d", Wrap_utils.Option.map d (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("n", Some(n |> Py.Int.of_int))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let rfftn ?s ?axes ?norm a =
   Py.Module.get_function_with_keywords __wrap_namespace "rfftn"
     [||]
     (Wrap_utils.keyword_args [("s", Wrap_utils.Option.map s (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("axes", Wrap_utils.Option.map axes (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("norm", Wrap_utils.Option.map norm Py.String.of_string); ("a", Some(a |> Obj.to_pyobject))])


end
module Linalg = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "numpy.linalg"

let get_py name = Py.Module.get __wrap_namespace name
module LinAlgError = struct
type tag = [`LinAlgError]
type t = [`BaseException | `LinAlgError | `Object] Obj.t
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
module Lapack_lite = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "numpy.linalg.lapack_lite"

let get_py name = Py.Module.get __wrap_namespace name
module LapackError = struct
type tag = [`LapackError]
type t = [`BaseException | `LapackError | `Object] Obj.t
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

end
module Linalg = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "numpy.linalg.linalg"

let get_py name = Py.Module.get __wrap_namespace name
                  let abs ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "abs"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let add ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "add"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let all ?axis ?out ?keepdims a =
   Py.Module.get_function_with_keywords __wrap_namespace "all"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let amax ?axis ?out ?keepdims ?initial ?where a =
                     Py.Module.get_function_with_keywords __wrap_namespace "amax"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("initial", Wrap_utils.Option.map initial (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("where", where); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let amin ?axis ?out ?keepdims ?initial ?where a =
                     Py.Module.get_function_with_keywords __wrap_namespace "amin"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("initial", Wrap_utils.Option.map initial (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("where", where); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let argsort ?axis ?kind ?order a =
                     Py.Module.get_function_with_keywords __wrap_namespace "argsort"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("kind", Wrap_utils.Option.map kind (function
| `Heapsort -> Py.String.of_string "heapsort"
| `Mergesort -> Py.String.of_string "mergesort"
| `Stable -> Py.String.of_string "stable"
| `Quicksort -> Py.String.of_string "quicksort"
)); ("order", Wrap_utils.Option.map order (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `S x -> Py.String.of_string x
)); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let array ?dtype ?copy ?order ?subok ?ndmin ~object_ () =
                     Py.Module.get_function_with_keywords __wrap_namespace "array"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("order", Wrap_utils.Option.map order (function
| `K -> Py.String.of_string "K"
| `A -> Py.String.of_string "A"
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("ndmin", Wrap_utils.Option.map ndmin Py.Int.of_int); ("object", Some(object_ |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let array_function_dispatch ?module_ ?verify ?docs_from_dispatcher ~dispatcher () =
   Py.Module.get_function_with_keywords __wrap_namespace "array_function_dispatch"
     [||]
     (Wrap_utils.keyword_args [("module", module_); ("verify", verify); ("docs_from_dispatcher", docs_from_dispatcher); ("dispatcher", Some(dispatcher ))])

                  let asanyarray ?dtype ?order a =
                     Py.Module.get_function_with_keywords __wrap_namespace "asanyarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("a", Some(a |> Obj.to_pyobject))])

                  let asarray ?dtype ?order a =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let atleast_2d arys =
   Py.Module.get_function_with_keywords __wrap_namespace "atleast_2d"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arys)])
     []

let cholesky a =
   Py.Module.get_function_with_keywords __wrap_namespace "cholesky"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a ))])

                  let cond ?p x =
                     Py.Module.get_function_with_keywords __wrap_namespace "cond"
                       [||]
                       (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p (function
| `Fro -> Py.String.of_string "fro"
| `One -> Py.Int.of_int 1
| `Two -> Py.Int.of_int 2
| `PyObject x -> Wrap_utils.id x
)); ("x", Some(x ))])

                  let count_nonzero ?axis ?keepdims a =
                     Py.Module.get_function_with_keywords __wrap_namespace "count_nonzero"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Tuple x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])

let det a =
   Py.Module.get_function_with_keywords __wrap_namespace "det"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a ))])

                  let divide ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "divide"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let dot ?out ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "dot"
     [||]
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Obj.to_pyobject); ("b", Some(b |> Obj.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let eig a =
   Py.Module.get_function_with_keywords __wrap_namespace "eig"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let eigh ?uplo a =
                     Py.Module.get_function_with_keywords __wrap_namespace "eigh"
                       [||]
                       (Wrap_utils.keyword_args [("UPLO", Wrap_utils.Option.map uplo (function
| `L -> Py.String.of_string "L"
| `U -> Py.String.of_string "U"
)); ("a", Some(a ))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let eigvals a =
   Py.Module.get_function_with_keywords __wrap_namespace "eigvals"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a ))])

                  let eigvalsh ?uplo a =
                     Py.Module.get_function_with_keywords __wrap_namespace "eigvalsh"
                       [||]
                       (Wrap_utils.keyword_args [("UPLO", Wrap_utils.Option.map uplo (function
| `L -> Py.String.of_string "L"
| `U -> Py.String.of_string "U"
)); ("a", Some(a ))])

                  let empty ?dtype ?order shape =
                     Py.Module.get_function_with_keywords __wrap_namespace "empty"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let empty_like ?dtype ?order ?subok ?shape ~prototype () =
                     Py.Module.get_function_with_keywords __wrap_namespace "empty_like"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `A -> Py.String.of_string "A"
| `F -> Py.String.of_string "F"
| `PyObject x -> Wrap_utils.id x
)); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("shape", Wrap_utils.Option.map shape (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("prototype", Some(prototype |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let eye ?m ?k ?dtype ?order ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "eye"
                       [||]
                       (Wrap_utils.keyword_args [("M", Wrap_utils.Option.map m Py.Int.of_int); ("k", Wrap_utils.Option.map k Py.Int.of_int); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("N", Some(n |> Py.Int.of_int))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let fastCopyAndTranspose a =
   Py.Module.get_function_with_keywords __wrap_namespace "fastCopyAndTranspose"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a ))])

let get_linalg_error_extobj callback =
   Py.Module.get_function_with_keywords __wrap_namespace "get_linalg_error_extobj"
     [||]
     (Wrap_utils.keyword_args [("callback", Some(callback ))])

let inv a =
   Py.Module.get_function_with_keywords __wrap_namespace "inv"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a ))])

let isComplexType t =
   Py.Module.get_function_with_keywords __wrap_namespace "isComplexType"
     [||]
     (Wrap_utils.keyword_args [("t", Some(t ))])

                  let isfinite ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "isfinite"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let isnan ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "isnan"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])

let lstsq ?rcond ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "lstsq"
     [||]
     (Wrap_utils.keyword_args [("rcond", Wrap_utils.Option.map rcond Py.Float.of_float); ("b", Some(b )); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3))))
let matmul ?out ?where x =
   Py.Module.get_function_with_keywords __wrap_namespace "matmul"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Obj.to_pyobject); ("where", where)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let matrix_power ~n a =
   Py.Module.get_function_with_keywords __wrap_namespace "matrix_power"
     [||]
     (Wrap_utils.keyword_args [("n", Some(n |> Py.Int.of_int)); ("a", Some(a ))])

                  let matrix_rank ?tol ?hermitian ~m () =
                     Py.Module.get_function_with_keywords __wrap_namespace "matrix_rank"
                       [||]
                       (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol (function
| `F x -> Py.Float.of_float x
| `T_array_like x -> Wrap_utils.id x
)); ("hermitian", Wrap_utils.Option.map hermitian Py.Bool.of_bool); ("M", Some(m ))])

                  let moveaxis ~source ~destination a =
                     Py.Module.get_function_with_keywords __wrap_namespace "moveaxis"
                       [||]
                       (Wrap_utils.keyword_args [("source", Some(source |> (function
| `Sequence_of_int x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("destination", Some(destination |> (function
| `Sequence_of_int x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("a", Some(a ))])

let multi_dot ?out ~arrays () =
   Py.Module.get_function_with_keywords __wrap_namespace "multi_dot"
     [||]
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Obj.to_pyobject); ("arrays", Some(arrays ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let multiply ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "multiply"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let norm ?ord ?axis ?keepdims x =
                     Py.Module.get_function_with_keywords __wrap_namespace "norm"
                       [||]
                       (Wrap_utils.keyword_args [("ord", Wrap_utils.Option.map ord (function
| `Nuc -> Py.String.of_string "nuc"
| `Fro -> Py.String.of_string "fro"
| `PyObject x -> Wrap_utils.id x
)); ("axis", Wrap_utils.Option.map axis (function
| `T2_tuple_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("x", Some(x |> Obj.to_pyobject))])

let normalize_axis_index ?msg_prefix ~axis ~ndim () =
   Py.Module.get_function_with_keywords __wrap_namespace "normalize_axis_index"
     [||]
     (Wrap_utils.keyword_args [("msg_prefix", Wrap_utils.Option.map msg_prefix Py.String.of_string); ("axis", Some(axis |> Py.Int.of_int)); ("ndim", Some(ndim |> Py.Int.of_int))])
     |> Py.Int.to_int
let pinv ?rcond ?hermitian a =
   Py.Module.get_function_with_keywords __wrap_namespace "pinv"
     [||]
     (Wrap_utils.keyword_args [("rcond", rcond); ("hermitian", Wrap_utils.Option.map hermitian Py.Bool.of_bool); ("a", Some(a ))])

let product ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "product"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

                  let qr ?mode a =
                     Py.Module.get_function_with_keywords __wrap_namespace "qr"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Reduced -> Py.String.of_string "reduced"
| `Complete -> Py.String.of_string "complete"
| `R -> Py.String.of_string "r"
| `Raw -> Py.String.of_string "raw"
)); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let set_module module_ =
   Py.Module.get_function_with_keywords __wrap_namespace "set_module"
     [||]
     (Wrap_utils.keyword_args [("module", Some(module_ ))])

                  let sign ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "sign"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let slogdet a =
   Py.Module.get_function_with_keywords __wrap_namespace "slogdet"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let solve ~b a =
                     Py.Module.get_function_with_keywords __wrap_namespace "solve"
                       [||]
                       (Wrap_utils.keyword_args [("b", Some(b |> (function
| `Ndarray x -> Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("a", Some(a ))])

                  let sort ?axis ?kind ?order a =
                     Py.Module.get_function_with_keywords __wrap_namespace "sort"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("kind", Wrap_utils.Option.map kind (function
| `Heapsort -> Py.String.of_string "heapsort"
| `Mergesort -> Py.String.of_string "mergesort"
| `Stable -> Py.String.of_string "stable"
| `Quicksort -> Py.String.of_string "quicksort"
)); ("order", Wrap_utils.Option.map order (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `S x -> Py.String.of_string x
)); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let sqrt ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "sqrt"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let sum ?axis ?dtype ?out ?keepdims ?initial ?where a =
                     Py.Module.get_function_with_keywords __wrap_namespace "sum"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("initial", Wrap_utils.Option.map initial (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("where", where); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let svd ?full_matrices ?compute_uv ?hermitian a =
   Py.Module.get_function_with_keywords __wrap_namespace "svd"
     [||]
     (Wrap_utils.keyword_args [("full_matrices", Wrap_utils.Option.map full_matrices Py.Bool.of_bool); ("compute_uv", Wrap_utils.Option.map compute_uv Py.Bool.of_bool); ("hermitian", Wrap_utils.Option.map hermitian Py.Bool.of_bool); ("a", Some(a ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
let swapaxes ~axis1 ~axis2 a =
   Py.Module.get_function_with_keywords __wrap_namespace "swapaxes"
     [||]
     (Wrap_utils.keyword_args [("axis1", Some(axis1 |> Py.Int.of_int)); ("axis2", Some(axis2 |> Py.Int.of_int)); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let tensorinv ?ind a =
   Py.Module.get_function_with_keywords __wrap_namespace "tensorinv"
     [||]
     (Wrap_utils.keyword_args [("ind", Wrap_utils.Option.map ind Py.Int.of_int); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let tensorsolve ?axes ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "tensorsolve"
     [||]
     (Wrap_utils.keyword_args [("axes", Wrap_utils.Option.map axes (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("b", Some(b |> Obj.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])

let transpose a =
   Py.Module.get_function_with_keywords __wrap_namespace "transpose"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a ))])

let triu ?k ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "triu"
     [||]
     (Wrap_utils.keyword_args [("k", k); ("m", Some(m ))])

                  let zeros ?dtype ?order shape =
                     Py.Module.get_function_with_keywords __wrap_namespace "zeros"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))

end
let cholesky a =
   Py.Module.get_function_with_keywords __wrap_namespace "cholesky"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a ))])

                  let cond ?p x =
                     Py.Module.get_function_with_keywords __wrap_namespace "cond"
                       [||]
                       (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p (function
| `Fro -> Py.String.of_string "fro"
| `One -> Py.Int.of_int 1
| `Two -> Py.Int.of_int 2
| `PyObject x -> Wrap_utils.id x
)); ("x", Some(x ))])

let det a =
   Py.Module.get_function_with_keywords __wrap_namespace "det"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a ))])

let eig a =
   Py.Module.get_function_with_keywords __wrap_namespace "eig"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let eigh ?uplo a =
                     Py.Module.get_function_with_keywords __wrap_namespace "eigh"
                       [||]
                       (Wrap_utils.keyword_args [("UPLO", Wrap_utils.Option.map uplo (function
| `L -> Py.String.of_string "L"
| `U -> Py.String.of_string "U"
)); ("a", Some(a ))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let eigvals a =
   Py.Module.get_function_with_keywords __wrap_namespace "eigvals"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a ))])

                  let eigvalsh ?uplo a =
                     Py.Module.get_function_with_keywords __wrap_namespace "eigvalsh"
                       [||]
                       (Wrap_utils.keyword_args [("UPLO", Wrap_utils.Option.map uplo (function
| `L -> Py.String.of_string "L"
| `U -> Py.String.of_string "U"
)); ("a", Some(a ))])

let inv a =
   Py.Module.get_function_with_keywords __wrap_namespace "inv"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a ))])

let lstsq ?rcond ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "lstsq"
     [||]
     (Wrap_utils.keyword_args [("rcond", Wrap_utils.Option.map rcond Py.Float.of_float); ("b", Some(b )); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3))))
let matrix_power ~n a =
   Py.Module.get_function_with_keywords __wrap_namespace "matrix_power"
     [||]
     (Wrap_utils.keyword_args [("n", Some(n |> Py.Int.of_int)); ("a", Some(a ))])

                  let matrix_rank ?tol ?hermitian ~m () =
                     Py.Module.get_function_with_keywords __wrap_namespace "matrix_rank"
                       [||]
                       (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol (function
| `F x -> Py.Float.of_float x
| `T_array_like x -> Wrap_utils.id x
)); ("hermitian", Wrap_utils.Option.map hermitian Py.Bool.of_bool); ("M", Some(m ))])

let multi_dot ?out ~arrays () =
   Py.Module.get_function_with_keywords __wrap_namespace "multi_dot"
     [||]
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Obj.to_pyobject); ("arrays", Some(arrays ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let norm ?ord ?axis ?keepdims x =
                     Py.Module.get_function_with_keywords __wrap_namespace "norm"
                       [||]
                       (Wrap_utils.keyword_args [("ord", Wrap_utils.Option.map ord (function
| `Nuc -> Py.String.of_string "nuc"
| `Fro -> Py.String.of_string "fro"
| `PyObject x -> Wrap_utils.id x
)); ("axis", Wrap_utils.Option.map axis (function
| `T2_tuple_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("x", Some(x |> Obj.to_pyobject))])

let pinv ?rcond ?hermitian a =
   Py.Module.get_function_with_keywords __wrap_namespace "pinv"
     [||]
     (Wrap_utils.keyword_args [("rcond", rcond); ("hermitian", Wrap_utils.Option.map hermitian Py.Bool.of_bool); ("a", Some(a ))])

                  let qr ?mode a =
                     Py.Module.get_function_with_keywords __wrap_namespace "qr"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Reduced -> Py.String.of_string "reduced"
| `Complete -> Py.String.of_string "complete"
| `R -> Py.String.of_string "r"
| `Raw -> Py.String.of_string "raw"
)); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let slogdet a =
   Py.Module.get_function_with_keywords __wrap_namespace "slogdet"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let solve ~b a =
                     Py.Module.get_function_with_keywords __wrap_namespace "solve"
                       [||]
                       (Wrap_utils.keyword_args [("b", Some(b |> (function
| `Ndarray x -> Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("a", Some(a ))])

let svd ?full_matrices ?compute_uv ?hermitian a =
   Py.Module.get_function_with_keywords __wrap_namespace "svd"
     [||]
     (Wrap_utils.keyword_args [("full_matrices", Wrap_utils.Option.map full_matrices Py.Bool.of_bool); ("compute_uv", Wrap_utils.Option.map compute_uv Py.Bool.of_bool); ("hermitian", Wrap_utils.Option.map hermitian Py.Bool.of_bool); ("a", Some(a ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
let tensorinv ?ind a =
   Py.Module.get_function_with_keywords __wrap_namespace "tensorinv"
     [||]
     (Wrap_utils.keyword_args [("ind", Wrap_utils.Option.map ind Py.Int.of_int); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let tensorsolve ?axes ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "tensorsolve"
     [||]
     (Wrap_utils.keyword_args [("axes", Wrap_utils.Option.map axes (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("b", Some(b |> Obj.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])


end
module Ma = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "numpy.ma"

let get_py name = Py.Module.get __wrap_namespace name
module MAError = struct
type tag = [`MAError]
type t = [`BaseException | `MAError | `Object] Obj.t
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
module MaskError = struct
type tag = [`MaskError]
type t = [`BaseException | `MaskError | `Object] Obj.t
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
module MaskedArray = struct
type tag = [`MaskedArray]
type t = [`ArrayLike | `MaskedArray | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?data ?mask ?dtype ?copy ?subok ?ndmin ?fill_value ?keep_mask ?hard_mask ?shrink ?order ?options () =
                     Py.Module.get_function_with_keywords __wrap_namespace "MaskedArray"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("data", Wrap_utils.Option.map data Obj.to_pyobject); ("mask", mask); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("ndmin", Wrap_utils.Option.map ndmin Py.Int.of_int); ("fill_value", Wrap_utils.Option.map fill_value (function
| `Bool x -> Py.Bool.of_bool x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
)); ("keep_mask", Wrap_utils.Option.map keep_mask Py.Bool.of_bool); ("hard_mask", Wrap_utils.Option.map hard_mask Py.Bool.of_bool); ("shrink", Wrap_utils.Option.map shrink Py.Bool.of_bool); ("order", Wrap_utils.Option.map order (function
| `A -> Py.String.of_string "A"
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
))]) (match options with None -> [] | Some x -> x))
                       |> of_pyobject
let __getitem__ ~indx self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("indx", Some(indx ))])

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let __setitem__ ~indx ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     [||]
     (Wrap_utils.keyword_args [("indx", Some(indx )); ("value", Some(value ))])

let all ?axis ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "all"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims)])

let anom ?axis ?dtype self =
   Py.Module.get_function_with_keywords (to_pyobject self) "anom"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject)])

let any ?axis ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "any"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims)])

let argmax ?axis ?fill_value ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "argmax"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("fill_value", fill_value); ("out", Wrap_utils.Option.map out Obj.to_pyobject)])

let argmin ?axis ?fill_value ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "argmin"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("fill_value", fill_value); ("out", Wrap_utils.Option.map out Obj.to_pyobject)])

let argpartition ?kwargs args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "argpartition"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

                  let argsort ?axis ?kind ?order ?endwith ?fill_value self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "argsort"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("kind", Wrap_utils.Option.map kind (function
| `Heapsort -> Py.String.of_string "heapsort"
| `Mergesort -> Py.String.of_string "mergesort"
| `Stable -> Py.String.of_string "stable"
| `Quicksort -> Py.String.of_string "quicksort"
)); ("order", Wrap_utils.Option.map order Obj.to_pyobject); ("endwith", Wrap_utils.Option.map endwith Py.Bool.of_bool); ("fill_value", fill_value)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
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
| `Dtype x -> Dtype.to_pyobject x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let byteswap ?inplace self =
   Py.Module.get_function_with_keywords (to_pyobject self) "byteswap"
     [||]
     (Wrap_utils.keyword_args [("inplace", Wrap_utils.Option.map inplace Py.Bool.of_bool)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
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
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", out); ("condition", Some(condition ))])

let compressed self =
   Py.Module.get_function_with_keywords (to_pyobject self) "compressed"
     [||]
     []
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let copy ?params args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "copy"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)

let count ?axis ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "count"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let cumprod ?axis ?dtype ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cumprod"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out)])

let cumsum ?axis ?dtype ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cumsum"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out)])

let diagonal ?params args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diagonal"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)

let dot ?out ?strict ~b self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("out", out); ("strict", Wrap_utils.Option.map strict Py.Bool.of_bool); ("b", Some(b ))])

                  let dump ~file self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "dump"
                       [||]
                       (Wrap_utils.keyword_args [("file", Some(file |> (function
| `Path x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))])

                  let fill ~value self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fill"
                       [||]
                       (Wrap_utils.keyword_args [("value", Some(value |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])

let filled ?fill_value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "filled"
     [||]
     (Wrap_utils.keyword_args [("fill_value", Wrap_utils.Option.map fill_value Obj.to_pyobject)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let flatten ?params args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "flatten"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let get_fill_value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_fill_value"
     [||]
     []

let get_imag self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_imag"
     [||]
     []

let get_real self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_real"
     [||]
     []

                  let getfield ?offset ~dtype self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "getfield"
                       [||]
                       (Wrap_utils.keyword_args [("offset", Wrap_utils.Option.map offset Py.Int.of_int); ("dtype", Some(dtype |> (function
| `Dtype x -> Dtype.to_pyobject x
| `S x -> Py.String.of_string x
)))])

let harden_mask self =
   Py.Module.get_function_with_keywords (to_pyobject self) "harden_mask"
     [||]
     []

let ids self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ids"
     [||]
     []

let iscontiguous self =
   Py.Module.get_function_with_keywords (to_pyobject self) "iscontiguous"
     [||]
     []

let item args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "item"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

let itemset args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "itemset"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

let max ?axis ?out ?fill_value ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "max"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("fill_value", fill_value); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let mean ?axis ?dtype ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "mean"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("keepdims", keepdims)])

let min ?axis ?out ?fill_value ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "min"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("fill_value", fill_value); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let mini ?axis self =
   Py.Module.get_function_with_keywords (to_pyobject self) "mini"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int)])

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let nonzero self =
   Py.Module.get_function_with_keywords (to_pyobject self) "nonzero"
     [||]
     []

let partition ?kwargs args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "partition"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

let prod ?axis ?dtype ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "prod"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("keepdims", keepdims)])

let ptp ?axis ?out ?fill_value ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ptp"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("fill_value", fill_value); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let put ?mode ~indices ~values self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "put"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Raise -> Py.String.of_string "raise"
| `Wrap -> Py.String.of_string "wrap"
| `Clip -> Py.String.of_string "clip"
)); ("indices", Some(indices )); ("values", Some(values |> Obj.to_pyobject))])

                  let ravel ?order self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "ravel"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
| `A -> Py.String.of_string "A"
| `K -> Py.String.of_string "K"
))])

let repeat ?params args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "repeat"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)

let reshape ?kwargs s self =
   Py.Module.get_function_with_keywords (to_pyobject self) "reshape"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id s)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let resize ?refcheck ?order ~newshape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "resize"
     [||]
     (Wrap_utils.keyword_args [("refcheck", refcheck); ("order", order); ("newshape", Some(newshape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])

let round ?decimals ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "round"
     [||]
     (Wrap_utils.keyword_args [("decimals", decimals); ("out", out)])

let searchsorted ?side ?sorter ~v self =
   Py.Module.get_function_with_keywords (to_pyobject self) "searchsorted"
     [||]
     (Wrap_utils.keyword_args [("side", side); ("sorter", sorter); ("v", Some(v ))])

let set_fill_value ?value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_fill_value"
     [||]
     (Wrap_utils.keyword_args [("value", value)])

let setfield ?offset ~val_ ~dtype self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setfield"
     [||]
     (Wrap_utils.keyword_args [("offset", Wrap_utils.Option.map offset Py.Int.of_int); ("val", Some(val_ )); ("dtype", Some(dtype ))])

let setflags ?write ?align ?uic self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setflags"
     [||]
     (Wrap_utils.keyword_args [("write", Wrap_utils.Option.map write Py.Bool.of_bool); ("align", Wrap_utils.Option.map align Py.Bool.of_bool); ("uic", Wrap_utils.Option.map uic Py.Bool.of_bool)])

let shrink_mask self =
   Py.Module.get_function_with_keywords (to_pyobject self) "shrink_mask"
     [||]
     []

let soften_mask self =
   Py.Module.get_function_with_keywords (to_pyobject self) "soften_mask"
     [||]
     []

                  let sort ?axis ?kind ?order ?endwith ?fill_value self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "sort"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("kind", Wrap_utils.Option.map kind (function
| `Heapsort -> Py.String.of_string "heapsort"
| `Mergesort -> Py.String.of_string "mergesort"
| `Stable -> Py.String.of_string "stable"
| `Quicksort -> Py.String.of_string "quicksort"
)); ("order", Wrap_utils.Option.map order Obj.to_pyobject); ("endwith", Wrap_utils.Option.map endwith Py.Bool.of_bool); ("fill_value", fill_value)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let squeeze ?params args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "squeeze"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)

let std ?axis ?dtype ?out ?ddof ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "std"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("ddof", ddof); ("keepdims", keepdims)])

let sum ?axis ?dtype ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sum"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("keepdims", keepdims)])

let swapaxes ?params args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "swapaxes"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)

let take ?axis ?out ?mode ~indices self =
   Py.Module.get_function_with_keywords (to_pyobject self) "take"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("mode", mode); ("indices", Some(indices ))])

                  let tobytes ?fill_value ?order self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "tobytes"
                       [||]
                       (Wrap_utils.keyword_args [("fill_value", Wrap_utils.Option.map fill_value (function
| `Bool x -> Py.Bool.of_bool x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
)); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
| `A -> Py.String.of_string "A"
))])

let tofile ?sep ?format ~fid self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tofile"
     [||]
     (Wrap_utils.keyword_args [("sep", sep); ("format", format); ("fid", Some(fid ))])

let toflex self =
   Py.Module.get_function_with_keywords (to_pyobject self) "toflex"
     [||]
     []
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let tolist ?fill_value self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "tolist"
                       [||]
                       (Wrap_utils.keyword_args [("fill_value", Wrap_utils.Option.map fill_value (function
| `Bool x -> Py.Bool.of_bool x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let tostring ?fill_value ?order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tostring"
     [||]
     (Wrap_utils.keyword_args [("fill_value", fill_value); ("order", order)])

let trace ?offset ?axis1 ?axis2 ?dtype ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "trace"
     [||]
     (Wrap_utils.keyword_args [("offset", offset); ("axis1", axis1); ("axis2", axis2); ("dtype", dtype); ("out", out)])

let transpose ?params args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let unshare_mask self =
   Py.Module.get_function_with_keywords (to_pyobject self) "unshare_mask"
     [||]
     []

let var ?axis ?dtype ?out ?ddof ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "var"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("ddof", Wrap_utils.Option.map ddof Py.Int.of_int); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let view ?dtype ?type_ ?fill_value self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "view"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype (function
| `Ndarray_sub_class x -> Wrap_utils.id x
| `Dtype x -> Dtype.to_pyobject x
)); ("type", type_); ("fill_value", Wrap_utils.Option.map fill_value (function
| `Bool x -> Py.Bool.of_bool x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Mvoid = struct
type tag = [`Mvoid]
type t = [`ArrayLike | `Mvoid | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?mask ?dtype ?fill_value ?hardmask ?copy ?subok ~data () =
   Py.Module.get_function_with_keywords __wrap_namespace "mvoid"
     [||]
     (Wrap_utils.keyword_args [("mask", mask); ("dtype", dtype); ("fill_value", fill_value); ("hardmask", hardmask); ("copy", copy); ("subok", subok); ("data", Some(data ))])
     |> of_pyobject
let __getitem__ ~indx self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("indx", Some(indx ))])

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let __setitem__ ~indx ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     [||]
     (Wrap_utils.keyword_args [("indx", Some(indx )); ("value", Some(value ))])

let all ?axis ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "all"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims)])

let anom ?axis ?dtype self =
   Py.Module.get_function_with_keywords (to_pyobject self) "anom"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject)])

let any ?axis ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "any"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims)])

let argmax ?axis ?fill_value ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "argmax"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("fill_value", fill_value); ("out", Wrap_utils.Option.map out Obj.to_pyobject)])

let argmin ?axis ?fill_value ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "argmin"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("fill_value", fill_value); ("out", Wrap_utils.Option.map out Obj.to_pyobject)])

let argpartition ?kwargs args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "argpartition"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

                  let argsort ?axis ?kind ?order ?endwith ?fill_value self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "argsort"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("kind", Wrap_utils.Option.map kind (function
| `Heapsort -> Py.String.of_string "heapsort"
| `Mergesort -> Py.String.of_string "mergesort"
| `Stable -> Py.String.of_string "stable"
| `Quicksort -> Py.String.of_string "quicksort"
)); ("order", Wrap_utils.Option.map order Obj.to_pyobject); ("endwith", Wrap_utils.Option.map endwith Py.Bool.of_bool); ("fill_value", fill_value)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
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
| `Dtype x -> Dtype.to_pyobject x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let byteswap ?inplace self =
   Py.Module.get_function_with_keywords (to_pyobject self) "byteswap"
     [||]
     (Wrap_utils.keyword_args [("inplace", Wrap_utils.Option.map inplace Py.Bool.of_bool)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
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
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", out); ("condition", Some(condition ))])

let compressed self =
   Py.Module.get_function_with_keywords (to_pyobject self) "compressed"
     [||]
     []
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let conj self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conj"
     [||]
     []

let conjugate self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conjugate"
     [||]
     []

let copy ?params args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "copy"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)

let count ?axis ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "count"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let cumprod ?axis ?dtype ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cumprod"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out)])

let cumsum ?axis ?dtype ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cumsum"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out)])

let diagonal ?params args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diagonal"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)

let dot ?out ?strict ~b self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("out", out); ("strict", Wrap_utils.Option.map strict Py.Bool.of_bool); ("b", Some(b ))])

                  let dump ~file self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "dump"
                       [||]
                       (Wrap_utils.keyword_args [("file", Some(file |> (function
| `Path x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
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

let filled ?fill_value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "filled"
     [||]
     (Wrap_utils.keyword_args [("fill_value", Wrap_utils.Option.map fill_value Obj.to_pyobject)])

let flatten ?params args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "flatten"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let get_fill_value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_fill_value"
     [||]
     []

let get_imag self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_imag"
     [||]
     []

let get_real self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_real"
     [||]
     []

                  let getfield ?offset ~dtype self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "getfield"
                       [||]
                       (Wrap_utils.keyword_args [("offset", Wrap_utils.Option.map offset Py.Int.of_int); ("dtype", Some(dtype |> (function
| `Dtype x -> Dtype.to_pyobject x
| `S x -> Py.String.of_string x
)))])

let harden_mask self =
   Py.Module.get_function_with_keywords (to_pyobject self) "harden_mask"
     [||]
     []

let ids self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ids"
     [||]
     []

let iscontiguous self =
   Py.Module.get_function_with_keywords (to_pyobject self) "iscontiguous"
     [||]
     []

let item args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "item"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

let itemset args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "itemset"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

let max ?axis ?out ?fill_value ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "max"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("fill_value", fill_value); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let mean ?axis ?dtype ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "mean"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("keepdims", keepdims)])

let min ?axis ?out ?fill_value ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "min"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("fill_value", fill_value); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let mini ?axis self =
   Py.Module.get_function_with_keywords (to_pyobject self) "mini"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int)])

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let nonzero self =
   Py.Module.get_function_with_keywords (to_pyobject self) "nonzero"
     [||]
     []

let partition ?kwargs args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "partition"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

let prod ?axis ?dtype ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "prod"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("keepdims", keepdims)])

let ptp ?axis ?out ?fill_value ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ptp"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("fill_value", fill_value); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let put ?mode ~indices ~values self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "put"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Raise -> Py.String.of_string "raise"
| `Wrap -> Py.String.of_string "wrap"
| `Clip -> Py.String.of_string "clip"
)); ("indices", Some(indices )); ("values", Some(values |> Obj.to_pyobject))])

                  let ravel ?order self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "ravel"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
| `A -> Py.String.of_string "A"
| `K -> Py.String.of_string "K"
))])

let repeat ?params args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "repeat"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)

let reshape ?kwargs s self =
   Py.Module.get_function_with_keywords (to_pyobject self) "reshape"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id s)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let resize ?refcheck ?order ~newshape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "resize"
     [||]
     (Wrap_utils.keyword_args [("refcheck", refcheck); ("order", order); ("newshape", Some(newshape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])

let round ?decimals ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "round"
     [||]
     (Wrap_utils.keyword_args [("decimals", decimals); ("out", out)])

let searchsorted ?side ?sorter ~v self =
   Py.Module.get_function_with_keywords (to_pyobject self) "searchsorted"
     [||]
     (Wrap_utils.keyword_args [("side", side); ("sorter", sorter); ("v", Some(v ))])

let set_fill_value ?value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_fill_value"
     [||]
     (Wrap_utils.keyword_args [("value", value)])

let setfield ?offset ~val_ ~dtype self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setfield"
     [||]
     (Wrap_utils.keyword_args [("offset", Wrap_utils.Option.map offset Py.Int.of_int); ("val", Some(val_ )); ("dtype", Some(dtype ))])

let setflags ?write ?align ?uic self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setflags"
     [||]
     (Wrap_utils.keyword_args [("write", Wrap_utils.Option.map write Py.Bool.of_bool); ("align", Wrap_utils.Option.map align Py.Bool.of_bool); ("uic", Wrap_utils.Option.map uic Py.Bool.of_bool)])

let shrink_mask self =
   Py.Module.get_function_with_keywords (to_pyobject self) "shrink_mask"
     [||]
     []

let soften_mask self =
   Py.Module.get_function_with_keywords (to_pyobject self) "soften_mask"
     [||]
     []

                  let sort ?axis ?kind ?order ?endwith ?fill_value self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "sort"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("kind", Wrap_utils.Option.map kind (function
| `Heapsort -> Py.String.of_string "heapsort"
| `Mergesort -> Py.String.of_string "mergesort"
| `Stable -> Py.String.of_string "stable"
| `Quicksort -> Py.String.of_string "quicksort"
)); ("order", Wrap_utils.Option.map order Obj.to_pyobject); ("endwith", Wrap_utils.Option.map endwith Py.Bool.of_bool); ("fill_value", fill_value)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let squeeze ?params args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "squeeze"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)

let std ?axis ?dtype ?out ?ddof ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "std"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("ddof", ddof); ("keepdims", keepdims)])

let sum ?axis ?dtype ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sum"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("keepdims", keepdims)])

let swapaxes ?params args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "swapaxes"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)

let take ?axis ?out ?mode ~indices self =
   Py.Module.get_function_with_keywords (to_pyobject self) "take"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("mode", mode); ("indices", Some(indices ))])

                  let tobytes ?fill_value ?order self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "tobytes"
                       [||]
                       (Wrap_utils.keyword_args [("fill_value", Wrap_utils.Option.map fill_value (function
| `Bool x -> Py.Bool.of_bool x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
)); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
| `A -> Py.String.of_string "A"
))])

let tofile ?sep ?format ~fid self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tofile"
     [||]
     (Wrap_utils.keyword_args [("sep", sep); ("format", format); ("fid", Some(fid ))])

let toflex self =
   Py.Module.get_function_with_keywords (to_pyobject self) "toflex"
     [||]
     []
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let tolist self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tolist"
     [||]
     []

let tostring ?fill_value ?order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tostring"
     [||]
     (Wrap_utils.keyword_args [("fill_value", fill_value); ("order", order)])

let trace ?offset ?axis1 ?axis2 ?dtype ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "trace"
     [||]
     (Wrap_utils.keyword_args [("offset", offset); ("axis1", axis1); ("axis2", axis2); ("dtype", dtype); ("out", out)])

let transpose ?params args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let unshare_mask self =
   Py.Module.get_function_with_keywords (to_pyobject self) "unshare_mask"
     [||]
     []

let var ?axis ?dtype ?out ?ddof ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "var"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("ddof", Wrap_utils.Option.map ddof Py.Int.of_int); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let view ?dtype ?type_ ?fill_value self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "view"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype (function
| `Ndarray_sub_class x -> Wrap_utils.id x
| `Dtype x -> Dtype.to_pyobject x
)); ("type", type_); ("fill_value", Wrap_utils.Option.map fill_value (function
| `Bool x -> Py.Bool.of_bool x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Extras = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "numpy.ma.extras"

let get_py name = Py.Module.get __wrap_namespace name
module AxisConcatenator = struct
type tag = [`AxisConcatenator]
type t = [`AxisConcatenator | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?axis ?matrix ?ndmin ?trans1d () =
   Py.Module.get_function_with_keywords __wrap_namespace "AxisConcatenator"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("matrix", matrix); ("ndmin", ndmin); ("trans1d", trans1d)])
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MAxisConcatenator = struct
type tag = [`MAxisConcatenator]
type t = [`MAxisConcatenator | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?axis ?matrix ?ndmin ?trans1d () =
   Py.Module.get_function_with_keywords __wrap_namespace "MAxisConcatenator"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("matrix", matrix); ("ndmin", ndmin); ("trans1d", trans1d)])
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key ))])

let makemat ~arr self =
   Py.Module.get_function_with_keywords (to_pyobject self) "makemat"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Mr_class = struct
type tag = [`Mr_class]
type t = [`Mr_class | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "mr_class"
     [||]
     []
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key ))])

let makemat ~arr self =
   Py.Module.get_function_with_keywords (to_pyobject self) "makemat"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Ma = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "numpy.ma.core"

let get_py name = Py.Module.get __wrap_namespace name
module MaskedArrayFutureWarning = struct
type tag = [`MaskedArrayFutureWarning]
type t = [`BaseException | `MaskedArrayFutureWarning | `Object] Obj.t
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
module MaskedConstant = struct
type tag = [`MaskedConstant]
type t = [`ArrayLike | `MaskedConstant | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "MaskedConstant"
     [||]
     []
     |> of_pyobject
let __getitem__ ~indx self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("indx", Some(indx ))])

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let __setitem__ ~indx ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     [||]
     (Wrap_utils.keyword_args [("indx", Some(indx )); ("value", Some(value ))])

let all ?axis ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "all"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims)])

let anom ?axis ?dtype self =
   Py.Module.get_function_with_keywords (to_pyobject self) "anom"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject)])

let any ?axis ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "any"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims)])

let argmax ?axis ?fill_value ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "argmax"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("fill_value", fill_value); ("out", Wrap_utils.Option.map out Obj.to_pyobject)])

let argmin ?axis ?fill_value ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "argmin"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("fill_value", fill_value); ("out", Wrap_utils.Option.map out Obj.to_pyobject)])

let argpartition ?kwargs args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "argpartition"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

                  let argsort ?axis ?kind ?order ?endwith ?fill_value self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "argsort"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("kind", Wrap_utils.Option.map kind (function
| `Heapsort -> Py.String.of_string "heapsort"
| `Mergesort -> Py.String.of_string "mergesort"
| `Stable -> Py.String.of_string "stable"
| `Quicksort -> Py.String.of_string "quicksort"
)); ("order", Wrap_utils.Option.map order Obj.to_pyobject); ("endwith", Wrap_utils.Option.map endwith Py.Bool.of_bool); ("fill_value", fill_value)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
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
| `Dtype x -> Dtype.to_pyobject x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let byteswap ?inplace self =
   Py.Module.get_function_with_keywords (to_pyobject self) "byteswap"
     [||]
     (Wrap_utils.keyword_args [("inplace", Wrap_utils.Option.map inplace Py.Bool.of_bool)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
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
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", out); ("condition", Some(condition ))])

let compressed self =
   Py.Module.get_function_with_keywords (to_pyobject self) "compressed"
     [||]
     []
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let copy ?kwargs args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "copy"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

let count ?axis ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "count"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let cumprod ?axis ?dtype ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cumprod"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out)])

let cumsum ?axis ?dtype ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cumsum"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out)])

let diagonal ?params args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diagonal"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)

let dot ?out ?strict ~b self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("out", out); ("strict", Wrap_utils.Option.map strict Py.Bool.of_bool); ("b", Some(b ))])

                  let dump ~file self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "dump"
                       [||]
                       (Wrap_utils.keyword_args [("file", Some(file |> (function
| `Path x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))])

                  let fill ~value self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fill"
                       [||]
                       (Wrap_utils.keyword_args [("value", Some(value |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])

let filled ?fill_value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "filled"
     [||]
     (Wrap_utils.keyword_args [("fill_value", Wrap_utils.Option.map fill_value Obj.to_pyobject)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let flatten ?params args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "flatten"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let get_fill_value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_fill_value"
     [||]
     []

let get_imag self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_imag"
     [||]
     []

let get_real self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_real"
     [||]
     []

                  let getfield ?offset ~dtype self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "getfield"
                       [||]
                       (Wrap_utils.keyword_args [("offset", Wrap_utils.Option.map offset Py.Int.of_int); ("dtype", Some(dtype |> (function
| `Dtype x -> Dtype.to_pyobject x
| `S x -> Py.String.of_string x
)))])

let harden_mask self =
   Py.Module.get_function_with_keywords (to_pyobject self) "harden_mask"
     [||]
     []

let ids self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ids"
     [||]
     []

let iscontiguous self =
   Py.Module.get_function_with_keywords (to_pyobject self) "iscontiguous"
     [||]
     []

let item args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "item"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

let itemset args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "itemset"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

let max ?axis ?out ?fill_value ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "max"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("fill_value", fill_value); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let mean ?axis ?dtype ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "mean"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("keepdims", keepdims)])

let min ?axis ?out ?fill_value ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "min"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("fill_value", fill_value); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let mini ?axis self =
   Py.Module.get_function_with_keywords (to_pyobject self) "mini"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int)])

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let nonzero self =
   Py.Module.get_function_with_keywords (to_pyobject self) "nonzero"
     [||]
     []

let partition ?kwargs args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "partition"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

let prod ?axis ?dtype ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "prod"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("keepdims", keepdims)])

let ptp ?axis ?out ?fill_value ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ptp"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("fill_value", fill_value); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let put ?mode ~indices ~values self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "put"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Raise -> Py.String.of_string "raise"
| `Wrap -> Py.String.of_string "wrap"
| `Clip -> Py.String.of_string "clip"
)); ("indices", Some(indices )); ("values", Some(values |> Obj.to_pyobject))])

                  let ravel ?order self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "ravel"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
| `A -> Py.String.of_string "A"
| `K -> Py.String.of_string "K"
))])

let repeat ?params args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "repeat"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)

let reshape ?kwargs s self =
   Py.Module.get_function_with_keywords (to_pyobject self) "reshape"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id s)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let resize ?refcheck ?order ~newshape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "resize"
     [||]
     (Wrap_utils.keyword_args [("refcheck", refcheck); ("order", order); ("newshape", Some(newshape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])

let round ?decimals ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "round"
     [||]
     (Wrap_utils.keyword_args [("decimals", decimals); ("out", out)])

let searchsorted ?side ?sorter ~v self =
   Py.Module.get_function_with_keywords (to_pyobject self) "searchsorted"
     [||]
     (Wrap_utils.keyword_args [("side", side); ("sorter", sorter); ("v", Some(v ))])

let set_fill_value ?value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_fill_value"
     [||]
     (Wrap_utils.keyword_args [("value", value)])

let setfield ?offset ~val_ ~dtype self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setfield"
     [||]
     (Wrap_utils.keyword_args [("offset", Wrap_utils.Option.map offset Py.Int.of_int); ("val", Some(val_ )); ("dtype", Some(dtype ))])

let setflags ?write ?align ?uic self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setflags"
     [||]
     (Wrap_utils.keyword_args [("write", Wrap_utils.Option.map write Py.Bool.of_bool); ("align", Wrap_utils.Option.map align Py.Bool.of_bool); ("uic", Wrap_utils.Option.map uic Py.Bool.of_bool)])

let shrink_mask self =
   Py.Module.get_function_with_keywords (to_pyobject self) "shrink_mask"
     [||]
     []

let soften_mask self =
   Py.Module.get_function_with_keywords (to_pyobject self) "soften_mask"
     [||]
     []

                  let sort ?axis ?kind ?order ?endwith ?fill_value self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "sort"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("kind", Wrap_utils.Option.map kind (function
| `Heapsort -> Py.String.of_string "heapsort"
| `Mergesort -> Py.String.of_string "mergesort"
| `Stable -> Py.String.of_string "stable"
| `Quicksort -> Py.String.of_string "quicksort"
)); ("order", Wrap_utils.Option.map order Obj.to_pyobject); ("endwith", Wrap_utils.Option.map endwith Py.Bool.of_bool); ("fill_value", fill_value)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let squeeze ?params args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "squeeze"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)

let std ?axis ?dtype ?out ?ddof ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "std"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("ddof", ddof); ("keepdims", keepdims)])

let sum ?axis ?dtype ?out ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sum"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("keepdims", keepdims)])

let swapaxes ?params args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "swapaxes"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)

let take ?axis ?out ?mode ~indices self =
   Py.Module.get_function_with_keywords (to_pyobject self) "take"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("mode", mode); ("indices", Some(indices ))])

                  let tobytes ?fill_value ?order self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "tobytes"
                       [||]
                       (Wrap_utils.keyword_args [("fill_value", Wrap_utils.Option.map fill_value (function
| `Bool x -> Py.Bool.of_bool x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
)); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
| `A -> Py.String.of_string "A"
))])

let tofile ?sep ?format ~fid self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tofile"
     [||]
     (Wrap_utils.keyword_args [("sep", sep); ("format", format); ("fid", Some(fid ))])

let toflex self =
   Py.Module.get_function_with_keywords (to_pyobject self) "toflex"
     [||]
     []
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let tolist ?fill_value self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "tolist"
                       [||]
                       (Wrap_utils.keyword_args [("fill_value", Wrap_utils.Option.map fill_value (function
| `Bool x -> Py.Bool.of_bool x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let tostring ?fill_value ?order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tostring"
     [||]
     (Wrap_utils.keyword_args [("fill_value", fill_value); ("order", order)])

let trace ?offset ?axis1 ?axis2 ?dtype ?out self =
   Py.Module.get_function_with_keywords (to_pyobject self) "trace"
     [||]
     (Wrap_utils.keyword_args [("offset", offset); ("axis1", axis1); ("axis2", axis2); ("dtype", dtype); ("out", out)])

let transpose ?params args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let unshare_mask self =
   Py.Module.get_function_with_keywords (to_pyobject self) "unshare_mask"
     [||]
     []

let var ?axis ?dtype ?out ?ddof ?keepdims self =
   Py.Module.get_function_with_keywords (to_pyobject self) "var"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("ddof", Wrap_utils.Option.map ddof Py.Int.of_int); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let view ?dtype ?type_ ?fill_value self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "view"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype (function
| `Ndarray_sub_class x -> Wrap_utils.id x
| `Dtype x -> Dtype.to_pyobject x
)); ("type", type_); ("fill_value", Wrap_utils.Option.map fill_value (function
| `Bool x -> Py.Bool.of_bool x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MaskedIterator = struct
type tag = [`MaskedIterator]
type t = [`MaskedIterator | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ma =
   Py.Module.get_function_with_keywords __wrap_namespace "MaskedIterator"
     [||]
     (Wrap_utils.keyword_args [("ma", Some(ma ))])
     |> of_pyobject
let __getitem__ ~indx self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("indx", Some(indx ))])

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let __setitem__ ~index ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index )); ("value", Some(value ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Bytes = struct
type tag = [`Bytes]
type t = [`Bytes | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create iterable_of_ints =
   Py.Module.get_function_with_keywords __wrap_namespace "bytes"
     [||]
     (Wrap_utils.keyword_args [("iterable_of_ints", Some(iterable_of_ints ))])
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let center ?fillchar ~width self =
   Py.Module.get_function_with_keywords (to_pyobject self) "center"
     (Array.of_list @@ List.concat [(match fillchar with None -> [] | Some x -> [x ]);[width ]])
     []

let count ?start ?end_ ~sub self =
   Py.Module.get_function_with_keywords (to_pyobject self) "count"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("sub", Some(sub ))])

let decode ?encoding ?errors self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decode"
     [||]
     (Wrap_utils.keyword_args [("encoding", encoding); ("errors", errors)])

let endswith ?start ?end_ ~suffix self =
   Py.Module.get_function_with_keywords (to_pyobject self) "endswith"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("suffix", Some(suffix ))])

let expandtabs ?tabsize self =
   Py.Module.get_function_with_keywords (to_pyobject self) "expandtabs"
     [||]
     (Wrap_utils.keyword_args [("tabsize", tabsize)])

let find ?start ?end_ ~sub self =
   Py.Module.get_function_with_keywords (to_pyobject self) "find"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("sub", Some(sub ))])

let fromhex ~string self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fromhex"
     (Array.of_list @@ List.concat [[string ]])
     []

let index ?start ?end_ ~sub self =
   Py.Module.get_function_with_keywords (to_pyobject self) "index"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("sub", Some(sub ))])

let join ~iterable_of_bytes self =
   Py.Module.get_function_with_keywords (to_pyobject self) "join"
     (Array.of_list @@ List.concat [[iterable_of_bytes ]])
     []

let ljust ?fillchar ~width self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ljust"
     (Array.of_list @@ List.concat [(match fillchar with None -> [] | Some x -> [x ]);[width ]])
     []

let lstrip ?bytes self =
   Py.Module.get_function_with_keywords (to_pyobject self) "lstrip"
     (Array.of_list @@ List.concat [(match bytes with None -> [] | Some x -> [x ])])
     []

let partition ~sep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "partition"
     (Array.of_list @@ List.concat [[sep ]])
     []

let replace ?count ~old ~new_ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "replace"
     (Array.of_list @@ List.concat [(match count with None -> [] | Some x -> [x ]);[old ];[new_ ]])
     []

let rfind ?start ?end_ ~sub self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rfind"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("sub", Some(sub ))])

let rindex ?start ?end_ ~sub self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rindex"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("sub", Some(sub ))])

let rjust ?fillchar ~width self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rjust"
     (Array.of_list @@ List.concat [(match fillchar with None -> [] | Some x -> [x ]);[width ]])
     []

let rpartition ~sep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rpartition"
     (Array.of_list @@ List.concat [[sep ]])
     []

let rsplit ?sep ?maxsplit self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rsplit"
     [||]
     (Wrap_utils.keyword_args [("sep", sep); ("maxsplit", maxsplit)])

let rstrip ?bytes self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rstrip"
     (Array.of_list @@ List.concat [(match bytes with None -> [] | Some x -> [x ])])
     []

let split ?sep ?maxsplit self =
   Py.Module.get_function_with_keywords (to_pyobject self) "split"
     [||]
     (Wrap_utils.keyword_args [("sep", sep); ("maxsplit", maxsplit)])

let splitlines ?keepends self =
   Py.Module.get_function_with_keywords (to_pyobject self) "splitlines"
     [||]
     (Wrap_utils.keyword_args [("keepends", keepends)])

let startswith ?start ?end_ ~prefix self =
   Py.Module.get_function_with_keywords (to_pyobject self) "startswith"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("end", end_); ("prefix", Some(prefix ))])

let strip ?bytes self =
   Py.Module.get_function_with_keywords (to_pyobject self) "strip"
     (Array.of_list @@ List.concat [(match bytes with None -> [] | Some x -> [x ])])
     []

let translate ?delete ~table self =
   Py.Module.get_function_with_keywords (to_pyobject self) "translate"
     (Array.of_list @@ List.concat [[table ]])
     (Wrap_utils.keyword_args [("delete", delete)])

let zfill ~width self =
   Py.Module.get_function_with_keywords (to_pyobject self) "zfill"
     (Array.of_list @@ List.concat [[width ]])
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Recursive = struct
type tag = [`Recursive]
type t = [`Object | `Recursive] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create func =
   Py.Module.get_function_with_keywords __wrap_namespace "recursive"
     [||]
     (Wrap_utils.keyword_args [("func", Some(func ))])
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let abs ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "abs"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let absolute ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "absolute"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let add ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "add"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let all ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "all"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let allclose ?masked_equal ?rtol ?atol ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "allclose"
     [||]
     (Wrap_utils.keyword_args [("masked_equal", Wrap_utils.Option.map masked_equal Py.Bool.of_bool); ("rtol", Wrap_utils.Option.map rtol Py.Float.of_float); ("atol", Wrap_utils.Option.map atol Py.Float.of_float); ("b", Some(b )); ("a", Some(a ))])
     |> Py.Bool.to_bool
let allequal ?fill_value ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "allequal"
     [||]
     (Wrap_utils.keyword_args [("fill_value", Wrap_utils.Option.map fill_value Py.Bool.of_bool); ("b", Some(b )); ("a", Some(a ))])
     |> Py.Bool.to_bool
let alltrue ?axis ?dtype ~target () =
   Py.Module.get_function_with_keywords __wrap_namespace "alltrue"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("target", Some(target ))])

                  let amax ?axis ?out ?keepdims ?initial ?where a =
                     Py.Module.get_function_with_keywords __wrap_namespace "amax"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("initial", Wrap_utils.Option.map initial (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("where", where); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let amin ?axis ?out ?keepdims ?initial ?where a =
                     Py.Module.get_function_with_keywords __wrap_namespace "amin"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("initial", Wrap_utils.Option.map initial (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("where", where); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let angle ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "angle"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let anom ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "anom"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let anomalies ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "anomalies"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let any ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "any"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let append ?axis ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "append"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("b", Some(b |> Obj.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])

let arange ?params args =
   Py.Module.get_function_with_keywords __wrap_namespace "arange"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let arccos ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "arccos"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let arccosh ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "arccosh"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let arcsin ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "arcsin"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let arcsinh ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "arcsinh"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let arctan ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "arctan"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let arctan2 ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "arctan2"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let arctanh ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "arctanh"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let argmax ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "argmax"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let argmin ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "argmin"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

                  let argsort ?axis ?kind ?order ?endwith ?fill_value a =
                     Py.Module.get_function_with_keywords __wrap_namespace "argsort"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("kind", Wrap_utils.Option.map kind (function
| `Heapsort -> Py.String.of_string "heapsort"
| `Mergesort -> Py.String.of_string "mergesort"
| `Stable -> Py.String.of_string "stable"
| `Quicksort -> Py.String.of_string "quicksort"
)); ("order", Wrap_utils.Option.map order Obj.to_pyobject); ("endwith", Wrap_utils.Option.map endwith Py.Bool.of_bool); ("fill_value", fill_value); ("a", Some(a ))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let around ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "around"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))

                  let array ?dtype ?copy ?order ?mask ?fill_value ?keep_mask ?hard_mask ?shrink ?subok ?ndmin ~data () =
                     Py.Module.get_function_with_keywords __wrap_namespace "array"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("order", Wrap_utils.Option.map order (function
| `A -> Py.String.of_string "A"
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("mask", mask); ("fill_value", Wrap_utils.Option.map fill_value (function
| `Bool x -> Py.Bool.of_bool x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
)); ("keep_mask", Wrap_utils.Option.map keep_mask Py.Bool.of_bool); ("hard_mask", Wrap_utils.Option.map hard_mask Py.Bool.of_bool); ("shrink", Wrap_utils.Option.map shrink Py.Bool.of_bool); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("ndmin", Wrap_utils.Option.map ndmin Py.Int.of_int); ("data", Some(data |> Obj.to_pyobject))])

let asanyarray ?dtype a =
   Py.Module.get_function_with_keywords __wrap_namespace "asanyarray"
     [||]
     (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("a", Some(a |> Obj.to_pyobject))])

                  let asarray ?dtype ?order a =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("a", Some(a |> Obj.to_pyobject))])

let bitwise_and ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "bitwise_and"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let bitwise_or ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "bitwise_or"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let bitwise_xor ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "bitwise_xor"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let ceil ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "ceil"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let choose ?out ?mode ~indices ~choices () =
                     Py.Module.get_function_with_keywords __wrap_namespace "choose"
                       [||]
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Obj.to_pyobject); ("mode", Wrap_utils.Option.map mode (function
| `Raise -> Py.String.of_string "raise"
| `Wrap -> Py.String.of_string "wrap"
| `Clip -> Py.String.of_string "clip"
)); ("indices", Some(indices )); ("choices", Some(choices ))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let clip ?out ?kwargs ~a_min ~a_max a =
                     Py.Module.get_function_with_keywords __wrap_namespace "clip"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Obj.to_pyobject); ("a_min", Some(a_min |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
| `None -> Py.none
))); ("a_max", Some(a_max |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
| `None -> Py.none
))); ("a", Some(a |> Obj.to_pyobject))]) (match kwargs with None -> [] | Some x -> x))
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let common_fill_value ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "common_fill_value"
     [||]
     (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))])
     |> (fun py -> if Py.is_none py then None else Some (Wrap_utils.id py))
let compress ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "compress"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let compressed x =
   Py.Module.get_function_with_keywords __wrap_namespace "compressed"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let concatenate ?axis ~arrays () =
   Py.Module.get_function_with_keywords __wrap_namespace "concatenate"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("arrays", Some(arrays ))])

let conjugate ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "conjugate"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let convolve ?mode ?propagate_mask ~v a =
                     Py.Module.get_function_with_keywords __wrap_namespace "convolve"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Valid -> Py.String.of_string "valid"
| `Same -> Py.String.of_string "same"
| `Full -> Py.String.of_string "full"
)); ("propagate_mask", Wrap_utils.Option.map propagate_mask Py.Bool.of_bool); ("v", Some(v )); ("a", Some(a ))])

let copy ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "copy"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

                  let correlate ?mode ?propagate_mask ~v a =
                     Py.Module.get_function_with_keywords __wrap_namespace "correlate"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Valid -> Py.String.of_string "valid"
| `Same -> Py.String.of_string "same"
| `Full -> Py.String.of_string "full"
)); ("propagate_mask", Wrap_utils.Option.map propagate_mask Py.Bool.of_bool); ("v", Some(v )); ("a", Some(a ))])

let cos ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "cos"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let cosh ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "cosh"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let count ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "count"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let cumprod ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "cumprod"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let cumsum ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "cumsum"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

                  let default_fill_value obj =
                     Py.Module.get_function_with_keywords __wrap_namespace "default_fill_value"
                       [||]
                       (Wrap_utils.keyword_args [("obj", Some(obj |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `Dtype x -> Dtype.to_pyobject x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
)))])

let diag ?k ~v () =
   Py.Module.get_function_with_keywords __wrap_namespace "diag"
     [||]
     (Wrap_utils.keyword_args [("k", k); ("v", Some(v ))])

let diagonal ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "diagonal"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let diff ?n ?axis ?prepend ?append a =
   Py.Module.get_function_with_keywords __wrap_namespace "diff"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("prepend", prepend); ("append", append); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let divide ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "divide"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let doc_note ~initialdoc ~note () =
   Py.Module.get_function_with_keywords __wrap_namespace "doc_note"
     [||]
     (Wrap_utils.keyword_args [("initialdoc", Some(initialdoc )); ("note", Some(note ))])

let dot ?strict ?out ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "dot"
     [||]
     (Wrap_utils.keyword_args [("strict", Wrap_utils.Option.map strict Py.Bool.of_bool); ("out", out); ("b", Some(b )); ("a", Some(a ))])

let empty ?params args =
   Py.Module.get_function_with_keywords __wrap_namespace "empty"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let empty_like ?params args =
   Py.Module.get_function_with_keywords __wrap_namespace "empty_like"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let equal ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "equal"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let exp ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "exp"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let expand_dims ~axis a =
   Py.Module.get_function_with_keywords __wrap_namespace "expand_dims"
     [||]
     (Wrap_utils.keyword_args [("axis", Some(axis |> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml))); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let fabs ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "fabs"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let filled ?fill_value a =
                     Py.Module.get_function_with_keywords __wrap_namespace "filled"
                       [||]
                       (Wrap_utils.keyword_args [("fill_value", Wrap_utils.Option.map fill_value Obj.to_pyobject); ("a", Some(a |> (function
| `Ndarray x -> Obj.to_pyobject x
| `MaskedArray x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let fix_invalid ?mask ?copy ?fill_value a =
                     Py.Module.get_function_with_keywords __wrap_namespace "fix_invalid"
                       [||]
                       (Wrap_utils.keyword_args [("mask", mask); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("fill_value", Wrap_utils.Option.map fill_value (function
| `Bool x -> Py.Bool.of_bool x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
)); ("a", Some(a |> Obj.to_pyobject))])

let flatten_mask mask =
   Py.Module.get_function_with_keywords __wrap_namespace "flatten_mask"
     [||]
     (Wrap_utils.keyword_args [("mask", Some(mask |> Obj.to_pyobject))])

let flatten_structured_array a =
   Py.Module.get_function_with_keywords __wrap_namespace "flatten_structured_array"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a ))])

let floor ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "floor"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let floor_divide ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "floor_divide"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let fmod ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "fmod"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let formatargspec ?varargs ?varkw ?defaults ?formatarg ?formatvarargs ?formatvarkw ?formatvalue ?join ~args () =
   Py.Module.get_function_with_keywords __wrap_namespace "formatargspec"
     [||]
     (Wrap_utils.keyword_args [("varargs", varargs); ("varkw", varkw); ("defaults", defaults); ("formatarg", formatarg); ("formatvarargs", formatvarargs); ("formatvarkw", formatvarkw); ("formatvalue", formatvalue); ("join", join); ("args", Some(args ))])

let frombuffer ?params args =
   Py.Module.get_function_with_keywords __wrap_namespace "frombuffer"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)

let fromfile ?dtype ?count ?sep ~file () =
   Py.Module.get_function_with_keywords __wrap_namespace "fromfile"
     [||]
     (Wrap_utils.keyword_args [("dtype", dtype); ("count", count); ("sep", sep); ("file", Some(file ))])

let fromflex fxarray =
   Py.Module.get_function_with_keywords __wrap_namespace "fromflex"
     [||]
     (Wrap_utils.keyword_args [("fxarray", Some(fxarray |> Obj.to_pyobject))])

let fromfunction ?params args =
   Py.Module.get_function_with_keywords __wrap_namespace "fromfunction"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)

let get_data ?subok a =
   Py.Module.get_function_with_keywords __wrap_namespace "get_data"
     [||]
     (Wrap_utils.keyword_args [("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])

let get_fill_value a =
   Py.Module.get_function_with_keywords __wrap_namespace "get_fill_value"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a ))])

let get_mask a =
   Py.Module.get_function_with_keywords __wrap_namespace "get_mask"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Obj.to_pyobject))])

let get_masked_subclass arrays =
   Py.Module.get_function_with_keywords __wrap_namespace "get_masked_subclass"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arrays)])
     []

let get_object_signature obj =
   Py.Module.get_function_with_keywords __wrap_namespace "get_object_signature"
     [||]
     (Wrap_utils.keyword_args [("obj", Some(obj ))])

let getargspec func =
   Py.Module.get_function_with_keywords __wrap_namespace "getargspec"
     [||]
     (Wrap_utils.keyword_args [("func", Some(func ))])

let getdata ?subok a =
   Py.Module.get_function_with_keywords __wrap_namespace "getdata"
     [||]
     (Wrap_utils.keyword_args [("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])

let getmask a =
   Py.Module.get_function_with_keywords __wrap_namespace "getmask"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Obj.to_pyobject))])

let getmaskarray arr =
   Py.Module.get_function_with_keywords __wrap_namespace "getmaskarray"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr |> Obj.to_pyobject))])

let greater ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "greater"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let greater_equal ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "greater_equal"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))

let harden_mask ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "harden_mask"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let hypot ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "hypot"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let identity ?params args =
   Py.Module.get_function_with_keywords __wrap_namespace "identity"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let ids ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "ids"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let indices ?dtype ?sparse ~dimensions () =
   Py.Module.get_function_with_keywords __wrap_namespace "indices"
     [||]
     (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("sparse", Wrap_utils.Option.map sparse Py.Bool.of_bool); ("dimensions", Some(dimensions |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])

let inner ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "inner"
     [||]
     (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let innerproduct ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "innerproduct"
     [||]
     (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let isMA x =
   Py.Module.get_function_with_keywords __wrap_namespace "isMA"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])
     |> Py.Bool.to_bool
let isMaskedArray x =
   Py.Module.get_function_with_keywords __wrap_namespace "isMaskedArray"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])
     |> Py.Bool.to_bool
let is_mask m =
   Py.Module.get_function_with_keywords __wrap_namespace "is_mask"
     [||]
     (Wrap_utils.keyword_args [("m", Some(m |> Obj.to_pyobject))])
     |> Py.Bool.to_bool
let is_masked x =
   Py.Module.get_function_with_keywords __wrap_namespace "is_masked"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Obj.to_pyobject))])
     |> Py.Bool.to_bool
let is_string_or_list_of_strings val_ =
   Py.Module.get_function_with_keywords __wrap_namespace "is_string_or_list_of_strings"
     [||]
     (Wrap_utils.keyword_args [("val", Some(val_ ))])

let isarray x =
   Py.Module.get_function_with_keywords __wrap_namespace "isarray"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])
     |> Py.Bool.to_bool
let iscomplexobj x =
   Py.Module.get_function_with_keywords __wrap_namespace "iscomplexobj"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])
     |> Py.Bool.to_bool
let left_shift ~n a =
   Py.Module.get_function_with_keywords __wrap_namespace "left_shift"
     [||]
     (Wrap_utils.keyword_args [("n", Some(n )); ("a", Some(a ))])

let less ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "less"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let less_equal ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "less_equal"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let log ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "log"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let log10 ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "log10"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let log2 ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "log2"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let logical_and ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "logical_and"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))

let logical_not ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "logical_not"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))

let logical_or ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "logical_or"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))

let logical_xor ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "logical_xor"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))

let make_mask ?copy ?shrink ?dtype ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "make_mask"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("shrink", Wrap_utils.Option.map shrink Py.Bool.of_bool); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("m", Some(m |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let make_mask_descr ndtype =
   Py.Module.get_function_with_keywords __wrap_namespace "make_mask_descr"
     [||]
     (Wrap_utils.keyword_args [("ndtype", Some(ndtype |> Dtype.to_pyobject))])
     |> Dtype.of_pyobject
let make_mask_none ?dtype ~newshape () =
   Py.Module.get_function_with_keywords __wrap_namespace "make_mask_none"
     [||]
     (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("newshape", Some(newshape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let mask_or ?copy ?shrink ~m1 ~m2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "mask_or"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("shrink", Wrap_utils.Option.map shrink Py.Bool.of_bool); ("m1", Some(m1 )); ("m2", Some(m2 ))])

                  let mask_rowcols ?axis a =
                     Py.Module.get_function_with_keywords __wrap_namespace "mask_rowcols"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("a", Some(a |> (function
| `Ndarray x -> Obj.to_pyobject x
| `MaskedArray x -> Wrap_utils.id x
)))])

let masked_equal ?copy ~value x =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_equal"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("value", Some(value )); ("x", Some(x ))])

let masked_greater ?copy ~value x =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_greater"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("value", Some(value )); ("x", Some(x ))])

let masked_greater_equal ?copy ~value x =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_greater_equal"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("value", Some(value )); ("x", Some(x ))])

let masked_inside ?copy ~v1 ~v2 x =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_inside"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("v1", Some(v1 )); ("v2", Some(v2 )); ("x", Some(x ))])

let masked_invalid ?copy a =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_invalid"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("a", Some(a ))])

let masked_less ?copy ~value x =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_less"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("value", Some(value )); ("x", Some(x ))])

let masked_less_equal ?copy ~value x =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_less_equal"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("value", Some(value )); ("x", Some(x ))])

let masked_not_equal ?copy ~value x =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_not_equal"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("value", Some(value )); ("x", Some(x ))])

let masked_object ?copy ?shrink ~value x =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_object"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("shrink", Wrap_utils.Option.map shrink Py.Bool.of_bool); ("value", Some(value )); ("x", Some(x |> Obj.to_pyobject))])

let masked_outside ?copy ~v1 ~v2 x =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_outside"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("v1", Some(v1 )); ("v2", Some(v2 )); ("x", Some(x ))])

let masked_values ?rtol ?atol ?copy ?shrink ~value x =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_values"
     [||]
     (Wrap_utils.keyword_args [("rtol", rtol); ("atol", atol); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("shrink", Wrap_utils.Option.map shrink Py.Bool.of_bool); ("value", Some(value |> Py.Float.of_float)); ("x", Some(x |> Obj.to_pyobject))])

let masked_where ?copy ~condition a =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_where"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("condition", Some(condition |> Obj.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])

let max ?axis ?out ?fill_value ?keepdims ~obj () =
   Py.Module.get_function_with_keywords __wrap_namespace "max"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("fill_value", fill_value); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("obj", Some(obj ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let maximum ?b a =
   Py.Module.get_function_with_keywords __wrap_namespace "maximum"
     [||]
     (Wrap_utils.keyword_args [("b", b); ("a", Some(a ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let maximum_fill_value obj =
                     Py.Module.get_function_with_keywords __wrap_namespace "maximum_fill_value"
                       [||]
                       (Wrap_utils.keyword_args [("obj", Some(obj |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `Dtype x -> Dtype.to_pyobject x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
)))])

let mean ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "mean"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let min ?axis ?out ?fill_value ?keepdims ~obj () =
   Py.Module.get_function_with_keywords __wrap_namespace "min"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("fill_value", fill_value); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("obj", Some(obj ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let minimum ?b a =
   Py.Module.get_function_with_keywords __wrap_namespace "minimum"
     [||]
     (Wrap_utils.keyword_args [("b", b); ("a", Some(a ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let minimum_fill_value obj =
                     Py.Module.get_function_with_keywords __wrap_namespace "minimum_fill_value"
                       [||]
                       (Wrap_utils.keyword_args [("obj", Some(obj |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `Dtype x -> Dtype.to_pyobject x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
)))])

let mod_ ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "mod"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let multiply ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "multiply"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let narray ?dtype ?copy ?order ?subok ?ndmin ~object_ () =
                     Py.Module.get_function_with_keywords __wrap_namespace "narray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("order", Wrap_utils.Option.map order (function
| `K -> Py.String.of_string "K"
| `A -> Py.String.of_string "A"
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("ndmin", Wrap_utils.Option.map ndmin Py.Int.of_int); ("object", Some(object_ |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let ndim obj =
   Py.Module.get_function_with_keywords __wrap_namespace "ndim"
     [||]
     (Wrap_utils.keyword_args [("obj", Some(obj ))])
     |> Py.Int.to_int
let negative ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "negative"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let nonzero ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "nonzero"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

                  let normalize_axis_tuple ?argname ?allow_duplicate ~axis ~ndim () =
                     Py.Module.get_function_with_keywords __wrap_namespace "normalize_axis_tuple"
                       [||]
                       (Wrap_utils.keyword_args [("argname", Wrap_utils.Option.map argname Py.String.of_string); ("allow_duplicate", Wrap_utils.Option.map allow_duplicate Py.Bool.of_bool); ("axis", Some(axis |> (function
| `Iterable_of_int x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("ndim", Some(ndim |> Py.Int.of_int))])

let not_equal ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "not_equal"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let ones ?params args =
   Py.Module.get_function_with_keywords __wrap_namespace "ones"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let ones_like ?dtype ?order ?subok ?shape a =
                     Py.Module.get_function_with_keywords __wrap_namespace "ones_like"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `A -> Py.String.of_string "A"
| `F -> Py.String.of_string "F"
| `PyObject x -> Wrap_utils.id x
)); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("shape", Wrap_utils.Option.map shape (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let outer ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "outer"
     [||]
     (Wrap_utils.keyword_args [("b", Some(b |> Obj.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let outerproduct ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "outerproduct"
     [||]
     (Wrap_utils.keyword_args [("b", Some(b |> Obj.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let power ?third ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "power"
     [||]
     (Wrap_utils.keyword_args [("third", third); ("b", Some(b )); ("a", Some(a ))])

let prod ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "prod"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let product ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "product"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let ptp ?axis ?out ?fill_value ?keepdims ~obj () =
   Py.Module.get_function_with_keywords __wrap_namespace "ptp"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("fill_value", fill_value); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("obj", Some(obj ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let put ?mode ~indices ~values a =
   Py.Module.get_function_with_keywords __wrap_namespace "put"
     [||]
     (Wrap_utils.keyword_args [("mode", mode); ("indices", Some(indices )); ("values", Some(values )); ("a", Some(a ))])

let putmask ~mask ~values a =
   Py.Module.get_function_with_keywords __wrap_namespace "putmask"
     [||]
     (Wrap_utils.keyword_args [("mask", Some(mask )); ("values", Some(values )); ("a", Some(a ))])

let ravel ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "ravel"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let reduce ?initial ~function_ ~sequence () =
   Py.Module.get_function_with_keywords __wrap_namespace "reduce"
     [||]
     (Wrap_utils.keyword_args [("initial", initial); ("function", Some(function_ )); ("sequence", Some(sequence ))])

let remainder ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "remainder"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let repeat ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "repeat"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let reshape ?order ~new_shape a =
   Py.Module.get_function_with_keywords __wrap_namespace "reshape"
     [||]
     (Wrap_utils.keyword_args [("order", order); ("new_shape", Some(new_shape )); ("a", Some(a ))])

let resize ~new_shape x =
   Py.Module.get_function_with_keywords __wrap_namespace "resize"
     [||]
     (Wrap_utils.keyword_args [("new_shape", Some(new_shape )); ("x", Some(x ))])

let right_shift ~n a =
   Py.Module.get_function_with_keywords __wrap_namespace "right_shift"
     [||]
     (Wrap_utils.keyword_args [("n", Some(n )); ("a", Some(a ))])

let round ?decimals ?out a =
   Py.Module.get_function_with_keywords __wrap_namespace "round"
     [||]
     (Wrap_utils.keyword_args [("decimals", Wrap_utils.Option.map decimals Py.Int.of_int); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("a", Some(a ))])

let set_fill_value ~fill_value a =
   Py.Module.get_function_with_keywords __wrap_namespace "set_fill_value"
     [||]
     (Wrap_utils.keyword_args [("fill_value", Some(fill_value |> Dtype.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])

        let shape obj =
           Py.Module.get_function_with_keywords __wrap_namespace "shape"
             [||]
             (Wrap_utils.keyword_args [("obj", Some(obj ))])
             |> (fun py -> let len = Py.Sequence.length py in Array.init len
(fun i -> Py.Int.to_int (Py.Sequence.get_item py i)))
let shrink_mask ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "shrink_mask"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let sin ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "sin"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let sinh ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "sinh"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let size ?axis ~obj () =
   Py.Module.get_function_with_keywords __wrap_namespace "size"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("obj", Some(obj ))])
     |> Py.Int.to_int
let soften_mask ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "soften_mask"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let sometrue ?axis ?dtype ~target () =
   Py.Module.get_function_with_keywords __wrap_namespace "sometrue"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("target", Some(target ))])

let sort ?axis ?kind ?order ?endwith ?fill_value a =
   Py.Module.get_function_with_keywords __wrap_namespace "sort"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("kind", kind); ("order", order); ("endwith", endwith); ("fill_value", fill_value); ("a", Some(a ))])

let sqrt ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "sqrt"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let squeeze ?axis a =
   Py.Module.get_function_with_keywords __wrap_namespace "squeeze"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let std ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "std"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let subtract ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "subtract"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let sum ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "sum"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let swapaxes ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "swapaxes"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let take ?axis ?out ?mode ~indices a =
   Py.Module.get_function_with_keywords __wrap_namespace "take"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("mode", mode); ("indices", Some(indices )); ("a", Some(a ))])

let tan ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "tan"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let tanh ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "tanh"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let trace ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "trace"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let transpose ?axes a =
   Py.Module.get_function_with_keywords __wrap_namespace "transpose"
     [||]
     (Wrap_utils.keyword_args [("axes", axes); ("a", Some(a ))])

let true_divide ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "true_divide"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let var ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "var"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a |> Obj.to_pyobject))]) (match params with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let where ?x ?y ~condition () =
                     Py.Module.get_function_with_keywords __wrap_namespace "where"
                       [||]
                       (Wrap_utils.keyword_args [("x", x); ("y", y); ("condition", Some(condition |> (function
| `Bool x -> Py.Bool.of_bool x
| `Ndarray x -> Obj.to_pyobject x
)))])

let zeros ?params args =
   Py.Module.get_function_with_keywords __wrap_namespace "zeros"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let zeros_like ?dtype ?order ?subok ?shape a =
                     Py.Module.get_function_with_keywords __wrap_namespace "zeros_like"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `A -> Py.String.of_string "A"
| `F -> Py.String.of_string "F"
| `PyObject x -> Wrap_utils.id x
)); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("shape", Wrap_utils.Option.map shape (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))

end
let add ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "add"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let apply_along_axis ?kwargs ~func1d ~axis ~arr args =
   Py.Module.get_function_with_keywords __wrap_namespace "apply_along_axis"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("func1d", Some(func1d )); ("axis", Some(axis |> Py.Int.of_int)); ("arr", Some(arr ))]) (match kwargs with None -> [] | Some x -> x))

let apply_over_axes ~func ~axes a =
   Py.Module.get_function_with_keywords __wrap_namespace "apply_over_axes"
     [||]
     (Wrap_utils.keyword_args [("func", Some(func )); ("axes", Some(axes |> Obj.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let array ?dtype ?copy ?order ?mask ?fill_value ?keep_mask ?hard_mask ?shrink ?subok ?ndmin ~data () =
                     Py.Module.get_function_with_keywords __wrap_namespace "array"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("order", Wrap_utils.Option.map order (function
| `A -> Py.String.of_string "A"
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("mask", mask); ("fill_value", Wrap_utils.Option.map fill_value (function
| `Bool x -> Py.Bool.of_bool x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
)); ("keep_mask", Wrap_utils.Option.map keep_mask Py.Bool.of_bool); ("hard_mask", Wrap_utils.Option.map hard_mask Py.Bool.of_bool); ("shrink", Wrap_utils.Option.map shrink Py.Bool.of_bool); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("ndmin", Wrap_utils.Option.map ndmin Py.Int.of_int); ("data", Some(data |> Obj.to_pyobject))])

                  let asarray ?dtype ?order a =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("a", Some(a |> Obj.to_pyobject))])

let atleast_1d ?params args =
   Py.Module.get_function_with_keywords __wrap_namespace "atleast_1d"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let atleast_2d ?params args =
   Py.Module.get_function_with_keywords __wrap_namespace "atleast_2d"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)

let atleast_3d ?params args =
   Py.Module.get_function_with_keywords __wrap_namespace "atleast_3d"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)

let average ?axis ?weights ?returned a =
   Py.Module.get_function_with_keywords __wrap_namespace "average"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("weights", Wrap_utils.Option.map weights Obj.to_pyobject); ("returned", Wrap_utils.Option.map returned Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])

let clump_masked a =
   Py.Module.get_function_with_keywords __wrap_namespace "clump_masked"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Obj.to_pyobject))])

let clump_unmasked a =
   Py.Module.get_function_with_keywords __wrap_namespace "clump_unmasked"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Obj.to_pyobject))])

let column_stack ?params x args =
   Py.Module.get_function_with_keywords __wrap_namespace "column_stack"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("x", Some(x ))]) (match params with None -> [] | Some x -> x))

let compress_cols a =
   Py.Module.get_function_with_keywords __wrap_namespace "compress_cols"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a ))])

                  let compress_nd ?axis x =
                     Py.Module.get_function_with_keywords __wrap_namespace "compress_nd"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("x", Some(x |> (function
| `Ndarray x -> Obj.to_pyobject x
| `MaskedArray x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let compress_rowcols ?axis x =
                     Py.Module.get_function_with_keywords __wrap_namespace "compress_rowcols"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("x", Some(x |> (function
| `Ndarray x -> Obj.to_pyobject x
| `MaskedArray x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let compress_rows a =
   Py.Module.get_function_with_keywords __wrap_namespace "compress_rows"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a ))])

let concatenate ?axis ~arrays () =
   Py.Module.get_function_with_keywords __wrap_namespace "concatenate"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("arrays", Some(arrays ))])

let corrcoef ?y ?rowvar ?bias ?allow_masked ?ddof x =
   Py.Module.get_function_with_keywords __wrap_namespace "corrcoef"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Obj.to_pyobject); ("rowvar", Wrap_utils.Option.map rowvar Py.Bool.of_bool); ("bias", bias); ("allow_masked", Wrap_utils.Option.map allow_masked Py.Bool.of_bool); ("ddof", ddof); ("x", Some(x |> Obj.to_pyobject))])

let count ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "count"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let count_masked ?axis ~arr () =
   Py.Module.get_function_with_keywords __wrap_namespace "count_masked"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("arr", Some(arr |> Obj.to_pyobject))])

let cov ?y ?rowvar ?bias ?allow_masked ?ddof x =
   Py.Module.get_function_with_keywords __wrap_namespace "cov"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Obj.to_pyobject); ("rowvar", Wrap_utils.Option.map rowvar Py.Bool.of_bool); ("bias", Wrap_utils.Option.map bias Py.Bool.of_bool); ("allow_masked", Wrap_utils.Option.map allow_masked Py.Bool.of_bool); ("ddof", Wrap_utils.Option.map ddof Py.Int.of_int); ("x", Some(x |> Obj.to_pyobject))])

let diagflat ?params x args =
   Py.Module.get_function_with_keywords __wrap_namespace "diagflat"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("x", Some(x ))]) (match params with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let dot ?strict ?out ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "dot"
     [||]
     (Wrap_utils.keyword_args [("strict", Wrap_utils.Option.map strict Py.Bool.of_bool); ("out", out); ("b", Some(b )); ("a", Some(a ))])

let dstack ?params x args =
   Py.Module.get_function_with_keywords __wrap_namespace "dstack"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("x", Some(x ))]) (match params with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let ediff1d ?to_end ?to_begin ~arr () =
   Py.Module.get_function_with_keywords __wrap_namespace "ediff1d"
     [||]
     (Wrap_utils.keyword_args [("to_end", to_end); ("to_begin", to_begin); ("arr", Some(arr ))])

                  let filled ?fill_value a =
                     Py.Module.get_function_with_keywords __wrap_namespace "filled"
                       [||]
                       (Wrap_utils.keyword_args [("fill_value", Wrap_utils.Option.map fill_value Obj.to_pyobject); ("a", Some(a |> (function
| `Ndarray x -> Obj.to_pyobject x
| `MaskedArray x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let flatnotmasked_contiguous a =
   Py.Module.get_function_with_keywords __wrap_namespace "flatnotmasked_contiguous"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let flatnotmasked_edges a =
   Py.Module.get_function_with_keywords __wrap_namespace "flatnotmasked_edges"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> if Py.is_none py then None else Some ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) py))
let flatten_inplace seq =
   Py.Module.get_function_with_keywords __wrap_namespace "flatten_inplace"
     [||]
     (Wrap_utils.keyword_args [("seq", Some(seq ))])

let get_masked_subclass arrays =
   Py.Module.get_function_with_keywords __wrap_namespace "get_masked_subclass"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arrays)])
     []

let getdata ?subok a =
   Py.Module.get_function_with_keywords __wrap_namespace "getdata"
     [||]
     (Wrap_utils.keyword_args [("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])

let getmask a =
   Py.Module.get_function_with_keywords __wrap_namespace "getmask"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Obj.to_pyobject))])

let getmaskarray arr =
   Py.Module.get_function_with_keywords __wrap_namespace "getmaskarray"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr |> Obj.to_pyobject))])

let hsplit ?params x args =
   Py.Module.get_function_with_keywords __wrap_namespace "hsplit"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("x", Some(x ))]) (match params with None -> [] | Some x -> x))

let hstack ?params x args =
   Py.Module.get_function_with_keywords __wrap_namespace "hstack"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("x", Some(x ))]) (match params with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let in1d ?assume_unique ?invert ~ar1 ~ar2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "in1d"
     [||]
     (Wrap_utils.keyword_args [("assume_unique", assume_unique); ("invert", invert); ("ar1", Some(ar1 )); ("ar2", Some(ar2 ))])

let intersect1d ?assume_unique ~ar1 ~ar2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "intersect1d"
     [||]
     (Wrap_utils.keyword_args [("assume_unique", assume_unique); ("ar1", Some(ar1 )); ("ar2", Some(ar2 ))])

let isin ?assume_unique ?invert ~element ~test_elements () =
   Py.Module.get_function_with_keywords __wrap_namespace "isin"
     [||]
     (Wrap_utils.keyword_args [("assume_unique", assume_unique); ("invert", invert); ("element", Some(element )); ("test_elements", Some(test_elements ))])

let issequence seq =
   Py.Module.get_function_with_keywords __wrap_namespace "issequence"
     [||]
     (Wrap_utils.keyword_args [("seq", Some(seq ))])

let make_mask_descr ndtype =
   Py.Module.get_function_with_keywords __wrap_namespace "make_mask_descr"
     [||]
     (Wrap_utils.keyword_args [("ndtype", Some(ndtype |> Dtype.to_pyobject))])
     |> Dtype.of_pyobject
let mask_cols ?axis a =
   Py.Module.get_function_with_keywords __wrap_namespace "mask_cols"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("a", Some(a ))])

let mask_or ?copy ?shrink ~m1 ~m2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "mask_or"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("shrink", Wrap_utils.Option.map shrink Py.Bool.of_bool); ("m1", Some(m1 )); ("m2", Some(m2 ))])

                  let mask_rowcols ?axis a =
                     Py.Module.get_function_with_keywords __wrap_namespace "mask_rowcols"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("a", Some(a |> (function
| `Ndarray x -> Obj.to_pyobject x
| `MaskedArray x -> Wrap_utils.id x
)))])

let mask_rows ?axis a =
   Py.Module.get_function_with_keywords __wrap_namespace "mask_rows"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("a", Some(a ))])

let masked_all ?dtype shape =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_all"
     [||]
     (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])

let masked_all_like arr =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_all_like"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr |> Obj.to_pyobject))])

let median ?axis ?out ?overwrite_input ?keepdims a =
   Py.Module.get_function_with_keywords __wrap_namespace "median"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("overwrite_input", Wrap_utils.Option.map overwrite_input Py.Bool.of_bool); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let normalize_axis_index ?msg_prefix ~axis ~ndim () =
   Py.Module.get_function_with_keywords __wrap_namespace "normalize_axis_index"
     [||]
     (Wrap_utils.keyword_args [("msg_prefix", Wrap_utils.Option.map msg_prefix Py.String.of_string); ("axis", Some(axis |> Py.Int.of_int)); ("ndim", Some(ndim |> Py.Int.of_int))])
     |> Py.Int.to_int
                  let normalize_axis_tuple ?argname ?allow_duplicate ~axis ~ndim () =
                     Py.Module.get_function_with_keywords __wrap_namespace "normalize_axis_tuple"
                       [||]
                       (Wrap_utils.keyword_args [("argname", Wrap_utils.Option.map argname Py.String.of_string); ("allow_duplicate", Wrap_utils.Option.map allow_duplicate Py.Bool.of_bool); ("axis", Some(axis |> (function
| `Iterable_of_int x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("ndim", Some(ndim |> Py.Int.of_int))])

let notmasked_contiguous ?axis a =
   Py.Module.get_function_with_keywords __wrap_namespace "notmasked_contiguous"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let notmasked_edges ?axis a =
   Py.Module.get_function_with_keywords __wrap_namespace "notmasked_edges"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let nxarray ?dtype ?copy ?order ?subok ?ndmin ~object_ () =
                     Py.Module.get_function_with_keywords __wrap_namespace "nxarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("order", Wrap_utils.Option.map order (function
| `K -> Py.String.of_string "K"
| `A -> Py.String.of_string "A"
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("ndmin", Wrap_utils.Option.map ndmin Py.Int.of_int); ("object", Some(object_ |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let ones ?params args =
   Py.Module.get_function_with_keywords __wrap_namespace "ones"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let polyfit ?rcond ?full ?w ?cov ~y ~deg x =
                     Py.Module.get_function_with_keywords __wrap_namespace "polyfit"
                       [||]
                       (Wrap_utils.keyword_args [("rcond", Wrap_utils.Option.map rcond Py.Float.of_float); ("full", Wrap_utils.Option.map full Py.Bool.of_bool); ("w", Wrap_utils.Option.map w Obj.to_pyobject); ("cov", Wrap_utils.Option.map cov (function
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("y", Some(y |> Obj.to_pyobject)); ("deg", Some(deg |> Py.Int.of_int)); ("x", Some(x |> Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 1))))
let row_stack ?params x args =
   Py.Module.get_function_with_keywords __wrap_namespace "row_stack"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("x", Some(x ))]) (match params with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let setdiff1d ?assume_unique ~ar1 ~ar2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "setdiff1d"
     [||]
     (Wrap_utils.keyword_args [("assume_unique", assume_unique); ("ar1", Some(ar1 )); ("ar2", Some(ar2 ))])

let setxor1d ?assume_unique ~ar1 ~ar2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "setxor1d"
     [||]
     (Wrap_utils.keyword_args [("assume_unique", assume_unique); ("ar1", Some(ar1 )); ("ar2", Some(ar2 ))])

let sort ?axis ?kind ?order ?endwith ?fill_value a =
   Py.Module.get_function_with_keywords __wrap_namespace "sort"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("kind", kind); ("order", order); ("endwith", endwith); ("fill_value", fill_value); ("a", Some(a ))])

let stack ?params x args =
   Py.Module.get_function_with_keywords __wrap_namespace "stack"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("x", Some(x ))]) (match params with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let union1d ~ar1 ~ar2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "union1d"
     [||]
     (Wrap_utils.keyword_args [("ar1", Some(ar1 )); ("ar2", Some(ar2 ))])

let unique ?return_index ?return_inverse ~ar1 () =
   Py.Module.get_function_with_keywords __wrap_namespace "unique"
     [||]
     (Wrap_utils.keyword_args [("return_index", return_index); ("return_inverse", return_inverse); ("ar1", Some(ar1 ))])

let vander ?n x =
   Py.Module.get_function_with_keywords __wrap_namespace "vander"
     [||]
     (Wrap_utils.keyword_args [("n", n); ("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let vstack ?params x args =
   Py.Module.get_function_with_keywords __wrap_namespace "vstack"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("x", Some(x ))]) (match params with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let zeros ?params args =
   Py.Module.get_function_with_keywords __wrap_namespace "zeros"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))

end
let abs ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "abs"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let absolute ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "absolute"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let add ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "add"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let all ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "all"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let allclose ?masked_equal ?rtol ?atol ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "allclose"
     [||]
     (Wrap_utils.keyword_args [("masked_equal", Wrap_utils.Option.map masked_equal Py.Bool.of_bool); ("rtol", Wrap_utils.Option.map rtol Py.Float.of_float); ("atol", Wrap_utils.Option.map atol Py.Float.of_float); ("b", Some(b )); ("a", Some(a ))])
     |> Py.Bool.to_bool
let allequal ?fill_value ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "allequal"
     [||]
     (Wrap_utils.keyword_args [("fill_value", Wrap_utils.Option.map fill_value Py.Bool.of_bool); ("b", Some(b )); ("a", Some(a ))])
     |> Py.Bool.to_bool
let alltrue ?axis ?dtype ~target () =
   Py.Module.get_function_with_keywords __wrap_namespace "alltrue"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("target", Some(target ))])

                  let amax ?axis ?out ?keepdims ?initial ?where a =
                     Py.Module.get_function_with_keywords __wrap_namespace "amax"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("initial", Wrap_utils.Option.map initial (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("where", where); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let amin ?axis ?out ?keepdims ?initial ?where a =
                     Py.Module.get_function_with_keywords __wrap_namespace "amin"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("initial", Wrap_utils.Option.map initial (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("where", where); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let angle ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "angle"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let anom ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "anom"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let anomalies ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "anomalies"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let any ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "any"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let append ?axis ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "append"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("b", Some(b |> Obj.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])

let apply_along_axis ?kwargs ~func1d ~axis ~arr args =
   Py.Module.get_function_with_keywords __wrap_namespace "apply_along_axis"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("func1d", Some(func1d )); ("axis", Some(axis |> Py.Int.of_int)); ("arr", Some(arr ))]) (match kwargs with None -> [] | Some x -> x))

let apply_over_axes ~func ~axes a =
   Py.Module.get_function_with_keywords __wrap_namespace "apply_over_axes"
     [||]
     (Wrap_utils.keyword_args [("func", Some(func )); ("axes", Some(axes |> Obj.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let arange ?params args =
   Py.Module.get_function_with_keywords __wrap_namespace "arange"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let arccos ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "arccos"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let arccosh ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "arccosh"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let arcsin ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "arcsin"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let arcsinh ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "arcsinh"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let arctan ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "arctan"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let arctan2 ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "arctan2"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let arctanh ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "arctanh"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let argmax ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "argmax"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let argmin ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "argmin"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

                  let argsort ?axis ?kind ?order ?endwith ?fill_value a =
                     Py.Module.get_function_with_keywords __wrap_namespace "argsort"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("kind", Wrap_utils.Option.map kind (function
| `Heapsort -> Py.String.of_string "heapsort"
| `Mergesort -> Py.String.of_string "mergesort"
| `Stable -> Py.String.of_string "stable"
| `Quicksort -> Py.String.of_string "quicksort"
)); ("order", Wrap_utils.Option.map order Obj.to_pyobject); ("endwith", Wrap_utils.Option.map endwith Py.Bool.of_bool); ("fill_value", fill_value); ("a", Some(a ))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let around ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "around"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))

                  let array ?dtype ?copy ?order ?mask ?fill_value ?keep_mask ?hard_mask ?shrink ?subok ?ndmin ~data () =
                     Py.Module.get_function_with_keywords __wrap_namespace "array"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("order", Wrap_utils.Option.map order (function
| `A -> Py.String.of_string "A"
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("mask", mask); ("fill_value", Wrap_utils.Option.map fill_value (function
| `Bool x -> Py.Bool.of_bool x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
)); ("keep_mask", Wrap_utils.Option.map keep_mask Py.Bool.of_bool); ("hard_mask", Wrap_utils.Option.map hard_mask Py.Bool.of_bool); ("shrink", Wrap_utils.Option.map shrink Py.Bool.of_bool); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("ndmin", Wrap_utils.Option.map ndmin Py.Int.of_int); ("data", Some(data |> Obj.to_pyobject))])

let asanyarray ?dtype a =
   Py.Module.get_function_with_keywords __wrap_namespace "asanyarray"
     [||]
     (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("a", Some(a |> Obj.to_pyobject))])

                  let asarray ?dtype ?order a =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("a", Some(a |> Obj.to_pyobject))])

let atleast_1d ?params args =
   Py.Module.get_function_with_keywords __wrap_namespace "atleast_1d"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let atleast_2d ?params args =
   Py.Module.get_function_with_keywords __wrap_namespace "atleast_2d"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)

let atleast_3d ?params args =
   Py.Module.get_function_with_keywords __wrap_namespace "atleast_3d"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)

let average ?axis ?weights ?returned a =
   Py.Module.get_function_with_keywords __wrap_namespace "average"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("weights", Wrap_utils.Option.map weights Obj.to_pyobject); ("returned", Wrap_utils.Option.map returned Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])

let bitwise_and ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "bitwise_and"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let bitwise_or ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "bitwise_or"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let bitwise_xor ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "bitwise_xor"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let ceil ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "ceil"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let choose ?out ?mode ~indices ~choices () =
                     Py.Module.get_function_with_keywords __wrap_namespace "choose"
                       [||]
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Obj.to_pyobject); ("mode", Wrap_utils.Option.map mode (function
| `Raise -> Py.String.of_string "raise"
| `Wrap -> Py.String.of_string "wrap"
| `Clip -> Py.String.of_string "clip"
)); ("indices", Some(indices )); ("choices", Some(choices ))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let clip ?out ?kwargs ~a_min ~a_max a =
                     Py.Module.get_function_with_keywords __wrap_namespace "clip"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Obj.to_pyobject); ("a_min", Some(a_min |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
| `None -> Py.none
))); ("a_max", Some(a_max |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
| `None -> Py.none
))); ("a", Some(a |> Obj.to_pyobject))]) (match kwargs with None -> [] | Some x -> x))
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let clump_masked a =
   Py.Module.get_function_with_keywords __wrap_namespace "clump_masked"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Obj.to_pyobject))])

let clump_unmasked a =
   Py.Module.get_function_with_keywords __wrap_namespace "clump_unmasked"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Obj.to_pyobject))])

let column_stack ?params x args =
   Py.Module.get_function_with_keywords __wrap_namespace "column_stack"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("x", Some(x ))]) (match params with None -> [] | Some x -> x))

let common_fill_value ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "common_fill_value"
     [||]
     (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))])
     |> (fun py -> if Py.is_none py then None else Some (Wrap_utils.id py))
let compress ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "compress"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let compress_cols a =
   Py.Module.get_function_with_keywords __wrap_namespace "compress_cols"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a ))])

                  let compress_nd ?axis x =
                     Py.Module.get_function_with_keywords __wrap_namespace "compress_nd"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("x", Some(x |> (function
| `Ndarray x -> Obj.to_pyobject x
| `MaskedArray x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let compress_rowcols ?axis x =
                     Py.Module.get_function_with_keywords __wrap_namespace "compress_rowcols"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("x", Some(x |> (function
| `Ndarray x -> Obj.to_pyobject x
| `MaskedArray x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let compress_rows a =
   Py.Module.get_function_with_keywords __wrap_namespace "compress_rows"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a ))])

let compressed x =
   Py.Module.get_function_with_keywords __wrap_namespace "compressed"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let concatenate ?axis ~arrays () =
   Py.Module.get_function_with_keywords __wrap_namespace "concatenate"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("arrays", Some(arrays ))])

let conjugate ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "conjugate"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let convolve ?mode ?propagate_mask ~v a =
                     Py.Module.get_function_with_keywords __wrap_namespace "convolve"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Valid -> Py.String.of_string "valid"
| `Same -> Py.String.of_string "same"
| `Full -> Py.String.of_string "full"
)); ("propagate_mask", Wrap_utils.Option.map propagate_mask Py.Bool.of_bool); ("v", Some(v )); ("a", Some(a ))])

let copy ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "copy"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let corrcoef ?y ?rowvar ?bias ?allow_masked ?ddof x =
   Py.Module.get_function_with_keywords __wrap_namespace "corrcoef"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Obj.to_pyobject); ("rowvar", Wrap_utils.Option.map rowvar Py.Bool.of_bool); ("bias", bias); ("allow_masked", Wrap_utils.Option.map allow_masked Py.Bool.of_bool); ("ddof", ddof); ("x", Some(x |> Obj.to_pyobject))])

                  let correlate ?mode ?propagate_mask ~v a =
                     Py.Module.get_function_with_keywords __wrap_namespace "correlate"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Valid -> Py.String.of_string "valid"
| `Same -> Py.String.of_string "same"
| `Full -> Py.String.of_string "full"
)); ("propagate_mask", Wrap_utils.Option.map propagate_mask Py.Bool.of_bool); ("v", Some(v )); ("a", Some(a ))])

let cos ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "cos"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let cosh ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "cosh"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let count ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "count"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let count_masked ?axis ~arr () =
   Py.Module.get_function_with_keywords __wrap_namespace "count_masked"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("arr", Some(arr |> Obj.to_pyobject))])

let cov ?y ?rowvar ?bias ?allow_masked ?ddof x =
   Py.Module.get_function_with_keywords __wrap_namespace "cov"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Obj.to_pyobject); ("rowvar", Wrap_utils.Option.map rowvar Py.Bool.of_bool); ("bias", Wrap_utils.Option.map bias Py.Bool.of_bool); ("allow_masked", Wrap_utils.Option.map allow_masked Py.Bool.of_bool); ("ddof", Wrap_utils.Option.map ddof Py.Int.of_int); ("x", Some(x |> Obj.to_pyobject))])

let cumprod ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "cumprod"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let cumsum ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "cumsum"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

                  let default_fill_value obj =
                     Py.Module.get_function_with_keywords __wrap_namespace "default_fill_value"
                       [||]
                       (Wrap_utils.keyword_args [("obj", Some(obj |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `Dtype x -> Dtype.to_pyobject x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
)))])

let diag ?k ~v () =
   Py.Module.get_function_with_keywords __wrap_namespace "diag"
     [||]
     (Wrap_utils.keyword_args [("k", k); ("v", Some(v ))])

let diagflat ?params x args =
   Py.Module.get_function_with_keywords __wrap_namespace "diagflat"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("x", Some(x ))]) (match params with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let diagonal ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "diagonal"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let diff ?n ?axis ?prepend ?append a =
   Py.Module.get_function_with_keywords __wrap_namespace "diff"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("prepend", prepend); ("append", append); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let divide ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "divide"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let dot ?strict ?out ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "dot"
     [||]
     (Wrap_utils.keyword_args [("strict", Wrap_utils.Option.map strict Py.Bool.of_bool); ("out", out); ("b", Some(b )); ("a", Some(a ))])

let dstack ?params x args =
   Py.Module.get_function_with_keywords __wrap_namespace "dstack"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("x", Some(x ))]) (match params with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let ediff1d ?to_end ?to_begin ~arr () =
   Py.Module.get_function_with_keywords __wrap_namespace "ediff1d"
     [||]
     (Wrap_utils.keyword_args [("to_end", to_end); ("to_begin", to_begin); ("arr", Some(arr ))])

let empty ?params args =
   Py.Module.get_function_with_keywords __wrap_namespace "empty"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let empty_like ?params args =
   Py.Module.get_function_with_keywords __wrap_namespace "empty_like"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let equal ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "equal"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let exp ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "exp"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let expand_dims ~axis a =
   Py.Module.get_function_with_keywords __wrap_namespace "expand_dims"
     [||]
     (Wrap_utils.keyword_args [("axis", Some(axis |> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml))); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let fabs ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "fabs"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let filled ?fill_value a =
                     Py.Module.get_function_with_keywords __wrap_namespace "filled"
                       [||]
                       (Wrap_utils.keyword_args [("fill_value", Wrap_utils.Option.map fill_value Obj.to_pyobject); ("a", Some(a |> (function
| `Ndarray x -> Obj.to_pyobject x
| `MaskedArray x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let fix_invalid ?mask ?copy ?fill_value a =
                     Py.Module.get_function_with_keywords __wrap_namespace "fix_invalid"
                       [||]
                       (Wrap_utils.keyword_args [("mask", mask); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("fill_value", Wrap_utils.Option.map fill_value (function
| `Bool x -> Py.Bool.of_bool x
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
)); ("a", Some(a |> Obj.to_pyobject))])

let flatnotmasked_contiguous a =
   Py.Module.get_function_with_keywords __wrap_namespace "flatnotmasked_contiguous"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let flatnotmasked_edges a =
   Py.Module.get_function_with_keywords __wrap_namespace "flatnotmasked_edges"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> if Py.is_none py then None else Some ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) py))
let flatten_mask mask =
   Py.Module.get_function_with_keywords __wrap_namespace "flatten_mask"
     [||]
     (Wrap_utils.keyword_args [("mask", Some(mask |> Obj.to_pyobject))])

let flatten_structured_array a =
   Py.Module.get_function_with_keywords __wrap_namespace "flatten_structured_array"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a ))])

let floor ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "floor"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let floor_divide ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "floor_divide"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let fmod ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "fmod"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let frombuffer ?params args =
   Py.Module.get_function_with_keywords __wrap_namespace "frombuffer"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)

let fromflex fxarray =
   Py.Module.get_function_with_keywords __wrap_namespace "fromflex"
     [||]
     (Wrap_utils.keyword_args [("fxarray", Some(fxarray |> Obj.to_pyobject))])

let fromfunction ?params args =
   Py.Module.get_function_with_keywords __wrap_namespace "fromfunction"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)

let getdata ?subok a =
   Py.Module.get_function_with_keywords __wrap_namespace "getdata"
     [||]
     (Wrap_utils.keyword_args [("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])

let getmask a =
   Py.Module.get_function_with_keywords __wrap_namespace "getmask"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Obj.to_pyobject))])

let getmaskarray arr =
   Py.Module.get_function_with_keywords __wrap_namespace "getmaskarray"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr |> Obj.to_pyobject))])

let greater ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "greater"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let greater_equal ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "greater_equal"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))

let harden_mask ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "harden_mask"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let hsplit ?params x args =
   Py.Module.get_function_with_keywords __wrap_namespace "hsplit"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("x", Some(x ))]) (match params with None -> [] | Some x -> x))

let hstack ?params x args =
   Py.Module.get_function_with_keywords __wrap_namespace "hstack"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("x", Some(x ))]) (match params with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hypot ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "hypot"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let identity ?params args =
   Py.Module.get_function_with_keywords __wrap_namespace "identity"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let ids ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "ids"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let in1d ?assume_unique ?invert ~ar1 ~ar2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "in1d"
     [||]
     (Wrap_utils.keyword_args [("assume_unique", assume_unique); ("invert", invert); ("ar1", Some(ar1 )); ("ar2", Some(ar2 ))])

let indices ?dtype ?sparse ~dimensions () =
   Py.Module.get_function_with_keywords __wrap_namespace "indices"
     [||]
     (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("sparse", Wrap_utils.Option.map sparse Py.Bool.of_bool); ("dimensions", Some(dimensions |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])

let inner ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "inner"
     [||]
     (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let innerproduct ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "innerproduct"
     [||]
     (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let intersect1d ?assume_unique ~ar1 ~ar2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "intersect1d"
     [||]
     (Wrap_utils.keyword_args [("assume_unique", assume_unique); ("ar1", Some(ar1 )); ("ar2", Some(ar2 ))])

let isMA x =
   Py.Module.get_function_with_keywords __wrap_namespace "isMA"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])
     |> Py.Bool.to_bool
let isMaskedArray x =
   Py.Module.get_function_with_keywords __wrap_namespace "isMaskedArray"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])
     |> Py.Bool.to_bool
let is_mask m =
   Py.Module.get_function_with_keywords __wrap_namespace "is_mask"
     [||]
     (Wrap_utils.keyword_args [("m", Some(m |> Obj.to_pyobject))])
     |> Py.Bool.to_bool
let is_masked x =
   Py.Module.get_function_with_keywords __wrap_namespace "is_masked"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Obj.to_pyobject))])
     |> Py.Bool.to_bool
let isarray x =
   Py.Module.get_function_with_keywords __wrap_namespace "isarray"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])
     |> Py.Bool.to_bool
let isin ?assume_unique ?invert ~element ~test_elements () =
   Py.Module.get_function_with_keywords __wrap_namespace "isin"
     [||]
     (Wrap_utils.keyword_args [("assume_unique", assume_unique); ("invert", invert); ("element", Some(element )); ("test_elements", Some(test_elements ))])

let left_shift ~n a =
   Py.Module.get_function_with_keywords __wrap_namespace "left_shift"
     [||]
     (Wrap_utils.keyword_args [("n", Some(n )); ("a", Some(a ))])

let less ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "less"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let less_equal ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "less_equal"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let log ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "log"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let log10 ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "log10"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let log2 ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "log2"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let logical_and ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "logical_and"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))

let logical_not ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "logical_not"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))

let logical_or ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "logical_or"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))

let logical_xor ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "logical_xor"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))

let make_mask ?copy ?shrink ?dtype ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "make_mask"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("shrink", Wrap_utils.Option.map shrink Py.Bool.of_bool); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("m", Some(m |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let make_mask_descr ndtype =
   Py.Module.get_function_with_keywords __wrap_namespace "make_mask_descr"
     [||]
     (Wrap_utils.keyword_args [("ndtype", Some(ndtype |> Dtype.to_pyobject))])
     |> Dtype.of_pyobject
let make_mask_none ?dtype ~newshape () =
   Py.Module.get_function_with_keywords __wrap_namespace "make_mask_none"
     [||]
     (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("newshape", Some(newshape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let mask_cols ?axis a =
   Py.Module.get_function_with_keywords __wrap_namespace "mask_cols"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("a", Some(a ))])

let mask_or ?copy ?shrink ~m1 ~m2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "mask_or"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("shrink", Wrap_utils.Option.map shrink Py.Bool.of_bool); ("m1", Some(m1 )); ("m2", Some(m2 ))])

                  let mask_rowcols ?axis a =
                     Py.Module.get_function_with_keywords __wrap_namespace "mask_rowcols"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("a", Some(a |> (function
| `Ndarray x -> Obj.to_pyobject x
| `MaskedArray x -> Wrap_utils.id x
)))])

let mask_rows ?axis a =
   Py.Module.get_function_with_keywords __wrap_namespace "mask_rows"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("a", Some(a ))])

let masked_all ?dtype shape =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_all"
     [||]
     (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])

let masked_all_like arr =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_all_like"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr |> Obj.to_pyobject))])

let masked_equal ?copy ~value x =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_equal"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("value", Some(value )); ("x", Some(x ))])

let masked_greater ?copy ~value x =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_greater"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("value", Some(value )); ("x", Some(x ))])

let masked_greater_equal ?copy ~value x =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_greater_equal"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("value", Some(value )); ("x", Some(x ))])

let masked_inside ?copy ~v1 ~v2 x =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_inside"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("v1", Some(v1 )); ("v2", Some(v2 )); ("x", Some(x ))])

let masked_invalid ?copy a =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_invalid"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("a", Some(a ))])

let masked_less ?copy ~value x =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_less"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("value", Some(value )); ("x", Some(x ))])

let masked_less_equal ?copy ~value x =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_less_equal"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("value", Some(value )); ("x", Some(x ))])

let masked_not_equal ?copy ~value x =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_not_equal"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("value", Some(value )); ("x", Some(x ))])

let masked_object ?copy ?shrink ~value x =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_object"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("shrink", Wrap_utils.Option.map shrink Py.Bool.of_bool); ("value", Some(value )); ("x", Some(x |> Obj.to_pyobject))])

let masked_outside ?copy ~v1 ~v2 x =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_outside"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("v1", Some(v1 )); ("v2", Some(v2 )); ("x", Some(x ))])

let masked_values ?rtol ?atol ?copy ?shrink ~value x =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_values"
     [||]
     (Wrap_utils.keyword_args [("rtol", rtol); ("atol", atol); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("shrink", Wrap_utils.Option.map shrink Py.Bool.of_bool); ("value", Some(value |> Py.Float.of_float)); ("x", Some(x |> Obj.to_pyobject))])

let masked_where ?copy ~condition a =
   Py.Module.get_function_with_keywords __wrap_namespace "masked_where"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("condition", Some(condition |> Obj.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])

let max ?axis ?out ?fill_value ?keepdims ~obj () =
   Py.Module.get_function_with_keywords __wrap_namespace "max"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("fill_value", fill_value); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("obj", Some(obj ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let maximum ?b a =
   Py.Module.get_function_with_keywords __wrap_namespace "maximum"
     [||]
     (Wrap_utils.keyword_args [("b", b); ("a", Some(a ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let maximum_fill_value obj =
                     Py.Module.get_function_with_keywords __wrap_namespace "maximum_fill_value"
                       [||]
                       (Wrap_utils.keyword_args [("obj", Some(obj |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `Dtype x -> Dtype.to_pyobject x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
)))])

let mean ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "mean"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let median ?axis ?out ?overwrite_input ?keepdims a =
   Py.Module.get_function_with_keywords __wrap_namespace "median"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("overwrite_input", Wrap_utils.Option.map overwrite_input Py.Bool.of_bool); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let min ?axis ?out ?fill_value ?keepdims ~obj () =
   Py.Module.get_function_with_keywords __wrap_namespace "min"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("fill_value", fill_value); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("obj", Some(obj ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let minimum ?b a =
   Py.Module.get_function_with_keywords __wrap_namespace "minimum"
     [||]
     (Wrap_utils.keyword_args [("b", b); ("a", Some(a ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let minimum_fill_value obj =
                     Py.Module.get_function_with_keywords __wrap_namespace "minimum_fill_value"
                       [||]
                       (Wrap_utils.keyword_args [("obj", Some(obj |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `Dtype x -> Dtype.to_pyobject x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
)))])

let mod_ ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "mod"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let multiply ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "multiply"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let ndim obj =
   Py.Module.get_function_with_keywords __wrap_namespace "ndim"
     [||]
     (Wrap_utils.keyword_args [("obj", Some(obj ))])
     |> Py.Int.to_int
let negative ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "negative"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let nonzero ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "nonzero"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let not_equal ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "not_equal"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let notmasked_contiguous ?axis a =
   Py.Module.get_function_with_keywords __wrap_namespace "notmasked_contiguous"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let notmasked_edges ?axis a =
   Py.Module.get_function_with_keywords __wrap_namespace "notmasked_edges"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let ones ?params args =
   Py.Module.get_function_with_keywords __wrap_namespace "ones"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let outer ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "outer"
     [||]
     (Wrap_utils.keyword_args [("b", Some(b |> Obj.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let outerproduct ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "outerproduct"
     [||]
     (Wrap_utils.keyword_args [("b", Some(b |> Obj.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let polyfit ?rcond ?full ?w ?cov ~y ~deg x =
                     Py.Module.get_function_with_keywords __wrap_namespace "polyfit"
                       [||]
                       (Wrap_utils.keyword_args [("rcond", Wrap_utils.Option.map rcond Py.Float.of_float); ("full", Wrap_utils.Option.map full Py.Bool.of_bool); ("w", Wrap_utils.Option.map w Obj.to_pyobject); ("cov", Wrap_utils.Option.map cov (function
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("y", Some(y |> Obj.to_pyobject)); ("deg", Some(deg |> Py.Int.of_int)); ("x", Some(x |> Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 1))))
let power ?third ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "power"
     [||]
     (Wrap_utils.keyword_args [("third", third); ("b", Some(b )); ("a", Some(a ))])

let prod ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "prod"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let product ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "product"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let ptp ?axis ?out ?fill_value ?keepdims ~obj () =
   Py.Module.get_function_with_keywords __wrap_namespace "ptp"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("fill_value", fill_value); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("obj", Some(obj ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let put ?mode ~indices ~values a =
   Py.Module.get_function_with_keywords __wrap_namespace "put"
     [||]
     (Wrap_utils.keyword_args [("mode", mode); ("indices", Some(indices )); ("values", Some(values )); ("a", Some(a ))])

let putmask ~mask ~values a =
   Py.Module.get_function_with_keywords __wrap_namespace "putmask"
     [||]
     (Wrap_utils.keyword_args [("mask", Some(mask )); ("values", Some(values )); ("a", Some(a ))])

let ravel ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "ravel"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let remainder ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "remainder"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let repeat ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "repeat"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let reshape ?order ~new_shape a =
   Py.Module.get_function_with_keywords __wrap_namespace "reshape"
     [||]
     (Wrap_utils.keyword_args [("order", order); ("new_shape", Some(new_shape )); ("a", Some(a ))])

let resize ~new_shape x =
   Py.Module.get_function_with_keywords __wrap_namespace "resize"
     [||]
     (Wrap_utils.keyword_args [("new_shape", Some(new_shape )); ("x", Some(x ))])

let right_shift ~n a =
   Py.Module.get_function_with_keywords __wrap_namespace "right_shift"
     [||]
     (Wrap_utils.keyword_args [("n", Some(n )); ("a", Some(a ))])

let round ?decimals ?out a =
   Py.Module.get_function_with_keywords __wrap_namespace "round"
     [||]
     (Wrap_utils.keyword_args [("decimals", Wrap_utils.Option.map decimals Py.Int.of_int); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("a", Some(a ))])

let row_stack ?params x args =
   Py.Module.get_function_with_keywords __wrap_namespace "row_stack"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("x", Some(x ))]) (match params with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let set_fill_value ~fill_value a =
   Py.Module.get_function_with_keywords __wrap_namespace "set_fill_value"
     [||]
     (Wrap_utils.keyword_args [("fill_value", Some(fill_value |> Dtype.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])

let setdiff1d ?assume_unique ~ar1 ~ar2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "setdiff1d"
     [||]
     (Wrap_utils.keyword_args [("assume_unique", assume_unique); ("ar1", Some(ar1 )); ("ar2", Some(ar2 ))])

let setxor1d ?assume_unique ~ar1 ~ar2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "setxor1d"
     [||]
     (Wrap_utils.keyword_args [("assume_unique", assume_unique); ("ar1", Some(ar1 )); ("ar2", Some(ar2 ))])

        let shape obj =
           Py.Module.get_function_with_keywords __wrap_namespace "shape"
             [||]
             (Wrap_utils.keyword_args [("obj", Some(obj ))])
             |> (fun py -> let len = Py.Sequence.length py in Array.init len
(fun i -> Py.Int.to_int (Py.Sequence.get_item py i)))
let sin ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "sin"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let sinh ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "sinh"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let size ?axis ~obj () =
   Py.Module.get_function_with_keywords __wrap_namespace "size"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("obj", Some(obj ))])
     |> Py.Int.to_int
let soften_mask ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "soften_mask"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let sometrue ?axis ?dtype ~target () =
   Py.Module.get_function_with_keywords __wrap_namespace "sometrue"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("target", Some(target ))])

let sort ?axis ?kind ?order ?endwith ?fill_value a =
   Py.Module.get_function_with_keywords __wrap_namespace "sort"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("kind", kind); ("order", order); ("endwith", endwith); ("fill_value", fill_value); ("a", Some(a ))])

let sqrt ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "sqrt"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let squeeze ?axis a =
   Py.Module.get_function_with_keywords __wrap_namespace "squeeze"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let stack ?params x args =
   Py.Module.get_function_with_keywords __wrap_namespace "stack"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("x", Some(x ))]) (match params with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let std ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "std"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let subtract ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "subtract"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let sum ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "sum"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let swapaxes ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "swapaxes"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let take ?axis ?out ?mode ~indices a =
   Py.Module.get_function_with_keywords __wrap_namespace "take"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("mode", mode); ("indices", Some(indices )); ("a", Some(a ))])

let tan ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "tan"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let tanh ?kwargs a args =
   Py.Module.get_function_with_keywords __wrap_namespace "tanh"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let trace ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "trace"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a ))]) (match params with None -> [] | Some x -> x))

let transpose ?axes a =
   Py.Module.get_function_with_keywords __wrap_namespace "transpose"
     [||]
     (Wrap_utils.keyword_args [("axes", axes); ("a", Some(a ))])

let true_divide ?kwargs ~b a args =
   Py.Module.get_function_with_keywords __wrap_namespace "true_divide"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))]) (match kwargs with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let union1d ~ar1 ~ar2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "union1d"
     [||]
     (Wrap_utils.keyword_args [("ar1", Some(ar1 )); ("ar2", Some(ar2 ))])

let unique ?return_index ?return_inverse ~ar1 () =
   Py.Module.get_function_with_keywords __wrap_namespace "unique"
     [||]
     (Wrap_utils.keyword_args [("return_index", return_index); ("return_inverse", return_inverse); ("ar1", Some(ar1 ))])

let vander ?n x =
   Py.Module.get_function_with_keywords __wrap_namespace "vander"
     [||]
     (Wrap_utils.keyword_args [("n", n); ("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let var ?params a args =
   Py.Module.get_function_with_keywords __wrap_namespace "var"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("a", Some(a |> Obj.to_pyobject))]) (match params with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let vstack ?params x args =
   Py.Module.get_function_with_keywords __wrap_namespace "vstack"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("x", Some(x ))]) (match params with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let where ?x ?y ~condition () =
                     Py.Module.get_function_with_keywords __wrap_namespace "where"
                       [||]
                       (Wrap_utils.keyword_args [("x", x); ("y", y); ("condition", Some(condition |> (function
| `Bool x -> Py.Bool.of_bool x
| `Ndarray x -> Obj.to_pyobject x
)))])

let zeros ?params args =
   Py.Module.get_function_with_keywords __wrap_namespace "zeros"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match params with None -> [] | Some x -> x)
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))

end
module Polynomial = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "numpy.polynomial"

let get_py name = Py.Module.get __wrap_namespace name
module Chebyshev = struct
type tag = [`Chebyshev]
type t = [`ABCPolyBase | `Chebyshev | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_abc_poly x = (x :> [`ABCPolyBase] Obj.t)
let create ?domain ?window ~coef () =
   Py.Module.get_function_with_keywords __wrap_namespace "Chebyshev"
     [||]
     (Wrap_utils.keyword_args [("domain", domain); ("window", window); ("coef", Some(coef |> Obj.to_pyobject))])
     |> of_pyobject
let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let basis ?domain ?window ~deg self =
   Py.Module.get_function_with_keywords (to_pyobject self) "basis"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("window", Wrap_utils.Option.map window Obj.to_pyobject); ("deg", Some(deg |> Py.Int.of_int))])

let cast ?domain ?window ~series self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cast"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("window", Wrap_utils.Option.map window Obj.to_pyobject); ("series", Some(series ))])

let convert ?domain ?kind ?window self =
   Py.Module.get_function_with_keywords (to_pyobject self) "convert"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("kind", kind); ("window", Wrap_utils.Option.map window Obj.to_pyobject)])

let copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "copy"
     [||]
     []

let cutdeg ~deg self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cutdeg"
     [||]
     (Wrap_utils.keyword_args [("deg", Some(deg ))])

let degree self =
   Py.Module.get_function_with_keywords (to_pyobject self) "degree"
     [||]
     []
     |> Py.Int.to_int
let deriv ?m self =
   Py.Module.get_function_with_keywords (to_pyobject self) "deriv"
     [||]
     (Wrap_utils.keyword_args [("m", m)])

                  let fit ?domain ?rcond ?full ?w ?window ~y ~deg x self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fit"
                       [||]
                       (Wrap_utils.keyword_args [("domain", domain); ("rcond", Wrap_utils.Option.map rcond Py.Float.of_float); ("full", Wrap_utils.Option.map full Py.Bool.of_bool); ("w", Wrap_utils.Option.map w Obj.to_pyobject); ("window", window); ("y", Some(y |> Obj.to_pyobject)); ("deg", Some(deg |> (function
| `T1_D_array_like x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("x", Some(x |> Obj.to_pyobject))])

                  let fromroots ?domain ?window ~roots self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fromroots"
                       [||]
                       (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain (function
| `Ndarray x -> Obj.to_pyobject x
| `T_ x -> Wrap_utils.id x
| `None -> Py.none
)); ("window", Wrap_utils.Option.map window Obj.to_pyobject); ("roots", Some(roots |> Obj.to_pyobject))])

let has_samecoef ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_samecoef"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let has_samedomain ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_samedomain"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let has_sametype ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_sametype"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let has_samewindow ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_samewindow"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let identity ?domain ?window self =
   Py.Module.get_function_with_keywords (to_pyobject self) "identity"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("window", Wrap_utils.Option.map window Obj.to_pyobject)])

let integ ?m ?k ?lbnd self =
   Py.Module.get_function_with_keywords (to_pyobject self) "integ"
     [||]
     (Wrap_utils.keyword_args [("m", m); ("k", Wrap_utils.Option.map k Obj.to_pyobject); ("lbnd", lbnd)])

let interpolate ?domain ?args ~func ~deg self =
   Py.Module.get_function_with_keywords (to_pyobject self) "interpolate"
     [||]
     (Wrap_utils.keyword_args [("domain", domain); ("args", args); ("func", Some(func )); ("deg", Some(deg |> Py.Int.of_int))])

let linspace ?n ?domain self =
   Py.Module.get_function_with_keywords (to_pyobject self) "linspace"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("domain", Wrap_utils.Option.map domain Obj.to_pyobject)])

let mapparms self =
   Py.Module.get_function_with_keywords (to_pyobject self) "mapparms"
     [||]
     []

let roots self =
   Py.Module.get_function_with_keywords (to_pyobject self) "roots"
     [||]
     []
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let trim ?tol self =
   Py.Module.get_function_with_keywords (to_pyobject self) "trim"
     [||]
     (Wrap_utils.keyword_args [("tol", tol)])

let truncate ~size self =
   Py.Module.get_function_with_keywords (to_pyobject self) "truncate"
     [||]
     (Wrap_utils.keyword_args [("size", Some(size ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Hermite = struct
type tag = [`Hermite]
type t = [`ABCPolyBase | `Hermite | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_abc_poly x = (x :> [`ABCPolyBase] Obj.t)
let create ?domain ?window ~coef () =
   Py.Module.get_function_with_keywords __wrap_namespace "Hermite"
     [||]
     (Wrap_utils.keyword_args [("domain", domain); ("window", window); ("coef", Some(coef |> Obj.to_pyobject))])
     |> of_pyobject
let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let basis ?domain ?window ~deg self =
   Py.Module.get_function_with_keywords (to_pyobject self) "basis"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("window", Wrap_utils.Option.map window Obj.to_pyobject); ("deg", Some(deg |> Py.Int.of_int))])

let cast ?domain ?window ~series self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cast"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("window", Wrap_utils.Option.map window Obj.to_pyobject); ("series", Some(series ))])

let convert ?domain ?kind ?window self =
   Py.Module.get_function_with_keywords (to_pyobject self) "convert"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("kind", kind); ("window", Wrap_utils.Option.map window Obj.to_pyobject)])

let copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "copy"
     [||]
     []

let cutdeg ~deg self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cutdeg"
     [||]
     (Wrap_utils.keyword_args [("deg", Some(deg ))])

let degree self =
   Py.Module.get_function_with_keywords (to_pyobject self) "degree"
     [||]
     []
     |> Py.Int.to_int
let deriv ?m self =
   Py.Module.get_function_with_keywords (to_pyobject self) "deriv"
     [||]
     (Wrap_utils.keyword_args [("m", m)])

                  let fit ?domain ?rcond ?full ?w ?window ~y ~deg x self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fit"
                       [||]
                       (Wrap_utils.keyword_args [("domain", domain); ("rcond", Wrap_utils.Option.map rcond Py.Float.of_float); ("full", Wrap_utils.Option.map full Py.Bool.of_bool); ("w", Wrap_utils.Option.map w Obj.to_pyobject); ("window", window); ("y", Some(y |> Obj.to_pyobject)); ("deg", Some(deg |> (function
| `T1_D_array_like x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("x", Some(x |> Obj.to_pyobject))])

                  let fromroots ?domain ?window ~roots self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fromroots"
                       [||]
                       (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain (function
| `Ndarray x -> Obj.to_pyobject x
| `T_ x -> Wrap_utils.id x
| `None -> Py.none
)); ("window", Wrap_utils.Option.map window Obj.to_pyobject); ("roots", Some(roots |> Obj.to_pyobject))])

let has_samecoef ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_samecoef"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let has_samedomain ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_samedomain"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let has_sametype ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_sametype"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let has_samewindow ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_samewindow"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let identity ?domain ?window self =
   Py.Module.get_function_with_keywords (to_pyobject self) "identity"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("window", Wrap_utils.Option.map window Obj.to_pyobject)])

let integ ?m ?k ?lbnd self =
   Py.Module.get_function_with_keywords (to_pyobject self) "integ"
     [||]
     (Wrap_utils.keyword_args [("m", m); ("k", Wrap_utils.Option.map k Obj.to_pyobject); ("lbnd", lbnd)])

let linspace ?n ?domain self =
   Py.Module.get_function_with_keywords (to_pyobject self) "linspace"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("domain", Wrap_utils.Option.map domain Obj.to_pyobject)])

let mapparms self =
   Py.Module.get_function_with_keywords (to_pyobject self) "mapparms"
     [||]
     []

let roots self =
   Py.Module.get_function_with_keywords (to_pyobject self) "roots"
     [||]
     []
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let trim ?tol self =
   Py.Module.get_function_with_keywords (to_pyobject self) "trim"
     [||]
     (Wrap_utils.keyword_args [("tol", tol)])

let truncate ~size self =
   Py.Module.get_function_with_keywords (to_pyobject self) "truncate"
     [||]
     (Wrap_utils.keyword_args [("size", Some(size ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module HermiteE = struct
type tag = [`HermiteE]
type t = [`ABCPolyBase | `HermiteE | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_abc_poly x = (x :> [`ABCPolyBase] Obj.t)
let create ?domain ?window ~coef () =
   Py.Module.get_function_with_keywords __wrap_namespace "HermiteE"
     [||]
     (Wrap_utils.keyword_args [("domain", domain); ("window", window); ("coef", Some(coef |> Obj.to_pyobject))])
     |> of_pyobject
let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let basis ?domain ?window ~deg self =
   Py.Module.get_function_with_keywords (to_pyobject self) "basis"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("window", Wrap_utils.Option.map window Obj.to_pyobject); ("deg", Some(deg |> Py.Int.of_int))])

let cast ?domain ?window ~series self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cast"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("window", Wrap_utils.Option.map window Obj.to_pyobject); ("series", Some(series ))])

let convert ?domain ?kind ?window self =
   Py.Module.get_function_with_keywords (to_pyobject self) "convert"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("kind", kind); ("window", Wrap_utils.Option.map window Obj.to_pyobject)])

let copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "copy"
     [||]
     []

let cutdeg ~deg self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cutdeg"
     [||]
     (Wrap_utils.keyword_args [("deg", Some(deg ))])

let degree self =
   Py.Module.get_function_with_keywords (to_pyobject self) "degree"
     [||]
     []
     |> Py.Int.to_int
let deriv ?m self =
   Py.Module.get_function_with_keywords (to_pyobject self) "deriv"
     [||]
     (Wrap_utils.keyword_args [("m", m)])

                  let fit ?domain ?rcond ?full ?w ?window ~y ~deg x self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fit"
                       [||]
                       (Wrap_utils.keyword_args [("domain", domain); ("rcond", Wrap_utils.Option.map rcond Py.Float.of_float); ("full", Wrap_utils.Option.map full Py.Bool.of_bool); ("w", Wrap_utils.Option.map w Obj.to_pyobject); ("window", window); ("y", Some(y |> Obj.to_pyobject)); ("deg", Some(deg |> (function
| `T1_D_array_like x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("x", Some(x |> Obj.to_pyobject))])

                  let fromroots ?domain ?window ~roots self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fromroots"
                       [||]
                       (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain (function
| `Ndarray x -> Obj.to_pyobject x
| `T_ x -> Wrap_utils.id x
| `None -> Py.none
)); ("window", Wrap_utils.Option.map window Obj.to_pyobject); ("roots", Some(roots |> Obj.to_pyobject))])

let has_samecoef ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_samecoef"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let has_samedomain ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_samedomain"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let has_sametype ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_sametype"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let has_samewindow ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_samewindow"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let identity ?domain ?window self =
   Py.Module.get_function_with_keywords (to_pyobject self) "identity"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("window", Wrap_utils.Option.map window Obj.to_pyobject)])

let integ ?m ?k ?lbnd self =
   Py.Module.get_function_with_keywords (to_pyobject self) "integ"
     [||]
     (Wrap_utils.keyword_args [("m", m); ("k", Wrap_utils.Option.map k Obj.to_pyobject); ("lbnd", lbnd)])

let linspace ?n ?domain self =
   Py.Module.get_function_with_keywords (to_pyobject self) "linspace"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("domain", Wrap_utils.Option.map domain Obj.to_pyobject)])

let mapparms self =
   Py.Module.get_function_with_keywords (to_pyobject self) "mapparms"
     [||]
     []

let roots self =
   Py.Module.get_function_with_keywords (to_pyobject self) "roots"
     [||]
     []
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let trim ?tol self =
   Py.Module.get_function_with_keywords (to_pyobject self) "trim"
     [||]
     (Wrap_utils.keyword_args [("tol", tol)])

let truncate ~size self =
   Py.Module.get_function_with_keywords (to_pyobject self) "truncate"
     [||]
     (Wrap_utils.keyword_args [("size", Some(size ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Laguerre = struct
type tag = [`Laguerre]
type t = [`ABCPolyBase | `Laguerre | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_abc_poly x = (x :> [`ABCPolyBase] Obj.t)
let create ?domain ?window ~coef () =
   Py.Module.get_function_with_keywords __wrap_namespace "Laguerre"
     [||]
     (Wrap_utils.keyword_args [("domain", domain); ("window", window); ("coef", Some(coef |> Obj.to_pyobject))])
     |> of_pyobject
let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let basis ?domain ?window ~deg self =
   Py.Module.get_function_with_keywords (to_pyobject self) "basis"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("window", Wrap_utils.Option.map window Obj.to_pyobject); ("deg", Some(deg |> Py.Int.of_int))])

let cast ?domain ?window ~series self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cast"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("window", Wrap_utils.Option.map window Obj.to_pyobject); ("series", Some(series ))])

let convert ?domain ?kind ?window self =
   Py.Module.get_function_with_keywords (to_pyobject self) "convert"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("kind", kind); ("window", Wrap_utils.Option.map window Obj.to_pyobject)])

let copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "copy"
     [||]
     []

let cutdeg ~deg self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cutdeg"
     [||]
     (Wrap_utils.keyword_args [("deg", Some(deg ))])

let degree self =
   Py.Module.get_function_with_keywords (to_pyobject self) "degree"
     [||]
     []
     |> Py.Int.to_int
let deriv ?m self =
   Py.Module.get_function_with_keywords (to_pyobject self) "deriv"
     [||]
     (Wrap_utils.keyword_args [("m", m)])

                  let fit ?domain ?rcond ?full ?w ?window ~y ~deg x self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fit"
                       [||]
                       (Wrap_utils.keyword_args [("domain", domain); ("rcond", Wrap_utils.Option.map rcond Py.Float.of_float); ("full", Wrap_utils.Option.map full Py.Bool.of_bool); ("w", Wrap_utils.Option.map w Obj.to_pyobject); ("window", window); ("y", Some(y |> Obj.to_pyobject)); ("deg", Some(deg |> (function
| `T1_D_array_like x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("x", Some(x |> Obj.to_pyobject))])

                  let fromroots ?domain ?window ~roots self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fromroots"
                       [||]
                       (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain (function
| `Ndarray x -> Obj.to_pyobject x
| `T_ x -> Wrap_utils.id x
| `None -> Py.none
)); ("window", Wrap_utils.Option.map window Obj.to_pyobject); ("roots", Some(roots |> Obj.to_pyobject))])

let has_samecoef ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_samecoef"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let has_samedomain ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_samedomain"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let has_sametype ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_sametype"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let has_samewindow ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_samewindow"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let identity ?domain ?window self =
   Py.Module.get_function_with_keywords (to_pyobject self) "identity"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("window", Wrap_utils.Option.map window Obj.to_pyobject)])

let integ ?m ?k ?lbnd self =
   Py.Module.get_function_with_keywords (to_pyobject self) "integ"
     [||]
     (Wrap_utils.keyword_args [("m", m); ("k", Wrap_utils.Option.map k Obj.to_pyobject); ("lbnd", lbnd)])

let linspace ?n ?domain self =
   Py.Module.get_function_with_keywords (to_pyobject self) "linspace"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("domain", Wrap_utils.Option.map domain Obj.to_pyobject)])

let mapparms self =
   Py.Module.get_function_with_keywords (to_pyobject self) "mapparms"
     [||]
     []

let roots self =
   Py.Module.get_function_with_keywords (to_pyobject self) "roots"
     [||]
     []
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let trim ?tol self =
   Py.Module.get_function_with_keywords (to_pyobject self) "trim"
     [||]
     (Wrap_utils.keyword_args [("tol", tol)])

let truncate ~size self =
   Py.Module.get_function_with_keywords (to_pyobject self) "truncate"
     [||]
     (Wrap_utils.keyword_args [("size", Some(size ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Legendre = struct
type tag = [`Legendre]
type t = [`ABCPolyBase | `Legendre | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_abc_poly x = (x :> [`ABCPolyBase] Obj.t)
let create ?domain ?window ~coef () =
   Py.Module.get_function_with_keywords __wrap_namespace "Legendre"
     [||]
     (Wrap_utils.keyword_args [("domain", domain); ("window", window); ("coef", Some(coef |> Obj.to_pyobject))])
     |> of_pyobject
let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let basis ?domain ?window ~deg self =
   Py.Module.get_function_with_keywords (to_pyobject self) "basis"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("window", Wrap_utils.Option.map window Obj.to_pyobject); ("deg", Some(deg |> Py.Int.of_int))])

let cast ?domain ?window ~series self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cast"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("window", Wrap_utils.Option.map window Obj.to_pyobject); ("series", Some(series ))])

let convert ?domain ?kind ?window self =
   Py.Module.get_function_with_keywords (to_pyobject self) "convert"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("kind", kind); ("window", Wrap_utils.Option.map window Obj.to_pyobject)])

let copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "copy"
     [||]
     []

let cutdeg ~deg self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cutdeg"
     [||]
     (Wrap_utils.keyword_args [("deg", Some(deg ))])

let degree self =
   Py.Module.get_function_with_keywords (to_pyobject self) "degree"
     [||]
     []
     |> Py.Int.to_int
let deriv ?m self =
   Py.Module.get_function_with_keywords (to_pyobject self) "deriv"
     [||]
     (Wrap_utils.keyword_args [("m", m)])

                  let fit ?domain ?rcond ?full ?w ?window ~y ~deg x self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fit"
                       [||]
                       (Wrap_utils.keyword_args [("domain", domain); ("rcond", Wrap_utils.Option.map rcond Py.Float.of_float); ("full", Wrap_utils.Option.map full Py.Bool.of_bool); ("w", Wrap_utils.Option.map w Obj.to_pyobject); ("window", window); ("y", Some(y |> Obj.to_pyobject)); ("deg", Some(deg |> (function
| `T1_D_array_like x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("x", Some(x |> Obj.to_pyobject))])

                  let fromroots ?domain ?window ~roots self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fromroots"
                       [||]
                       (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain (function
| `Ndarray x -> Obj.to_pyobject x
| `T_ x -> Wrap_utils.id x
| `None -> Py.none
)); ("window", Wrap_utils.Option.map window Obj.to_pyobject); ("roots", Some(roots |> Obj.to_pyobject))])

let has_samecoef ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_samecoef"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let has_samedomain ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_samedomain"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let has_sametype ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_sametype"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let has_samewindow ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_samewindow"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let identity ?domain ?window self =
   Py.Module.get_function_with_keywords (to_pyobject self) "identity"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("window", Wrap_utils.Option.map window Obj.to_pyobject)])

let integ ?m ?k ?lbnd self =
   Py.Module.get_function_with_keywords (to_pyobject self) "integ"
     [||]
     (Wrap_utils.keyword_args [("m", m); ("k", Wrap_utils.Option.map k Obj.to_pyobject); ("lbnd", lbnd)])

let linspace ?n ?domain self =
   Py.Module.get_function_with_keywords (to_pyobject self) "linspace"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("domain", Wrap_utils.Option.map domain Obj.to_pyobject)])

let mapparms self =
   Py.Module.get_function_with_keywords (to_pyobject self) "mapparms"
     [||]
     []

let roots self =
   Py.Module.get_function_with_keywords (to_pyobject self) "roots"
     [||]
     []
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let trim ?tol self =
   Py.Module.get_function_with_keywords (to_pyobject self) "trim"
     [||]
     (Wrap_utils.keyword_args [("tol", tol)])

let truncate ~size self =
   Py.Module.get_function_with_keywords (to_pyobject self) "truncate"
     [||]
     (Wrap_utils.keyword_args [("size", Some(size ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Polynomial = struct
type tag = [`Polynomial]
type t = [`ABCPolyBase | `Object | `Polynomial] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_abc_poly x = (x :> [`ABCPolyBase] Obj.t)
let create ?domain ?window ~coef () =
   Py.Module.get_function_with_keywords __wrap_namespace "Polynomial"
     [||]
     (Wrap_utils.keyword_args [("domain", domain); ("window", window); ("coef", Some(coef |> Obj.to_pyobject))])
     |> of_pyobject
let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let basis ?domain ?window ~deg self =
   Py.Module.get_function_with_keywords (to_pyobject self) "basis"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("window", Wrap_utils.Option.map window Obj.to_pyobject); ("deg", Some(deg |> Py.Int.of_int))])

let cast ?domain ?window ~series self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cast"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("window", Wrap_utils.Option.map window Obj.to_pyobject); ("series", Some(series ))])

let convert ?domain ?kind ?window self =
   Py.Module.get_function_with_keywords (to_pyobject self) "convert"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("kind", kind); ("window", Wrap_utils.Option.map window Obj.to_pyobject)])

let copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "copy"
     [||]
     []

let cutdeg ~deg self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cutdeg"
     [||]
     (Wrap_utils.keyword_args [("deg", Some(deg ))])

let degree self =
   Py.Module.get_function_with_keywords (to_pyobject self) "degree"
     [||]
     []
     |> Py.Int.to_int
let deriv ?m self =
   Py.Module.get_function_with_keywords (to_pyobject self) "deriv"
     [||]
     (Wrap_utils.keyword_args [("m", m)])

                  let fit ?domain ?rcond ?full ?w ?window ~y ~deg x self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fit"
                       [||]
                       (Wrap_utils.keyword_args [("domain", domain); ("rcond", Wrap_utils.Option.map rcond Py.Float.of_float); ("full", Wrap_utils.Option.map full Py.Bool.of_bool); ("w", Wrap_utils.Option.map w Obj.to_pyobject); ("window", window); ("y", Some(y |> Obj.to_pyobject)); ("deg", Some(deg |> (function
| `T1_D_array_like x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("x", Some(x |> Obj.to_pyobject))])

                  let fromroots ?domain ?window ~roots self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fromroots"
                       [||]
                       (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain (function
| `Ndarray x -> Obj.to_pyobject x
| `T_ x -> Wrap_utils.id x
| `None -> Py.none
)); ("window", Wrap_utils.Option.map window Obj.to_pyobject); ("roots", Some(roots |> Obj.to_pyobject))])

let has_samecoef ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_samecoef"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let has_samedomain ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_samedomain"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let has_sametype ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_sametype"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let has_samewindow ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_samewindow"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let identity ?domain ?window self =
   Py.Module.get_function_with_keywords (to_pyobject self) "identity"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("window", Wrap_utils.Option.map window Obj.to_pyobject)])

let integ ?m ?k ?lbnd self =
   Py.Module.get_function_with_keywords (to_pyobject self) "integ"
     [||]
     (Wrap_utils.keyword_args [("m", m); ("k", Wrap_utils.Option.map k Obj.to_pyobject); ("lbnd", lbnd)])

let linspace ?n ?domain self =
   Py.Module.get_function_with_keywords (to_pyobject self) "linspace"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("domain", Wrap_utils.Option.map domain Obj.to_pyobject)])

let mapparms self =
   Py.Module.get_function_with_keywords (to_pyobject self) "mapparms"
     [||]
     []

let roots self =
   Py.Module.get_function_with_keywords (to_pyobject self) "roots"
     [||]
     []
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let trim ?tol self =
   Py.Module.get_function_with_keywords (to_pyobject self) "trim"
     [||]
     (Wrap_utils.keyword_args [("tol", tol)])

let truncate ~size self =
   Py.Module.get_function_with_keywords (to_pyobject self) "truncate"
     [||]
     (Wrap_utils.keyword_args [("size", Some(size ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Chebyshev' = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "numpy.polynomial.chebyshev"

let get_py name = Py.Module.get __wrap_namespace name
module ABCPolyBase = struct
type tag = [`ABCPolyBase]
type t = [`ABCPolyBase | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let basis ?domain ?window ~deg self =
   Py.Module.get_function_with_keywords (to_pyobject self) "basis"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("window", Wrap_utils.Option.map window Obj.to_pyobject); ("deg", Some(deg |> Py.Int.of_int))])

let cast ?domain ?window ~series self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cast"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("window", Wrap_utils.Option.map window Obj.to_pyobject); ("series", Some(series ))])

let convert ?domain ?kind ?window self =
   Py.Module.get_function_with_keywords (to_pyobject self) "convert"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("kind", kind); ("window", Wrap_utils.Option.map window Obj.to_pyobject)])

let copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "copy"
     [||]
     []

let cutdeg ~deg self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cutdeg"
     [||]
     (Wrap_utils.keyword_args [("deg", Some(deg ))])

let degree self =
   Py.Module.get_function_with_keywords (to_pyobject self) "degree"
     [||]
     []
     |> Py.Int.to_int
let deriv ?m self =
   Py.Module.get_function_with_keywords (to_pyobject self) "deriv"
     [||]
     (Wrap_utils.keyword_args [("m", m)])

                  let fit ?domain ?rcond ?full ?w ?window ~y ~deg x self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fit"
                       [||]
                       (Wrap_utils.keyword_args [("domain", domain); ("rcond", Wrap_utils.Option.map rcond Py.Float.of_float); ("full", Wrap_utils.Option.map full Py.Bool.of_bool); ("w", Wrap_utils.Option.map w Obj.to_pyobject); ("window", window); ("y", Some(y |> Obj.to_pyobject)); ("deg", Some(deg |> (function
| `T1_D_array_like x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("x", Some(x |> Obj.to_pyobject))])

                  let fromroots ?domain ?window ~roots self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fromroots"
                       [||]
                       (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain (function
| `Ndarray x -> Obj.to_pyobject x
| `T_ x -> Wrap_utils.id x
| `None -> Py.none
)); ("window", Wrap_utils.Option.map window Obj.to_pyobject); ("roots", Some(roots |> Obj.to_pyobject))])

let has_samecoef ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_samecoef"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let has_samedomain ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_samedomain"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let has_sametype ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_sametype"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let has_samewindow ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_samewindow"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let identity ?domain ?window self =
   Py.Module.get_function_with_keywords (to_pyobject self) "identity"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("window", Wrap_utils.Option.map window Obj.to_pyobject)])

let integ ?m ?k ?lbnd self =
   Py.Module.get_function_with_keywords (to_pyobject self) "integ"
     [||]
     (Wrap_utils.keyword_args [("m", m); ("k", Wrap_utils.Option.map k Obj.to_pyobject); ("lbnd", lbnd)])

let linspace ?n ?domain self =
   Py.Module.get_function_with_keywords (to_pyobject self) "linspace"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("domain", Wrap_utils.Option.map domain Obj.to_pyobject)])

let mapparms self =
   Py.Module.get_function_with_keywords (to_pyobject self) "mapparms"
     [||]
     []

let roots self =
   Py.Module.get_function_with_keywords (to_pyobject self) "roots"
     [||]
     []
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let trim ?tol self =
   Py.Module.get_function_with_keywords (to_pyobject self) "trim"
     [||]
     (Wrap_utils.keyword_args [("tol", tol)])

let truncate ~size self =
   Py.Module.get_function_with_keywords (to_pyobject self) "truncate"
     [||]
     (Wrap_utils.keyword_args [("size", Some(size ))])


let coef_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "coef" with
  | None -> failwith "attribute coef not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) x)

let coef self = match coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let domain_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "domain" with
  | None -> failwith "attribute domain not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let domain self = match domain_opt self with
  | None -> raise Not_found
  | Some x -> x

let window_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "window" with
  | None -> failwith "attribute window not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let window self = match window_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let cheb2poly c =
   Py.Module.get_function_with_keywords __wrap_namespace "cheb2poly"
     [||]
     (Wrap_utils.keyword_args [("c", Some(c |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let chebadd ~c1 ~c2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "chebadd"
     [||]
     (Wrap_utils.keyword_args [("c1", Some(c1 )); ("c2", Some(c2 ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let chebcompanion c =
   Py.Module.get_function_with_keywords __wrap_namespace "chebcompanion"
     [||]
     (Wrap_utils.keyword_args [("c", Some(c |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let chebder ?m ?scl ?axis ~c () =
                     Py.Module.get_function_with_keywords __wrap_namespace "chebder"
                       [||]
                       (Wrap_utils.keyword_args [("m", Wrap_utils.Option.map m Py.Int.of_int); ("scl", Wrap_utils.Option.map scl (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("c", Some(c |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let chebdiv ~c1 ~c2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "chebdiv"
     [||]
     (Wrap_utils.keyword_args [("c1", Some(c1 )); ("c2", Some(c2 ))])

                  let chebfit ?rcond ?full ?w ~y ~deg x =
                     Py.Module.get_function_with_keywords __wrap_namespace "chebfit"
                       [||]
                       (Wrap_utils.keyword_args [("rcond", Wrap_utils.Option.map rcond Py.Float.of_float); ("full", Wrap_utils.Option.map full Py.Bool.of_bool); ("w", Wrap_utils.Option.map w Obj.to_pyobject); ("y", Some(y |> Obj.to_pyobject)); ("deg", Some(deg |> (function
| `T1_D_array_like x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("x", Some(x |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let chebfromroots roots =
   Py.Module.get_function_with_keywords __wrap_namespace "chebfromroots"
     [||]
     (Wrap_utils.keyword_args [("roots", Some(roots |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let chebgauss deg =
   Py.Module.get_function_with_keywords __wrap_namespace "chebgauss"
     [||]
     (Wrap_utils.keyword_args [("deg", Some(deg |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 1))))
let chebgrid2d ~y ~c x =
   Py.Module.get_function_with_keywords __wrap_namespace "chebgrid2d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x ))])

let chebgrid3d ~y ~z ~c x =
   Py.Module.get_function_with_keywords __wrap_namespace "chebgrid3d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("z", Some(z )); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x ))])

                  let chebint ?m ?k ?lbnd ?scl ?axis ~c () =
                     Py.Module.get_function_with_keywords __wrap_namespace "chebint"
                       [||]
                       (Wrap_utils.keyword_args [("m", Wrap_utils.Option.map m Py.Int.of_int); ("k", Wrap_utils.Option.map k (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `T_ x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
)); ("lbnd", Wrap_utils.Option.map lbnd (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("scl", Wrap_utils.Option.map scl (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("c", Some(c |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let chebinterpolate ?args ~func ~deg () =
   Py.Module.get_function_with_keywords __wrap_namespace "chebinterpolate"
     [||]
     (Wrap_utils.keyword_args [("args", args); ("func", Some(func )); ("deg", Some(deg |> Py.Int.of_int))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let chebline ~off ~scl () =
   Py.Module.get_function_with_keywords __wrap_namespace "chebline"
     [||]
     (Wrap_utils.keyword_args [("off", Some(off )); ("scl", Some(scl ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let chebmul ~c1 ~c2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "chebmul"
     [||]
     (Wrap_utils.keyword_args [("c1", Some(c1 )); ("c2", Some(c2 ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let chebmulx c =
   Py.Module.get_function_with_keywords __wrap_namespace "chebmulx"
     [||]
     (Wrap_utils.keyword_args [("c", Some(c |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let chebpow ?maxpower ~c ~pow () =
   Py.Module.get_function_with_keywords __wrap_namespace "chebpow"
     [||]
     (Wrap_utils.keyword_args [("maxpower", Wrap_utils.Option.map maxpower Py.Int.of_int); ("c", Some(c |> Obj.to_pyobject)); ("pow", Some(pow |> Py.Int.of_int))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let chebpts1 npts =
   Py.Module.get_function_with_keywords __wrap_namespace "chebpts1"
     [||]
     (Wrap_utils.keyword_args [("npts", Some(npts |> Py.Int.of_int))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let chebpts2 npts =
   Py.Module.get_function_with_keywords __wrap_namespace "chebpts2"
     [||]
     (Wrap_utils.keyword_args [("npts", Some(npts |> Py.Int.of_int))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let chebroots c =
   Py.Module.get_function_with_keywords __wrap_namespace "chebroots"
     [||]
     (Wrap_utils.keyword_args [("c", Some(c ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let chebsub ~c1 ~c2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "chebsub"
     [||]
     (Wrap_utils.keyword_args [("c1", Some(c1 )); ("c2", Some(c2 ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let chebtrim ?tol ~c () =
                     Py.Module.get_function_with_keywords __wrap_namespace "chebtrim"
                       [||]
                       (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("c", Some(c |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let chebval ?tensor ~c x =
                     Py.Module.get_function_with_keywords __wrap_namespace "chebval"
                       [||]
                       (Wrap_utils.keyword_args [("tensor", Wrap_utils.Option.map tensor Py.Bool.of_bool); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x |> (function
| `Compatible_object x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)))])

let chebval2d ~y ~c x =
   Py.Module.get_function_with_keywords __wrap_namespace "chebval2d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x ))])

let chebval3d ~y ~z ~c x =
   Py.Module.get_function_with_keywords __wrap_namespace "chebval3d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("z", Some(z )); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x ))])

let chebvander ~deg x =
   Py.Module.get_function_with_keywords __wrap_namespace "chebvander"
     [||]
     (Wrap_utils.keyword_args [("deg", Some(deg |> Py.Int.of_int)); ("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let chebvander2d ~y ~deg x =
   Py.Module.get_function_with_keywords __wrap_namespace "chebvander2d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("deg", Some(deg )); ("x", Some(x ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let chebvander3d ~y ~z ~deg x =
   Py.Module.get_function_with_keywords __wrap_namespace "chebvander3d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("z", Some(z )); ("deg", Some(deg )); ("x", Some(x ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let chebweight x =
   Py.Module.get_function_with_keywords __wrap_namespace "chebweight"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let normalize_axis_index ?msg_prefix ~axis ~ndim () =
   Py.Module.get_function_with_keywords __wrap_namespace "normalize_axis_index"
     [||]
     (Wrap_utils.keyword_args [("msg_prefix", Wrap_utils.Option.map msg_prefix Py.String.of_string); ("axis", Some(axis |> Py.Int.of_int)); ("ndim", Some(ndim |> Py.Int.of_int))])
     |> Py.Int.to_int
let poly2cheb pol =
   Py.Module.get_function_with_keywords __wrap_namespace "poly2cheb"
     [||]
     (Wrap_utils.keyword_args [("pol", Some(pol |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))

end
module Hermite' = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "numpy.polynomial.hermite"

let get_py name = Py.Module.get __wrap_namespace name
module ABCPolyBase = struct
type tag = [`ABCPolyBase]
type t = [`ABCPolyBase | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let basis ?domain ?window ~deg self =
   Py.Module.get_function_with_keywords (to_pyobject self) "basis"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("window", Wrap_utils.Option.map window Obj.to_pyobject); ("deg", Some(deg |> Py.Int.of_int))])

let cast ?domain ?window ~series self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cast"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("window", Wrap_utils.Option.map window Obj.to_pyobject); ("series", Some(series ))])

let convert ?domain ?kind ?window self =
   Py.Module.get_function_with_keywords (to_pyobject self) "convert"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("kind", kind); ("window", Wrap_utils.Option.map window Obj.to_pyobject)])

let copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "copy"
     [||]
     []

let cutdeg ~deg self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cutdeg"
     [||]
     (Wrap_utils.keyword_args [("deg", Some(deg ))])

let degree self =
   Py.Module.get_function_with_keywords (to_pyobject self) "degree"
     [||]
     []
     |> Py.Int.to_int
let deriv ?m self =
   Py.Module.get_function_with_keywords (to_pyobject self) "deriv"
     [||]
     (Wrap_utils.keyword_args [("m", m)])

                  let fit ?domain ?rcond ?full ?w ?window ~y ~deg x self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fit"
                       [||]
                       (Wrap_utils.keyword_args [("domain", domain); ("rcond", Wrap_utils.Option.map rcond Py.Float.of_float); ("full", Wrap_utils.Option.map full Py.Bool.of_bool); ("w", Wrap_utils.Option.map w Obj.to_pyobject); ("window", window); ("y", Some(y |> Obj.to_pyobject)); ("deg", Some(deg |> (function
| `T1_D_array_like x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("x", Some(x |> Obj.to_pyobject))])

                  let fromroots ?domain ?window ~roots self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "fromroots"
                       [||]
                       (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain (function
| `Ndarray x -> Obj.to_pyobject x
| `T_ x -> Wrap_utils.id x
| `None -> Py.none
)); ("window", Wrap_utils.Option.map window Obj.to_pyobject); ("roots", Some(roots |> Obj.to_pyobject))])

let has_samecoef ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_samecoef"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let has_samedomain ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_samedomain"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let has_sametype ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_sametype"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let has_samewindow ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "has_samewindow"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])
     |> Py.Bool.to_bool
let identity ?domain ?window self =
   Py.Module.get_function_with_keywords (to_pyobject self) "identity"
     [||]
     (Wrap_utils.keyword_args [("domain", Wrap_utils.Option.map domain Obj.to_pyobject); ("window", Wrap_utils.Option.map window Obj.to_pyobject)])

let integ ?m ?k ?lbnd self =
   Py.Module.get_function_with_keywords (to_pyobject self) "integ"
     [||]
     (Wrap_utils.keyword_args [("m", m); ("k", Wrap_utils.Option.map k Obj.to_pyobject); ("lbnd", lbnd)])

let linspace ?n ?domain self =
   Py.Module.get_function_with_keywords (to_pyobject self) "linspace"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("domain", Wrap_utils.Option.map domain Obj.to_pyobject)])

let mapparms self =
   Py.Module.get_function_with_keywords (to_pyobject self) "mapparms"
     [||]
     []

let roots self =
   Py.Module.get_function_with_keywords (to_pyobject self) "roots"
     [||]
     []
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let trim ?tol self =
   Py.Module.get_function_with_keywords (to_pyobject self) "trim"
     [||]
     (Wrap_utils.keyword_args [("tol", tol)])

let truncate ~size self =
   Py.Module.get_function_with_keywords (to_pyobject self) "truncate"
     [||]
     (Wrap_utils.keyword_args [("size", Some(size ))])


let coef_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "coef" with
  | None -> failwith "attribute coef not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) x)

let coef self = match coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let domain_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "domain" with
  | None -> failwith "attribute domain not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let domain self = match domain_opt self with
  | None -> raise Not_found
  | Some x -> x

let window_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "window" with
  | None -> failwith "attribute window not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let window self = match window_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let herm2poly c =
   Py.Module.get_function_with_keywords __wrap_namespace "herm2poly"
     [||]
     (Wrap_utils.keyword_args [("c", Some(c |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermadd ~c1 ~c2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "hermadd"
     [||]
     (Wrap_utils.keyword_args [("c1", Some(c1 )); ("c2", Some(c2 ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermcompanion c =
   Py.Module.get_function_with_keywords __wrap_namespace "hermcompanion"
     [||]
     (Wrap_utils.keyword_args [("c", Some(c |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let hermder ?m ?scl ?axis ~c () =
                     Py.Module.get_function_with_keywords __wrap_namespace "hermder"
                       [||]
                       (Wrap_utils.keyword_args [("m", Wrap_utils.Option.map m Py.Int.of_int); ("scl", Wrap_utils.Option.map scl (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("c", Some(c |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermdiv ~c1 ~c2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "hermdiv"
     [||]
     (Wrap_utils.keyword_args [("c1", Some(c1 )); ("c2", Some(c2 ))])

                  let hermfit ?rcond ?full ?w ~y ~deg x =
                     Py.Module.get_function_with_keywords __wrap_namespace "hermfit"
                       [||]
                       (Wrap_utils.keyword_args [("rcond", Wrap_utils.Option.map rcond Py.Float.of_float); ("full", Wrap_utils.Option.map full Py.Bool.of_bool); ("w", Wrap_utils.Option.map w Obj.to_pyobject); ("y", Some(y |> Obj.to_pyobject)); ("deg", Some(deg |> (function
| `T1_D_array_like x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("x", Some(x |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermfromroots roots =
   Py.Module.get_function_with_keywords __wrap_namespace "hermfromroots"
     [||]
     (Wrap_utils.keyword_args [("roots", Some(roots |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermgauss deg =
   Py.Module.get_function_with_keywords __wrap_namespace "hermgauss"
     [||]
     (Wrap_utils.keyword_args [("deg", Some(deg |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 1))))
let hermgrid2d ~y ~c x =
   Py.Module.get_function_with_keywords __wrap_namespace "hermgrid2d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x ))])

let hermgrid3d ~y ~z ~c x =
   Py.Module.get_function_with_keywords __wrap_namespace "hermgrid3d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("z", Some(z )); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x ))])

                  let hermint ?m ?k ?lbnd ?scl ?axis ~c () =
                     Py.Module.get_function_with_keywords __wrap_namespace "hermint"
                       [||]
                       (Wrap_utils.keyword_args [("m", Wrap_utils.Option.map m Py.Int.of_int); ("k", Wrap_utils.Option.map k (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `T_ x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
)); ("lbnd", Wrap_utils.Option.map lbnd (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("scl", Wrap_utils.Option.map scl (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("c", Some(c |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermline ~off ~scl () =
   Py.Module.get_function_with_keywords __wrap_namespace "hermline"
     [||]
     (Wrap_utils.keyword_args [("off", Some(off )); ("scl", Some(scl ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermmul ~c1 ~c2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "hermmul"
     [||]
     (Wrap_utils.keyword_args [("c1", Some(c1 )); ("c2", Some(c2 ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermmulx c =
   Py.Module.get_function_with_keywords __wrap_namespace "hermmulx"
     [||]
     (Wrap_utils.keyword_args [("c", Some(c |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermpow ?maxpower ~c ~pow () =
   Py.Module.get_function_with_keywords __wrap_namespace "hermpow"
     [||]
     (Wrap_utils.keyword_args [("maxpower", Wrap_utils.Option.map maxpower Py.Int.of_int); ("c", Some(c |> Obj.to_pyobject)); ("pow", Some(pow |> Py.Int.of_int))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermroots c =
   Py.Module.get_function_with_keywords __wrap_namespace "hermroots"
     [||]
     (Wrap_utils.keyword_args [("c", Some(c ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermsub ~c1 ~c2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "hermsub"
     [||]
     (Wrap_utils.keyword_args [("c1", Some(c1 )); ("c2", Some(c2 ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let hermtrim ?tol ~c () =
                     Py.Module.get_function_with_keywords __wrap_namespace "hermtrim"
                       [||]
                       (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("c", Some(c |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let hermval ?tensor ~c x =
                     Py.Module.get_function_with_keywords __wrap_namespace "hermval"
                       [||]
                       (Wrap_utils.keyword_args [("tensor", Wrap_utils.Option.map tensor Py.Bool.of_bool); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x |> (function
| `Compatible_object x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)))])

let hermval2d ~y ~c x =
   Py.Module.get_function_with_keywords __wrap_namespace "hermval2d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x ))])

let hermval3d ~y ~z ~c x =
   Py.Module.get_function_with_keywords __wrap_namespace "hermval3d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("z", Some(z )); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x ))])

let hermvander ~deg x =
   Py.Module.get_function_with_keywords __wrap_namespace "hermvander"
     [||]
     (Wrap_utils.keyword_args [("deg", Some(deg |> Py.Int.of_int)); ("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermvander2d ~y ~deg x =
   Py.Module.get_function_with_keywords __wrap_namespace "hermvander2d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("deg", Some(deg )); ("x", Some(x ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermvander3d ~y ~z ~deg x =
   Py.Module.get_function_with_keywords __wrap_namespace "hermvander3d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("z", Some(z )); ("deg", Some(deg )); ("x", Some(x ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermweight x =
   Py.Module.get_function_with_keywords __wrap_namespace "hermweight"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let normalize_axis_index ?msg_prefix ~axis ~ndim () =
   Py.Module.get_function_with_keywords __wrap_namespace "normalize_axis_index"
     [||]
     (Wrap_utils.keyword_args [("msg_prefix", Wrap_utils.Option.map msg_prefix Py.String.of_string); ("axis", Some(axis |> Py.Int.of_int)); ("ndim", Some(ndim |> Py.Int.of_int))])
     |> Py.Int.to_int
let poly2herm pol =
   Py.Module.get_function_with_keywords __wrap_namespace "poly2herm"
     [||]
     (Wrap_utils.keyword_args [("pol", Some(pol |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))

end
module Hermite_e = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "numpy.polynomial.hermite_e"

let get_py name = Py.Module.get __wrap_namespace name
let herme2poly c =
   Py.Module.get_function_with_keywords __wrap_namespace "herme2poly"
     [||]
     (Wrap_utils.keyword_args [("c", Some(c |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermeadd ~c1 ~c2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "hermeadd"
     [||]
     (Wrap_utils.keyword_args [("c1", Some(c1 )); ("c2", Some(c2 ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermecompanion c =
   Py.Module.get_function_with_keywords __wrap_namespace "hermecompanion"
     [||]
     (Wrap_utils.keyword_args [("c", Some(c |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let hermeder ?m ?scl ?axis ~c () =
                     Py.Module.get_function_with_keywords __wrap_namespace "hermeder"
                       [||]
                       (Wrap_utils.keyword_args [("m", Wrap_utils.Option.map m Py.Int.of_int); ("scl", Wrap_utils.Option.map scl (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("c", Some(c |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermediv ~c1 ~c2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "hermediv"
     [||]
     (Wrap_utils.keyword_args [("c1", Some(c1 )); ("c2", Some(c2 ))])

                  let hermefit ?rcond ?full ?w ~y ~deg x =
                     Py.Module.get_function_with_keywords __wrap_namespace "hermefit"
                       [||]
                       (Wrap_utils.keyword_args [("rcond", Wrap_utils.Option.map rcond Py.Float.of_float); ("full", Wrap_utils.Option.map full Py.Bool.of_bool); ("w", Wrap_utils.Option.map w Obj.to_pyobject); ("y", Some(y |> Obj.to_pyobject)); ("deg", Some(deg |> (function
| `T1_D_array_like x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("x", Some(x |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermefromroots roots =
   Py.Module.get_function_with_keywords __wrap_namespace "hermefromroots"
     [||]
     (Wrap_utils.keyword_args [("roots", Some(roots |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermegauss deg =
   Py.Module.get_function_with_keywords __wrap_namespace "hermegauss"
     [||]
     (Wrap_utils.keyword_args [("deg", Some(deg |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 1))))
let hermegrid2d ~y ~c x =
   Py.Module.get_function_with_keywords __wrap_namespace "hermegrid2d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x ))])

let hermegrid3d ~y ~z ~c x =
   Py.Module.get_function_with_keywords __wrap_namespace "hermegrid3d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("z", Some(z )); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x ))])

                  let hermeint ?m ?k ?lbnd ?scl ?axis ~c () =
                     Py.Module.get_function_with_keywords __wrap_namespace "hermeint"
                       [||]
                       (Wrap_utils.keyword_args [("m", Wrap_utils.Option.map m Py.Int.of_int); ("k", Wrap_utils.Option.map k (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `T_ x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
)); ("lbnd", Wrap_utils.Option.map lbnd (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("scl", Wrap_utils.Option.map scl (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("c", Some(c |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermeline ~off ~scl () =
   Py.Module.get_function_with_keywords __wrap_namespace "hermeline"
     [||]
     (Wrap_utils.keyword_args [("off", Some(off )); ("scl", Some(scl ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermemul ~c1 ~c2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "hermemul"
     [||]
     (Wrap_utils.keyword_args [("c1", Some(c1 )); ("c2", Some(c2 ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermemulx c =
   Py.Module.get_function_with_keywords __wrap_namespace "hermemulx"
     [||]
     (Wrap_utils.keyword_args [("c", Some(c |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermepow ?maxpower ~c ~pow () =
   Py.Module.get_function_with_keywords __wrap_namespace "hermepow"
     [||]
     (Wrap_utils.keyword_args [("maxpower", Wrap_utils.Option.map maxpower Py.Int.of_int); ("c", Some(c |> Obj.to_pyobject)); ("pow", Some(pow |> Py.Int.of_int))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermeroots c =
   Py.Module.get_function_with_keywords __wrap_namespace "hermeroots"
     [||]
     (Wrap_utils.keyword_args [("c", Some(c ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermesub ~c1 ~c2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "hermesub"
     [||]
     (Wrap_utils.keyword_args [("c1", Some(c1 )); ("c2", Some(c2 ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let hermetrim ?tol ~c () =
                     Py.Module.get_function_with_keywords __wrap_namespace "hermetrim"
                       [||]
                       (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("c", Some(c |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let hermeval ?tensor ~c x =
                     Py.Module.get_function_with_keywords __wrap_namespace "hermeval"
                       [||]
                       (Wrap_utils.keyword_args [("tensor", Wrap_utils.Option.map tensor Py.Bool.of_bool); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x |> (function
| `Compatible_object x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)))])

let hermeval2d ~y ~c x =
   Py.Module.get_function_with_keywords __wrap_namespace "hermeval2d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x ))])

let hermeval3d ~y ~z ~c x =
   Py.Module.get_function_with_keywords __wrap_namespace "hermeval3d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("z", Some(z )); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x ))])

let hermevander ~deg x =
   Py.Module.get_function_with_keywords __wrap_namespace "hermevander"
     [||]
     (Wrap_utils.keyword_args [("deg", Some(deg |> Py.Int.of_int)); ("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermevander2d ~y ~deg x =
   Py.Module.get_function_with_keywords __wrap_namespace "hermevander2d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("deg", Some(deg )); ("x", Some(x ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermevander3d ~y ~z ~deg x =
   Py.Module.get_function_with_keywords __wrap_namespace "hermevander3d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("z", Some(z )); ("deg", Some(deg )); ("x", Some(x ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hermeweight x =
   Py.Module.get_function_with_keywords __wrap_namespace "hermeweight"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let normalize_axis_index ?msg_prefix ~axis ~ndim () =
   Py.Module.get_function_with_keywords __wrap_namespace "normalize_axis_index"
     [||]
     (Wrap_utils.keyword_args [("msg_prefix", Wrap_utils.Option.map msg_prefix Py.String.of_string); ("axis", Some(axis |> Py.Int.of_int)); ("ndim", Some(ndim |> Py.Int.of_int))])
     |> Py.Int.to_int
let poly2herme pol =
   Py.Module.get_function_with_keywords __wrap_namespace "poly2herme"
     [||]
     (Wrap_utils.keyword_args [("pol", Some(pol |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))

end
module Laguerre' = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "numpy.polynomial.laguerre"

let get_py name = Py.Module.get __wrap_namespace name
let lag2poly c =
   Py.Module.get_function_with_keywords __wrap_namespace "lag2poly"
     [||]
     (Wrap_utils.keyword_args [("c", Some(c |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let lagadd ~c1 ~c2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "lagadd"
     [||]
     (Wrap_utils.keyword_args [("c1", Some(c1 )); ("c2", Some(c2 ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let lagcompanion c =
   Py.Module.get_function_with_keywords __wrap_namespace "lagcompanion"
     [||]
     (Wrap_utils.keyword_args [("c", Some(c |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let lagder ?m ?scl ?axis ~c () =
                     Py.Module.get_function_with_keywords __wrap_namespace "lagder"
                       [||]
                       (Wrap_utils.keyword_args [("m", Wrap_utils.Option.map m Py.Int.of_int); ("scl", Wrap_utils.Option.map scl (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("c", Some(c |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let lagdiv ~c1 ~c2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "lagdiv"
     [||]
     (Wrap_utils.keyword_args [("c1", Some(c1 )); ("c2", Some(c2 ))])

                  let lagfit ?rcond ?full ?w ~y ~deg x =
                     Py.Module.get_function_with_keywords __wrap_namespace "lagfit"
                       [||]
                       (Wrap_utils.keyword_args [("rcond", Wrap_utils.Option.map rcond Py.Float.of_float); ("full", Wrap_utils.Option.map full Py.Bool.of_bool); ("w", Wrap_utils.Option.map w Obj.to_pyobject); ("y", Some(y |> Obj.to_pyobject)); ("deg", Some(deg |> (function
| `T1_D_array_like x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("x", Some(x |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let lagfromroots roots =
   Py.Module.get_function_with_keywords __wrap_namespace "lagfromroots"
     [||]
     (Wrap_utils.keyword_args [("roots", Some(roots |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let laggauss deg =
   Py.Module.get_function_with_keywords __wrap_namespace "laggauss"
     [||]
     (Wrap_utils.keyword_args [("deg", Some(deg |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 1))))
let laggrid2d ~y ~c x =
   Py.Module.get_function_with_keywords __wrap_namespace "laggrid2d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x ))])

let laggrid3d ~y ~z ~c x =
   Py.Module.get_function_with_keywords __wrap_namespace "laggrid3d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("z", Some(z )); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x ))])

                  let lagint ?m ?k ?lbnd ?scl ?axis ~c () =
                     Py.Module.get_function_with_keywords __wrap_namespace "lagint"
                       [||]
                       (Wrap_utils.keyword_args [("m", Wrap_utils.Option.map m Py.Int.of_int); ("k", Wrap_utils.Option.map k (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `T_ x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
)); ("lbnd", Wrap_utils.Option.map lbnd (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("scl", Wrap_utils.Option.map scl (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("c", Some(c |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let lagline ~off ~scl () =
   Py.Module.get_function_with_keywords __wrap_namespace "lagline"
     [||]
     (Wrap_utils.keyword_args [("off", Some(off )); ("scl", Some(scl ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let lagmul ~c1 ~c2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "lagmul"
     [||]
     (Wrap_utils.keyword_args [("c1", Some(c1 )); ("c2", Some(c2 ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let lagmulx c =
   Py.Module.get_function_with_keywords __wrap_namespace "lagmulx"
     [||]
     (Wrap_utils.keyword_args [("c", Some(c |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let lagpow ?maxpower ~c ~pow () =
   Py.Module.get_function_with_keywords __wrap_namespace "lagpow"
     [||]
     (Wrap_utils.keyword_args [("maxpower", Wrap_utils.Option.map maxpower Py.Int.of_int); ("c", Some(c |> Obj.to_pyobject)); ("pow", Some(pow |> Py.Int.of_int))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let lagroots c =
   Py.Module.get_function_with_keywords __wrap_namespace "lagroots"
     [||]
     (Wrap_utils.keyword_args [("c", Some(c ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let lagsub ~c1 ~c2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "lagsub"
     [||]
     (Wrap_utils.keyword_args [("c1", Some(c1 )); ("c2", Some(c2 ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let lagtrim ?tol ~c () =
                     Py.Module.get_function_with_keywords __wrap_namespace "lagtrim"
                       [||]
                       (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("c", Some(c |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let lagval ?tensor ~c x =
                     Py.Module.get_function_with_keywords __wrap_namespace "lagval"
                       [||]
                       (Wrap_utils.keyword_args [("tensor", Wrap_utils.Option.map tensor Py.Bool.of_bool); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x |> (function
| `Compatible_object x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)))])

let lagval2d ~y ~c x =
   Py.Module.get_function_with_keywords __wrap_namespace "lagval2d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x ))])

let lagval3d ~y ~z ~c x =
   Py.Module.get_function_with_keywords __wrap_namespace "lagval3d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("z", Some(z )); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x ))])

let lagvander ~deg x =
   Py.Module.get_function_with_keywords __wrap_namespace "lagvander"
     [||]
     (Wrap_utils.keyword_args [("deg", Some(deg |> Py.Int.of_int)); ("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let lagvander2d ~y ~deg x =
   Py.Module.get_function_with_keywords __wrap_namespace "lagvander2d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("deg", Some(deg )); ("x", Some(x ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let lagvander3d ~y ~z ~deg x =
   Py.Module.get_function_with_keywords __wrap_namespace "lagvander3d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("z", Some(z )); ("deg", Some(deg )); ("x", Some(x ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let lagweight x =
   Py.Module.get_function_with_keywords __wrap_namespace "lagweight"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let normalize_axis_index ?msg_prefix ~axis ~ndim () =
   Py.Module.get_function_with_keywords __wrap_namespace "normalize_axis_index"
     [||]
     (Wrap_utils.keyword_args [("msg_prefix", Wrap_utils.Option.map msg_prefix Py.String.of_string); ("axis", Some(axis |> Py.Int.of_int)); ("ndim", Some(ndim |> Py.Int.of_int))])
     |> Py.Int.to_int
let poly2lag pol =
   Py.Module.get_function_with_keywords __wrap_namespace "poly2lag"
     [||]
     (Wrap_utils.keyword_args [("pol", Some(pol |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))

end
module Legendre' = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "numpy.polynomial.legendre"

let get_py name = Py.Module.get __wrap_namespace name
let leg2poly c =
   Py.Module.get_function_with_keywords __wrap_namespace "leg2poly"
     [||]
     (Wrap_utils.keyword_args [("c", Some(c |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let legadd ~c1 ~c2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "legadd"
     [||]
     (Wrap_utils.keyword_args [("c1", Some(c1 )); ("c2", Some(c2 ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let legcompanion c =
   Py.Module.get_function_with_keywords __wrap_namespace "legcompanion"
     [||]
     (Wrap_utils.keyword_args [("c", Some(c |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let legder ?m ?scl ?axis ~c () =
                     Py.Module.get_function_with_keywords __wrap_namespace "legder"
                       [||]
                       (Wrap_utils.keyword_args [("m", Wrap_utils.Option.map m Py.Int.of_int); ("scl", Wrap_utils.Option.map scl (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("c", Some(c |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let legdiv ~c1 ~c2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "legdiv"
     [||]
     (Wrap_utils.keyword_args [("c1", Some(c1 )); ("c2", Some(c2 ))])

                  let legfit ?rcond ?full ?w ~y ~deg x =
                     Py.Module.get_function_with_keywords __wrap_namespace "legfit"
                       [||]
                       (Wrap_utils.keyword_args [("rcond", Wrap_utils.Option.map rcond Py.Float.of_float); ("full", Wrap_utils.Option.map full Py.Bool.of_bool); ("w", Wrap_utils.Option.map w Obj.to_pyobject); ("y", Some(y |> Obj.to_pyobject)); ("deg", Some(deg |> (function
| `T1_D_array_like x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("x", Some(x |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let legfromroots roots =
   Py.Module.get_function_with_keywords __wrap_namespace "legfromroots"
     [||]
     (Wrap_utils.keyword_args [("roots", Some(roots |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let leggauss deg =
   Py.Module.get_function_with_keywords __wrap_namespace "leggauss"
     [||]
     (Wrap_utils.keyword_args [("deg", Some(deg |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 1))))
let leggrid2d ~y ~c x =
   Py.Module.get_function_with_keywords __wrap_namespace "leggrid2d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x ))])

let leggrid3d ~y ~z ~c x =
   Py.Module.get_function_with_keywords __wrap_namespace "leggrid3d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("z", Some(z )); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x ))])

                  let legint ?m ?k ?lbnd ?scl ?axis ~c () =
                     Py.Module.get_function_with_keywords __wrap_namespace "legint"
                       [||]
                       (Wrap_utils.keyword_args [("m", Wrap_utils.Option.map m Py.Int.of_int); ("k", Wrap_utils.Option.map k (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `T_ x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
)); ("lbnd", Wrap_utils.Option.map lbnd (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("scl", Wrap_utils.Option.map scl (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("c", Some(c |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let legline ~off ~scl () =
   Py.Module.get_function_with_keywords __wrap_namespace "legline"
     [||]
     (Wrap_utils.keyword_args [("off", Some(off )); ("scl", Some(scl ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let legmul ~c1 ~c2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "legmul"
     [||]
     (Wrap_utils.keyword_args [("c1", Some(c1 )); ("c2", Some(c2 ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let legmulx c =
   Py.Module.get_function_with_keywords __wrap_namespace "legmulx"
     [||]
     (Wrap_utils.keyword_args [("c", Some(c |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let legpow ?maxpower ~c ~pow () =
   Py.Module.get_function_with_keywords __wrap_namespace "legpow"
     [||]
     (Wrap_utils.keyword_args [("maxpower", Wrap_utils.Option.map maxpower Py.Int.of_int); ("c", Some(c |> Obj.to_pyobject)); ("pow", Some(pow |> Py.Int.of_int))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let legroots c =
   Py.Module.get_function_with_keywords __wrap_namespace "legroots"
     [||]
     (Wrap_utils.keyword_args [("c", Some(c ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let legsub ~c1 ~c2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "legsub"
     [||]
     (Wrap_utils.keyword_args [("c1", Some(c1 )); ("c2", Some(c2 ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let legtrim ?tol ~c () =
                     Py.Module.get_function_with_keywords __wrap_namespace "legtrim"
                       [||]
                       (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("c", Some(c |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let legval ?tensor ~c x =
                     Py.Module.get_function_with_keywords __wrap_namespace "legval"
                       [||]
                       (Wrap_utils.keyword_args [("tensor", Wrap_utils.Option.map tensor Py.Bool.of_bool); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x |> (function
| `Compatible_object x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)))])

let legval2d ~y ~c x =
   Py.Module.get_function_with_keywords __wrap_namespace "legval2d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x ))])

let legval3d ~y ~z ~c x =
   Py.Module.get_function_with_keywords __wrap_namespace "legval3d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("z", Some(z )); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x ))])

let legvander ~deg x =
   Py.Module.get_function_with_keywords __wrap_namespace "legvander"
     [||]
     (Wrap_utils.keyword_args [("deg", Some(deg |> Py.Int.of_int)); ("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let legvander2d ~y ~deg x =
   Py.Module.get_function_with_keywords __wrap_namespace "legvander2d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("deg", Some(deg )); ("x", Some(x ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let legvander3d ~y ~z ~deg x =
   Py.Module.get_function_with_keywords __wrap_namespace "legvander3d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("z", Some(z )); ("deg", Some(deg )); ("x", Some(x ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let legweight x =
   Py.Module.get_function_with_keywords __wrap_namespace "legweight"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let normalize_axis_index ?msg_prefix ~axis ~ndim () =
   Py.Module.get_function_with_keywords __wrap_namespace "normalize_axis_index"
     [||]
     (Wrap_utils.keyword_args [("msg_prefix", Wrap_utils.Option.map msg_prefix Py.String.of_string); ("axis", Some(axis |> Py.Int.of_int)); ("ndim", Some(ndim |> Py.Int.of_int))])
     |> Py.Int.to_int
let poly2leg pol =
   Py.Module.get_function_with_keywords __wrap_namespace "poly2leg"
     [||]
     (Wrap_utils.keyword_args [("pol", Some(pol |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))

end
module Polynomial' = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "numpy.polynomial.polynomial"

let get_py name = Py.Module.get __wrap_namespace name
let normalize_axis_index ?msg_prefix ~axis ~ndim () =
   Py.Module.get_function_with_keywords __wrap_namespace "normalize_axis_index"
     [||]
     (Wrap_utils.keyword_args [("msg_prefix", Wrap_utils.Option.map msg_prefix Py.String.of_string); ("axis", Some(axis |> Py.Int.of_int)); ("ndim", Some(ndim |> Py.Int.of_int))])
     |> Py.Int.to_int
let polyadd ~c1 ~c2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "polyadd"
     [||]
     (Wrap_utils.keyword_args [("c1", Some(c1 )); ("c2", Some(c2 ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let polycompanion c =
   Py.Module.get_function_with_keywords __wrap_namespace "polycompanion"
     [||]
     (Wrap_utils.keyword_args [("c", Some(c |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let polyder ?m ?scl ?axis ~c () =
                     Py.Module.get_function_with_keywords __wrap_namespace "polyder"
                       [||]
                       (Wrap_utils.keyword_args [("m", Wrap_utils.Option.map m Py.Int.of_int); ("scl", Wrap_utils.Option.map scl (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("c", Some(c |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let polydiv ~c1 ~c2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "polydiv"
     [||]
     (Wrap_utils.keyword_args [("c1", Some(c1 )); ("c2", Some(c2 ))])

                  let polyfit ?rcond ?full ?w ~y ~deg x =
                     Py.Module.get_function_with_keywords __wrap_namespace "polyfit"
                       [||]
                       (Wrap_utils.keyword_args [("rcond", Wrap_utils.Option.map rcond Py.Float.of_float); ("full", Wrap_utils.Option.map full Py.Bool.of_bool); ("w", Wrap_utils.Option.map w Obj.to_pyobject); ("y", Some(y |> Obj.to_pyobject)); ("deg", Some(deg |> (function
| `T1_D_array_like x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("x", Some(x |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let polyfromroots roots =
   Py.Module.get_function_with_keywords __wrap_namespace "polyfromroots"
     [||]
     (Wrap_utils.keyword_args [("roots", Some(roots |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let polygrid2d ~y ~c x =
   Py.Module.get_function_with_keywords __wrap_namespace "polygrid2d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x ))])

let polygrid3d ~y ~z ~c x =
   Py.Module.get_function_with_keywords __wrap_namespace "polygrid3d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("z", Some(z )); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x ))])

                  let polyint ?m ?k ?lbnd ?scl ?axis ~c () =
                     Py.Module.get_function_with_keywords __wrap_namespace "polyint"
                       [||]
                       (Wrap_utils.keyword_args [("m", Wrap_utils.Option.map m Py.Int.of_int); ("k", Wrap_utils.Option.map k (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `T_ x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
)); ("lbnd", Wrap_utils.Option.map lbnd (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("scl", Wrap_utils.Option.map scl (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("c", Some(c |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let polyline ~off ~scl () =
   Py.Module.get_function_with_keywords __wrap_namespace "polyline"
     [||]
     (Wrap_utils.keyword_args [("off", Some(off )); ("scl", Some(scl ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let polymul ~c1 ~c2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "polymul"
     [||]
     (Wrap_utils.keyword_args [("c1", Some(c1 )); ("c2", Some(c2 ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let polymulx c =
   Py.Module.get_function_with_keywords __wrap_namespace "polymulx"
     [||]
     (Wrap_utils.keyword_args [("c", Some(c |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let polypow ?maxpower ~c ~pow () =
   Py.Module.get_function_with_keywords __wrap_namespace "polypow"
     [||]
     (Wrap_utils.keyword_args [("maxpower", Wrap_utils.Option.map maxpower Py.Int.of_int); ("c", Some(c |> Obj.to_pyobject)); ("pow", Some(pow |> Py.Int.of_int))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let polyroots c =
   Py.Module.get_function_with_keywords __wrap_namespace "polyroots"
     [||]
     (Wrap_utils.keyword_args [("c", Some(c ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let polysub ~c1 ~c2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "polysub"
     [||]
     (Wrap_utils.keyword_args [("c1", Some(c1 )); ("c2", Some(c2 ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let polytrim ?tol ~c () =
                     Py.Module.get_function_with_keywords __wrap_namespace "polytrim"
                       [||]
                       (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("c", Some(c |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let polyval ?tensor ~c x =
                     Py.Module.get_function_with_keywords __wrap_namespace "polyval"
                       [||]
                       (Wrap_utils.keyword_args [("tensor", Wrap_utils.Option.map tensor Py.Bool.of_bool); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x |> (function
| `Compatible_object x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)))])

let polyval2d ~y ~c x =
   Py.Module.get_function_with_keywords __wrap_namespace "polyval2d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x ))])

let polyval3d ~y ~z ~c x =
   Py.Module.get_function_with_keywords __wrap_namespace "polyval3d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("z", Some(z )); ("c", Some(c |> Obj.to_pyobject)); ("x", Some(x ))])

                  let polyvalfromroots ?tensor ~r x =
                     Py.Module.get_function_with_keywords __wrap_namespace "polyvalfromroots"
                       [||]
                       (Wrap_utils.keyword_args [("tensor", Wrap_utils.Option.map tensor Py.Bool.of_bool); ("r", Some(r |> Obj.to_pyobject)); ("x", Some(x |> (function
| `Compatible_object x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)))])

let polyvander ~deg x =
   Py.Module.get_function_with_keywords __wrap_namespace "polyvander"
     [||]
     (Wrap_utils.keyword_args [("deg", Some(deg |> Py.Int.of_int)); ("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let polyvander2d ~y ~deg x =
   Py.Module.get_function_with_keywords __wrap_namespace "polyvander2d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("deg", Some(deg )); ("x", Some(x ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let polyvander3d ~y ~z ~deg x =
   Py.Module.get_function_with_keywords __wrap_namespace "polyvander3d"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y )); ("z", Some(z )); ("deg", Some(deg )); ("x", Some(x ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))

end
module Polyutils = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "numpy.polynomial.polyutils"

let get_py name = Py.Module.get __wrap_namespace name
module PolyBase = struct
type tag = [`PolyBase]
type t = [`Object | `PolyBase] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "PolyBase"
     [||]
     []
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module PolyDomainError = struct
type tag = [`PolyDomainError]
type t = [`BaseException | `Object | `PolyDomainError] Obj.t
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
module PolyError = struct
type tag = [`PolyError]
type t = [`BaseException | `Object | `PolyError] Obj.t
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
module RankWarning = struct
type tag = [`RankWarning]
type t = [`BaseException | `Object | `RankWarning] Obj.t
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
let as_series ?trim ~alist () =
   Py.Module.get_function_with_keywords __wrap_namespace "as_series"
     [||]
     (Wrap_utils.keyword_args [("trim", Wrap_utils.Option.map trim Py.Bool.of_bool); ("alist", Some(alist |> Obj.to_pyobject))])

let getdomain x =
   Py.Module.get_function_with_keywords __wrap_namespace "getdomain"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let mapdomain ~old ~new_ x =
   Py.Module.get_function_with_keywords __wrap_namespace "mapdomain"
     [||]
     (Wrap_utils.keyword_args [("old", Some(old )); ("new", Some(new_ )); ("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let mapparms ~old ~new_ () =
   Py.Module.get_function_with_keywords __wrap_namespace "mapparms"
     [||]
     (Wrap_utils.keyword_args [("old", Some(old )); ("new", Some(new_ ))])

                  let trimcoef ?tol ~c () =
                     Py.Module.get_function_with_keywords __wrap_namespace "trimcoef"
                       [||]
                       (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("c", Some(c |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let trimseq seq =
   Py.Module.get_function_with_keywords __wrap_namespace "trimseq"
     [||]
     (Wrap_utils.keyword_args [("seq", Some(seq ))])


end

end
module Random = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "numpy.random"

let get_py name = Py.Module.get __wrap_namespace name
module BitGenerator = struct
type tag = [`BitGenerator]
type t = [`BitGenerator | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?seed () =
                     Py.Module.get_function_with_keywords __wrap_namespace "BitGenerator"
                       [||]
                       (Wrap_utils.keyword_args [("seed", Wrap_utils.Option.map seed (function
| `I x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
))])
                       |> of_pyobject
let random_raw ?size self =
   Py.Module.get_function_with_keywords (to_pyobject self) "random_raw"
     [||]
     (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])


let lock_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "lock" with
  | None -> failwith "attribute lock not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let lock self = match lock_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Generator = struct
type tag = [`Generator]
type t = [`Generator | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create bit_generator =
   Py.Module.get_function_with_keywords __wrap_namespace "Generator"
     [||]
     (Wrap_utils.keyword_args [("bit_generator", Some(bit_generator ))])
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MT19937 = struct
type tag = [`MT19937]
type t = [`MT19937 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?seed () =
                     Py.Module.get_function_with_keywords __wrap_namespace "MT19937"
                       [||]
                       (Wrap_utils.keyword_args [("seed", Wrap_utils.Option.map seed (function
| `I x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
))])
                       |> of_pyobject
                  let jumped ?jumps self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "jumped"
                       [||]
                       (Wrap_utils.keyword_args [("jumps", Wrap_utils.Option.map jumps (function
| `Positive x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))])


let lock_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "lock" with
  | None -> failwith "attribute lock not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let lock self = match lock_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module PCG64 = struct
type tag = [`PCG64]
type t = [`Object | `PCG64] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?seed () =
                     Py.Module.get_function_with_keywords __wrap_namespace "PCG64"
                       [||]
                       (Wrap_utils.keyword_args [("seed", Wrap_utils.Option.map seed (function
| `I x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
))])
                       |> of_pyobject
                  let advance ~delta self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "advance"
                       [||]
                       (Wrap_utils.keyword_args [("delta", Some(delta |> (function
| `Positive x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)))])
                       |> of_pyobject
                  let jumped ?jumps self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "jumped"
                       [||]
                       (Wrap_utils.keyword_args [("jumps", Wrap_utils.Option.map jumps (function
| `Positive x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Philox = struct
type tag = [`Philox]
type t = [`Object | `Philox] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?seed ?counter ?key () =
                     Py.Module.get_function_with_keywords __wrap_namespace "Philox"
                       [||]
                       (Wrap_utils.keyword_args [("seed", Wrap_utils.Option.map seed (function
| `I x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)); ("counter", Wrap_utils.Option.map counter (function
| `Ndarray x -> Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("key", Wrap_utils.Option.map key (function
| `Ndarray x -> Obj.to_pyobject x
| `I x -> Py.Int.of_int x
))])
                       |> of_pyobject
                  let advance ~delta self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "advance"
                       [||]
                       (Wrap_utils.keyword_args [("delta", Some(delta |> (function
| `Positive x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)))])
                       |> of_pyobject
                  let jumped ?jumps self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "jumped"
                       [||]
                       (Wrap_utils.keyword_args [("jumps", Wrap_utils.Option.map jumps (function
| `Positive x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))])


let lock_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "lock" with
  | None -> failwith "attribute lock not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let lock self = match lock_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RandomState = struct
type tag = [`RandomState]
type t = [`Object | `RandomState] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?seed () =
                     Py.Module.get_function_with_keywords __wrap_namespace "RandomState"
                       [||]
                       (Wrap_utils.keyword_args [("seed", Wrap_utils.Option.map seed (function
| `BitGenerator x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
| `I x -> Py.Int.of_int x
))])
                       |> of_pyobject
                  let beta ?size ~b a self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "beta"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("b", Some(b |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("a", Some(a |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let binomial ?size ~n ~p self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "binomial"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("n", Some(n |> (function
| `Array_like_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("p", Some(p |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let bytes ~length self =
   Py.Module.get_function_with_keywords (to_pyobject self) "bytes"
     [||]
     (Wrap_utils.keyword_args [("length", Some(length |> Py.Int.of_int))])
     |> Py.String.to_string
                  let chisquare ?size ~df self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "chisquare"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("df", Some(df |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let choice ?size ?replace ?p a self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "choice"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("replace", Wrap_utils.Option.map replace Py.Bool.of_bool); ("p", p); ("a", Some(a |> (function
| `T1_D_array_like x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)))])

                  let dirichlet ?size ~alpha self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "dirichlet"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("alpha", Some(alpha |> (function
| `Ndarray x -> Obj.to_pyobject x
| `Length_k x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let exponential ?scale ?size self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "exponential"
                       [||]
                       (Wrap_utils.keyword_args [("scale", Wrap_utils.Option.map scale (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let f ?size ~dfnum ~dfden self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "f"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("dfnum", Some(dfnum |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("dfden", Some(dfden |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let gamma ?scale ?size shape self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "gamma"
                       [||]
                       (Wrap_utils.keyword_args [("scale", Wrap_utils.Option.map scale (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let geometric ?size ~p self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "geometric"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("p", Some(p |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let gumbel ?loc ?scale ?size self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "gumbel"
                       [||]
                       (Wrap_utils.keyword_args [("loc", Wrap_utils.Option.map loc (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("scale", Wrap_utils.Option.map scale (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let hypergeometric ?size ~ngood ~nbad ~nsample self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "hypergeometric"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("ngood", Some(ngood |> (function
| `Array_like_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("nbad", Some(nbad |> (function
| `Array_like_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("nsample", Some(nsample |> (function
| `Array_like_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let laplace ?loc ?scale ?size self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "laplace"
                       [||]
                       (Wrap_utils.keyword_args [("loc", Wrap_utils.Option.map loc (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("scale", Wrap_utils.Option.map scale (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let logistic ?loc ?scale ?size self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "logistic"
                       [||]
                       (Wrap_utils.keyword_args [("loc", Wrap_utils.Option.map loc (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("scale", Wrap_utils.Option.map scale (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let lognormal ?mean ?sigma ?size self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "lognormal"
                       [||]
                       (Wrap_utils.keyword_args [("mean", Wrap_utils.Option.map mean (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("sigma", Wrap_utils.Option.map sigma (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let logseries ?size ~p self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "logseries"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("p", Some(p |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let multinomial ?size ~n ~pvals self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "multinomial"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("n", Some(n |> Py.Int.of_int)); ("pvals", Some(pvals |> (function
| `Ndarray x -> Obj.to_pyobject x
| `Length_p x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let multivariate_normal ?size ?check_valid ?tol ~mean ~cov self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "multivariate_normal"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("check_valid", Wrap_utils.Option.map check_valid (function
| `Warn -> Py.String.of_string "warn"
| `Raise -> Py.String.of_string "raise"
| `Ignore -> Py.String.of_string "ignore"
)); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("mean", Some(mean )); ("cov", Some(cov ))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let negative_binomial ?size ~n ~p self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "negative_binomial"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("n", Some(n |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("p", Some(p |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let noncentral_chisquare ?size ~df ~nonc self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "noncentral_chisquare"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("df", Some(df |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("nonc", Some(nonc |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let noncentral_f ?size ~dfnum ~dfden ~nonc self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "noncentral_f"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("dfnum", Some(dfnum |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("dfden", Some(dfden |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("nonc", Some(nonc |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let normal ?loc ?scale ?size self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "normal"
                       [||]
                       (Wrap_utils.keyword_args [("loc", Wrap_utils.Option.map loc (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("scale", Wrap_utils.Option.map scale (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let pareto ?size a self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "pareto"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("a", Some(a |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let permutation x self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "permutation"
                       [||]
                       (Wrap_utils.keyword_args [("x", Some(x |> (function
| `Ndarray x -> Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let poisson ?lam ?size self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "poisson"
                       [||]
                       (Wrap_utils.keyword_args [("lam", Wrap_utils.Option.map lam (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let power ?size a self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "power"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("a", Some(a |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let rand ~d self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rand"
     [||]
     (Wrap_utils.keyword_args [("d", Some(d ))])

                  let randint ?high ?size ?dtype ~low self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "randint"
                       [||]
                       (Wrap_utils.keyword_args [("high", Wrap_utils.Option.map high (function
| `Array_like_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("low", Some(low |> (function
| `Array_like_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)))])

let randn ~d self =
   Py.Module.get_function_with_keywords (to_pyobject self) "randn"
     [||]
     (Wrap_utils.keyword_args [("d", Some(d ))])

let random ?size self =
   Py.Module.get_function_with_keywords (to_pyobject self) "random"
     [||]
     (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])

let random_integers ?high ?size ~low self =
   Py.Module.get_function_with_keywords (to_pyobject self) "random_integers"
     [||]
     (Wrap_utils.keyword_args [("high", Wrap_utils.Option.map high Py.Int.of_int); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("low", Some(low |> Py.Int.of_int))])

let random_sample ?size self =
   Py.Module.get_function_with_keywords (to_pyobject self) "random_sample"
     [||]
     (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])

                  let rayleigh ?scale ?size self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "rayleigh"
                       [||]
                       (Wrap_utils.keyword_args [("scale", Wrap_utils.Option.map scale (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let set_state ~state self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_state"
     [||]
     (Wrap_utils.keyword_args [("state", Some(state ))])

let shuffle x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "shuffle"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Obj.to_pyobject))])

let standard_cauchy ?size self =
   Py.Module.get_function_with_keywords (to_pyobject self) "standard_cauchy"
     [||]
     (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let standard_exponential ?size self =
   Py.Module.get_function_with_keywords (to_pyobject self) "standard_exponential"
     [||]
     (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])

let standard_gamma ?size shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "standard_gamma"
     [||]
     (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let standard_normal ?size self =
   Py.Module.get_function_with_keywords (to_pyobject self) "standard_normal"
     [||]
     (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])

                  let standard_t ?size ~df self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "standard_t"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("df", Some(df |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let tomaxint ?size self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tomaxint"
     [||]
     (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let triangular ?size ~left ~mode ~right self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "triangular"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("left", Some(left |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("mode", Some(mode |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("right", Some(right |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let uniform ?low ?high ?size self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "uniform"
                       [||]
                       (Wrap_utils.keyword_args [("low", Wrap_utils.Option.map low (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("high", Wrap_utils.Option.map high (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let vonmises ?size ~mu ~kappa self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "vonmises"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("mu", Some(mu |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("kappa", Some(kappa |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let wald ?size ~mean ~scale self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "wald"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("mean", Some(mean |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("scale", Some(scale |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let weibull ?size a self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "weibull"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("a", Some(a |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let zipf ?size a self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "zipf"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("a", Some(a |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SFC64 = struct
type tag = [`SFC64]
type t = [`Object | `SFC64] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?seed () =
                     Py.Module.get_function_with_keywords __wrap_namespace "SFC64"
                       [||]
                       (Wrap_utils.keyword_args [("seed", Wrap_utils.Option.map seed (function
| `I x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
))])
                       |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SeedSequence = struct
type tag = [`SeedSequence]
type t = [`Object | `SeedSequence] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?entropy ?spawn_key ?pool_size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "SeedSequence"
                       [||]
                       (Wrap_utils.keyword_args [("entropy", Wrap_utils.Option.map entropy (function
| `Sequence_int_ x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("spawn_key", spawn_key); ("pool_size", pool_size)])
                       |> of_pyobject
let generate_state ?dtype ~n_words self =
   Py.Module.get_function_with_keywords (to_pyobject self) "generate_state"
     [||]
     (Wrap_utils.keyword_args [("dtype", dtype); ("n_words", Some(n_words |> Py.Int.of_int))])

let spawn ~n_children self =
   Py.Module.get_function_with_keywords (to_pyobject self) "spawn"
     [||]
     (Wrap_utils.keyword_args [("n_children", Some(n_children |> Py.Int.of_int))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Bit_generator = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "numpy.random.bit_generator"

let get_py name = Py.Module.get __wrap_namespace name
module ISeedSequence = struct
type tag = [`ISeedSequence]
type t = [`ISeedSequence | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let generate_state ?dtype ~n_words self =
   Py.Module.get_function_with_keywords (to_pyobject self) "generate_state"
     [||]
     (Wrap_utils.keyword_args [("dtype", dtype); ("n_words", Some(n_words |> Py.Int.of_int))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ISpawnableSeedSequence = struct
type tag = [`ISpawnableSeedSequence]
type t = [`ISpawnableSeedSequence | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let generate_state ?dtype ~n_words self =
   Py.Module.get_function_with_keywords (to_pyobject self) "generate_state"
     [||]
     (Wrap_utils.keyword_args [("dtype", dtype); ("n_words", Some(n_words |> Py.Int.of_int))])

let spawn ~n_children self =
   Py.Module.get_function_with_keywords (to_pyobject self) "spawn"
     [||]
     (Wrap_utils.keyword_args [("n_children", Some(n_children |> Py.Int.of_int))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SeedlessSeedSequence = struct
type tag = [`SeedlessSeedSequence]
type t = [`Object | `SeedlessSeedSequence] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SeedlessSequence = struct
type tag = [`SeedlessSequence]
type t = [`Object | `SeedlessSequence] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Cycle = struct
type tag = [`Cycle]
type t = [`Cycle | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create iterable =
   Py.Module.get_function_with_keywords __wrap_namespace "cycle"
     (Array.of_list @@ List.concat [[iterable ]])
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
let randbits k =
   Py.Module.get_function_with_keywords __wrap_namespace "randbits"
     [||]
     (Wrap_utils.keyword_args [("k", Some(k ))])


end
module Mtrand = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "numpy.random.mtrand"

let get_py name = Py.Module.get __wrap_namespace name
                  let beta ?size ~b a =
                     Py.Module.get_function_with_keywords __wrap_namespace "beta"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("b", Some(b |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("a", Some(a |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let binomial ?size ~n ~p () =
                     Py.Module.get_function_with_keywords __wrap_namespace "binomial"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("n", Some(n |> (function
| `Array_like_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("p", Some(p |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let bytes length =
   Py.Module.get_function_with_keywords __wrap_namespace "bytes"
     [||]
     (Wrap_utils.keyword_args [("length", Some(length |> Py.Int.of_int))])
     |> Py.String.to_string
                  let chisquare ?size ~df () =
                     Py.Module.get_function_with_keywords __wrap_namespace "chisquare"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("df", Some(df |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let choice ?size ?replace ?p a =
                     Py.Module.get_function_with_keywords __wrap_namespace "choice"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("replace", Wrap_utils.Option.map replace Py.Bool.of_bool); ("p", p); ("a", Some(a |> (function
| `T1_D_array_like x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)))])

                  let dirichlet ?size ~alpha () =
                     Py.Module.get_function_with_keywords __wrap_namespace "dirichlet"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("alpha", Some(alpha |> (function
| `Ndarray x -> Obj.to_pyobject x
| `Length_k x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let exponential ?scale ?size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "exponential"
                       [||]
                       (Wrap_utils.keyword_args [("scale", Wrap_utils.Option.map scale (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let f ?size ~dfnum ~dfden () =
                     Py.Module.get_function_with_keywords __wrap_namespace "f"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("dfnum", Some(dfnum |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("dfden", Some(dfden |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let gamma ?scale ?size shape =
                     Py.Module.get_function_with_keywords __wrap_namespace "gamma"
                       [||]
                       (Wrap_utils.keyword_args [("scale", Wrap_utils.Option.map scale (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let geometric ?size ~p () =
                     Py.Module.get_function_with_keywords __wrap_namespace "geometric"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("p", Some(p |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let gumbel ?loc ?scale ?size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "gumbel"
                       [||]
                       (Wrap_utils.keyword_args [("loc", Wrap_utils.Option.map loc (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("scale", Wrap_utils.Option.map scale (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let hypergeometric ?size ~ngood ~nbad ~nsample () =
                     Py.Module.get_function_with_keywords __wrap_namespace "hypergeometric"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("ngood", Some(ngood |> (function
| `Array_like_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("nbad", Some(nbad |> (function
| `Array_like_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("nsample", Some(nsample |> (function
| `Array_like_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let laplace ?loc ?scale ?size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "laplace"
                       [||]
                       (Wrap_utils.keyword_args [("loc", Wrap_utils.Option.map loc (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("scale", Wrap_utils.Option.map scale (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let logistic ?loc ?scale ?size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "logistic"
                       [||]
                       (Wrap_utils.keyword_args [("loc", Wrap_utils.Option.map loc (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("scale", Wrap_utils.Option.map scale (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let lognormal ?mean ?sigma ?size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "lognormal"
                       [||]
                       (Wrap_utils.keyword_args [("mean", Wrap_utils.Option.map mean (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("sigma", Wrap_utils.Option.map sigma (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let logseries ?size ~p () =
                     Py.Module.get_function_with_keywords __wrap_namespace "logseries"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("p", Some(p |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let multinomial ?size ~n ~pvals () =
                     Py.Module.get_function_with_keywords __wrap_namespace "multinomial"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("n", Some(n |> Py.Int.of_int)); ("pvals", Some(pvals |> (function
| `Ndarray x -> Obj.to_pyobject x
| `Length_p x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let multivariate_normal ?size ?check_valid ?tol ~mean ~cov () =
                     Py.Module.get_function_with_keywords __wrap_namespace "multivariate_normal"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("check_valid", Wrap_utils.Option.map check_valid (function
| `Warn -> Py.String.of_string "warn"
| `Raise -> Py.String.of_string "raise"
| `Ignore -> Py.String.of_string "ignore"
)); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("mean", Some(mean )); ("cov", Some(cov ))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let negative_binomial ?size ~n ~p () =
                     Py.Module.get_function_with_keywords __wrap_namespace "negative_binomial"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("n", Some(n |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("p", Some(p |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let noncentral_chisquare ?size ~df ~nonc () =
                     Py.Module.get_function_with_keywords __wrap_namespace "noncentral_chisquare"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("df", Some(df |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("nonc", Some(nonc |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let noncentral_f ?size ~dfnum ~dfden ~nonc () =
                     Py.Module.get_function_with_keywords __wrap_namespace "noncentral_f"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("dfnum", Some(dfnum |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("dfden", Some(dfden |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("nonc", Some(nonc |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let normal ?loc ?scale ?size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "normal"
                       [||]
                       (Wrap_utils.keyword_args [("loc", Wrap_utils.Option.map loc (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("scale", Wrap_utils.Option.map scale (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let pareto ?size a =
                     Py.Module.get_function_with_keywords __wrap_namespace "pareto"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("a", Some(a |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let permutation x =
                     Py.Module.get_function_with_keywords __wrap_namespace "permutation"
                       [||]
                       (Wrap_utils.keyword_args [("x", Some(x |> (function
| `Ndarray x -> Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let poisson ?lam ?size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "poisson"
                       [||]
                       (Wrap_utils.keyword_args [("lam", Wrap_utils.Option.map lam (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let power ?size a =
                     Py.Module.get_function_with_keywords __wrap_namespace "power"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("a", Some(a |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let rand d =
   Py.Module.get_function_with_keywords __wrap_namespace "rand"
     [||]
     (Wrap_utils.keyword_args [("d", Some(d ))])

                  let randint ?high ?size ?dtype ~low () =
                     Py.Module.get_function_with_keywords __wrap_namespace "randint"
                       [||]
                       (Wrap_utils.keyword_args [("high", Wrap_utils.Option.map high (function
| `Array_like_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("low", Some(low |> (function
| `Array_like_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)))])

let randn d =
   Py.Module.get_function_with_keywords __wrap_namespace "randn"
     [||]
     (Wrap_utils.keyword_args [("d", Some(d ))])

let random ?size () =
   Py.Module.get_function_with_keywords __wrap_namespace "random"
     [||]
     (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])

let random_integers ?high ?size ~low () =
   Py.Module.get_function_with_keywords __wrap_namespace "random_integers"
     [||]
     (Wrap_utils.keyword_args [("high", Wrap_utils.Option.map high Py.Int.of_int); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("low", Some(low |> Py.Int.of_int))])

let random_sample ?size () =
   Py.Module.get_function_with_keywords __wrap_namespace "random_sample"
     [||]
     (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])

                  let rayleigh ?scale ?size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "rayleigh"
                       [||]
                       (Wrap_utils.keyword_args [("scale", Wrap_utils.Option.map scale (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let set_state state =
   Py.Module.get_function_with_keywords __wrap_namespace "set_state"
     [||]
     (Wrap_utils.keyword_args [("state", Some(state ))])

let shuffle x =
   Py.Module.get_function_with_keywords __wrap_namespace "shuffle"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Obj.to_pyobject))])

let standard_cauchy ?size () =
   Py.Module.get_function_with_keywords __wrap_namespace "standard_cauchy"
     [||]
     (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let standard_exponential ?size () =
   Py.Module.get_function_with_keywords __wrap_namespace "standard_exponential"
     [||]
     (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])

let standard_gamma ?size shape =
   Py.Module.get_function_with_keywords __wrap_namespace "standard_gamma"
     [||]
     (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let standard_normal ?size () =
   Py.Module.get_function_with_keywords __wrap_namespace "standard_normal"
     [||]
     (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])

                  let standard_t ?size ~df () =
                     Py.Module.get_function_with_keywords __wrap_namespace "standard_t"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("df", Some(df |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let triangular ?size ~left ~mode ~right () =
                     Py.Module.get_function_with_keywords __wrap_namespace "triangular"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("left", Some(left |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("mode", Some(mode |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("right", Some(right |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let uniform ?low ?high ?size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "uniform"
                       [||]
                       (Wrap_utils.keyword_args [("low", Wrap_utils.Option.map low (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("high", Wrap_utils.Option.map high (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let vonmises ?size ~mu ~kappa () =
                     Py.Module.get_function_with_keywords __wrap_namespace "vonmises"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("mu", Some(mu |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("kappa", Some(kappa |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let wald ?size ~mean ~scale () =
                     Py.Module.get_function_with_keywords __wrap_namespace "wald"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("mean", Some(mean |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("scale", Some(scale |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let weibull ?size a =
                     Py.Module.get_function_with_keywords __wrap_namespace "weibull"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("a", Some(a |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let zipf ?size a =
                     Py.Module.get_function_with_keywords __wrap_namespace "zipf"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("a", Some(a |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))

end
                  let beta ?size ~b a =
                     Py.Module.get_function_with_keywords __wrap_namespace "beta"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("b", Some(b |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("a", Some(a |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let binomial ?size ~n ~p () =
                     Py.Module.get_function_with_keywords __wrap_namespace "binomial"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("n", Some(n |> (function
| `Array_like_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("p", Some(p |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let bytes length =
   Py.Module.get_function_with_keywords __wrap_namespace "bytes"
     [||]
     (Wrap_utils.keyword_args [("length", Some(length |> Py.Int.of_int))])
     |> Py.String.to_string
                  let chisquare ?size ~df () =
                     Py.Module.get_function_with_keywords __wrap_namespace "chisquare"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("df", Some(df |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let choice ?size ?replace ?p a =
                     Py.Module.get_function_with_keywords __wrap_namespace "choice"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("replace", Wrap_utils.Option.map replace Py.Bool.of_bool); ("p", p); ("a", Some(a |> (function
| `T1_D_array_like x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)))])

                  let dirichlet ?size ~alpha () =
                     Py.Module.get_function_with_keywords __wrap_namespace "dirichlet"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("alpha", Some(alpha |> (function
| `Ndarray x -> Obj.to_pyobject x
| `Length_k x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let exponential ?scale ?size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "exponential"
                       [||]
                       (Wrap_utils.keyword_args [("scale", Wrap_utils.Option.map scale (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let f ?size ~dfnum ~dfden () =
                     Py.Module.get_function_with_keywords __wrap_namespace "f"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("dfnum", Some(dfnum |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("dfden", Some(dfden |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let gamma ?scale ?size shape =
                     Py.Module.get_function_with_keywords __wrap_namespace "gamma"
                       [||]
                       (Wrap_utils.keyword_args [("scale", Wrap_utils.Option.map scale (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let geometric ?size ~p () =
                     Py.Module.get_function_with_keywords __wrap_namespace "geometric"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("p", Some(p |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let gumbel ?loc ?scale ?size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "gumbel"
                       [||]
                       (Wrap_utils.keyword_args [("loc", Wrap_utils.Option.map loc (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("scale", Wrap_utils.Option.map scale (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let hypergeometric ?size ~ngood ~nbad ~nsample () =
                     Py.Module.get_function_with_keywords __wrap_namespace "hypergeometric"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("ngood", Some(ngood |> (function
| `Array_like_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("nbad", Some(nbad |> (function
| `Array_like_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("nsample", Some(nsample |> (function
| `Array_like_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let laplace ?loc ?scale ?size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "laplace"
                       [||]
                       (Wrap_utils.keyword_args [("loc", Wrap_utils.Option.map loc (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("scale", Wrap_utils.Option.map scale (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let logistic ?loc ?scale ?size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "logistic"
                       [||]
                       (Wrap_utils.keyword_args [("loc", Wrap_utils.Option.map loc (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("scale", Wrap_utils.Option.map scale (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let lognormal ?mean ?sigma ?size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "lognormal"
                       [||]
                       (Wrap_utils.keyword_args [("mean", Wrap_utils.Option.map mean (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("sigma", Wrap_utils.Option.map sigma (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let logseries ?size ~p () =
                     Py.Module.get_function_with_keywords __wrap_namespace "logseries"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("p", Some(p |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let multinomial ?size ~n ~pvals () =
                     Py.Module.get_function_with_keywords __wrap_namespace "multinomial"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("n", Some(n |> Py.Int.of_int)); ("pvals", Some(pvals |> (function
| `Ndarray x -> Obj.to_pyobject x
| `Length_p x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let multivariate_normal ?size ?check_valid ?tol ~mean ~cov () =
                     Py.Module.get_function_with_keywords __wrap_namespace "multivariate_normal"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("check_valid", Wrap_utils.Option.map check_valid (function
| `Warn -> Py.String.of_string "warn"
| `Raise -> Py.String.of_string "raise"
| `Ignore -> Py.String.of_string "ignore"
)); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("mean", Some(mean )); ("cov", Some(cov ))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let negative_binomial ?size ~n ~p () =
                     Py.Module.get_function_with_keywords __wrap_namespace "negative_binomial"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("n", Some(n |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("p", Some(p |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let noncentral_chisquare ?size ~df ~nonc () =
                     Py.Module.get_function_with_keywords __wrap_namespace "noncentral_chisquare"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("df", Some(df |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("nonc", Some(nonc |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let noncentral_f ?size ~dfnum ~dfden ~nonc () =
                     Py.Module.get_function_with_keywords __wrap_namespace "noncentral_f"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("dfnum", Some(dfnum |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("dfden", Some(dfden |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("nonc", Some(nonc |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let normal ?loc ?scale ?size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "normal"
                       [||]
                       (Wrap_utils.keyword_args [("loc", Wrap_utils.Option.map loc (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("scale", Wrap_utils.Option.map scale (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let pareto ?size a =
                     Py.Module.get_function_with_keywords __wrap_namespace "pareto"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("a", Some(a |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let permutation x =
                     Py.Module.get_function_with_keywords __wrap_namespace "permutation"
                       [||]
                       (Wrap_utils.keyword_args [("x", Some(x |> (function
| `Ndarray x -> Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let poisson ?lam ?size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "poisson"
                       [||]
                       (Wrap_utils.keyword_args [("lam", Wrap_utils.Option.map lam (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let power ?size a =
                     Py.Module.get_function_with_keywords __wrap_namespace "power"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("a", Some(a |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let rand d =
   Py.Module.get_function_with_keywords __wrap_namespace "rand"
     [||]
     (Wrap_utils.keyword_args [("d", Some(d ))])

                  let randint ?high ?size ?dtype ~low () =
                     Py.Module.get_function_with_keywords __wrap_namespace "randint"
                       [||]
                       (Wrap_utils.keyword_args [("high", Wrap_utils.Option.map high (function
| `Array_like_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("low", Some(low |> (function
| `Array_like_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)))])

let randn d =
   Py.Module.get_function_with_keywords __wrap_namespace "randn"
     (Array.of_list @@ List.concat [(List.map Py.Int.of_int d)])
     []
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let random ?size () =
   Py.Module.get_function_with_keywords __wrap_namespace "random"
     [||]
     (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])

let random_integers ?high ?size ~low () =
   Py.Module.get_function_with_keywords __wrap_namespace "random_integers"
     [||]
     (Wrap_utils.keyword_args [("high", Wrap_utils.Option.map high Py.Int.of_int); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("low", Some(low |> Py.Int.of_int))])

let random_sample ?size () =
   Py.Module.get_function_with_keywords __wrap_namespace "random_sample"
     [||]
     (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let rayleigh ?scale ?size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "rayleigh"
                       [||]
                       (Wrap_utils.keyword_args [("scale", Wrap_utils.Option.map scale (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let seed seed =
   Py.Module.get_function_with_keywords __wrap_namespace "seed"
     [||]
     (Wrap_utils.keyword_args [("seed", Some(seed |> Py.Int.of_int))])
     |> (fun _ -> ())
let set_state state =
   Py.Module.get_function_with_keywords __wrap_namespace "set_state"
     [||]
     (Wrap_utils.keyword_args [("state", Some(state ))])

let shuffle x =
   Py.Module.get_function_with_keywords __wrap_namespace "shuffle"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Obj.to_pyobject))])

let standard_cauchy ?size () =
   Py.Module.get_function_with_keywords __wrap_namespace "standard_cauchy"
     [||]
     (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let standard_exponential ?size () =
   Py.Module.get_function_with_keywords __wrap_namespace "standard_exponential"
     [||]
     (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])

let standard_gamma ?size shape =
   Py.Module.get_function_with_keywords __wrap_namespace "standard_gamma"
     [||]
     (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let standard_normal ?size () =
   Py.Module.get_function_with_keywords __wrap_namespace "standard_normal"
     [||]
     (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])

                  let standard_t ?size ~df () =
                     Py.Module.get_function_with_keywords __wrap_namespace "standard_t"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("df", Some(df |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let triangular ?size ~left ~mode ~right () =
                     Py.Module.get_function_with_keywords __wrap_namespace "triangular"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("left", Some(left |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("mode", Some(mode |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("right", Some(right |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let uniform ?low ?high ?size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "uniform"
                       [||]
                       (Wrap_utils.keyword_args [("low", Wrap_utils.Option.map low (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("high", Wrap_utils.Option.map high (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)); ("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let vonmises ?size ~mu ~kappa () =
                     Py.Module.get_function_with_keywords __wrap_namespace "vonmises"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("mu", Some(mu |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("kappa", Some(kappa |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let wald ?size ~mean ~scale () =
                     Py.Module.get_function_with_keywords __wrap_namespace "wald"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("mean", Some(mean |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
))); ("scale", Some(scale |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let weibull ?size a =
                     Py.Module.get_function_with_keywords __wrap_namespace "weibull"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("a", Some(a |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let zipf ?size a =
                     Py.Module.get_function_with_keywords __wrap_namespace "zipf"
                       [||]
                       (Wrap_utils.keyword_args [("size", Wrap_utils.Option.map size (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("a", Some(a |> (function
| `Ndarray x -> Obj.to_pyobject x
| `F x -> Py.Float.of_float x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))

end
module Version = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "numpy.version"

let get_py name = Py.Module.get __wrap_namespace name

end
                  let abs ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "abs"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let absolute ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "absolute"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let add ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "add"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let add_docstring ~obj ~docstring () =
   Py.Module.get_function_with_keywords __wrap_namespace "add_docstring"
     [||]
     (Wrap_utils.keyword_args [("obj", Some(obj )); ("docstring", Some(docstring ))])

                  let add_newdoc ?warn_on_python ~place ~obj ~doc () =
                     Py.Module.get_function_with_keywords __wrap_namespace "add_newdoc"
                       [||]
                       (Wrap_utils.keyword_args [("warn_on_python", Wrap_utils.Option.map warn_on_python Py.Bool.of_bool); ("place", Some(place |> Py.String.of_string)); ("obj", Some(obj |> Py.String.of_string)); ("doc", Some(doc |> (function
| `S x -> Py.String.of_string x
| `PyObject x -> Wrap_utils.id x
)))])

let add_newdoc_ufunc ~ufunc ~new_docstring () =
   Py.Module.get_function_with_keywords __wrap_namespace "add_newdoc_ufunc"
     [||]
     (Wrap_utils.keyword_args [("ufunc", Some(ufunc )); ("new_docstring", Some(new_docstring |> Py.String.of_string))])

let alen a =
   Py.Module.get_function_with_keywords __wrap_namespace "alen"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Obj.to_pyobject))])
     |> Py.Int.to_int
let all ?axis ?out ?keepdims a =
   Py.Module.get_function_with_keywords __wrap_namespace "all"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let allclose ?rtol ?atol ?equal_nan ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "allclose"
     [||]
     (Wrap_utils.keyword_args [("rtol", Wrap_utils.Option.map rtol Py.Float.of_float); ("atol", Wrap_utils.Option.map atol Py.Float.of_float); ("equal_nan", Wrap_utils.Option.map equal_nan Py.Bool.of_bool); ("b", Some(b )); ("a", Some(a ))])
     |> Py.Bool.to_bool
let alltrue ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "alltrue"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

                  let amax ?axis ?out ?keepdims ?initial ?where a =
                     Py.Module.get_function_with_keywords __wrap_namespace "amax"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("initial", Wrap_utils.Option.map initial (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("where", where); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let amin ?axis ?out ?keepdims ?initial ?where a =
                     Py.Module.get_function_with_keywords __wrap_namespace "amin"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("initial", Wrap_utils.Option.map initial (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("where", where); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let angle ?deg ~z () =
   Py.Module.get_function_with_keywords __wrap_namespace "angle"
     [||]
     (Wrap_utils.keyword_args [("deg", Wrap_utils.Option.map deg Py.Bool.of_bool); ("z", Some(z |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let any ?axis ?out ?keepdims a =
   Py.Module.get_function_with_keywords __wrap_namespace "any"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])

let append ?axis ~arr ~values () =
   Py.Module.get_function_with_keywords __wrap_namespace "append"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("arr", Some(arr |> Obj.to_pyobject)); ("values", Some(values |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let apply_along_axis ?kwargs ~func1d ~axis ~arr args =
   Py.Module.get_function_with_keywords __wrap_namespace "apply_along_axis"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("func1d", Some(func1d )); ("axis", Some(axis |> Py.Int.of_int)); ("arr", Some(arr ))]) (match kwargs with None -> [] | Some x -> x))

let apply_over_axes ~func ~axes a =
   Py.Module.get_function_with_keywords __wrap_namespace "apply_over_axes"
     [||]
     (Wrap_utils.keyword_args [("func", Some(func )); ("axes", Some(axes |> Obj.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let arange ?start ?step ?dtype ~stop () =
                     Py.Module.get_function_with_keywords __wrap_namespace "arange"
                       [||]
                       (Wrap_utils.keyword_args [("start", Wrap_utils.Option.map start (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("step", Wrap_utils.Option.map step (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("stop", Some(stop |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let arccos ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "arccos"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let arccosh ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "arccosh"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let arcsin ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "arcsin"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let arcsinh ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "arcsinh"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let arctan ?out ?where x =
   Py.Module.get_function_with_keywords __wrap_namespace "arctan"
     (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let arctan2 ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "arctan2"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let arctanh ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "arctanh"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let argmax ?axis ?out a =
   Py.Module.get_function_with_keywords __wrap_namespace "argmax"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("a", Some(a |> Obj.to_pyobject))])

let argmin ?axis ?out a =
   Py.Module.get_function_with_keywords __wrap_namespace "argmin"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("a", Some(a |> Obj.to_pyobject))])

                  let argpartition ?axis ?kind ?order ~kth a =
                     Py.Module.get_function_with_keywords __wrap_namespace "argpartition"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("kind", Wrap_utils.Option.map kind (function
| `Introselect -> Py.String.of_string "introselect"
)); ("order", Wrap_utils.Option.map order (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `S x -> Py.String.of_string x
)); ("kth", Some(kth |> (function
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
| `I x -> Py.Int.of_int x
))); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let argsort ?axis ?kind ?order a =
                     Py.Module.get_function_with_keywords __wrap_namespace "argsort"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("kind", Wrap_utils.Option.map kind (function
| `Heapsort -> Py.String.of_string "heapsort"
| `Mergesort -> Py.String.of_string "mergesort"
| `Stable -> Py.String.of_string "stable"
| `Quicksort -> Py.String.of_string "quicksort"
)); ("order", Wrap_utils.Option.map order (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `S x -> Py.String.of_string x
)); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let argwhere a =
   Py.Module.get_function_with_keywords __wrap_namespace "argwhere"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Obj.to_pyobject))])

let around ?decimals ?out a =
   Py.Module.get_function_with_keywords __wrap_namespace "around"
     [||]
     (Wrap_utils.keyword_args [("decimals", Wrap_utils.Option.map decimals Py.Int.of_int); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let array ?dtype ?copy ?order ?subok ?ndmin ~object_ () =
                     Py.Module.get_function_with_keywords __wrap_namespace "array"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("order", Wrap_utils.Option.map order (function
| `K -> Py.String.of_string "K"
| `A -> Py.String.of_string "A"
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("ndmin", Wrap_utils.Option.map ndmin Py.Int.of_int); ("object", Some(object_ |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let array2string ?max_line_width ?precision ?suppress_small ?separator ?prefix ?style ?formatter ?threshold ?edgeitems ?sign ?floatmode ?suffix ?legacy a =
                     Py.Module.get_function_with_keywords __wrap_namespace "array2string"
                       [||]
                       (Wrap_utils.keyword_args [("max_line_width", Wrap_utils.Option.map max_line_width Py.Int.of_int); ("precision", Wrap_utils.Option.map precision Py.Int.of_int); ("suppress_small", Wrap_utils.Option.map suppress_small Py.Bool.of_bool); ("separator", Wrap_utils.Option.map separator Py.String.of_string); ("prefix", Wrap_utils.Option.map prefix Py.String.of_string); ("style", style); ("formatter", formatter); ("threshold", Wrap_utils.Option.map threshold Py.Int.of_int); ("edgeitems", Wrap_utils.Option.map edgeitems Py.Int.of_int); ("sign", Wrap_utils.Option.map sign (function
| `Space -> Py.String.of_string " "
| `Plus -> Py.String.of_string "+"
| `Minus -> Py.String.of_string "-"
)); ("floatmode", Wrap_utils.Option.map floatmode Py.String.of_string); ("suffix", suffix); ("legacy", Wrap_utils.Option.map legacy (function
| `T_False_ x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("a", Some(a |> Obj.to_pyobject))])
                       |> Py.String.to_string
let array_equal ?equal_nan ~a1 ~a2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "array_equal"
     [||]
     (Wrap_utils.keyword_args [("equal_nan", Wrap_utils.Option.map equal_nan Py.Bool.of_bool); ("a1", Some(a1 )); ("a2", Some(a2 ))])
     |> Py.Bool.to_bool
let array_equiv ~a1 ~a2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "array_equiv"
     [||]
     (Wrap_utils.keyword_args [("a1", Some(a1 )); ("a2", Some(a2 ))])
     |> Py.Bool.to_bool
let array_repr ?max_line_width ?precision ?suppress_small ~arr () =
   Py.Module.get_function_with_keywords __wrap_namespace "array_repr"
     [||]
     (Wrap_utils.keyword_args [("max_line_width", Wrap_utils.Option.map max_line_width Py.Int.of_int); ("precision", Wrap_utils.Option.map precision Py.Int.of_int); ("suppress_small", Wrap_utils.Option.map suppress_small Py.Bool.of_bool); ("arr", Some(arr |> Obj.to_pyobject))])
     |> Py.String.to_string
let array_split ?axis ~ary ~indices_or_sections () =
   Py.Module.get_function_with_keywords __wrap_namespace "array_split"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("ary", Some(ary )); ("indices_or_sections", Some(indices_or_sections ))])

let array_str ?max_line_width ?precision ?suppress_small a =
   Py.Module.get_function_with_keywords __wrap_namespace "array_str"
     [||]
     (Wrap_utils.keyword_args [("max_line_width", Wrap_utils.Option.map max_line_width Py.Int.of_int); ("precision", Wrap_utils.Option.map precision Py.Int.of_int); ("suppress_small", Wrap_utils.Option.map suppress_small Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])

                  let asanyarray ?dtype ?order a =
                     Py.Module.get_function_with_keywords __wrap_namespace "asanyarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("a", Some(a |> Obj.to_pyobject))])

                  let asarray ?dtype ?order a =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let asarray_chkfinite ?dtype ?order a =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray_chkfinite"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let ascontiguousarray ?dtype a =
                     Py.Module.get_function_with_keywords __wrap_namespace "ascontiguousarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype (function
| `Dtype_object x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let asfarray ?dtype a =
                     Py.Module.get_function_with_keywords __wrap_namespace "asfarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype (function
| `Dtype_object x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let asfortranarray ?dtype a =
                     Py.Module.get_function_with_keywords __wrap_namespace "asfortranarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype (function
| `Dtype_object x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let asmatrix ?dtype ~data () =
   Py.Module.get_function_with_keywords __wrap_namespace "asmatrix"
     [||]
     (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("data", Some(data |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let asscalar a =
   Py.Module.get_function_with_keywords __wrap_namespace "asscalar"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Obj.to_pyobject))])

let atleast_1d arys =
   Py.Module.get_function_with_keywords __wrap_namespace "atleast_1d"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arys)])
     []
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let atleast_2d arys =
   Py.Module.get_function_with_keywords __wrap_namespace "atleast_2d"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arys)])
     []

let atleast_3d arys =
   Py.Module.get_function_with_keywords __wrap_namespace "atleast_3d"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arys)])
     []

let average ?axis ?weights ?returned a =
   Py.Module.get_function_with_keywords __wrap_namespace "average"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("weights", Wrap_utils.Option.map weights Obj.to_pyobject); ("returned", Wrap_utils.Option.map returned Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])

let bartlett m =
   Py.Module.get_function_with_keywords __wrap_namespace "bartlett"
     [||]
     (Wrap_utils.keyword_args [("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let base_repr ?base ?padding ~number () =
   Py.Module.get_function_with_keywords __wrap_namespace "base_repr"
     [||]
     (Wrap_utils.keyword_args [("base", Wrap_utils.Option.map base Py.Int.of_int); ("padding", Wrap_utils.Option.map padding Py.Int.of_int); ("number", Some(number |> Py.Int.of_int))])
     |> Py.String.to_string
let binary_repr ?width ~num () =
   Py.Module.get_function_with_keywords __wrap_namespace "binary_repr"
     [||]
     (Wrap_utils.keyword_args [("width", Wrap_utils.Option.map width Py.Int.of_int); ("num", Some(num |> Py.Int.of_int))])
     |> Py.String.to_string
                  let bincount ?weights ?minlength x =
                     Py.Module.get_function_with_keywords __wrap_namespace "bincount"
                       [||]
                       (Wrap_utils.keyword_args [("weights", Wrap_utils.Option.map weights Obj.to_pyobject); ("minlength", Wrap_utils.Option.map minlength Py.Int.of_int); ("x", Some(x |> (function
| `Ndarray x -> Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])

                  let bitwise_and ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "bitwise_and"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let bitwise_not ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "bitwise_not"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let bitwise_or ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "bitwise_or"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let bitwise_xor ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "bitwise_xor"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let blackman m =
   Py.Module.get_function_with_keywords __wrap_namespace "blackman"
     [||]
     (Wrap_utils.keyword_args [("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let block arrays =
   Py.Module.get_function_with_keywords __wrap_namespace "block"
     [||]
     (Wrap_utils.keyword_args [("arrays", Some(arrays ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let bmat ?ldict ?gdict ~obj () =
                     Py.Module.get_function_with_keywords __wrap_namespace "bmat"
                       [||]
                       (Wrap_utils.keyword_args [("ldict", ldict); ("gdict", gdict); ("obj", Some(obj |> (function
| `Ndarray x -> Obj.to_pyobject x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let broadcast_arrays ?subok args =
   Py.Module.get_function_with_keywords __wrap_namespace "broadcast_arrays"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (Wrap_utils.keyword_args [("subok", Wrap_utils.Option.map subok Py.Bool.of_bool)])

let broadcast_to ?subok ~array shape =
   Py.Module.get_function_with_keywords __wrap_namespace "broadcast_to"
     [||]
     (Wrap_utils.keyword_args [("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("array", Some(array |> Obj.to_pyobject)); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let busday_count ?weekmask ?holidays ?busdaycal ?out ~begindates ~enddates () =
                     Py.Module.get_function_with_keywords __wrap_namespace "busday_count"
                       [||]
                       (Wrap_utils.keyword_args [("weekmask", Wrap_utils.Option.map weekmask (function
| `Array_like_of_bool x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("holidays", holidays); ("busdaycal", busdaycal); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("begindates", Some(begindates )); ("enddates", Some(enddates ))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let busday_offset ?roll ?weekmask ?holidays ?busdaycal ?out ~dates ~offsets () =
                     Py.Module.get_function_with_keywords __wrap_namespace "busday_offset"
                       [||]
                       (Wrap_utils.keyword_args [("roll", Wrap_utils.Option.map roll (function
| `Raise -> Py.String.of_string "raise"
| `Nat -> Py.String.of_string "nat"
| `Forward -> Py.String.of_string "forward"
| `Following -> Py.String.of_string "following"
| `Backward -> Py.String.of_string "backward"
| `Preceding -> Py.String.of_string "preceding"
| `Modifiedfollowing -> Py.String.of_string "modifiedfollowing"
| `Modifiedpreceding -> Py.String.of_string "modifiedpreceding"
)); ("weekmask", Wrap_utils.Option.map weekmask (function
| `Array_like_of_bool x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("holidays", holidays); ("busdaycal", busdaycal); ("out", out); ("dates", Some(dates )); ("offsets", Some(offsets ))])

let byte_bounds a =
   Py.Module.get_function_with_keywords __wrap_namespace "byte_bounds"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Obj.to_pyobject))])

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
| `Dtype_specifier x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
| `Dtype x -> Dtype.to_pyobject x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
))); ("to", Some(to_ |> (function
| `Dtype x -> Dtype.to_pyobject x
| `Dtype_specifier x -> Wrap_utils.id x
)))])
                       |> Py.Bool.to_bool
                  let cbrt ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "cbrt"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let ceil ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "ceil"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let choose ?out ?mode ~choices a =
                     Py.Module.get_function_with_keywords __wrap_namespace "choose"
                       [||]
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Obj.to_pyobject); ("mode", Wrap_utils.Option.map mode (function
| `Raise -> Py.String.of_string "raise"
| `Wrap -> Py.String.of_string "wrap"
| `Clip -> Py.String.of_string "clip"
)); ("choices", Some(choices )); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let clip ?out ?kwargs ~a_min ~a_max a =
                     Py.Module.get_function_with_keywords __wrap_namespace "clip"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Obj.to_pyobject); ("a_min", Some(a_min |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
| `None -> Py.none
))); ("a_max", Some(a_max |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
| `None -> Py.none
))); ("a", Some(a |> Obj.to_pyobject))]) (match kwargs with None -> [] | Some x -> x))
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let column_stack tup =
   Py.Module.get_function_with_keywords __wrap_namespace "column_stack"
     [||]
     (Wrap_utils.keyword_args [("tup", Some(tup ))])

let common_type arrays =
   Py.Module.get_function_with_keywords __wrap_namespace "common_type"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arrays)])
     []

                  let compare_chararrays ~b ~cmp_op ~rstrip a =
                     Py.Module.get_function_with_keywords __wrap_namespace "compare_chararrays"
                       [||]
                       (Wrap_utils.keyword_args [("b", Some(b )); ("cmp_op", Some(cmp_op |> (function
| `Lt -> Py.String.of_string "<"
| `Lte -> Py.String.of_string "<="
| `Eq -> Py.String.of_string "=="
| `Gte -> Py.String.of_string ">="
| `Gt -> Py.String.of_string ">"
| `Neq -> Py.String.of_string "!="
))); ("rstrip", Some(rstrip |> Py.Bool.of_bool)); ("a", Some(a ))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let compress ?axis ?out ~condition a =
   Py.Module.get_function_with_keywords __wrap_namespace "compress"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("condition", Some(condition )); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let concatenate ?axis ?out a =
   Py.Module.get_function_with_keywords __wrap_namespace "concatenate"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("a", Some(a ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let conj ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "conj"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let conjugate ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "conjugate"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let convolve ?mode ~v a =
                     Py.Module.get_function_with_keywords __wrap_namespace "convolve"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Full -> Py.String.of_string "full"
| `Valid -> Py.String.of_string "valid"
| `Same -> Py.String.of_string "same"
)); ("v", Some(v |> Obj.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let copy ?order ?subok a =
                     Py.Module.get_function_with_keywords __wrap_namespace "copy"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
| `A -> Py.String.of_string "A"
| `K -> Py.String.of_string "K"
)); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let copysign ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "copysign"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let copyto ?casting ?where ~dst ~src () =
                     Py.Module.get_function_with_keywords __wrap_namespace "copyto"
                       [||]
                       (Wrap_utils.keyword_args [("casting", Wrap_utils.Option.map casting (function
| `No -> Py.String.of_string "no"
| `Equiv -> Py.String.of_string "equiv"
| `Safe -> Py.String.of_string "safe"
| `Same_kind -> Py.String.of_string "same_kind"
| `Unsafe -> Py.String.of_string "unsafe"
)); ("where", where); ("dst", Some(dst |> Obj.to_pyobject)); ("src", Some(src |> Obj.to_pyobject))])

let corrcoef ?y ?rowvar ?bias ?ddof x =
   Py.Module.get_function_with_keywords __wrap_namespace "corrcoef"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Obj.to_pyobject); ("rowvar", Wrap_utils.Option.map rowvar Py.Bool.of_bool); ("bias", bias); ("ddof", ddof); ("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let correlate ?mode ~v a =
                     Py.Module.get_function_with_keywords __wrap_namespace "correlate"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Valid -> Py.String.of_string "valid"
| `Same -> Py.String.of_string "same"
| `Full -> Py.String.of_string "full"
)); ("v", Some(v )); ("a", Some(a ))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let cos ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "cos"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let cosh ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "cosh"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let count_nonzero ?axis ?keepdims a =
                     Py.Module.get_function_with_keywords __wrap_namespace "count_nonzero"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Tuple x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])

                  let cov ?y ?rowvar ?bias ?ddof ?fweights ?aweights ~m () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cov"
                       [||]
                       (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Obj.to_pyobject); ("rowvar", Wrap_utils.Option.map rowvar Py.Bool.of_bool); ("bias", Wrap_utils.Option.map bias Py.Bool.of_bool); ("ddof", Wrap_utils.Option.map ddof Py.Int.of_int); ("fweights", Wrap_utils.Option.map fweights (function
| `Ndarray x -> Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("aweights", Wrap_utils.Option.map aweights Obj.to_pyobject); ("m", Some(m |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let cross ?axisa ?axisb ?axisc ?axis ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "cross"
     [||]
     (Wrap_utils.keyword_args [("axisa", Wrap_utils.Option.map axisa Py.Int.of_int); ("axisb", Wrap_utils.Option.map axisb Py.Int.of_int); ("axisc", Wrap_utils.Option.map axisc Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("b", Some(b |> Obj.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let cumprod ?axis ?dtype ?out a =
   Py.Module.get_function_with_keywords __wrap_namespace "cumprod"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let cumproduct ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "cumproduct"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

let cumsum ?axis ?dtype ?out a =
   Py.Module.get_function_with_keywords __wrap_namespace "cumsum"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let datetime_as_string ?unit ?timezone ?casting ~arr () =
                     Py.Module.get_function_with_keywords __wrap_namespace "datetime_as_string"
                       [||]
                       (Wrap_utils.keyword_args [("unit", Wrap_utils.Option.map unit Py.String.of_string); ("timezone", Wrap_utils.Option.map timezone (function
| `Tzinfo x -> Wrap_utils.id x
| `UTC -> Py.String.of_string "UTC"
| `Local -> Py.String.of_string "local"
| `Naive -> Py.String.of_string "naive"
)); ("casting", Wrap_utils.Option.map casting (function
| `No -> Py.String.of_string "no"
| `Equiv -> Py.String.of_string "equiv"
| `Safe -> Py.String.of_string "safe"
| `Same_kind -> Py.String.of_string "same_kind"
| `Unsafe -> Py.String.of_string "unsafe"
)); ("arr", Some(arr ))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let datetime_data dtype =
   Py.Module.get_function_with_keywords __wrap_namespace "datetime_data"
     (Array.of_list @@ List.concat [[dtype |> Dtype.to_pyobject]])
     []
     |> (fun x -> ((Py.String.to_string (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let deg2rad ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "deg2rad"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let degrees ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "degrees"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let delete ?axis ~arr ~obj () =
                     Py.Module.get_function_with_keywords __wrap_namespace "delete"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("arr", Some(arr |> Obj.to_pyobject)); ("obj", Some(obj |> (function
| `Slice x -> Wrap_utils.Slice.to_pyobject x
| `I x -> Py.Int.of_int x
| `Array_of_ints x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let deprecate ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "deprecate"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

let deprecate_with_doc msg =
   Py.Module.get_function_with_keywords __wrap_namespace "deprecate_with_doc"
     [||]
     (Wrap_utils.keyword_args [("msg", Some(msg ))])

let diag ?k ~v () =
   Py.Module.get_function_with_keywords __wrap_namespace "diag"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("v", Some(v |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let diag_indices ?ndim ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "diag_indices"
     [||]
     (Wrap_utils.keyword_args [("ndim", Wrap_utils.Option.map ndim Py.Int.of_int); ("n", Some(n |> Py.Int.of_int))])

                  let diag_indices_from arr =
                     Py.Module.get_function_with_keywords __wrap_namespace "diag_indices_from"
                       [||]
                       (Wrap_utils.keyword_args [("arr", Some(arr |> (function
| `At_least_2_D x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)))])

let diagflat ?k ~v () =
   Py.Module.get_function_with_keywords __wrap_namespace "diagflat"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("v", Some(v |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let diagonal ?offset ?axis1 ?axis2 a =
   Py.Module.get_function_with_keywords __wrap_namespace "diagonal"
     [||]
     (Wrap_utils.keyword_args [("offset", Wrap_utils.Option.map offset Py.Int.of_int); ("axis1", Wrap_utils.Option.map axis1 Py.Int.of_int); ("axis2", Wrap_utils.Option.map axis2 Py.Int.of_int); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let diff ?n ?axis ?prepend ?append a =
   Py.Module.get_function_with_keywords __wrap_namespace "diff"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("prepend", prepend); ("append", append); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let digitize ?right ~bins x =
   Py.Module.get_function_with_keywords __wrap_namespace "digitize"
     [||]
     (Wrap_utils.keyword_args [("right", Wrap_utils.Option.map right Py.Bool.of_bool); ("bins", Some(bins |> Obj.to_pyobject)); ("x", Some(x |> Obj.to_pyobject))])

let disp ?device ?linefeed ~mesg () =
   Py.Module.get_function_with_keywords __wrap_namespace "disp"
     [||]
     (Wrap_utils.keyword_args [("device", device); ("linefeed", Wrap_utils.Option.map linefeed Py.Bool.of_bool); ("mesg", Some(mesg |> Py.String.of_string))])

                  let divide ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "divide"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let divmod ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "divmod"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun x -> (((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 1))))
let dot ?out ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "dot"
     [||]
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Obj.to_pyobject); ("b", Some(b |> Obj.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let dsplit ~ary ~indices_or_sections () =
   Py.Module.get_function_with_keywords __wrap_namespace "dsplit"
     [||]
     (Wrap_utils.keyword_args [("ary", Some(ary )); ("indices_or_sections", Some(indices_or_sections ))])

let dstack tup =
   Py.Module.get_function_with_keywords __wrap_namespace "dstack"
     [||]
     (Wrap_utils.keyword_args [("tup", Some(tup ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let ediff1d ?to_end ?to_begin ~ary () =
   Py.Module.get_function_with_keywords __wrap_namespace "ediff1d"
     [||]
     (Wrap_utils.keyword_args [("to_end", Wrap_utils.Option.map to_end Obj.to_pyobject); ("to_begin", Wrap_utils.Option.map to_begin Obj.to_pyobject); ("ary", Some(ary |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let einsum ?out ?optimize ?kwargs operands =
                     Py.Module.get_function_with_keywords __wrap_namespace "einsum"
                       (Array.of_list @@ List.concat [(List.map Wrap_utils.id operands)])
                       (List.rev_append (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Obj.to_pyobject); ("optimize", Wrap_utils.Option.map optimize (function
| `Optimal -> Py.String.of_string "optimal"
| `Bool x -> Py.Bool.of_bool x
| `Greedy -> Py.String.of_string "greedy"
))]) (match kwargs with None -> [] | Some x -> x))
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let einsum_path ?optimize ?einsum_call operands =
                     Py.Module.get_function_with_keywords __wrap_namespace "einsum_path"
                       (Array.of_list @@ List.concat [(List.map Wrap_utils.id operands)])
                       (Wrap_utils.keyword_args [("optimize", Wrap_utils.Option.map optimize (function
| `Optimal -> Py.String.of_string "optimal"
| `Bool x -> Py.Bool.of_bool x
| `Greedy -> Py.String.of_string "greedy"
| `Ndarray x -> Obj.to_pyobject x
| `Tuple x -> Wrap_utils.id x
)); ("einsum_call", einsum_call)])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Py.String.to_string (Py.Tuple.get x 1))))
                  let empty ?dtype ?order shape =
                     Py.Module.get_function_with_keywords __wrap_namespace "empty"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let empty_like ?dtype ?order ?subok ?shape ~prototype () =
                     Py.Module.get_function_with_keywords __wrap_namespace "empty_like"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `A -> Py.String.of_string "A"
| `F -> Py.String.of_string "F"
| `PyObject x -> Wrap_utils.id x
)); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("shape", Wrap_utils.Option.map shape (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("prototype", Some(prototype |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let equal ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "equal"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let exp ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "exp"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let exp2 ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "exp2"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let expand_dims ~axis a =
   Py.Module.get_function_with_keywords __wrap_namespace "expand_dims"
     [||]
     (Wrap_utils.keyword_args [("axis", Some(axis |> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml))); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let expm1 ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "expm1"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let extract ~condition ~arr () =
   Py.Module.get_function_with_keywords __wrap_namespace "extract"
     [||]
     (Wrap_utils.keyword_args [("condition", Some(condition |> Obj.to_pyobject)); ("arr", Some(arr |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let eye ?m ?k ?dtype ?order ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "eye"
                       [||]
                       (Wrap_utils.keyword_args [("M", Wrap_utils.Option.map m Py.Int.of_int); ("k", Wrap_utils.Option.map k Py.Int.of_int); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("N", Some(n |> Py.Int.of_int))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let fabs ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "fabs"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let fastCopyAndTranspose a =
   Py.Module.get_function_with_keywords __wrap_namespace "fastCopyAndTranspose"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a ))])

                  let fill_diagonal ?wrap ~val_ a =
                     Py.Module.get_function_with_keywords __wrap_namespace "fill_diagonal"
                       [||]
                       (Wrap_utils.keyword_args [("wrap", Wrap_utils.Option.map wrap Py.Bool.of_bool); ("val", Some(val_ |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
))); ("a", Some(a |> (function
| `At_least_2_D x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)))])

let find_common_type ~array_types ~scalar_types () =
   Py.Module.get_function_with_keywords __wrap_namespace "find_common_type"
     [||]
     (Wrap_utils.keyword_args [("array_types", Some(array_types )); ("scalar_types", Some(scalar_types ))])
     |> Dtype.of_pyobject
let fix ?out x =
   Py.Module.get_function_with_keywords __wrap_namespace "fix"
     [||]
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Obj.to_pyobject); ("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let flatnonzero a =
   Py.Module.get_function_with_keywords __wrap_namespace "flatnonzero"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let flip ?axis ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "flip"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("m", Some(m |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let fliplr m =
   Py.Module.get_function_with_keywords __wrap_namespace "fliplr"
     [||]
     (Wrap_utils.keyword_args [("m", Some(m |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let flipud m =
   Py.Module.get_function_with_keywords __wrap_namespace "flipud"
     [||]
     (Wrap_utils.keyword_args [("m", Some(m |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let float_power ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "float_power"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let floor ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "floor"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let floor_divide ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "floor_divide"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let fmax ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "fmax"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let fmin ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "fmin"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let fmod ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "fmod"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let format_float_positional ?precision ?unique ?fractional ?trim ?sign ?pad_left ?pad_right x =
                     Py.Module.get_function_with_keywords __wrap_namespace "format_float_positional"
                       [||]
                       (Wrap_utils.keyword_args [("precision", precision); ("unique", Wrap_utils.Option.map unique Py.Bool.of_bool); ("fractional", Wrap_utils.Option.map fractional Py.Bool.of_bool); ("trim", Wrap_utils.Option.map trim (function
| `One_of_k_ x -> Wrap_utils.id x
| `T0 -> Py.String.of_string "0"
| `T_ -> Py.String.of_string "."
| `Minus -> Py.String.of_string "-"
)); ("sign", sign); ("pad_left", pad_left); ("pad_right", pad_right); ("x", Some(x ))])
                       |> Py.String.to_string
                  let format_float_scientific ?precision ?unique ?trim ?sign ?pad_left ?exp_digits x =
                     Py.Module.get_function_with_keywords __wrap_namespace "format_float_scientific"
                       [||]
                       (Wrap_utils.keyword_args [("precision", precision); ("unique", Wrap_utils.Option.map unique Py.Bool.of_bool); ("trim", Wrap_utils.Option.map trim (function
| `One_of_k_ x -> Wrap_utils.id x
| `T0 -> Py.String.of_string "0"
| `T_ -> Py.String.of_string "."
| `Minus -> Py.String.of_string "-"
)); ("sign", sign); ("pad_left", pad_left); ("exp_digits", exp_digits); ("x", Some(x ))])
                       |> Py.String.to_string
                  let frexp ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "frexp"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun x -> (((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 1))))
let frombuffer ?dtype ?count ?offset ~buffer () =
   Py.Module.get_function_with_keywords __wrap_namespace "frombuffer"
     [||]
     (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("count", Wrap_utils.Option.map count Py.Int.of_int); ("offset", Wrap_utils.Option.map offset Py.Int.of_int); ("buffer", Some(buffer ))])

                  let fromfile ?dtype ?count ?sep ?offset ~file () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fromfile"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("count", Wrap_utils.Option.map count Py.Int.of_int); ("sep", Wrap_utils.Option.map sep Py.String.of_string); ("offset", Wrap_utils.Option.map offset Py.Int.of_int); ("file", Some(file |> (function
| `S x -> Py.String.of_string x
| `PyObject x -> Wrap_utils.id x
)))])

let fromfunction ?dtype ?kwargs ~function_ shape =
   Py.Module.get_function_with_keywords __wrap_namespace "fromfunction"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("function", Some(function_ )); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))]) (match kwargs with None -> [] | Some x -> x))

let fromiter ?count ~iterable ~dtype () =
   Py.Module.get_function_with_keywords __wrap_namespace "fromiter"
     [||]
     (Wrap_utils.keyword_args [("count", Wrap_utils.Option.map count Py.Int.of_int); ("iterable", Some(iterable )); ("dtype", Some(dtype |> Dtype.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let frompyfunc ?identity ~func ~nin ~nout () =
   Py.Module.get_function_with_keywords __wrap_namespace "frompyfunc"
     [||]
     (Wrap_utils.keyword_args [("identity", identity); ("func", Some(func )); ("nin", Some(nin |> Py.Int.of_int)); ("nout", Some(nout |> Py.Int.of_int))])

                  let fromregex ?encoding ~file ~regexp ~dtype () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fromregex"
                       [||]
                       (Wrap_utils.keyword_args [("encoding", Wrap_utils.Option.map encoding Py.String.of_string); ("file", Some(file |> (function
| `File x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
))); ("regexp", Some(regexp |> (function
| `Regexp x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
))); ("dtype", Some(dtype |> (function
| `List_of_dtypes x -> Wrap_utils.id x
| `Dtype x -> Dtype.to_pyobject x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let fromstring ?dtype ?count ?sep ~string () =
   Py.Module.get_function_with_keywords __wrap_namespace "fromstring"
     [||]
     (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("count", Wrap_utils.Option.map count Py.Int.of_int); ("sep", Wrap_utils.Option.map sep Py.String.of_string); ("string", Some(string |> Py.String.of_string))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let full ?dtype ?order ~fill_value shape =
                     Py.Module.get_function_with_keywords __wrap_namespace "full"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("fill_value", Some(fill_value |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
))); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let full_like ?dtype ?order ?subok ?shape ~fill_value a =
                     Py.Module.get_function_with_keywords __wrap_namespace "full_like"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `A -> Py.String.of_string "A"
| `F -> Py.String.of_string "F"
| `PyObject x -> Wrap_utils.id x
)); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("shape", Wrap_utils.Option.map shape (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("fill_value", Some(fill_value |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
))); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let fv ?when_ ~rate ~nper ~pmt ~pv () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fv"
                       [||]
                       (Wrap_utils.keyword_args [("when", Wrap_utils.Option.map when_ (function
| `I x -> Py.Int.of_int x
| `Begin -> Py.String.of_string "begin"
| `PyObject x -> Wrap_utils.id x
)); ("rate", Some(rate |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
))); ("nper", Some(nper |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
))); ("pmt", Some(pmt |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
))); ("pv", Some(pv |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let gcd ?out ?where x =
   Py.Module.get_function_with_keywords __wrap_namespace "gcd"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let genfromtxt ?dtype ?comments ?delimiter ?skip_header ?skip_footer ?converters ?missing_values ?filling_values ?usecols ?names ?excludelist ?deletechars ?replace_space ?autostrip ?case_sensitive ?defaultfmt ?unpack ?usemask ?loose ?invalid_raise ?max_rows ?encoding ~fname () =
                     Py.Module.get_function_with_keywords __wrap_namespace "genfromtxt"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("comments", Wrap_utils.Option.map comments Py.String.of_string); ("delimiter", Wrap_utils.Option.map delimiter (function
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
)); ("skip_header", Wrap_utils.Option.map skip_header Py.Int.of_int); ("skip_footer", Wrap_utils.Option.map skip_footer Py.Int.of_int); ("converters", converters); ("missing_values", missing_values); ("filling_values", filling_values); ("usecols", usecols); ("names", Wrap_utils.Option.map names (function
| `S x -> Py.String.of_string x
| `Sequence x -> Wrap_utils.id x
| `True -> Py.Bool.t
)); ("excludelist", excludelist); ("deletechars", Wrap_utils.Option.map deletechars Py.String.of_string); ("replace_space", replace_space); ("autostrip", Wrap_utils.Option.map autostrip Py.Bool.of_bool); ("case_sensitive", Wrap_utils.Option.map case_sensitive (function
| `Bool x -> Py.Bool.of_bool x
| `Upper -> Py.String.of_string "upper"
| `Lower -> Py.String.of_string "lower"
)); ("defaultfmt", Wrap_utils.Option.map defaultfmt Py.String.of_string); ("unpack", Wrap_utils.Option.map unpack Py.Bool.of_bool); ("usemask", Wrap_utils.Option.map usemask Py.Bool.of_bool); ("loose", Wrap_utils.Option.map loose Py.Bool.of_bool); ("invalid_raise", Wrap_utils.Option.map invalid_raise Py.Bool.of_bool); ("max_rows", Wrap_utils.Option.map max_rows Py.Int.of_int); ("encoding", Wrap_utils.Option.map encoding Py.String.of_string); ("fname", Some(fname |> (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `S x -> Py.String.of_string x
| `PyObject x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let geomspace ?num ?endpoint ?dtype ?axis ~start ~stop () =
   Py.Module.get_function_with_keywords __wrap_namespace "geomspace"
     [||]
     (Wrap_utils.keyword_args [("num", Wrap_utils.Option.map num Py.Int.of_int); ("endpoint", Wrap_utils.Option.map endpoint Py.Bool.of_bool); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("start", Some(start |> Obj.to_pyobject)); ("stop", Some(stop |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let get_array_wrap args =
   Py.Module.get_function_with_keywords __wrap_namespace "get_array_wrap"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

let get_include () =
   Py.Module.get_function_with_keywords __wrap_namespace "get_include"
     [||]
     []

let get_printoptions () =
   Py.Module.get_function_with_keywords __wrap_namespace "get_printoptions"
     [||]
     []

let getbufsize () =
   Py.Module.get_function_with_keywords __wrap_namespace "getbufsize"
     [||]
     []
     |> Py.Int.to_int
let geterr () =
   Py.Module.get_function_with_keywords __wrap_namespace "geterr"
     [||]
     []

let geterrcall () =
   Py.Module.get_function_with_keywords __wrap_namespace "geterrcall"
     [||]
     []
     |> (fun py -> if Py.is_none py then None else Some (Wrap_utils.id py))
                  let gradient ?axis ?edge_order ~f varargs =
                     Py.Module.get_function_with_keywords __wrap_namespace "gradient"
                       (Array.of_list @@ List.concat [(List.map Wrap_utils.id varargs)])
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("edge_order", Wrap_utils.Option.map edge_order (function
| `Two -> Py.Int.of_int 2
| `One -> Py.Int.of_int 1
)); ("f", Some(f |> Obj.to_pyobject))])

                  let greater ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "greater"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let greater_equal ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "greater_equal"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])

let hamming m =
   Py.Module.get_function_with_keywords __wrap_namespace "hamming"
     [||]
     (Wrap_utils.keyword_args [("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let hanning m =
   Py.Module.get_function_with_keywords __wrap_namespace "hanning"
     [||]
     (Wrap_utils.keyword_args [("M", Some(m |> Py.Int.of_int))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let heaviside ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "heaviside"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let histogram ?bins ?range ?normed ?weights ?density a =
                     Py.Module.get_function_with_keywords __wrap_namespace "histogram"
                       [||]
                       (Wrap_utils.keyword_args [("bins", Wrap_utils.Option.map bins (function
| `Sequence_of_scalars x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
)); ("range", Wrap_utils.Option.map range (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.Float.of_float ml_0); (Py.Float.of_float ml_1)])); ("normed", Wrap_utils.Option.map normed Py.Bool.of_bool); ("weights", Wrap_utils.Option.map weights Obj.to_pyobject); ("density", Wrap_utils.Option.map density Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let histogram2d ?bins ?range ?normed ?weights ?density ~y x =
                     Py.Module.get_function_with_keywords __wrap_namespace "histogram2d"
                       [||]
                       (Wrap_utils.keyword_args [("bins", Wrap_utils.Option.map bins (function
| `Ndarray x -> Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)); ("range", Wrap_utils.Option.map range Obj.to_pyobject); ("normed", Wrap_utils.Option.map normed Py.Bool.of_bool); ("weights", Wrap_utils.Option.map weights Obj.to_pyobject); ("density", Wrap_utils.Option.map density Py.Bool.of_bool); ("y", Some(y |> Obj.to_pyobject)); ("x", Some(x |> Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 2))))
                  let histogram_bin_edges ?bins ?range ?weights a =
                     Py.Module.get_function_with_keywords __wrap_namespace "histogram_bin_edges"
                       [||]
                       (Wrap_utils.keyword_args [("bins", Wrap_utils.Option.map bins (function
| `Sequence_of_scalars x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
)); ("range", Wrap_utils.Option.map range (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.Float.of_float ml_0); (Py.Float.of_float ml_1)])); ("weights", Wrap_utils.Option.map weights Obj.to_pyobject); ("a", Some(a |> Obj.to_pyobject))])

                  let histogramdd ?bins ?range ?normed ?weights ?density ~sample () =
                     Py.Module.get_function_with_keywords __wrap_namespace "histogramdd"
                       [||]
                       (Wrap_utils.keyword_args [("bins", Wrap_utils.Option.map bins (function
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("range", range); ("normed", Wrap_utils.Option.map normed Py.Bool.of_bool); ("weights", Wrap_utils.Option.map weights Obj.to_pyobject); ("density", Wrap_utils.Option.map density Py.Bool.of_bool); ("sample", Some(sample ))])
                       |> (fun x -> (((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 1))))
let hsplit ~ary ~indices_or_sections () =
   Py.Module.get_function_with_keywords __wrap_namespace "hsplit"
     [||]
     (Wrap_utils.keyword_args [("ary", Some(ary )); ("indices_or_sections", Some(indices_or_sections ))])

let hstack tup =
   Py.Module.get_function_with_keywords __wrap_namespace "hstack"
     [||]
     (Wrap_utils.keyword_args [("tup", Some(tup |> (fun ml -> Py.List.of_list_map Obj.to_pyobject ml)))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let hypot ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "hypot"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let i0 x =
                     Py.Module.get_function_with_keywords __wrap_namespace "i0"
                       [||]
                       (Wrap_utils.keyword_args [("x", Some(x |> (function
| `Ndarray x -> Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])

let identity ?dtype ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "identity"
     [||]
     (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("n", Some(n |> Py.Int.of_int))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let imag val_ =
   Py.Module.get_function_with_keywords __wrap_namespace "imag"
     [||]
     (Wrap_utils.keyword_args [("val", Some(val_ |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let in1d ?assume_unique ?invert ~ar1 ~ar2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "in1d"
     [||]
     (Wrap_utils.keyword_args [("assume_unique", Wrap_utils.Option.map assume_unique Py.Bool.of_bool); ("invert", Wrap_utils.Option.map invert Py.Bool.of_bool); ("ar1", Some(ar1 |> Obj.to_pyobject)); ("ar2", Some(ar2 |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let indices ?dtype ?sparse ~dimensions () =
   Py.Module.get_function_with_keywords __wrap_namespace "indices"
     [||]
     (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("sparse", Wrap_utils.Option.map sparse Py.Bool.of_bool); ("dimensions", Some(dimensions |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])

                  let info ?object_ ?maxwidth ?output ?toplevel () =
                     Py.Module.get_function_with_keywords __wrap_namespace "info"
                       [||]
                       (Wrap_utils.keyword_args [("object", Wrap_utils.Option.map object_ (function
| `PyObject x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("maxwidth", Wrap_utils.Option.map maxwidth Py.Int.of_int); ("output", output); ("toplevel", Wrap_utils.Option.map toplevel Py.String.of_string)])

let inner ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "inner"
     [||]
     (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let insert ?axis ~arr ~obj ~values () =
                     Py.Module.get_function_with_keywords __wrap_namespace "insert"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("arr", Some(arr |> Obj.to_pyobject)); ("obj", Some(obj |> (function
| `Slice x -> Wrap_utils.Slice.to_pyobject x
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
| `I x -> Py.Int.of_int x
))); ("values", Some(values |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let interp ?left ?right ?period ~xp ~fp x =
                     Py.Module.get_function_with_keywords __wrap_namespace "interp"
                       [||]
                       (Wrap_utils.keyword_args [("left", Wrap_utils.Option.map left (function
| `Complex_corresponding_to_fp x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)); ("right", Wrap_utils.Option.map right (function
| `Complex_corresponding_to_fp x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)); ("period", Wrap_utils.Option.map period Py.Float.of_float); ("xp", Some(xp )); ("fp", Some(fp )); ("x", Some(x |> Obj.to_pyobject))])

let intersect1d ?assume_unique ?return_indices ~ar1 ~ar2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "intersect1d"
     [||]
     (Wrap_utils.keyword_args [("assume_unique", Wrap_utils.Option.map assume_unique Py.Bool.of_bool); ("return_indices", Wrap_utils.Option.map return_indices Py.Bool.of_bool); ("ar1", Some(ar1 )); ("ar2", Some(ar2 ))])
     |> (fun x -> (((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 2))))
                  let invert ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "invert"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let ipmt ?fv ?when_ ~rate ~per ~nper ~pv () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ipmt"
                       [||]
                       (Wrap_utils.keyword_args [("fv", Wrap_utils.Option.map fv (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
)); ("when", Wrap_utils.Option.map when_ (function
| `I x -> Py.Int.of_int x
| `Begin -> Py.String.of_string "begin"
| `PyObject x -> Wrap_utils.id x
)); ("rate", Some(rate |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
))); ("per", Some(per |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
))); ("nper", Some(nper |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
))); ("pv", Some(pv |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let irr values =
   Py.Module.get_function_with_keywords __wrap_namespace "irr"
     [||]
     (Wrap_utils.keyword_args [("values", Some(values |> Obj.to_pyobject))])
     |> Py.Float.to_float
                  let is_busday ?weekmask ?holidays ?busdaycal ?out ~dates () =
                     Py.Module.get_function_with_keywords __wrap_namespace "is_busday"
                       [||]
                       (Wrap_utils.keyword_args [("weekmask", Wrap_utils.Option.map weekmask (function
| `Array_like_of_bool x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("holidays", holidays); ("busdaycal", busdaycal); ("out", out); ("dates", Some(dates ))])

let isclose ?rtol ?atol ?equal_nan ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "isclose"
     [||]
     (Wrap_utils.keyword_args [("rtol", Wrap_utils.Option.map rtol Py.Float.of_float); ("atol", Wrap_utils.Option.map atol Py.Float.of_float); ("equal_nan", Wrap_utils.Option.map equal_nan Py.Bool.of_bool); ("b", Some(b )); ("a", Some(a ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let iscomplex x =
   Py.Module.get_function_with_keywords __wrap_namespace "iscomplex"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Obj.to_pyobject))])

let iscomplexobj x =
   Py.Module.get_function_with_keywords __wrap_namespace "iscomplexobj"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])
     |> Py.Bool.to_bool
                  let isfinite ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "isfinite"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let isfortran a =
   Py.Module.get_function_with_keywords __wrap_namespace "isfortran"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Obj.to_pyobject))])
     |> Py.Bool.to_bool
let isin ?assume_unique ?invert ~element ~test_elements () =
   Py.Module.get_function_with_keywords __wrap_namespace "isin"
     [||]
     (Wrap_utils.keyword_args [("assume_unique", Wrap_utils.Option.map assume_unique Py.Bool.of_bool); ("invert", Wrap_utils.Option.map invert Py.Bool.of_bool); ("element", Some(element |> Obj.to_pyobject)); ("test_elements", Some(test_elements |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let isinf ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "isinf"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let isnan ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "isnan"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])

                  let isnat ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "isnat"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])

let isneginf ?out x =
   Py.Module.get_function_with_keywords __wrap_namespace "isneginf"
     [||]
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Obj.to_pyobject); ("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let isposinf ?out x =
   Py.Module.get_function_with_keywords __wrap_namespace "isposinf"
     [||]
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Obj.to_pyobject); ("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let isreal x =
   Py.Module.get_function_with_keywords __wrap_namespace "isreal"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let isrealobj x =
   Py.Module.get_function_with_keywords __wrap_namespace "isrealobj"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])
     |> Py.Bool.to_bool
let isscalar element =
   Py.Module.get_function_with_keywords __wrap_namespace "isscalar"
     [||]
     (Wrap_utils.keyword_args [("element", Some(element ))])
     |> Py.Bool.to_bool
let issctype rep =
   Py.Module.get_function_with_keywords __wrap_namespace "issctype"
     [||]
     (Wrap_utils.keyword_args [("rep", Some(rep ))])
     |> Py.Bool.to_bool
let issubdtype ~arg1 ~arg2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "issubdtype"
     [||]
     (Wrap_utils.keyword_args [("arg1", Some(arg1 )); ("arg2", Some(arg2 ))])
     |> Py.Bool.to_bool
let issubsctype ~arg1 ~arg2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "issubsctype"
     [||]
     (Wrap_utils.keyword_args [("arg1", Some(arg1 )); ("arg2", Some(arg2 ))])
     |> Py.Bool.to_bool
let iterable y =
   Py.Module.get_function_with_keywords __wrap_namespace "iterable"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y ))])
     |> Py.Bool.to_bool
let kaiser ~m ~beta () =
   Py.Module.get_function_with_keywords __wrap_namespace "kaiser"
     [||]
     (Wrap_utils.keyword_args [("M", Some(m |> Py.Int.of_int)); ("beta", Some(beta |> Py.Float.of_float))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let kron ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "kron"
     [||]
     (Wrap_utils.keyword_args [("b", Some(b )); ("a", Some(a ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let lcm ?out ?where x =
   Py.Module.get_function_with_keywords __wrap_namespace "lcm"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let ldexp ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "ldexp"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let left_shift ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "left_shift"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let less ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "less"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let less_equal ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "less_equal"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let lexsort ?axis ~keys () =
   Py.Module.get_function_with_keywords __wrap_namespace "lexsort"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("keys", Some(keys ))])

let linspace ?num ?endpoint ?retstep ?dtype ?axis ~start ~stop () =
   Py.Module.get_function_with_keywords __wrap_namespace "linspace"
     [||]
     (Wrap_utils.keyword_args [("num", Wrap_utils.Option.map num Py.Int.of_int); ("endpoint", Wrap_utils.Option.map endpoint Py.Bool.of_bool); ("retstep", Wrap_utils.Option.map retstep Py.Bool.of_bool); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("start", Some(start |> Obj.to_pyobject)); ("stop", Some(stop |> Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))
                  let load ?mmap_mode ?allow_pickle ?fix_imports ?encoding ~file () =
                     Py.Module.get_function_with_keywords __wrap_namespace "load"
                       [||]
                       (Wrap_utils.keyword_args [("mmap_mode", Wrap_utils.Option.map mmap_mode (function
| `R_plus -> Py.String.of_string "r+"
| `R -> Py.String.of_string "r"
| `C -> Py.String.of_string "c"
| `W_plus -> Py.String.of_string "w+"
)); ("allow_pickle", Wrap_utils.Option.map allow_pickle Py.Bool.of_bool); ("fix_imports", Wrap_utils.Option.map fix_imports Py.Bool.of_bool); ("encoding", Wrap_utils.Option.map encoding Py.String.of_string); ("file", Some(file |> (function
| `S x -> Py.String.of_string x
| `PyObject x -> Wrap_utils.id x
)))])

let loads ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "loads"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

                  let loadtxt ?dtype ?comments ?delimiter ?converters ?skiprows ?usecols ?unpack ?ndmin ?encoding ?max_rows ~fname () =
                     Py.Module.get_function_with_keywords __wrap_namespace "loadtxt"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("comments", Wrap_utils.Option.map comments (function
| `Sequence_of_str x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("delimiter", Wrap_utils.Option.map delimiter Py.String.of_string); ("converters", converters); ("skiprows", Wrap_utils.Option.map skiprows Py.Int.of_int); ("usecols", Wrap_utils.Option.map usecols (function
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("unpack", Wrap_utils.Option.map unpack Py.Bool.of_bool); ("ndmin", Wrap_utils.Option.map ndmin Py.Int.of_int); ("encoding", Wrap_utils.Option.map encoding Py.String.of_string); ("max_rows", Wrap_utils.Option.map max_rows Py.Int.of_int); ("fname", Some(fname |> (function
| `S x -> Py.String.of_string x
| `PyObject x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let log ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "log"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let log10 ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "log10"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let log1p ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "log1p"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let log2 ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "log2"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let logaddexp ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "logaddexp"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let logaddexp2 ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "logaddexp2"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let logical_and ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "logical_and"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])

                  let logical_not ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "logical_not"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])

                  let logical_or ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "logical_or"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])

                  let logical_xor ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "logical_xor"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])

let logspace ?num ?endpoint ?base ?dtype ?axis ~start ~stop () =
   Py.Module.get_function_with_keywords __wrap_namespace "logspace"
     [||]
     (Wrap_utils.keyword_args [("num", Wrap_utils.Option.map num Py.Int.of_int); ("endpoint", Wrap_utils.Option.map endpoint Py.Bool.of_bool); ("base", Wrap_utils.Option.map base Py.Float.of_float); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("start", Some(start |> Obj.to_pyobject)); ("stop", Some(stop |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let lookfor ?module_ ?import_modules ?regenerate ?output ~what () =
                     Py.Module.get_function_with_keywords __wrap_namespace "lookfor"
                       [||]
                       (Wrap_utils.keyword_args [("module", Wrap_utils.Option.map module_ (function
| `Ndarray x -> Obj.to_pyobject x
| `S x -> Py.String.of_string x
)); ("import_modules", Wrap_utils.Option.map import_modules Py.Bool.of_bool); ("regenerate", Wrap_utils.Option.map regenerate Py.Bool.of_bool); ("output", output); ("what", Some(what |> Py.String.of_string))])

let mafromtxt ?kwargs ~fname () =
   Py.Module.get_function_with_keywords __wrap_namespace "mafromtxt"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("fname", Some(fname ))]) (match kwargs with None -> [] | Some x -> x))

                  let mask_indices ?k ~n ~mask_func () =
                     Py.Module.get_function_with_keywords __wrap_namespace "mask_indices"
                       [||]
                       (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("n", Some(n |> Py.Int.of_int)); ("mask_func", Some(mask_func ))])

let mat ?dtype ~data () =
   Py.Module.get_function_with_keywords __wrap_namespace "mat"
     [||]
     (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("data", Some(data |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let matmul ?out ?where x =
   Py.Module.get_function_with_keywords __wrap_namespace "matmul"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Obj.to_pyobject); ("where", where)])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let max ?axis ?out ?keepdims ?initial ?where a =
                     Py.Module.get_function_with_keywords __wrap_namespace "max"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("initial", Wrap_utils.Option.map initial (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("where", where); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let maximum ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "maximum"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let maximum_sctype t =
                     Py.Module.get_function_with_keywords __wrap_namespace "maximum_sctype"
                       [||]
                       (Wrap_utils.keyword_args [("t", Some(t |> (function
| `Dtype x -> Dtype.to_pyobject x
| `Dtype_specifier x -> Wrap_utils.id x
)))])
                       |> Dtype.of_pyobject
let may_share_memory ?max_work ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "may_share_memory"
     [||]
     (Wrap_utils.keyword_args [("max_work", Wrap_utils.Option.map max_work Py.Int.of_int); ("b", Some(b )); ("a", Some(a ))])
     |> Py.Bool.to_bool
let mean ?axis ?dtype ?out ?keepdims a =
   Py.Module.get_function_with_keywords __wrap_namespace "mean"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let median ?axis ?out ?overwrite_input ?keepdims a =
                     Py.Module.get_function_with_keywords __wrap_namespace "median"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Sequence_of_int x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("overwrite_input", Wrap_utils.Option.map overwrite_input Py.Bool.of_bool); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let meshgrid ?copy ?sparse ?indexing xi =
                     Py.Module.get_function_with_keywords __wrap_namespace "meshgrid"
                       (Array.of_list @@ List.concat [(List.map Wrap_utils.id xi)])
                       (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("sparse", Wrap_utils.Option.map sparse Py.Bool.of_bool); ("indexing", Wrap_utils.Option.map indexing (function
| `Xy -> Py.String.of_string "xy"
| `Ij -> Py.String.of_string "ij"
))])

                  let min ?axis ?out ?keepdims ?initial ?where a =
                     Py.Module.get_function_with_keywords __wrap_namespace "min"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("initial", Wrap_utils.Option.map initial (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("where", where); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let min_scalar_type a =
                     Py.Module.get_function_with_keywords __wrap_namespace "min_scalar_type"
                       [||]
                       (Wrap_utils.keyword_args [("a", Some(a |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
)))])
                       |> Dtype.of_pyobject
                  let minimum ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "minimum"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let mintypecode ?typeset ?default ~typechars () =
                     Py.Module.get_function_with_keywords __wrap_namespace "mintypecode"
                       [||]
                       (Wrap_utils.keyword_args [("typeset", Wrap_utils.Option.map typeset (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `S x -> Py.String.of_string x
)); ("default", Wrap_utils.Option.map default Py.String.of_string); ("typechars", Some(typechars |> (function
| `Ndarray x -> Obj.to_pyobject x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
)))])
                       |> Py.String.to_string
                  let mirr ~values ~finance_rate ~reinvest_rate () =
                     Py.Module.get_function_with_keywords __wrap_namespace "mirr"
                       [||]
                       (Wrap_utils.keyword_args [("values", Some(values |> Obj.to_pyobject)); ("finance_rate", Some(finance_rate |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
))); ("reinvest_rate", Some(reinvest_rate |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])
                       |> Py.Float.to_float
                  let mod_ ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "mod"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let modf ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "modf"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun x -> (((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 1))))
                  let moveaxis ~source ~destination a =
                     Py.Module.get_function_with_keywords __wrap_namespace "moveaxis"
                       [||]
                       (Wrap_utils.keyword_args [("source", Some(source |> (function
| `Sequence_of_int x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("destination", Some(destination |> (function
| `Sequence_of_int x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("a", Some(a ))])

let msort a =
   Py.Module.get_function_with_keywords __wrap_namespace "msort"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let multiply ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "multiply"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let nan_to_num ?copy ?nan ?posinf ?neginf x =
                     Py.Module.get_function_with_keywords __wrap_namespace "nan_to_num"
                       [||]
                       (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("nan", Wrap_utils.Option.map nan (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("posinf", Wrap_utils.Option.map posinf (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("neginf", Wrap_utils.Option.map neginf (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("x", Some(x |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let nanargmax ?axis a =
   Py.Module.get_function_with_keywords __wrap_namespace "nanargmax"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let nanargmin ?axis a =
   Py.Module.get_function_with_keywords __wrap_namespace "nanargmin"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let nancumprod ?axis ?dtype ?out a =
   Py.Module.get_function_with_keywords __wrap_namespace "nancumprod"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let nancumsum ?axis ?dtype ?out a =
   Py.Module.get_function_with_keywords __wrap_namespace "nancumsum"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let nanmax ?axis ?out ?keepdims a =
                     Py.Module.get_function_with_keywords __wrap_namespace "nanmax"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Tuple_of_int x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let nanmean ?axis ?dtype ?out ?keepdims a =
                     Py.Module.get_function_with_keywords __wrap_namespace "nanmean"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Tuple_of_int x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let nanmedian ?axis ?out ?overwrite_input ?keepdims a =
                     Py.Module.get_function_with_keywords __wrap_namespace "nanmedian"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Sequence_of_int x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("overwrite_input", Wrap_utils.Option.map overwrite_input Py.Bool.of_bool); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let nanmin ?axis ?out ?keepdims a =
                     Py.Module.get_function_with_keywords __wrap_namespace "nanmin"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Tuple_of_int x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let nanpercentile ?axis ?out ?overwrite_input ?interpolation ?keepdims ~q a =
                     Py.Module.get_function_with_keywords __wrap_namespace "nanpercentile"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Tuple_of_int x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("overwrite_input", Wrap_utils.Option.map overwrite_input Py.Bool.of_bool); ("interpolation", Wrap_utils.Option.map interpolation (function
| `Linear -> Py.String.of_string "linear"
| `Lower -> Py.String.of_string "lower"
| `Higher -> Py.String.of_string "higher"
| `Midpoint -> Py.String.of_string "midpoint"
| `Nearest -> Py.String.of_string "nearest"
)); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("q", Some(q |> Obj.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])

                  let nanprod ?axis ?dtype ?out ?keepdims a =
                     Py.Module.get_function_with_keywords __wrap_namespace "nanprod"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Tuple_of_int x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let nanquantile ?axis ?out ?overwrite_input ?interpolation ?keepdims ~q a =
                     Py.Module.get_function_with_keywords __wrap_namespace "nanquantile"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Tuple_of_int x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("overwrite_input", Wrap_utils.Option.map overwrite_input Py.Bool.of_bool); ("interpolation", Wrap_utils.Option.map interpolation (function
| `Linear -> Py.String.of_string "linear"
| `Lower -> Py.String.of_string "lower"
| `Higher -> Py.String.of_string "higher"
| `Midpoint -> Py.String.of_string "midpoint"
| `Nearest -> Py.String.of_string "nearest"
)); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("q", Some(q |> Obj.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])

                  let nanstd ?axis ?dtype ?out ?ddof ?keepdims a =
                     Py.Module.get_function_with_keywords __wrap_namespace "nanstd"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Tuple_of_int x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("ddof", Wrap_utils.Option.map ddof Py.Int.of_int); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let nansum ?axis ?dtype ?out ?keepdims a =
                     Py.Module.get_function_with_keywords __wrap_namespace "nansum"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Tuple_of_int x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let nanvar ?axis ?dtype ?out ?ddof ?keepdims a =
                     Py.Module.get_function_with_keywords __wrap_namespace "nanvar"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Tuple_of_int x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("ddof", Wrap_utils.Option.map ddof Py.Int.of_int); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let ndfromtxt ?kwargs ~fname () =
   Py.Module.get_function_with_keywords __wrap_namespace "ndfromtxt"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("fname", Some(fname ))]) (match kwargs with None -> [] | Some x -> x))

let ndim a =
   Py.Module.get_function_with_keywords __wrap_namespace "ndim"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Obj.to_pyobject))])
     |> Py.Int.to_int
                  let negative ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "negative"
                       (Array.of_list @@ List.concat [[x |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
)]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let nextafter ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "nextafter"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let nonzero a =
   Py.Module.get_function_with_keywords __wrap_namespace "nonzero"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Obj.to_pyobject))])

                  let not_equal ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "not_equal"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let nper ?fv ?when_ ~rate ~pmt ~pv () =
                     Py.Module.get_function_with_keywords __wrap_namespace "nper"
                       [||]
                       (Wrap_utils.keyword_args [("fv", Wrap_utils.Option.map fv Obj.to_pyobject); ("when", Wrap_utils.Option.map when_ (function
| `I x -> Py.Int.of_int x
| `Begin -> Py.String.of_string "begin"
| `PyObject x -> Wrap_utils.id x
)); ("rate", Some(rate |> Obj.to_pyobject)); ("pmt", Some(pmt |> Obj.to_pyobject)); ("pv", Some(pv |> Obj.to_pyobject))])

                  let npv ~rate ~values () =
                     Py.Module.get_function_with_keywords __wrap_namespace "npv"
                       [||]
                       (Wrap_utils.keyword_args [("rate", Some(rate |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
))); ("values", Some(values |> Obj.to_pyobject))])
                       |> Py.Float.to_float
let obj2sctype ?default ~rep () =
   Py.Module.get_function_with_keywords __wrap_namespace "obj2sctype"
     [||]
     (Wrap_utils.keyword_args [("default", default); ("rep", Some(rep ))])

                  let ones ?dtype ?order shape =
                     Py.Module.get_function_with_keywords __wrap_namespace "ones"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let ones_like ?dtype ?order ?subok ?shape a =
                     Py.Module.get_function_with_keywords __wrap_namespace "ones_like"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `A -> Py.String.of_string "A"
| `F -> Py.String.of_string "F"
| `PyObject x -> Wrap_utils.id x
)); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("shape", Wrap_utils.Option.map shape (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let outer ?out ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "outer"
     [||]
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Obj.to_pyobject); ("b", Some(b |> Obj.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let packbits ?axis ?bitorder a =
                     Py.Module.get_function_with_keywords __wrap_namespace "packbits"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("bitorder", Wrap_utils.Option.map bitorder (function
| `Big -> Py.String.of_string "big"
| `Little -> Py.String.of_string "little"
)); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let pad ?mode ?kwargs ~array ~pad_width () =
                     Py.Module.get_function_with_keywords __wrap_namespace "pad"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("array", Some(array )); ("pad_width", Some(pad_width |> (function
| `Ndarray x -> Obj.to_pyobject x
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)))]) (match kwargs with None -> [] | Some x -> x))
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let partition ?axis ?kind ?order ~kth a =
                     Py.Module.get_function_with_keywords __wrap_namespace "partition"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("kind", Wrap_utils.Option.map kind (function
| `Introselect -> Py.String.of_string "introselect"
)); ("order", Wrap_utils.Option.map order (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `S x -> Py.String.of_string x
)); ("kth", Some(kth |> (function
| `Is x -> (fun ml -> Py.List.of_list_map Py.Int.of_int ml) x
| `I x -> Py.Int.of_int x
))); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let percentile ?axis ?out ?overwrite_input ?interpolation ?keepdims ~q a =
                     Py.Module.get_function_with_keywords __wrap_namespace "percentile"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Tuple_of_int x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("overwrite_input", Wrap_utils.Option.map overwrite_input Py.Bool.of_bool); ("interpolation", Wrap_utils.Option.map interpolation (function
| `Linear -> Py.String.of_string "linear"
| `Lower -> Py.String.of_string "lower"
| `Higher -> Py.String.of_string "higher"
| `Midpoint -> Py.String.of_string "midpoint"
| `Nearest -> Py.String.of_string "nearest"
)); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("q", Some(q |> Obj.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])

let piecewise ?kw ~condlist ~funclist x args =
   Py.Module.get_function_with_keywords __wrap_namespace "piecewise"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("condlist", Some(condlist )); ("funclist", Some(funclist )); ("x", Some(x |> Obj.to_pyobject))]) (match kw with None -> [] | Some x -> x))
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let place ~arr ~mask ~vals () =
   Py.Module.get_function_with_keywords __wrap_namespace "place"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr |> Obj.to_pyobject)); ("mask", Some(mask |> Obj.to_pyobject)); ("vals", Some(vals ))])

                  let pmt ?fv ?when_ ~rate ~nper ~pv () =
                     Py.Module.get_function_with_keywords __wrap_namespace "pmt"
                       [||]
                       (Wrap_utils.keyword_args [("fv", Wrap_utils.Option.map fv Obj.to_pyobject); ("when", Wrap_utils.Option.map when_ (function
| `I x -> Py.Int.of_int x
| `Begin -> Py.String.of_string "begin"
| `PyObject x -> Wrap_utils.id x
)); ("rate", Some(rate |> Obj.to_pyobject)); ("nper", Some(nper |> Obj.to_pyobject)); ("pv", Some(pv |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let poly seq_of_zeros =
   Py.Module.get_function_with_keywords __wrap_namespace "poly"
     [||]
     (Wrap_utils.keyword_args [("seq_of_zeros", Some(seq_of_zeros |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let polyadd ~a1 ~a2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "polyadd"
     [||]
     (Wrap_utils.keyword_args [("a1", Some(a1 )); ("a2", Some(a2 ))])

let polyder ?m ~p () =
   Py.Module.get_function_with_keywords __wrap_namespace "polyder"
     [||]
     (Wrap_utils.keyword_args [("m", Wrap_utils.Option.map m Py.Int.of_int); ("p", Some(p ))])

                  let polydiv ~u ~v () =
                     Py.Module.get_function_with_keywords __wrap_namespace "polydiv"
                       [||]
                       (Wrap_utils.keyword_args [("u", Some(u |> (function
| `Ndarray x -> Obj.to_pyobject x
| `Poly1d x -> Wrap_utils.id x
))); ("v", Some(v |> (function
| `Ndarray x -> Obj.to_pyobject x
| `Poly1d x -> Wrap_utils.id x
)))])
                       |> (fun x -> (((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 1))))
                  let polyfit ?rcond ?full ?w ?cov ~y ~deg x =
                     Py.Module.get_function_with_keywords __wrap_namespace "polyfit"
                       [||]
                       (Wrap_utils.keyword_args [("rcond", Wrap_utils.Option.map rcond Py.Float.of_float); ("full", Wrap_utils.Option.map full Py.Bool.of_bool); ("w", Wrap_utils.Option.map w Obj.to_pyobject); ("cov", Wrap_utils.Option.map cov (function
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("y", Some(y |> Obj.to_pyobject)); ("deg", Some(deg |> Py.Int.of_int)); ("x", Some(x |> Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 1))))
                  let polyint ?m ?k ~p () =
                     Py.Module.get_function_with_keywords __wrap_namespace "polyint"
                       [||]
                       (Wrap_utils.keyword_args [("m", Wrap_utils.Option.map m Py.Int.of_int); ("k", Wrap_utils.Option.map k (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `List_of_m_scalars x -> Wrap_utils.id x
| `F x -> Py.Float.of_float x
)); ("p", Some(p |> (function
| `Ndarray x -> Obj.to_pyobject x
| `Poly1d x -> Wrap_utils.id x
)))])

let polymul ~a1 ~a2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "polymul"
     [||]
     (Wrap_utils.keyword_args [("a1", Some(a1 )); ("a2", Some(a2 ))])

let polysub ~a1 ~a2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "polysub"
     [||]
     (Wrap_utils.keyword_args [("a1", Some(a1 )); ("a2", Some(a2 ))])

                  let polyval ~p x =
                     Py.Module.get_function_with_keywords __wrap_namespace "polyval"
                       [||]
                       (Wrap_utils.keyword_args [("p", Some(p |> (function
| `Poly1d_object x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
))); ("x", Some(x |> (function
| `Poly1d_object x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)))])

                  let positive ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "positive"
                       (Array.of_list @@ List.concat [[x |> (function
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `S x -> Py.String.of_string x
| `F x -> Py.Float.of_float x
| `Ndarray x -> Obj.to_pyobject x
)]])
                       (Wrap_utils.keyword_args [("out", out); ("where", where)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let power ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "power"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let ppmt ?fv ?when_ ~rate ~per ~nper ~pv () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ppmt"
                       [||]
                       (Wrap_utils.keyword_args [("fv", Wrap_utils.Option.map fv Obj.to_pyobject); ("when", Wrap_utils.Option.map when_ (function
| `I x -> Py.Int.of_int x
| `Begin -> Py.String.of_string "begin"
| `PyObject x -> Wrap_utils.id x
)); ("rate", Some(rate |> Obj.to_pyobject)); ("per", Some(per |> (function
| `Ndarray x -> Obj.to_pyobject x
| `I x -> Py.Int.of_int x
))); ("nper", Some(nper |> Obj.to_pyobject)); ("pv", Some(pv |> Obj.to_pyobject))])

let printoptions ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "printoptions"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

                  let prod ?axis ?dtype ?out ?keepdims ?initial ?where a =
                     Py.Module.get_function_with_keywords __wrap_namespace "prod"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("initial", Wrap_utils.Option.map initial (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("where", where); ("a", Some(a |> Obj.to_pyobject))])

let product ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "product"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

                  let promote_types ~type1 ~type2 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "promote_types"
                       [||]
                       (Wrap_utils.keyword_args [("type1", Some(type1 |> (function
| `Dtype x -> Dtype.to_pyobject x
| `Dtype_specifier x -> Wrap_utils.id x
))); ("type2", Some(type2 |> (function
| `Dtype x -> Dtype.to_pyobject x
| `Dtype_specifier x -> Wrap_utils.id x
)))])
                       |> Dtype.of_pyobject
let ptp ?axis ?out ?keepdims a =
   Py.Module.get_function_with_keywords __wrap_namespace "ptp"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let put ?mode ~ind ~v a =
                     Py.Module.get_function_with_keywords __wrap_namespace "put"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Raise -> Py.String.of_string "raise"
| `Wrap -> Py.String.of_string "wrap"
| `Clip -> Py.String.of_string "clip"
)); ("ind", Some(ind |> Obj.to_pyobject)); ("v", Some(v |> Obj.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])

let put_along_axis ~arr ~indices ~values ~axis () =
   Py.Module.get_function_with_keywords __wrap_namespace "put_along_axis"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr )); ("indices", Some(indices )); ("values", Some(values )); ("axis", Some(axis |> Py.Int.of_int))])

let putmask ~mask ~values a =
   Py.Module.get_function_with_keywords __wrap_namespace "putmask"
     [||]
     (Wrap_utils.keyword_args [("mask", Some(mask |> Obj.to_pyobject)); ("values", Some(values |> Obj.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])

                  let pv ?fv ?when_ ~rate ~nper ~pmt () =
                     Py.Module.get_function_with_keywords __wrap_namespace "pv"
                       [||]
                       (Wrap_utils.keyword_args [("fv", Wrap_utils.Option.map fv Obj.to_pyobject); ("when", Wrap_utils.Option.map when_ (function
| `I x -> Py.Int.of_int x
| `Begin -> Py.String.of_string "begin"
| `PyObject x -> Wrap_utils.id x
)); ("rate", Some(rate |> Obj.to_pyobject)); ("nper", Some(nper |> Obj.to_pyobject)); ("pmt", Some(pmt |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let quantile ?axis ?out ?overwrite_input ?interpolation ?keepdims ~q a =
                     Py.Module.get_function_with_keywords __wrap_namespace "quantile"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Tuple_of_int x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("overwrite_input", Wrap_utils.Option.map overwrite_input Py.Bool.of_bool); ("interpolation", Wrap_utils.Option.map interpolation (function
| `Linear -> Py.String.of_string "linear"
| `Lower -> Py.String.of_string "lower"
| `Higher -> Py.String.of_string "higher"
| `Midpoint -> Py.String.of_string "midpoint"
| `Nearest -> Py.String.of_string "nearest"
)); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("q", Some(q |> Obj.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])

                  let rad2deg ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "rad2deg"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let radians ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "radians"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let rate ?when_ ?guess ?tol ?maxiter ~nper ~pmt ~pv ~fv () =
                     Py.Module.get_function_with_keywords __wrap_namespace "rate"
                       [||]
                       (Wrap_utils.keyword_args [("when", Wrap_utils.Option.map when_ (function
| `I x -> Py.Int.of_int x
| `Begin -> Py.String.of_string "begin"
| `PyObject x -> Wrap_utils.id x
)); ("guess", guess); ("tol", tol); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("nper", Some(nper |> Obj.to_pyobject)); ("pmt", Some(pmt |> Obj.to_pyobject)); ("pv", Some(pv |> Obj.to_pyobject)); ("fv", Some(fv |> Obj.to_pyobject))])

                  let ravel ?order a =
                     Py.Module.get_function_with_keywords __wrap_namespace "ravel"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
| `A -> Py.String.of_string "A"
| `K -> Py.String.of_string "K"
)); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let ravel_multi_index ?mode ?order ~multi_index ~dims () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ravel_multi_index"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Raise -> Py.String.of_string "raise"
| `Wrap -> Py.String.of_string "wrap"
| `Clip -> Py.String.of_string "clip"
)); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("multi_index", Some(multi_index )); ("dims", Some(dims |> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let real val_ =
   Py.Module.get_function_with_keywords __wrap_namespace "real"
     [||]
     (Wrap_utils.keyword_args [("val", Some(val_ |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let real_if_close ?tol a =
   Py.Module.get_function_with_keywords __wrap_namespace "real_if_close"
     [||]
     (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let recfromcsv ?kwargs ~fname () =
   Py.Module.get_function_with_keywords __wrap_namespace "recfromcsv"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("fname", Some(fname ))]) (match kwargs with None -> [] | Some x -> x))

let recfromtxt ?kwargs ~fname () =
   Py.Module.get_function_with_keywords __wrap_namespace "recfromtxt"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("fname", Some(fname ))]) (match kwargs with None -> [] | Some x -> x))

                  let reciprocal ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "reciprocal"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let remainder ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "remainder"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let repeat ?axis ~repeats a =
                     Py.Module.get_function_with_keywords __wrap_namespace "repeat"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("repeats", Some(repeats |> (function
| `I x -> Py.Int.of_int x
| `Array_of_ints x -> Wrap_utils.id x
))); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let require ?dtype ?requirements a =
                     Py.Module.get_function_with_keywords __wrap_namespace "require"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("requirements", Wrap_utils.Option.map requirements (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `S x -> Py.String.of_string x
)); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let reshape ?order ~newshape a =
                     Py.Module.get_function_with_keywords __wrap_namespace "reshape"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
| `A -> Py.String.of_string "A"
)); ("newshape", Some(newshape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml))); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let resize ~new_shape a =
                     Py.Module.get_function_with_keywords __wrap_namespace "resize"
                       [||]
                       (Wrap_utils.keyword_args [("new_shape", Some(new_shape |> (function
| `Tuple_of_int x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
))); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let result_type arrays_and_dtypes =
   Py.Module.get_function_with_keywords __wrap_namespace "result_type"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arrays_and_dtypes)])
     []
     |> Dtype.of_pyobject
                  let right_shift ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "right_shift"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let rint ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "rint"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let roll ?axis ~shift a =
   Py.Module.get_function_with_keywords __wrap_namespace "roll"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("shift", Some(shift |> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml))); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let rollaxis ?start ~axis a =
   Py.Module.get_function_with_keywords __wrap_namespace "rollaxis"
     [||]
     (Wrap_utils.keyword_args [("start", Wrap_utils.Option.map start Py.Int.of_int); ("axis", Some(axis |> Py.Int.of_int)); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let roots p =
   Py.Module.get_function_with_keywords __wrap_namespace "roots"
     [||]
     (Wrap_utils.keyword_args [("p", Some(p |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let rot90 ?k ?axes ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "rot90"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("axes", axes); ("m", Some(m |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let round ?decimals ?out a =
   Py.Module.get_function_with_keywords __wrap_namespace "round"
     [||]
     (Wrap_utils.keyword_args [("decimals", decimals); ("out", out); ("a", Some(a ))])

let row_stack tup =
   Py.Module.get_function_with_keywords __wrap_namespace "row_stack"
     [||]
     (Wrap_utils.keyword_args [("tup", Some(tup |> (fun ml -> Py.List.of_list_map Obj.to_pyobject ml)))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let safe_eval source =
   Py.Module.get_function_with_keywords __wrap_namespace "safe_eval"
     [||]
     (Wrap_utils.keyword_args [("source", Some(source |> Py.String.of_string))])

                  let save ?allow_pickle ?fix_imports ~file ~arr () =
                     Py.Module.get_function_with_keywords __wrap_namespace "save"
                       [||]
                       (Wrap_utils.keyword_args [("allow_pickle", Wrap_utils.Option.map allow_pickle Py.Bool.of_bool); ("fix_imports", Wrap_utils.Option.map fix_imports Py.Bool.of_bool); ("file", Some(file |> (function
| `S x -> Py.String.of_string x
| `PyObject x -> Wrap_utils.id x
))); ("arr", Some(arr |> Obj.to_pyobject))])

                  let savetxt ?fmt ?delimiter ?newline ?header ?footer ?comments ?encoding ~fname ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "savetxt"
                       [||]
                       (Wrap_utils.keyword_args [("fmt", Wrap_utils.Option.map fmt (function
| `Sequence_of_strs x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("delimiter", Wrap_utils.Option.map delimiter Py.String.of_string); ("newline", Wrap_utils.Option.map newline Py.String.of_string); ("header", Wrap_utils.Option.map header Py.String.of_string); ("footer", Wrap_utils.Option.map footer Py.String.of_string); ("comments", Wrap_utils.Option.map comments Py.String.of_string); ("encoding", Wrap_utils.Option.map encoding Py.String.of_string); ("fname", Some(fname )); ("X", Some(x ))])

                  let savez ?kwds ~file args =
                     Py.Module.get_function_with_keywords __wrap_namespace "savez"
                       (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
                       (List.rev_append (Wrap_utils.keyword_args [("file", Some(file |> (function
| `File x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))]) (match kwds with None -> [] | Some x -> x))

                  let savez_compressed ?kwds ~file args =
                     Py.Module.get_function_with_keywords __wrap_namespace "savez_compressed"
                       (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
                       (List.rev_append (Wrap_utils.keyword_args [("file", Some(file |> (function
| `File x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))]) (match kwds with None -> [] | Some x -> x))

let sctype2char sctype =
   Py.Module.get_function_with_keywords __wrap_namespace "sctype2char"
     [||]
     (Wrap_utils.keyword_args [("sctype", Some(sctype ))])
     |> Py.String.to_string
                  let searchsorted ?side ?sorter ~v a =
                     Py.Module.get_function_with_keywords __wrap_namespace "searchsorted"
                       [||]
                       (Wrap_utils.keyword_args [("side", Wrap_utils.Option.map side (function
| `Left -> Py.String.of_string "left"
| `Right -> Py.String.of_string "right"
)); ("sorter", sorter); ("v", Some(v |> Obj.to_pyobject)); ("a", Some(a ))])

                  let select ?default ~condlist ~choicelist () =
                     Py.Module.get_function_with_keywords __wrap_namespace "select"
                       [||]
                       (Wrap_utils.keyword_args [("default", Wrap_utils.Option.map default (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("condlist", Some(condlist )); ("choicelist", Some(choicelist ))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let set_printoptions ?precision ?threshold ?edgeitems ?linewidth ?suppress ?nanstr ?infstr ?formatter ?sign ?floatmode ?legacy () =
                     Py.Module.get_function_with_keywords __wrap_namespace "set_printoptions"
                       [||]
                       (Wrap_utils.keyword_args [("precision", Wrap_utils.Option.map precision Py.Int.of_int); ("threshold", Wrap_utils.Option.map threshold Py.Int.of_int); ("edgeitems", Wrap_utils.Option.map edgeitems Py.Int.of_int); ("linewidth", Wrap_utils.Option.map linewidth Py.Int.of_int); ("suppress", Wrap_utils.Option.map suppress Py.Bool.of_bool); ("nanstr", Wrap_utils.Option.map nanstr Py.String.of_string); ("infstr", Wrap_utils.Option.map infstr Py.String.of_string); ("formatter", formatter); ("sign", Wrap_utils.Option.map sign (function
| `Space -> Py.String.of_string " "
| `Plus -> Py.String.of_string "+"
| `Minus -> Py.String.of_string "-"
)); ("floatmode", Wrap_utils.Option.map floatmode Py.String.of_string); ("legacy", Wrap_utils.Option.map legacy (function
| `T_False_ x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
))])

                  let set_string_function ?repr ~f () =
                     Py.Module.get_function_with_keywords __wrap_namespace "set_string_function"
                       [||]
                       (Wrap_utils.keyword_args [("repr", Wrap_utils.Option.map repr Py.Bool.of_bool); ("f", Some(f |> (function
| `Callable x -> Wrap_utils.id x
| `None -> Py.none
)))])

let setbufsize size =
   Py.Module.get_function_with_keywords __wrap_namespace "setbufsize"
     [||]
     (Wrap_utils.keyword_args [("size", Some(size |> Py.Int.of_int))])

let setdiff1d ?assume_unique ~ar1 ~ar2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "setdiff1d"
     [||]
     (Wrap_utils.keyword_args [("assume_unique", Wrap_utils.Option.map assume_unique Py.Bool.of_bool); ("ar1", Some(ar1 |> Obj.to_pyobject)); ("ar2", Some(ar2 |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let seterr ?all ?divide ?over ?under ?invalid () =
                     Py.Module.get_function_with_keywords __wrap_namespace "seterr"
                       [||]
                       (Wrap_utils.keyword_args [("all", Wrap_utils.Option.map all (function
| `Warn -> Py.String.of_string "warn"
| `Print -> Py.String.of_string "print"
| `Ignore -> Py.String.of_string "ignore"
| `Raise -> Py.String.of_string "raise"
| `Log -> Py.String.of_string "log"
| `Call -> Py.String.of_string "call"
)); ("divide", Wrap_utils.Option.map divide (function
| `Warn -> Py.String.of_string "warn"
| `Print -> Py.String.of_string "print"
| `Ignore -> Py.String.of_string "ignore"
| `Raise -> Py.String.of_string "raise"
| `Log -> Py.String.of_string "log"
| `Call -> Py.String.of_string "call"
)); ("over", Wrap_utils.Option.map over (function
| `Warn -> Py.String.of_string "warn"
| `Print -> Py.String.of_string "print"
| `Ignore -> Py.String.of_string "ignore"
| `Raise -> Py.String.of_string "raise"
| `Log -> Py.String.of_string "log"
| `Call -> Py.String.of_string "call"
)); ("under", Wrap_utils.Option.map under (function
| `Warn -> Py.String.of_string "warn"
| `Print -> Py.String.of_string "print"
| `Ignore -> Py.String.of_string "ignore"
| `Raise -> Py.String.of_string "raise"
| `Log -> Py.String.of_string "log"
| `Call -> Py.String.of_string "call"
)); ("invalid", Wrap_utils.Option.map invalid (function
| `Warn -> Py.String.of_string "warn"
| `Print -> Py.String.of_string "print"
| `Ignore -> Py.String.of_string "ignore"
| `Raise -> Py.String.of_string "raise"
| `Log -> Py.String.of_string "log"
| `Call -> Py.String.of_string "call"
))])

let seterrcall func =
   Py.Module.get_function_with_keywords __wrap_namespace "seterrcall"
     [||]
     (Wrap_utils.keyword_args [("func", Some(func ))])
     |> (fun py -> if Py.is_none py then None else Some (Wrap_utils.id py))
let seterrobj errobj =
   Py.Module.get_function_with_keywords __wrap_namespace "seterrobj"
     [||]
     (Wrap_utils.keyword_args [("errobj", Some(errobj |> Obj.to_pyobject))])

let setxor1d ?assume_unique ~ar1 ~ar2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "setxor1d"
     [||]
     (Wrap_utils.keyword_args [("assume_unique", Wrap_utils.Option.map assume_unique Py.Bool.of_bool); ("ar1", Some(ar1 )); ("ar2", Some(ar2 ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
        let shape a =
           Py.Module.get_function_with_keywords __wrap_namespace "shape"
             [||]
             (Wrap_utils.keyword_args [("a", Some(a |> Obj.to_pyobject))])
             |> (fun py -> let len = Py.Sequence.length py in Array.init len
(fun i -> Py.Int.to_int (Py.Sequence.get_item py i)))
let shares_memory ?max_work ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "shares_memory"
     [||]
     (Wrap_utils.keyword_args [("max_work", Wrap_utils.Option.map max_work Py.Int.of_int); ("b", Some(b )); ("a", Some(a ))])
     |> Py.Bool.to_bool
let show_config () =
   Py.Module.get_function_with_keywords __wrap_namespace "show_config"
     [||]
     []

                  let sign ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "sign"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let signbit ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "signbit"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let sin ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "sin"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let sinc x =
   Py.Module.get_function_with_keywords __wrap_namespace "sinc"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let sinh ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "sinh"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let size ?axis a =
   Py.Module.get_function_with_keywords __wrap_namespace "size"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("a", Some(a |> Obj.to_pyobject))])
     |> Py.Int.to_int
let sometrue ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "sometrue"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

                  let sort ?axis ?kind ?order a =
                     Py.Module.get_function_with_keywords __wrap_namespace "sort"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("kind", Wrap_utils.Option.map kind (function
| `Heapsort -> Py.String.of_string "heapsort"
| `Mergesort -> Py.String.of_string "mergesort"
| `Stable -> Py.String.of_string "stable"
| `Quicksort -> Py.String.of_string "quicksort"
)); ("order", Wrap_utils.Option.map order (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `S x -> Py.String.of_string x
)); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let sort_complex a =
   Py.Module.get_function_with_keywords __wrap_namespace "sort_complex"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Obj.to_pyobject))])

let source ?output ~object_ () =
   Py.Module.get_function_with_keywords __wrap_namespace "source"
     [||]
     (Wrap_utils.keyword_args [("output", output); ("object", Some(object_ ))])

                  let spacing ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "spacing"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let split ?axis ~ary ~indices_or_sections () =
                     Py.Module.get_function_with_keywords __wrap_namespace "split"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("ary", Some(ary |> Obj.to_pyobject)); ("indices_or_sections", Some(indices_or_sections |> (function
| `I x -> Py.Int.of_int x
| `T1_D_array x -> Wrap_utils.id x
)))])

                  let sqrt ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "sqrt"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let square ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "square"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let squeeze ?axis a =
   Py.Module.get_function_with_keywords __wrap_namespace "squeeze"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let stack ?axis ?out ~arrays () =
   Py.Module.get_function_with_keywords __wrap_namespace "stack"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("arrays", Some(arrays ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let std ?axis ?dtype ?out ?ddof ?keepdims a =
   Py.Module.get_function_with_keywords __wrap_namespace "std"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("ddof", Wrap_utils.Option.map ddof Py.Int.of_int); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let subtract ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "subtract"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let sum ?axis ?dtype ?out ?keepdims ?initial ?where a =
                     Py.Module.get_function_with_keywords __wrap_namespace "sum"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("initial", Wrap_utils.Option.map initial (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("where", where); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let swapaxes ~axis1 ~axis2 a =
   Py.Module.get_function_with_keywords __wrap_namespace "swapaxes"
     [||]
     (Wrap_utils.keyword_args [("axis1", Some(axis1 |> Py.Int.of_int)); ("axis2", Some(axis2 |> Py.Int.of_int)); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let take ?axis ?out ?mode ~indices a =
                     Py.Module.get_function_with_keywords __wrap_namespace "take"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("out", Wrap_utils.Option.map out (function
| `T_Ni_Nj_Nk_ x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("mode", Wrap_utils.Option.map mode (function
| `Raise -> Py.String.of_string "raise"
| `Wrap -> Py.String.of_string "wrap"
| `Clip -> Py.String.of_string "clip"
)); ("indices", Some(indices )); ("a", Some(a ))])

let take_along_axis ~arr ~indices ~axis () =
   Py.Module.get_function_with_keywords __wrap_namespace "take_along_axis"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr )); ("indices", Some(indices )); ("axis", Some(axis |> Py.Int.of_int))])

                  let tan ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "tan"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let tanh ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "tanh"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let tensordot ?axes ~b a =
                     Py.Module.get_function_with_keywords __wrap_namespace "tensordot"
                       [||]
                       (Wrap_utils.keyword_args [("axes", Wrap_utils.Option.map axes (function
| `I x -> Py.Int.of_int x
| `T_2_array_like x -> Wrap_utils.id x
)); ("b", Some(b )); ("a", Some(a ))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let tile ~a ~reps () =
   Py.Module.get_function_with_keywords __wrap_namespace "tile"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Obj.to_pyobject)); ("reps", Some(reps |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let trace ?offset ?axis1 ?axis2 ?dtype ?out a =
   Py.Module.get_function_with_keywords __wrap_namespace "trace"
     [||]
     (Wrap_utils.keyword_args [("offset", Wrap_utils.Option.map offset Py.Int.of_int); ("axis1", axis1); ("axis2", axis2); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let transpose ?axes a =
   Py.Module.get_function_with_keywords __wrap_namespace "transpose"
     [||]
     (Wrap_utils.keyword_args [("axes", axes); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let trapz ?x ?dx ?axis ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "trapz"
                       [||]
                       (Wrap_utils.keyword_args [("x", Wrap_utils.Option.map x Obj.to_pyobject); ("dx", Wrap_utils.Option.map dx (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("y", Some(y |> Obj.to_pyobject))])
                       |> Py.Float.to_float
let tri ?m ?k ?dtype ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "tri"
     [||]
     (Wrap_utils.keyword_args [("M", Wrap_utils.Option.map m Py.Int.of_int); ("k", Wrap_utils.Option.map k Py.Int.of_int); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("N", Some(n |> Py.Int.of_int))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let tril ?k ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "tril"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("m", Some(m |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let tril_indices ?k ?m ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "tril_indices"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("m", Wrap_utils.Option.map m Py.Int.of_int); ("n", Some(n |> Py.Int.of_int))])

let tril_indices_from ?k ~arr () =
   Py.Module.get_function_with_keywords __wrap_namespace "tril_indices_from"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("arr", Some(arr |> Obj.to_pyobject))])

let trim_zeros ?trim ~filt () =
   Py.Module.get_function_with_keywords __wrap_namespace "trim_zeros"
     [||]
     (Wrap_utils.keyword_args [("trim", Wrap_utils.Option.map trim Py.String.of_string); ("filt", Some(filt ))])

let triu ?k ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "triu"
     [||]
     (Wrap_utils.keyword_args [("k", k); ("m", Some(m ))])

let triu_indices ?k ?m ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "triu_indices"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("m", Wrap_utils.Option.map m Py.Int.of_int); ("n", Some(n |> Py.Int.of_int))])

let triu_indices_from ?k ~arr () =
   Py.Module.get_function_with_keywords __wrap_namespace "triu_indices_from"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("arr", Some(arr |> Obj.to_pyobject))])

                  let true_divide ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "true_divide"
                       (Array.of_list @@ List.concat [[x ]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let trunc ?out ?where x =
                     Py.Module.get_function_with_keywords __wrap_namespace "trunc"
                       (Array.of_list @@ List.concat [[x |> Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)); ("where", Wrap_utils.Option.map where Obj.to_pyobject)])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let typename char =
   Py.Module.get_function_with_keywords __wrap_namespace "typename"
     [||]
     (Wrap_utils.keyword_args [("char", Some(char |> Py.String.of_string))])
     |> Py.String.to_string
let union1d ~ar1 ~ar2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "union1d"
     [||]
     (Wrap_utils.keyword_args [("ar1", Some(ar1 )); ("ar2", Some(ar2 ))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let unique ?return_index ?return_inverse ?return_counts ?axis ~ar () =
   Py.Module.get_function_with_keywords __wrap_namespace "unique"
     [||]
     (Wrap_utils.keyword_args [("return_index", Wrap_utils.Option.map return_index Py.Bool.of_bool); ("return_inverse", Wrap_utils.Option.map return_inverse Py.Bool.of_bool); ("return_counts", Wrap_utils.Option.map return_counts Py.Bool.of_bool); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("ar", Some(ar |> Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 2)), ((fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t)) (Py.Tuple.get x 3))))
                  let unpackbits ?axis ?count ?bitorder a =
                     Py.Module.get_function_with_keywords __wrap_namespace "unpackbits"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("count", Wrap_utils.Option.map count Py.Int.of_int); ("bitorder", Wrap_utils.Option.map bitorder (function
| `Big -> Py.String.of_string "big"
| `Little -> Py.String.of_string "little"
)); ("a", Some(a |> (function
| `Uint8_type x -> Wrap_utils.id x
| `Ndarray x -> Obj.to_pyobject x
)))])

                  let unravel_index ?order ~indices shape =
                     Py.Module.get_function_with_keywords __wrap_namespace "unravel_index"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("indices", Some(indices |> Obj.to_pyobject)); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])

let unwrap ?discont ?axis ~p () =
   Py.Module.get_function_with_keywords __wrap_namespace "unwrap"
     [||]
     (Wrap_utils.keyword_args [("discont", Wrap_utils.Option.map discont Py.Float.of_float); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("p", Some(p |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let vander ?n ?increasing x =
   Py.Module.get_function_with_keywords __wrap_namespace "vander"
     [||]
     (Wrap_utils.keyword_args [("N", Wrap_utils.Option.map n Py.Int.of_int); ("increasing", Wrap_utils.Option.map increasing Py.Bool.of_bool); ("x", Some(x |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let var ?axis ?dtype ?out ?ddof ?keepdims a =
   Py.Module.get_function_with_keywords __wrap_namespace "var"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Obj.to_pyobject); ("ddof", Wrap_utils.Option.map ddof Py.Int.of_int); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let vdot ~b a =
   Py.Module.get_function_with_keywords __wrap_namespace "vdot"
     [||]
     (Wrap_utils.keyword_args [("b", Some(b |> Obj.to_pyobject)); ("a", Some(a |> Obj.to_pyobject))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let vsplit ~ary ~indices_or_sections () =
   Py.Module.get_function_with_keywords __wrap_namespace "vsplit"
     [||]
     (Wrap_utils.keyword_args [("ary", Some(ary )); ("indices_or_sections", Some(indices_or_sections ))])

let vstack tup =
   Py.Module.get_function_with_keywords __wrap_namespace "vstack"
     [||]
     (Wrap_utils.keyword_args [("tup", Some(tup |> (fun ml -> Py.List.of_list_map Obj.to_pyobject ml)))])
     |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let where ?x ?y ~condition () =
                     Py.Module.get_function_with_keywords __wrap_namespace "where"
                       [||]
                       (Wrap_utils.keyword_args [("x", x); ("y", y); ("condition", Some(condition |> (function
| `Bool x -> Py.Bool.of_bool x
| `Ndarray x -> Obj.to_pyobject x
)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
let who ?vardict () =
   Py.Module.get_function_with_keywords __wrap_namespace "who"
     [||]
     (Wrap_utils.keyword_args [("vardict", vardict)])

                  let zeros ?dtype ?order shape =
                     Py.Module.get_function_with_keywords __wrap_namespace "zeros"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
                  let zeros_like ?dtype ?order ?subok ?shape a =
                     Py.Module.get_function_with_keywords __wrap_namespace "zeros_like"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `A -> Py.String.of_string "A"
| `F -> Py.String.of_string "F"
| `PyObject x -> Wrap_utils.id x
)); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("shape", Wrap_utils.Option.map shape (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("a", Some(a |> Obj.to_pyobject))])
                       |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
