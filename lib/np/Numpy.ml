module type Numpy_no_ndarray = module type of struct
  include NumpyRaw
end
with module Ndarray := NumpyRaw.Ndarray

include (NumpyRaw : Numpy_no_ndarray)

include Wrap_utils

module Ndarray = struct
  include NumpyRaw.Ndarray

  type ndarray = t
  
  let raw_array ?dtype ?copy ?order ?subok ?ndmin object_ =
    Py.Module.get_function_with_keywords Types.numpy "array" [||]
      (Wrap_utils.keyword_args
         [
           ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject);
           ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool);
           ( "order",
             Wrap_utils.Option.map order (function
                 | `K -> Py.String.of_string "K"
                 | `A -> Py.String.of_string "A"
                 | `C -> Py.String.of_string "C"
                 | `F -> Py.String.of_string "F") );
           ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool);
           ("ndmin", Wrap_utils.Option.map ndmin Py.Int.of_int);
           ("object", Some (object_));
         ])
    |> fun py -> (Obj.of_pyobject py : [ `ArrayLike | `Ndarray | `Object ] Obj.t)

  let of_int_list x = raw_array (Py.List.of_list_map Py.Int.of_int x)

  let of_int_array x = raw_array (Py.List.of_array_map Py.Int.of_int x)

  let vectori = of_int_array

  let of_float_list x =
    raw_array (Py.List.of_list_map Py.Float.of_float x)

  let of_float_array x =
    raw_array (Py.List.of_array_map Py.Float.of_float x)

  let vectorf = of_float_array
  
  let of_string_list x =
    raw_array (Py.List.of_list_map Py.String.of_string x)

  let of_string_array x =
    raw_array (Py.List.of_array_map Py.String.of_string x)

  let vectors = of_string_array

  let py_of_object = function
    | `F x -> Py.Float.of_float x
    | `I x -> Py.Int.of_int x
    | `S x -> Py.String.of_string x
    | `B x -> Py.Bool.of_bool x

  let of_object_list x =
    raw_array ~dtype:`Object (Py.List.of_list_map py_of_object x)

  let of_object_array x =
    raw_array ~dtype:`Object(Py.List.of_array_map py_of_object x)

  let vectoro = of_object_array
  
  (* let arrayi vals =
   *   Py.Array.of_array Py.Int.of_int Py.Int.to_int vals
   * 
   * let arrayf vals =
   *   Py.Array.of_array Py.Float.of_float Py.Float.to_float vals *)
  (* let matrixi vals = raw_array (Py.List.of_array_map arrayi vals) *)

  
  let matrixi vals = raw_array (Py.List.of_array_map (Py.List.of_array_map Py.Int.of_int) vals)
  let matrixf vals = raw_array (Py.List.of_array_map (Py.List.of_array_map Py.Float.of_float) vals)
  let matrixs vals = raw_array (Py.List.of_array_map (Py.List.of_array_map Py.String.of_string) vals)
  let matrixo vals = raw_array ~dtype:`Object (Py.List.of_array_map (Py.List.of_array_map py_of_object) vals)

  let float x = raw_array (Py.Float.of_float x)
  let int x = raw_array (Py.Int.of_int x)
  let string x = raw_array (Py.String.of_string x)
  let bool x = raw_array (Py.Bool.of_bool x)

  let to_int_array x =
    flatten x |> to_pyobject |> Py.Sequence.to_array_map Py.Int.to_int

  let to_float_array x =
    flatten x |> to_pyobject |> Py.Sequence.to_array_map Py.Float.to_float

  let to_string_array x =
    flatten x |> to_pyobject |> Py.Sequence.to_array_map Py.String.to_string

  module List = struct
    include PyList.Make(NumpyRaw.Ndarray)
    let vectori xs = of_list_map vectori xs
    let vectors xs = of_list_map vectors xs
  end

  let iter x =
    Py.Iter.to_seq (__iter__ x) |> Seq.map of_pyobject

  let slice = slice
  let mask = mask
end

module Obj = Obj

let vectori = Ndarray.vectori
let vectorf = Ndarray.vectorf
let vectors = Ndarray.vectors
let vectoro = Ndarray.vectoro
let matrixi = Ndarray.matrixi
let matrixf = Ndarray.matrixf
let matrixs = Ndarray.matrixs
let matrixo = Ndarray.matrixo
let float = Ndarray.float
let int = Ndarray.int
let string = Ndarray.string
let bool = Ndarray.bool

module Ops = struct
  let operator = Py.import "operator"

  let binop name =
    let f = Py.Module.get_function operator name in fun a b ->
      Ndarray.(of_pyobject @@ f [|to_pyobject a; to_pyobject b|])

  let ( - ) a b = binop "sub" a b
  let ( + ) a b = binop "add" a b
  let ( * ) a b = binop "mul" a b
  let ( / ) a b = binop "truediv" a b

  let ( < ) a b = binop "lt" a b
  let ( <= ) a b = binop "le" a b
  let ( > ) a b = binop "gt" a b
  let ( >= ) a b = binop "get" a b

  let ( = ) a b = binop "eq" a b
  let ( != ) a b = binop "ne" a b
end

include Ops

let pp fmt x = Ndarray.pp fmt x

let py_of_number = function
  | `F x -> Py.Float.of_float x
  | `I x -> Py.Int.of_int x

(* Wrapping for this is weird. Passing just stop as a named param
   fails, if you want just stop, you need to pass it as an unnamed
   param. It seems easier to fix this here than to attempt to fix it
   in a more general and complicated way in skdoc.py. *)
let arange ?start ?step ?dtype stop =
  let f = Py.Module.get_function_with_keywords Types.numpy "arange" in
  match start with
  | None ->
    f [|py_of_number stop|]
      (Wrap_utils.keyword_args [("step", Wrap_utils.Option.map step py_of_number);
                                ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject)])
    |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))
  | Some start ->
    f [|py_of_number start; py_of_number stop|]
      (Wrap_utils.keyword_args [("step", Wrap_utils.Option.map step py_of_number);
                                ("dtype", Wrap_utils.Option.map dtype Dtype.to_pyobject)])
    |> (fun py -> (Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Obj.t))

