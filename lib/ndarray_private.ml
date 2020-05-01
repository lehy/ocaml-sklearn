Wrap_utils.init ()
let numpy = Py.import "numpy"

module M = struct
  type t = Py.Object.t
  let show x = Py.Object.to_string x
  let pp fmt x = Format.fprintf fmt "%s" (Py.Object.to_string x)
  let of_pyobject x = x
  let to_pyobject x = x
  let of_bigarray ba = Numpy.of_bigarray ba
end

include M

let shape self =
  match Py.Object.get_attr_string self "shape" with
  | None -> raise (Wrap_utils.Attribute_not_found "shape")
  | Some x -> Py.List.to_array_map Py.Int.to_int x

let arange ?start ?step stop =
  let stop = Py.Int.of_int stop in
  let args = match start, step with
    | None, None -> [| stop |]
    | Some start, None -> [| Py.Int.of_int start; stop |]
    | None, Some step -> [| Py.Int.of_int 0; stop; Py.Int.of_int step |]
    | Some start, Some step -> [| Py.Int.of_int start; stop; Py.Int.of_int step |]
  in
  Py.Module.get_function numpy "arange" args
  |> of_pyobject

let reshape ~shape x =
  Py.Object.call_method x "reshape" [| Py.List.of_array_map Py.Int.of_int shape |] |> of_pyobject  

module List = PyList.Make(M)

module Float = struct
  let matrix values =
    Bigarray.Array2.of_array Bigarray.float64 Bigarray.c_layout values
    |> Bigarray.genarray_of_array2 |> Numpy.of_bigarray
  let vector values =
    Bigarray.Array1.of_array Bigarray.float64 Bigarray.c_layout values
    |> Bigarray.genarray_of_array1 |> Numpy.of_bigarray
  let of_bigarray ba = Numpy.of_bigarray ba

  let to_bigarray x = Numpy.to_bigarray Bigarray.float64 Bigarray.c_layout x
 
  let matrices values =
    List.of_list_map matrix values
  let vectors values =
    List.of_list_map vector values
end

module Int = struct
  let vector ia =
    Bigarray.Array1.of_array Bigarray.nativeint Bigarray.c_layout (Array.map Nativeint.of_int ia)
    |> Bigarray.genarray_of_array1 |> Numpy.of_bigarray
  let matrix ia =
    Bigarray.Array2.of_array Bigarray.nativeint Bigarray.c_layout (Array.map (Array.map Nativeint.of_int) ia)
    |> Bigarray.genarray_of_array2 |> Numpy.of_bigarray
  let of_bigarray ba = Numpy.of_bigarray ba

  let to_bigarray x = Numpy.to_bigarray Bigarray.nativeint Bigarray.c_layout x
  
  let matrices values =
    List.of_list_map matrix values
  let vectors values =
    List.of_list_map vector values
end

module Dtype = struct
  type t = [`Object | `S of string]
  let rec to_pyobject = function
    | `S s -> Py.Module.get_function numpy "dtype" [|Py.String.of_string s|]
    | `Object -> to_pyobject (`S "object")
end

let numpy_array ?dtype a =
  match dtype with
  | None -> Py.Module.get_function numpy "array" [|a|]
  | Some dtype ->
    Py.Module.get_function_with_keywords numpy "array" [|a|] ["dtype", Dtype.to_pyobject dtype]

module String = struct
  let py_of_array a = Py.List.of_array_map Py.String.of_string a

  let vector ia =
    (* XXX TODO figure out a way to do this with one copy less *)
    numpy_array @@ py_of_array ia

  let matrix aa =
    numpy_array @@ Py.List.of_array_map py_of_array aa
  
  let vectors values =
    List.of_list_map vector values
end

module Object = struct
  type elt = [`I of int | `F of float | `S of string | `Arr of t]
  let py_of_elt x = match x with
    | `I x -> Py.Int.of_int x
    | `F x -> Py.Float.of_float x
    | `S x -> Py.String.of_string x
    | `Arr x -> of_pyobject x
  let py_of_array a = Py.List.of_array_map py_of_elt a
  let vector a =
    numpy_array ~dtype:`Object @@ py_of_array a
  let matrix aa =
    numpy_array ~dtype:`Object @@ Py.List.of_array_map py_of_array aa
end

(* [`Colon | `I of int] -> [`I of int | `F of float | `S of string] -> t -> unit *)
let set idx value self =
  let py_of_idx = function
    | `Colon -> Wrap_utils.Slice.create ()
    | (`I _) as i -> Wrap_utils.Slice.create ~i ()
  in
  let _ = Py.Object.call_method self "__setitem__" [| Py.Tuple.of_array (Array.map py_of_idx idx) ; Object.py_of_elt value|]
  in ()

let get_int idx self =
  match Py.Object.get_item self (Py.Tuple.of_list_map Py.Int.of_int idx) with
  | None -> raise (invalid_arg "Sklearn.Ndarray.get_int")
  | Some x -> Py.Int.to_int x

let get_float idx self =
  match Py.Object.get_item self (Py.Tuple.of_list_map Py.Int.of_int idx) with
  | None -> raise (invalid_arg "Sklearn.Ndarray.get_float")
  | Some x -> Py.Float.to_float x

let ones ?dtype shape =
  match dtype with
  | None -> Py.Module.get_function numpy "ones" [|Py.Tuple.of_list_map Py.Int.of_int shape|]
  | Some dtype ->
    Py.Module.get_function numpy "ones"
      [|Py.Tuple.of_list_map Py.Int.of_int shape; Dtype.to_pyobject dtype|]

let zeros ?dtype shape =
  match dtype with
  | None -> Py.Module.get_function numpy "ones" [|Py.Tuple.of_list_map Py.Int.of_int shape|]
  | Some dtype ->
    Py.Module.get_function numpy "zeros"
      [|Py.Tuple.of_list_map Py.Int.of_int shape; Dtype.to_pyobject dtype|]

module Ops = struct
  let operator = Py.import "operator"

  let int x = numpy_array (Py.Int.of_int x)
  let float x = numpy_array (Py.Float.of_float x)
  let bool x = numpy_array (Py.Bool.of_bool x)
  let string x = numpy_array (Py.String.of_string x)

  let binop name a b =
    of_pyobject @@ Py.Module.get_function operator name [|to_pyobject a; to_pyobject b|]
  
  let ( - ) = binop "sub"
  let ( + ) = binop "add"
  let ( * ) = binop "mul"
  let ( / ) = binop "truediv"
      
  let ( < ) = binop "lt"
  let ( <= ) = binop "le"
  let ( > ) = binop "gt"
  let ( >= ) = binop "get"

  let ( = ) = binop "eq"
  let ( != ) = binop "ne"
end

let ravel x =
  Py.Module.get_function numpy "ravel" [|x|]

let to_int_array x =
  let x = ravel x in
  let len = (shape x).(0) in
  Array.init len (fun i -> get_int [i] x)

let to_float_array x =
  let x = ravel x in
  let len = (shape x).(0) in
  Array.init len (fun i -> get_float [i] x)

let slice ?i ?j ?step () =
  `Slice (Wrap_utils.Slice.create_options ?i ?j ?step ())

let get_sub indices self =
  let index_of_tag = function
    | `I i -> Py.Int.of_int i
    | `Slice s -> Wrap_utils.Slice.to_pyobject s
    | `Arr x -> to_pyobject x
  in
  match Py.Object.get_item self (Py.Tuple.of_list_map index_of_tag indices) with
  | None -> raise (invalid_arg "Sklearn.Ndarray.get_sub")
  | Some x -> of_pyobject x

let min x =
  Py.Module.get_function numpy "min" [|x|] |> Py.Float.to_float

let max x =
  Py.Module.get_function numpy "max" [|x|] |> Py.Float.to_float

let argsort x =
  Py.Module.get_function numpy "argsort" [|x|] |> of_pyobject
