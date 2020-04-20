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
  let rec to_pyobject = function
    | `String s -> Py.Module.get_function numpy "dtype" [|Py.String.of_string s|]
    | `Object -> to_pyobject (`String "object")
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
  type elt = [`I of int | `F of float | `S of string]
  let py_of_elt x = match x with
    | `I x -> Py.Int.of_int x
    | `F x -> Py.Float.of_float x
    | `S x -> Py.String.of_string x
  let py_of_array a = Py.List.of_array_map py_of_elt a
  let vector a =
    numpy_array ~dtype:`Object @@ py_of_array a
  let matrix aa =
    numpy_array ~dtype:`Object @@ Py.List.of_array_map py_of_array aa
end
