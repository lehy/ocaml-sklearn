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

module String = struct
  let vector ia =
    (* XXX TODO figure out a way to do this with one copy less *)
    Py.Module.get_function numpy "array" [|Py.List.of_array_map Py.String.of_string ia|]

  let vectors values =
    List.of_list_map vector values
end


let shape self =
  match Py.Object.get_attr_string self "shape" with
  | None -> raise (Wrap_utils.Attribute_not_found "shape")
  | Some x -> Py.List.to_array_map Py.Int.to_int x
