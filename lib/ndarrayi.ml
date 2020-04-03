type t = (nativeint, Bigarray.nativeint_elt, Bigarray.c_layout) Bigarray.Genarray.t
let pp fmt x = Format.fprintf fmt "%s" (Py.Object.to_string (Numpy.of_bigarray x))
let vector ia =
  Bigarray.Array1.of_array Bigarray.nativeint Bigarray.c_layout (Array.map Nativeint.of_int ia) |> Bigarray.genarray_of_array1
let matrix ia =
  Bigarray.Array2.of_array Bigarray.nativeint Bigarray.c_layout
    (Array.map (Array.map Nativeint.of_int) ia) |> Bigarray.genarray_of_array2;
