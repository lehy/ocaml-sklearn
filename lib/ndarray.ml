type t = (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Genarray.t
let pp fmt x = Format.fprintf fmt "%s" (Py.Object.to_string (Numpy.of_bigarray x))
let of_pyobject x = Numpy.to_bigarray Bigarray.float64 Bigarray.c_layout x
let to_pyobject x = Numpy.of_bigarray x
let matrix values =
  Bigarray.Array2.of_array Bigarray.float64 Bigarray.c_layout values |> Bigarray.genarray_of_array2
let vector values =
  Bigarray.Array1.of_array Bigarray.float64 Bigarray.c_layout values |> Bigarray.genarray_of_array1
