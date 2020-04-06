type t = Py.Object.t
let show x = Py.Object.to_string x
let pp fmt x = Format.fprintf fmt "%s" (Py.Object.to_string x)
let of_pyobject x = (*  XXX assert that isinstance(x, np.ndarray) at least *) x
let to_pyobject x = x
let of_bigarray ba = Numpy.of_bigarray ba

module Float = struct
  let matrix values =
    Bigarray.Array2.of_array Bigarray.float64 Bigarray.c_layout values
    |> Bigarray.genarray_of_array2 |> Numpy.of_bigarray
  let vector values =
    Bigarray.Array1.of_array Bigarray.float64 Bigarray.c_layout values
    |> Bigarray.genarray_of_array1 |> Numpy.of_bigarray
  let of_bigarray ba = Numpy.of_bigarray ba
end

module Int = struct
  let vector ia =
    Bigarray.Array1.of_array Bigarray.nativeint Bigarray.c_layout (Array.map Nativeint.of_int ia)
    |> Bigarray.genarray_of_array1 |> Numpy.of_bigarray
  let matrix ia =
    Bigarray.Array2.of_array Bigarray.nativeint Bigarray.c_layout (Array.map (Array.map Nativeint.of_int) ia)
    |> Bigarray.genarray_of_array2 |> Numpy.of_bigarray
  let of_bigarray ba = Numpy.of_bigarray ba
end
