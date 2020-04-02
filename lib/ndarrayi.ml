type t = (nativeint, Bigarray.nativeint_elt, Bigarray.c_layout) Bigarray.Genarray.t
let pp fmt x = Format.fprintf fmt "%s" (Py.Object.to_string (Numpy.of_bigarray x))
