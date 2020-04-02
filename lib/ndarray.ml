type t = (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Genarray.t
(*  sucks to depend on Owl just for that  *)
let pp fmt x = Format.fprintf fmt "%s" (Py.Object.to_string (Numpy.of_bigarray x))
