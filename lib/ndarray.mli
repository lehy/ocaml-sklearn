type t = (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Genarray.t
val pp : Format.formatter -> t -> unit
val to_pyobject : t -> Py.Object.t
val of_pyobject : Py.Object.t -> t
val vector : float array -> t
val matrix : float array array -> t
