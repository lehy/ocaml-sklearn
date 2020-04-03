type t = (nativeint, Bigarray.nativeint_elt, Bigarray.c_layout) Bigarray.Genarray.t
val pp : Format.formatter -> t -> unit
val vector : int array -> t
val matrix : int array array -> t
