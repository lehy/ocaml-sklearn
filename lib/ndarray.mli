type t
val show : t -> string
val pp : Format.formatter -> t -> unit
val to_pyobject : t -> Py.Object.t
val of_pyobject : Py.Object.t -> t
val of_bigarray : ('a, 'b, 'c) Bigarray.Genarray.t -> t
module Float : sig
  val vector : float array -> t
  val matrix : float array array -> t
  val of_bigarray : (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Genarray.t -> t
end
module Int : sig
  val vector : int array -> t
  val matrix : int array array -> t
  val of_bigarray : (nativeint, Bigarray.nativeint_elt, Bigarray.c_layout) Bigarray.Genarray.t -> t
end

