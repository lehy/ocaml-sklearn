type t
val show : t -> string
val pp : Format.formatter -> t -> unit
val to_pyobject : t -> Py.Object.t
val of_pyobject : Py.Object.t -> t
val of_bigarray : ('a, 'b, 'c) Bigarray.Genarray.t -> t

module List : sig
  type t
  val to_pyobject : t -> Py.Object.t
  val of_pyobject : Py.Object.t -> t
end

module Float : sig
  val vector : float array -> t
  val matrix : float array array -> t
  val of_bigarray : (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Genarray.t -> t

  val matrices : float array array list -> List.t
  val vectors : float array list -> List.t
end

module Int : sig
  val vector : int array -> t
  val matrix : int array array -> t
  val of_bigarray : (nativeint, Bigarray.nativeint_elt, Bigarray.c_layout) Bigarray.Genarray.t -> t

  val matrices : int array array list -> List.t
  val vectors : int array list -> List.t
end

module String : sig
  val vector : string array -> t
  (* val matrix : string array array -> t *)
  val vectors : string array list -> List.t
end

val shape : t -> int array
