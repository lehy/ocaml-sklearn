module type ELEMENT = sig
  type t
  val of_pyobject : Py.Object.t -> t
  val to_pyobject : t -> Py.Object.t
end


module Make : functor (X : ELEMENT) -> sig
  type t = Py.Object.t

  val of_pyobject : Py.Object.t -> t
  val to_pyobject : t -> Py.Object.t

  val of_list : X.t list -> t

  val of_list_map : ('a -> X.t) -> 'a list -> t

  val append : t -> X.t -> unit

  val show : t -> string
  val pp : Format.formatter -> t -> unit
end
                         
