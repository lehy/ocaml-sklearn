module type ELEMENT = sig
  type t
  val of_pyobject : Py.Object.t -> t
  val to_pyobject : t -> Py.Object.t
end

module type S = sig
  type t
  type elt
  
  val of_pyobject : Py.Object.t -> t
  val to_pyobject : t -> Py.Object.t

  val create : unit -> t
  val of_list : elt list -> t

  val of_list_map : ('a -> elt) -> 'a list -> t

  val append : t -> elt -> unit

  val show : t -> string
  val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]
end

module Make : functor (X : ELEMENT) -> (S with type elt := X.t)
