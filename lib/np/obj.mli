type -'a t

val to_pyobject : 'a t -> Py.Object.t
val of_pyobject : Py.Object.t -> 'a t
val print : 'a t -> unit
val to_string : 'a t -> string
val pp : Format.formatter -> 'a t -> unit [@@ocaml.toplevel_printer]
