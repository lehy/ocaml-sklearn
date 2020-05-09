type t

module type BUILD = sig
  type t
  val of_pyobject : Py.Object.t -> t
end

type param_grid = [`Ints of int list | `Floats of float list |
                   `Strings of string list | `Bools of bool list]

type param_distributions = [`Ints of int list | `Floats of float list |
                            `Strings of string list | `Bools of bool list
                           | `Dist of Scipy.BaseTypes.Rv_generic.t]

val of_param_grid_alist : (string * param_grid) list -> t
val of_param_distributions_alist : (string * param_distributions) list -> t

(*  raises Not_found  *)
val get : (module BUILD with type t = 'a) -> name : string -> t -> 'a

val get_opt : (module BUILD with type t = 'a) -> name : string -> t -> 'a option
val fold_py : f : ('acc -> string -> Py.Object.t -> 'acc) -> init : 'acc -> t -> 'acc
val fold : (module BUILD with type t = 'a) -> f : ('acc -> string -> 'a -> 'acc) -> init : 'acc -> t -> 'acc

val keys : t -> string list

val of_pyobject : Py.Object.t -> t
val to_pyobject : t -> Py.Object.t
val pp : Format.formatter -> t -> unit
val to_string : t -> string
val show : t -> string
