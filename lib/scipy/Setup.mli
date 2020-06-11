(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val configuration : ?parent_package:Py.Object.t -> ?top_path:Py.Object.t -> unit -> Py.Object.t
(**
None
*)

