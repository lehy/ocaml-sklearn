(** Get an attribute of this module as a Py.Object.t. This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

