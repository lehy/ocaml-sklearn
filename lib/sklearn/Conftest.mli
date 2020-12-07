(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val pyplot : unit -> Py.Object.t
(**
Setup and teardown fixture for matplotlib.

This fixture checks if we can import matplotlib. If not, the tests will be
skipped. Otherwise, we setup matplotlib backend and close the figures
after running the functions.

Returns
-------
pyplot : module
    The ``matplotlib.pyplot`` module.
*)

