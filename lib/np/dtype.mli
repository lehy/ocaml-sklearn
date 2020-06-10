type t = [`Object | `S of string]
val to_pyobject : t -> Py.Object.t
val of_pyobject : Py.Object.t -> t
