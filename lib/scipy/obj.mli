type -'a t

val to_pyobject : 'a t -> Py.Object.t
val of_pyobject : Py.Object.t -> 'a t
