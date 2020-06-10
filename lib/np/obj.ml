type -'a t = Py.Object.t

let to_pyobject x = x
let of_pyobject x = x
let to_string x = Py.Object.to_string x
let print x = print_endline (to_string x)
let pp fmt x = Format.fprintf fmt "%s" (to_string x)
