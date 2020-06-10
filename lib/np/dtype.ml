let numpy = Py.import "numpy"

type t = [`Object | `S of string]
let rec to_pyobject = function
  | `S s -> Py.Module.get_function numpy "dtype" [|Py.String.of_string s|]
  | `Object -> to_pyobject (`S "object")
let of_pyobject x = match Py.Object.to_string x with
  | "object" -> `Object
  | x -> `S x

