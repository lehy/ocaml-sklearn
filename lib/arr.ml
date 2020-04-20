Wrap_utils.init ()

include Ndarray_private

let isinstance =
  let builtins = Py.Module.builtins() in
  fun x klass ->
    Py.Bool.to_bool @@ Py.Module.get_function builtins "isinstance" [|x; klass|]

let get =
  let csr_matrix = Py.Module.get (Py.import "sklearn.metrics.pairwise") "csr_matrix" in
  let ndarray = Py.Module.get (Py.import "numpy") "ndarray" in
  fun x ->
    if isinstance x ndarray then `Ndarray (Ndarray.of_pyobject x)
    else if isinstance x csr_matrix then `Csr_matrix (Csr_matrix.of_pyobject x)
    else failwith (Printf.sprintf "Arr.get: unexpected type: %s" (Py.Object.to_string x))

let get_ndarray x = match get x with
  | `Ndarray x -> x
  | `Csr_matrix _ -> invalid_arg "get_ndarray: not an Ndarray"

let get_csr_matrix x = match get x with
  | `Csr_matrix x -> x
  | `Ndarray _ -> invalid_arg "get_ndarray: not a Csr_matrix"

let of_ndarray x = Ndarray.to_pyobject x
let of_csr_matrix x = Csr_matrix.to_pyobject x
