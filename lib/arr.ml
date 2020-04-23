Wrap_utils.init ()

include Ndarray_private

let of_pyobject x =
  assert (Wrap_utils.isinstance Wrap_utils.csr_matrix x || Wrap_utils.isinstance Wrap_utils.ndarray x);
  x

let get =
  fun x ->
  if Wrap_utils.isinstance Wrap_utils.ndarray x then `Ndarray (Ndarray.of_pyobject x)
  else if Wrap_utils.isinstance Wrap_utils.csr_matrix x then `Csr_matrix (Csr_matrix.of_pyobject x)
  else failwith (Printf.sprintf "Arr.get: unexpected type: %s" (Py.Object.to_string x))

let get_ndarray x = match get x with
  | `Ndarray x -> x
  | `Csr_matrix _ -> invalid_arg "get_ndarray: not an Ndarray"

let get_csr_matrix x = match get x with
  | `Csr_matrix x -> x
  | `Ndarray _ -> invalid_arg "get_ndarray: not a Csr_matrix"

let of_ndarray x = Ndarray.to_pyobject x
let of_csr_matrix x = Csr_matrix.to_pyobject x
