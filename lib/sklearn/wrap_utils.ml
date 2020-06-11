include Np.Wrap_utils_common

let init () =
  init "sklearn" Wrap_version.version

let () = init ()

module Types = struct
  include BaseTypes()
  let numpy = Py.import "numpy"
  let string = Py.Module.get (Py.Module.builtins ()) "str"
  let dict = Py.Module.get (Py.Module.builtins ()) "dict"
  let ndarray = Py.Module.get numpy "ndarray"
  let np_floating = Py.Module.get numpy "floating"
  let float = Py.Module.get (Py.Module.builtins ()) "float"
  let np_integer = Py.Module.get numpy "integer"
  let int = Py.Module.get (Py.Module.builtins ()) "int"
  let np_bool = Py.Module.get numpy "bool_"
  let bool = Py.Module.get (Py.Module.builtins ()) "bool"
  let np_object = Py.Module.get numpy "object_"
  let csr_matrix = Py.Module.get (Py.import "sklearn.metrics.pairwise") "csr_matrix"
  let spmatrix = Py.Module.get (Py.import "scipy.sparse.base") "spmatrix"
end

let check_int x =
  isinstance Types.[np_integer; int] x

let check_float x =
  isinstance Types.[np_floating; float] x

let check_bool x =
  isinstance Types.[np_bool; bool] x

let check_array x =
  isinstance Types.[ndarray] x

let check_csr_matrix x =
  isinstance Types.[csr_matrix] x

let check_arr x =
  isinstance Types.[ndarray; spmatrix; np_integer; np_floating; np_bool; np_object] x
