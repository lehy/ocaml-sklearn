include Wrap_utils_common

let init () =
  init "numpy" Wrap_version.version

let () = init ()

module Types = struct
  include BaseTypes()
  let numpy = Py.import "numpy"
  let builtins = Py.Module.builtins ()
  let string = Py.Module.get builtins "str"
  let dict = Py.Module.get builtins "dict"
  let ndarray = Py.Module.get numpy "ndarray"
  let np_floating = Py.Module.get numpy "floating"
  let float = Py.Module.get builtins "float"
  let np_integer = Py.Module.get numpy "integer"
  let int = Py.Module.get builtins "int"
  let np_bool = Py.Module.get numpy "bool_"
  let bool = Py.Module.get builtins "bool"
  let np_object = Py.Module.get numpy "object_"
  (* let csr_matrix = Py.Module.get (Py.import "sklearn.metrics.pairwise") "csr_matrix" *)
end

let check_int x =
  isinstance Types.[np_integer; int] x

let check_float x =
  isinstance Types.[np_floating; float] x

let check_bool x =
  isinstance Types.[np_bool; bool] x

let check_array x =
  isinstance Types.[ndarray] x

let check_arr x =
  isinstance Types.[ndarray; np_integer; np_floating; np_bool; np_object] x

module Index = struct
  let ellipsis = Option.get @@ Py.Object.get_attr_string Types.builtins "Ellipsis"

  module Element = struct
    type t = [
      | `I of int
      | `Slice of Slice.t
      | `Arr of [`Ndarray] Obj.t
      | `Newaxis
      | `Ellipsis
    ]

    (* type t =
     *   | I : int -> t
     *   | Slice : Slice.t -> t
     *   | Arr : [> `Ndarray] Obj.t -> t
     *   | Newaxis : t
     *   | Ellipsis : t *)

    let to_pyobject x = match x with
      | `I i -> Py.Int.of_int i
      | `Slice s -> Slice.to_pyobject s
      | `Arr x -> Obj.to_pyobject x
      | `Newaxis -> Py.none
      | `Ellipsis -> ellipsis
  end

  type t = Element.t list
  let to_pyobject x = Py.Tuple.of_list_map Element.to_pyobject x
end

let slice ?i ?j ?step () =
  `Slice (Slice.create_options ?i ?j ?step ())

let mask : 'a. ([> `Ndarray] as 'a) Obj.t -> Index.Element.t =
  fun a -> `Arr (a :> [`Ndarray] Obj.t)
