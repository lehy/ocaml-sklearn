type t = Py.Object.t

module type BUILD = sig
  type t
  val of_pyobject : Py.Object.t -> t
end

let of_param = function
  | `Ints x -> Py.List.of_list_map Py.Int.of_int x
  | `Floats x -> Py.List.of_list_map Py.Float.of_float x
  | `Strings x -> Py.List.of_list_map Py.String.of_string x
  | `Bools x -> Py.List.of_list_map Py.Bool.of_bool x

let of_distribution = function
  | `Ints x -> Py.List.of_list_map Py.Int.of_int x
  | `Floats x -> Py.List.of_list_map Py.Float.of_float x
  | `Strings x -> Py.List.of_list_map Py.String.of_string x
  | `Bools x -> Py.List.of_list_map Py.Bool.of_bool x
  | `Dist x -> x

let of_param_grid_alist param_grid =
  Py.Dict.of_bindings_map Py.String.of_string of_param param_grid

let of_param_distributions_alist param_grid =
  Py.Dict.of_bindings_map Py.String.of_string of_distribution param_grid

let fold_py ~f ~init self =
  Py.Dict.fold (fun k v acc -> f acc (Py.String.to_string k) v) self init

let fold (type a) (module B : BUILD with type t = a) ~f ~init self =
  fold_py ~f:(fun acc k v -> f acc k (B.of_pyobject v)) ~init self

let get_opt (type a) (module B : BUILD with type t = a) ~name self =
  match Py.Dict.get_item_string self name with
  | None -> None
  | Some x -> Some (B.of_pyobject x)

let get (type a) (module B : BUILD with type t = a) ~name self =
  match get_opt (module B) ~name self with
  | None -> raise Not_found
  | Some x -> x

let keys x =
  fold_py ~f:(fun acc k _v -> k::acc) ~init:[] x |> List.rev

let of_pyobject self = self
let to_pyobject self = self
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)
