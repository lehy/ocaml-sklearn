type t = Py.Object.t

module type BUILD = sig
  type t
  val of_pyobject : Py.Object.t -> t
end

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

let of_pyobject self = self
let to_pyobject self = self
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)
