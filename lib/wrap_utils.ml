let id x = x

exception Attribute_not_found of string

let keyword_args : (string * 'a option) list -> (string * Py.Object.t) list = fun l ->
  List.fold_left (fun acc (name, e) -> match e with
      | None -> acc
      | Some x -> (name, x)::acc) [] l;;

module Option = struct
  let get x = match x with
    | Some x -> x
    | None -> raise Not_found
  let map x f = match x with
    | None -> None
    | Some x -> Some (f x)
end

let init () =
  if not @@ Py.is_initialized () then
    Py.initialize ()
