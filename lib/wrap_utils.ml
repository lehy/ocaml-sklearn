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

module Slice = struct
  type t = Py.Object.t
  let create_py i j step =
    Py.Module.get_function (Py.Module.builtins ()) "slice" [|i; j; step|]

  let of_variant s =
    let of_tag = function
      | `None -> Py.none
      | `Int i -> Py.Int.of_int i
    in match s with
    | `Slice(i, j, step) -> create_py (of_tag i) (of_tag j) (of_tag step)
end 

(* XXX at some point it would be nice to create a PyObject tuple
   directly instead of going through an array (but we would need to
   create the keword args as a PyObject dict also, so we can use
   Py.Callable.to_function_as_tuple_and_dict) *)
let pos_arg f arg_list =
  let arr = Array.make (List.length arg_list) Py.none in
  let _ = List.fold_left (fun i e -> arr.(i) <- f e; succ i) 0 arg_list in
  arr
