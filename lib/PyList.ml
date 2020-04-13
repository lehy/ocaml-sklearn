module type ELEMENT = sig
  type t
  val of_pyobject : Py.Object.t -> t
  val to_pyobject : t -> Py.Object.t
end


module Make(X : ELEMENT) = struct
  type t = Py.Object.t

  let of_pyobject x = x
  let to_pyobject x = x
  
  let of_list : X.t list -> t = fun l ->
    Py.List.of_list_map X.to_pyobject l

  let of_list_map : ('a -> X.t) -> 'a list -> t = fun f l ->
    Py.List.of_list_map (fun e -> X.to_pyobject @@ f e) l
    
  let append : t -> X.t -> unit = fun l x ->
    let _ = Py.Object.call_method l "append" [| X.to_pyobject x |] in ()

  let show x = Py.Object.to_string x
  let pp fmt x = Format.fprintf fmt "%s" (show x)
end
