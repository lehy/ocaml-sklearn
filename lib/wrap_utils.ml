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

let runtime_sklearn_version () =
  match Py.Object.get_attr_string (Py.import "sklearn") "__version__" with
  | None -> raise Not_found
  | Some version ->
    let version = Py.String.to_string version in
    match String.split_on_char '.' version with
    | major::minor::revision::_ ->
      Ok (int_of_string major, int_of_string minor, int_of_string revision)
    | _ -> Error version

exception OCaml_sklearn_version_error of string

let do_check_version () =
  let (wanted_major, wanted_minor) = Version.version in
  match runtime_sklearn_version () with
  | Ok (major, minor, _) ->
    if (major, minor) = (wanted_major, wanted_minor) then `Ok
    else `Version_mismatch (Printf.sprintf "ocaml-sklearn version error: wanted: %d.%d, running: %d.%d"
                              wanted_major wanted_minor major minor)
  | Error version -> `Cannot_determine_runtime_version (Printf.sprintf "cannot parse runtime sklearn version %s" version)

let version_checked = ref false

let check_version () =
  if not !version_checked then begin
    version_checked := true;
    match do_check_version () with
    | `Ok -> ()
    | `Version_mismatch msg -> raise (OCaml_sklearn_version_error msg)
    | `Cannot_determine_runtime_version msg ->
      Printf.eprintf "ocaml-sklearn: warning: %s\n" msg
  end

let init () =
  if not @@ Py.is_initialized () then begin
    Py.initialize ();
  end;
  check_version ();;

let () = init ()
let builtins = Py.Module.builtins()
  
module Slice = struct
  type t = Py.Object.t

  let to_pyobject x = x
  let of_pyobject x = x
  
  let create_py i j step =
    Py.Module.get_function builtins "slice" [|i; j; step|]

  let py_of_tag = function
    | `None -> Py.none
    | `I i -> Py.Int.of_int i

  let py_of_option = function
    | None -> Py.none
    | Some i -> Py.Int.of_int i
  
  let of_variant s =
    match s with
    | `Slice(i, j, step) -> create_py (py_of_tag i) (py_of_tag j) (py_of_tag step)

  let create ?(i=`None) ?(j=`None) ?(step=`None) () =
    create_py (py_of_tag i) (py_of_tag j) (py_of_tag step)

  let create_options ?i ?j ?step () =
    create_py (py_of_option i) (py_of_option j) (py_of_option step)
end

(* XXX at some point it would be nice to create a PyObject tuple
   directly instead of going through an array (but we would need to
   create the keword args as a PyObject dict also, so we can use
   Py.Callable.to_function_as_tuple_and_dict) *)
let pos_arg f arg_list =
  let arr = Array.make (List.length arg_list) Py.none in
  let _ = List.fold_left (fun i e -> arr.(i) <- f e; succ i) 0 arg_list in
  arr

(*  call this to print the Python part of a traceback when Py.E is caught  *)
let print_python_traceback =
  init();
  let _ = Py.Run.eval ~start: Py.File "import traceback" in fun () ->
    match Py.Err.fetched () with
    | None -> ()
    | Some (exc, exc_val, tb) ->
      let exc_string_list = Py.Module.get_function (Py.import "traceback") "format_exception" [|exc; exc_val; tb|] in
      let exc_string_list = Py.List.to_list_map Py.String.to_string exc_string_list in
      List.iter (Format.printf "%s") exc_string_list

let type_ x =
  Py.Module.get_function builtins "type" [|x|]

let type_string x =
  Py.Object.to_string @@ type_ x

let isinstance =
  fun klasses x ->
    Py.Bool.to_bool @@ Py.Module.get_function builtins "isinstance" [|x; Py.Tuple.of_list klasses|]

module Types = struct
  let numpy = Py.import "numpy"
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
