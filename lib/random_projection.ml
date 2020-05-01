let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.random_projection"

let get_py name = Py.Module.get ns name
module ABCMeta = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?kwargs ~name ~bases ~namespace () =
   Py.Module.get_function_with_keywords ns "ABCMeta"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("name", Some(name )); ("bases", Some(bases )); ("namespace", Some(namespace ))]) (match kwargs with None -> [] | Some x -> x))

let mro self =
   Py.Module.get_function_with_keywords self "mro"
     [||]
     []

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module BaseEstimator = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create () =
   Py.Module.get_function_with_keywords ns "BaseEstimator"
     [||]
     []

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module GaussianRandomProjection = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_components ?eps ?random_state () =
                     Py.Module.get_function_with_keywords ns "GaussianRandomProjection"
                       [||]
                       (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components (function
| `I x -> Py.Int.of_int x
| `Auto -> Py.String.of_string "auto"
)); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let n_components_opt self =
  match Py.Object.get_attr_string self "n_components_" with
  | None -> failwith "attribute n_components_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_components_ self = match n_components_opt self with
  | None -> raise Not_found
  | Some x -> x

let components_opt self =
  match Py.Object.get_attr_string self "components_" with
  | None -> failwith "attribute components_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let components_ self = match components_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SparseRandomProjection = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_components ?density ?eps ?dense_output ?random_state () =
                     Py.Module.get_function_with_keywords ns "SparseRandomProjection"
                       [||]
                       (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components (function
| `I x -> Py.Int.of_int x
| `Auto -> Py.String.of_string "auto"
)); ("density", Wrap_utils.Option.map density Py.Float.of_float); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("dense_output", Wrap_utils.Option.map dense_output Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let n_components_opt self =
  match Py.Object.get_attr_string self "n_components_" with
  | None -> failwith "attribute n_components_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_components_ self = match n_components_opt self with
  | None -> raise Not_found
  | Some x -> x

let components_opt self =
  match Py.Object.get_attr_string self "components_" with
  | None -> failwith "attribute components_ not found"
  | Some x -> if Py.is_none x then None else Some (Csr_matrix.of_pyobject x)

let components_ self = match components_opt self with
  | None -> raise Not_found
  | Some x -> x

let density_opt self =
  match Py.Object.get_attr_string self "density_" with
  | None -> failwith "attribute density_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let density_ self = match density_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module TransformerMixin = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create () =
   Py.Module.get_function_with_keywords ns "TransformerMixin"
     [||]
     []

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Arr.of_pyobject
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let abstractmethod ~funcobj () =
   Py.Module.get_function_with_keywords ns "abstractmethod"
     [||]
     (Wrap_utils.keyword_args [("funcobj", Some(funcobj ))])

                  let check_array ?accept_sparse ?accept_large_sparse ?dtype ?order ?copy ?force_all_finite ?ensure_2d ?allow_nd ?ensure_min_samples ?ensure_min_features ?warn_on_dtype ?estimator ~array () =
                     Py.Module.get_function_with_keywords ns "check_array"
                       [||]
                       (Wrap_utils.keyword_args [("accept_sparse", Wrap_utils.Option.map accept_sparse (function
| `S x -> Py.String.of_string x
| `Bool x -> Py.Bool.of_bool x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
)); ("accept_large_sparse", Wrap_utils.Option.map accept_large_sparse Py.Bool.of_bool); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Wrap_utils.id x
| `TypeList x -> Wrap_utils.id x
| `None -> Py.none
)); ("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("force_all_finite", Wrap_utils.Option.map force_all_finite (function
| `Bool x -> Py.Bool.of_bool x
| `Allow_nan -> Py.String.of_string "allow-nan"
)); ("ensure_2d", Wrap_utils.Option.map ensure_2d Py.Bool.of_bool); ("allow_nd", Wrap_utils.Option.map allow_nd Py.Bool.of_bool); ("ensure_min_samples", Wrap_utils.Option.map ensure_min_samples Py.Int.of_int); ("ensure_min_features", Wrap_utils.Option.map ensure_min_features Py.Int.of_int); ("warn_on_dtype", Wrap_utils.Option.map warn_on_dtype Py.Bool.of_bool); ("estimator", Wrap_utils.Option.map estimator (function
| `S x -> Py.String.of_string x
| `Estimator x -> Wrap_utils.id x
)); ("array", Some(array ))])

                  let check_is_fitted ?attributes ?msg ?all_or_any ~estimator () =
                     Py.Module.get_function_with_keywords ns "check_is_fitted"
                       [||]
                       (Wrap_utils.keyword_args [("attributes", Wrap_utils.Option.map attributes (function
| `S x -> Py.String.of_string x
| `Arr x -> Arr.to_pyobject x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
)); ("msg", Wrap_utils.Option.map msg Py.String.of_string); ("all_or_any", Wrap_utils.Option.map all_or_any (function
| `Callable x -> Wrap_utils.id x
| `PyObject x -> Wrap_utils.id x
)); ("estimator", Some(estimator ))])

                  let check_random_state ~seed () =
                     Py.Module.get_function_with_keywords ns "check_random_state"
                       [||]
                       (Wrap_utils.keyword_args [("seed", Some(seed |> (function
| `I x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.none
)))])

module Deprecated = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?extra () =
   Py.Module.get_function_with_keywords ns "deprecated"
     [||]
     (Wrap_utils.keyword_args [("extra", Wrap_utils.Option.map extra Py.String.of_string)])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let gaussian_random_matrix ?random_state ~n_components ~n_features () =
   Py.Module.get_function_with_keywords ns "gaussian_random_matrix"
     [||]
     (Wrap_utils.keyword_args [("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("n_components", Some(n_components )); ("n_features", Some(n_features ))])

                  let johnson_lindenstrauss_min_dim ?eps ~n_samples () =
                     Py.Module.get_function_with_keywords ns "johnson_lindenstrauss_min_dim"
                       [||]
                       (Wrap_utils.keyword_args [("eps", Wrap_utils.Option.map eps (function
| `F x -> Py.Float.of_float x
| `Arr x -> Arr.to_pyobject x
)); ("n_samples", Some(n_samples |> (function
| `I x -> Py.Int.of_int x
| `Arr x -> Arr.to_pyobject x
)))])
                       |> (fun x -> if Py.Int.check x then `I (Py.Int.to_int x) else if (fun x -> (Wrap_utils.isinstance Wrap_utils.ndarray x) || (Wrap_utils.isinstance Wrap_utils.csr_matrix x)) x then `Arr (Arr.of_pyobject x) else failwith "could not identify type from Python value")
let safe_sparse_dot ?dense_output ~a ~b () =
   Py.Module.get_function_with_keywords ns "safe_sparse_dot"
     [||]
     (Wrap_utils.keyword_args [("dense_output", dense_output); ("a", Some(a |> Arr.to_pyobject)); ("b", Some(b ))])
     |> Arr.of_pyobject
let sparse_random_matrix ?density ?random_state ~n_components ~n_features () =
   Py.Module.get_function_with_keywords ns "sparse_random_matrix"
     [||]
     (Wrap_utils.keyword_args [("density", density); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("n_components", Some(n_components )); ("n_features", Some(n_features ))])

