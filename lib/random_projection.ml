let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.random_projection"

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
| `Int x -> Py.Int.of_int x
| `Auto -> Py.String.of_string "auto"
)); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])

                  let fit ?y ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

                  let transform ~x self =
                     Py.Module.get_function_with_keywords self "transform"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])
                       |> Ndarray.of_pyobject
let n_components_ self =
  match Py.Object.get_attr_string self "n_components_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_components_")
| Some x -> Py.Int.to_int x
let components_ self =
  match Py.Object.get_attr_string self "components_" with
| None -> raise (Wrap_utils.Attribute_not_found "components_")
| Some x -> Ndarray.of_pyobject x
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
| `Int x -> Py.Int.of_int x
| `Auto -> Py.String.of_string "auto"
)); ("density", Wrap_utils.Option.map density Py.Float.of_float); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("dense_output", Wrap_utils.Option.map dense_output Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])

                  let fit ?y ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

                  let transform ~x self =
                     Py.Module.get_function_with_keywords self "transform"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])
                       |> Ndarray.of_pyobject
let n_components_ self =
  match Py.Object.get_attr_string self "n_components_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_components_")
| Some x -> Py.Int.to_int x
let components_ self =
  match Py.Object.get_attr_string self "components_" with
| None -> raise (Wrap_utils.Attribute_not_found "components_")
| Some x -> Wrap_utils.id x
let density_ self =
  match Py.Object.get_attr_string self "density_" with
| None -> raise (Wrap_utils.Attribute_not_found "density_")
| Some x -> Wrap_utils.id x
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
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Ndarray.of_pyobject
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
| `String x -> Py.String.of_string x
| `Bool x -> Py.Bool.of_bool x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
)); ("accept_large_sparse", Wrap_utils.Option.map accept_large_sparse Py.Bool.of_bool); ("dtype", Wrap_utils.Option.map dtype (function
| `String x -> Py.String.of_string x
| `Dtype x -> Wrap_utils.id x
| `TypeList x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
| `None -> Py.String.of_string "None"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("force_all_finite", Wrap_utils.Option.map force_all_finite (function
| `Bool x -> Py.Bool.of_bool x
| `Allow_nan -> Py.String.of_string "allow-nan"
)); ("ensure_2d", Wrap_utils.Option.map ensure_2d Py.Bool.of_bool); ("allow_nd", Wrap_utils.Option.map allow_nd Py.Bool.of_bool); ("ensure_min_samples", Wrap_utils.Option.map ensure_min_samples Py.Int.of_int); ("ensure_min_features", Wrap_utils.Option.map ensure_min_features Py.Int.of_int); ("warn_on_dtype", Wrap_utils.Option.map warn_on_dtype (function
| `Bool x -> Py.Bool.of_bool x
| `None -> Py.String.of_string "None"
)); ("estimator", Wrap_utils.Option.map estimator (function
| `String x -> Py.String.of_string x
| `Estimator x -> Wrap_utils.id x
)); ("array", Some(array ))])

                  let check_is_fitted ?attributes ?msg ?all_or_any ~estimator () =
                     Py.Module.get_function_with_keywords ns "check_is_fitted"
                       [||]
                       (Wrap_utils.keyword_args [("attributes", Wrap_utils.Option.map attributes (function
| `String x -> Py.String.of_string x
| `ArrayLike x -> Wrap_utils.id x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
)); ("msg", Wrap_utils.Option.map msg Py.String.of_string); ("all_or_any", Wrap_utils.Option.map all_or_any (function
| `Callable x -> Wrap_utils.id x
| `PyObject x -> Wrap_utils.id x
)); ("estimator", Some(estimator ))])

                  let check_random_state ~seed () =
                     Py.Module.get_function_with_keywords ns "check_random_state"
                       [||]
                       (Wrap_utils.keyword_args [("seed", Some(seed |> (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
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
     (Wrap_utils.keyword_args [("random_state", random_state); ("n_components", Some(n_components )); ("n_features", Some(n_features ))])

                  let johnson_lindenstrauss_min_dim ?eps ~n_samples () =
                     Py.Module.get_function_with_keywords ns "johnson_lindenstrauss_min_dim"
                       [||]
                       (Wrap_utils.keyword_args [("eps", Wrap_utils.Option.map eps (function
| `Float x -> Py.Float.of_float x
| `PyObject x -> Wrap_utils.id x
)); ("n_samples", Some(n_samples |> (function
| `Int x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)))])

                  let safe_sparse_dot ?dense_output ~a ~b () =
                     Py.Module.get_function_with_keywords ns "safe_sparse_dot"
                       [||]
                       (Wrap_utils.keyword_args [("dense_output", dense_output); ("a", Some(a |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
))); ("b", Some(b ))])

let sparse_random_matrix ?density ?random_state ~n_components ~n_features () =
   Py.Module.get_function_with_keywords ns "sparse_random_matrix"
     [||]
     (Wrap_utils.keyword_args [("density", density); ("random_state", random_state); ("n_components", Some(n_components )); ("n_features", Some(n_features ))])

