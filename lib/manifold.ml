let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.manifold"

module Isomap = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_neighbors ?n_components ?eigen_solver ?tol ?max_iter ?path_method ?neighbors_algorithm ?n_jobs ?metric ?p ?metric_params () =
                     Py.Module.get_function_with_keywords ns "Isomap"
                       [||]
                       (Wrap_utils.keyword_args [("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("eigen_solver", Wrap_utils.Option.map eigen_solver (function
| `Auto -> Py.String.of_string "auto"
| `Arpack -> Py.String.of_string "arpack"
| `Dense -> Py.String.of_string "dense"
)); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("path_method", Wrap_utils.Option.map path_method (function
| `Auto -> Py.String.of_string "auto"
| `FW -> Py.String.of_string "FW"
| `D -> Py.String.of_string "D"
)); ("neighbors_algorithm", Wrap_utils.Option.map neighbors_algorithm (function
| `Auto -> Py.String.of_string "auto"
| `Brute -> Py.String.of_string "brute"
| `Kd_tree -> Py.String.of_string "kd_tree"
| `Ball_tree -> Py.String.of_string "ball_tree"
)); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("metric", Wrap_utils.Option.map metric (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("p", Wrap_utils.Option.map p Py.Int.of_int); ("metric_params", metric_params)])

                  let fit ?y ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])

                  let fit_transform ?y ~x self =
                     Py.Module.get_function_with_keywords self "fit_transform"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])
                       |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let reconstruction_error self =
   Py.Module.get_function_with_keywords self "reconstruction_error"
     [||]
     []
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let embedding_ self =
  match Py.Object.get_attr_string self "embedding_" with
| None -> raise (Wrap_utils.Attribute_not_found "embedding_")
| Some x -> Ndarray.of_pyobject x
let kernel_pca_ self =
  match Py.Object.get_attr_string self "kernel_pca_" with
| None -> raise (Wrap_utils.Attribute_not_found "kernel_pca_")
| Some x -> Wrap_utils.id x
let nbrs_ self =
  match Py.Object.get_attr_string self "nbrs_" with
| None -> raise (Wrap_utils.Attribute_not_found "nbrs_")
| Some x -> Wrap_utils.id x
let dist_matrix_ self =
  match Py.Object.get_attr_string self "dist_matrix_" with
| None -> raise (Wrap_utils.Attribute_not_found "dist_matrix_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LocallyLinearEmbedding = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_neighbors ?n_components ?reg ?eigen_solver ?tol ?max_iter ?method_ ?hessian_tol ?modified_tol ?neighbors_algorithm ?random_state ?n_jobs () =
                     Py.Module.get_function_with_keywords ns "LocallyLinearEmbedding"
                       [||]
                       (Wrap_utils.keyword_args [("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("reg", Wrap_utils.Option.map reg Py.Float.of_float); ("eigen_solver", Wrap_utils.Option.map eigen_solver (function
| `Auto -> Py.String.of_string "auto"
| `Arpack -> Py.String.of_string "arpack"
| `Dense -> Py.String.of_string "dense"
)); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("method", Wrap_utils.Option.map method_ (function
| `Standard -> Py.String.of_string "standard"
| `Hessian -> Py.String.of_string "hessian"
| `Modified -> Py.String.of_string "modified"
| `Ltsa -> Py.String.of_string "ltsa"
)); ("hessian_tol", Wrap_utils.Option.map hessian_tol Py.Float.of_float); ("modified_tol", Wrap_utils.Option.map modified_tol Py.Float.of_float); ("neighbors_algorithm", Wrap_utils.Option.map neighbors_algorithm (function
| `Auto -> Py.String.of_string "auto"
| `Brute -> Py.String.of_string "brute"
| `Kd_tree -> Py.String.of_string "kd_tree"
| `Ball_tree -> Py.String.of_string "ball_tree"
)); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
))])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])
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
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let embedding_ self =
  match Py.Object.get_attr_string self "embedding_" with
| None -> raise (Wrap_utils.Attribute_not_found "embedding_")
| Some x -> Ndarray.of_pyobject x
let reconstruction_error_ self =
  match Py.Object.get_attr_string self "reconstruction_error_" with
| None -> raise (Wrap_utils.Attribute_not_found "reconstruction_error_")
| Some x -> Py.Float.to_float x
let nbrs_ self =
  match Py.Object.get_attr_string self "nbrs_" with
| None -> raise (Wrap_utils.Attribute_not_found "nbrs_")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MDS = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_components ?metric ?n_init ?max_iter ?verbose ?eps ?n_jobs ?random_state ?dissimilarity () =
                     Py.Module.get_function_with_keywords ns "MDS"
                       [||]
                       (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("metric", Wrap_utils.Option.map metric Py.Bool.of_bool); ("n_init", Wrap_utils.Option.map n_init Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("dissimilarity", Wrap_utils.Option.map dissimilarity (function
| `Euclidean -> Py.String.of_string "euclidean"
| `Precomputed -> Py.String.of_string "precomputed"
))])

let fit ?y ?init ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("init", Wrap_utils.Option.map init Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))])

let fit_transform ?y ?init ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("init", Wrap_utils.Option.map init Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let embedding_ self =
  match Py.Object.get_attr_string self "embedding_" with
| None -> raise (Wrap_utils.Attribute_not_found "embedding_")
| Some x -> Ndarray.of_pyobject x
let stress_ self =
  match Py.Object.get_attr_string self "stress_" with
| None -> raise (Wrap_utils.Attribute_not_found "stress_")
| Some x -> Py.Float.to_float x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module SpectralEmbedding = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_components ?affinity ?gamma ?random_state ?eigen_solver ?n_neighbors ?n_jobs () =
                     Py.Module.get_function_with_keywords ns "SpectralEmbedding"
                       [||]
                       (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("affinity", Wrap_utils.Option.map affinity (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("gamma", Wrap_utils.Option.map gamma Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("eigen_solver", Wrap_utils.Option.map eigen_solver (function
| `Arpack -> Py.String.of_string "arpack"
| `Lobpcg -> Py.String.of_string "lobpcg"
| `Amg -> Py.String.of_string "amg"
| `None -> Py.String.of_string "None"
)); ("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
))])

                  let fit ?y ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

                  let fit_transform ?y ~x self =
                     Py.Module.get_function_with_keywords self "fit_transform"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let embedding_ self =
  match Py.Object.get_attr_string self "embedding_" with
| None -> raise (Wrap_utils.Attribute_not_found "embedding_")
| Some x -> Ndarray.of_pyobject x
let affinity_matrix_ self =
  match Py.Object.get_attr_string self "affinity_matrix_" with
| None -> raise (Wrap_utils.Attribute_not_found "affinity_matrix_")
| Some x -> Ndarray.of_pyobject x
let n_neighbors_ self =
  match Py.Object.get_attr_string self "n_neighbors_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_neighbors_")
| Some x -> Py.Int.to_int x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module TSNE = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_components ?perplexity ?early_exaggeration ?learning_rate ?n_iter ?n_iter_without_progress ?min_grad_norm ?metric ?init ?verbose ?random_state ?method_ ?angle ?n_jobs () =
                     Py.Module.get_function_with_keywords ns "TSNE"
                       [||]
                       (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("perplexity", Wrap_utils.Option.map perplexity Py.Float.of_float); ("early_exaggeration", Wrap_utils.Option.map early_exaggeration Py.Float.of_float); ("learning_rate", Wrap_utils.Option.map learning_rate Py.Float.of_float); ("n_iter", Wrap_utils.Option.map n_iter Py.Int.of_int); ("n_iter_without_progress", Wrap_utils.Option.map n_iter_without_progress Py.Int.of_int); ("min_grad_norm", Wrap_utils.Option.map min_grad_norm Py.Float.of_float); ("metric", Wrap_utils.Option.map metric (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("init", Wrap_utils.Option.map init (function
| `String x -> Py.String.of_string x
| `Ndarray x -> Ndarray.to_pyobject x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("method", Wrap_utils.Option.map method_ Py.String.of_string); ("angle", Wrap_utils.Option.map angle Py.Float.of_float); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
))])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let embedding_ self =
  match Py.Object.get_attr_string self "embedding_" with
| None -> raise (Wrap_utils.Attribute_not_found "embedding_")
| Some x -> Ndarray.of_pyobject x
let kl_divergence_ self =
  match Py.Object.get_attr_string self "kl_divergence_" with
| None -> raise (Wrap_utils.Attribute_not_found "kl_divergence_")
| Some x -> Py.Float.to_float x
let n_iter_ self =
  match Py.Object.get_attr_string self "n_iter_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_iter_")
| Some x -> Py.Int.to_int x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let locally_linear_embedding ?reg ?eigen_solver ?tol ?max_iter ?method_ ?hessian_tol ?modified_tol ?random_state ?n_jobs ~x ~n_neighbors ~n_components () =
                     Py.Module.get_function_with_keywords ns "locally_linear_embedding"
                       [||]
                       (Wrap_utils.keyword_args [("reg", Wrap_utils.Option.map reg Py.Float.of_float); ("eigen_solver", Wrap_utils.Option.map eigen_solver (function
| `Auto -> Py.String.of_string "auto"
| `Arpack -> Py.String.of_string "arpack"
| `Dense -> Py.String.of_string "dense"
)); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("method", Wrap_utils.Option.map method_ (function
| `Standard -> Py.String.of_string "standard"
| `Hessian -> Py.String.of_string "hessian"
| `Modified -> Py.String.of_string "modified"
| `Ltsa -> Py.String.of_string "ltsa"
)); ("hessian_tol", Wrap_utils.Option.map hessian_tol Py.Float.of_float); ("modified_tol", Wrap_utils.Option.map modified_tol Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("n_neighbors", Some(n_neighbors |> Py.Int.of_int)); ("n_components", Some(n_components |> Py.Int.of_int))])
                       |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))
                  let smacof ?metric ?n_components ?init ?n_init ?n_jobs ?max_iter ?verbose ?eps ?random_state ?return_n_iter ~dissimilarities () =
                     Py.Module.get_function_with_keywords ns "smacof"
                       [||]
                       (Wrap_utils.keyword_args [("metric", Wrap_utils.Option.map metric Py.Bool.of_bool); ("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("init", Wrap_utils.Option.map init Ndarray.to_pyobject); ("n_init", Wrap_utils.Option.map n_init Py.Int.of_int); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("return_n_iter", Wrap_utils.Option.map return_n_iter Py.Bool.of_bool); ("dissimilarities", Some(dissimilarities |> Ndarray.to_pyobject))])
                       |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2))))
                  let spectral_embedding ?n_components ?eigen_solver ?random_state ?eigen_tol ?norm_laplacian ?drop_first ~adjacency () =
                     Py.Module.get_function_with_keywords ns "spectral_embedding"
                       [||]
                       (Wrap_utils.keyword_args [("n_components", Wrap_utils.Option.map n_components Py.Int.of_int); ("eigen_solver", Wrap_utils.Option.map eigen_solver (function
| `Arpack -> Py.String.of_string "arpack"
| `Lobpcg -> Py.String.of_string "lobpcg"
| `Amg -> Py.String.of_string "amg"
| `None -> Py.String.of_string "None"
)); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("eigen_tol", Wrap_utils.Option.map eigen_tol Py.Float.of_float); ("norm_laplacian", Wrap_utils.Option.map norm_laplacian Py.Bool.of_bool); ("drop_first", Wrap_utils.Option.map drop_first Py.Bool.of_bool); ("adjacency", Some(adjacency |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])
                       |> Ndarray.of_pyobject
                  let trustworthiness ?n_neighbors ?metric ~x ~x_embedded () =
                     Py.Module.get_function_with_keywords ns "trustworthiness"
                       [||]
                       (Wrap_utils.keyword_args [("n_neighbors", Wrap_utils.Option.map n_neighbors Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("X", Some(x |> Ndarray.to_pyobject)); ("X_embedded", Some(x_embedded |> Ndarray.to_pyobject))])
                       |> Py.Float.to_float
