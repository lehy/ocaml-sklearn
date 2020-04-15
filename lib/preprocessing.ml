let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.preprocessing"

module Binarizer = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?threshold ?copy () =
                     Py.Module.get_function_with_keywords ns "Binarizer"
                       [||]
                       (Wrap_utils.keyword_args [("threshold", Wrap_utils.Option.map threshold (function
| `Float x -> Py.Float.of_float x
| `PyObject x -> Wrap_utils.id x
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

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

                  let transform ?copy ~x self =
                     Py.Module.get_function_with_keywords self "transform"
                       [||]
                       (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module FunctionTransformer = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?func ?inverse_func ?validate ?accept_sparse ?check_inverse ?kw_args ?inv_kw_args () =
   Py.Module.get_function_with_keywords ns "FunctionTransformer"
     [||]
     (Wrap_utils.keyword_args [("func", func); ("inverse_func", inverse_func); ("validate", Wrap_utils.Option.map validate Py.Bool.of_bool); ("accept_sparse", Wrap_utils.Option.map accept_sparse Py.Bool.of_bool); ("check_inverse", Wrap_utils.Option.map check_inverse Py.Bool.of_bool); ("kw_args", kw_args); ("inv_kw_args", inv_kw_args)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module KBinsDiscretizer = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_bins ?encode ?strategy () =
                     Py.Module.get_function_with_keywords ns "KBinsDiscretizer"
                       [||]
                       (Wrap_utils.keyword_args [("n_bins", Wrap_utils.Option.map n_bins (function
| `Int x -> Py.Int.of_int x
| `Ndarray x -> Ndarray.to_pyobject x
)); ("encode", Wrap_utils.Option.map encode (function
| `Onehot -> Py.String.of_string "onehot"
| `Onehot_dense -> Py.String.of_string "onehot-dense"
| `Ordinal -> Py.String.of_string "ordinal"
)); ("strategy", Wrap_utils.Option.map strategy (function
| `Uniform -> Py.String.of_string "uniform"
| `Quantile -> Py.String.of_string "quantile"
| `Kmeans -> Py.String.of_string "kmeans"
))])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let inverse_transform ~xt self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("Xt", Some(xt |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let n_bins_ self =
  match Py.Object.get_attr_string self "n_bins_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_bins_")
| Some x -> Wrap_utils.id x
let bin_edges_ self =
  match Py.Object.get_attr_string self "bin_edges_" with
| None -> raise (Wrap_utils.Attribute_not_found "bin_edges_")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module KernelCenterer = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create () =
   Py.Module.get_function_with_keywords ns "KernelCenterer"
     [||]
     []

let fit ?y ~k self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("K", Some(k |> Ndarray.to_pyobject))])

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

let transform ?copy ~k self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("K", Some(k |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let k_fit_rows_ self =
  match Py.Object.get_attr_string self "K_fit_rows_" with
| None -> raise (Wrap_utils.Attribute_not_found "K_fit_rows_")
| Some x -> Ndarray.of_pyobject x
let k_fit_all_ self =
  match Py.Object.get_attr_string self "K_fit_all_" with
| None -> raise (Wrap_utils.Attribute_not_found "K_fit_all_")
| Some x -> Py.Float.to_float x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LabelBinarizer = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?neg_label ?pos_label ?sparse_output () =
   Py.Module.get_function_with_keywords ns "LabelBinarizer"
     [||]
     (Wrap_utils.keyword_args [("neg_label", Wrap_utils.Option.map neg_label Py.Int.of_int); ("pos_label", Wrap_utils.Option.map pos_label Py.Int.of_int); ("sparse_output", Wrap_utils.Option.map sparse_output Py.Bool.of_bool)])

let fit ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Ndarray.to_pyobject))])

                  let fit_transform ~y self =
                     Py.Module.get_function_with_keywords self "fit_transform"
                       [||]
                       (Wrap_utils.keyword_args [("y", Some(y |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

                  let inverse_transform ?threshold ~y self =
                     Py.Module.get_function_with_keywords self "inverse_transform"
                       [||]
                       (Wrap_utils.keyword_args [("threshold", Wrap_utils.Option.map threshold (function
| `Float x -> Py.Float.of_float x
| `None -> Py.String.of_string "None"
)); ("Y", Some(y |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

                  let transform ~y self =
                     Py.Module.get_function_with_keywords self "transform"
                       [||]
                       (Wrap_utils.keyword_args [("y", Some(y |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let classes_ self =
  match Py.Object.get_attr_string self "classes_" with
| None -> raise (Wrap_utils.Attribute_not_found "classes_")
| Some x -> Ndarray.of_pyobject x
let y_type_ self =
  match Py.Object.get_attr_string self "y_type_" with
| None -> raise (Wrap_utils.Attribute_not_found "y_type_")
| Some x -> Py.String.to_string x
let sparse_input_ self =
  match Py.Object.get_attr_string self "sparse_input_" with
| None -> raise (Wrap_utils.Attribute_not_found "sparse_input_")
| Some x -> Py.Bool.to_bool x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LabelEncoder = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create () =
   Py.Module.get_function_with_keywords ns "LabelEncoder"
     [||]
     []

let fit ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Ndarray.to_pyobject))])

let fit_transform ~y self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let inverse_transform ~y self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~y self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let classes_ self =
  match Py.Object.get_attr_string self "classes_" with
| None -> raise (Wrap_utils.Attribute_not_found "classes_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MaxAbsScaler = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?copy () =
   Py.Module.get_function_with_keywords ns "MaxAbsScaler"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

                  let fit ?y ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
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

                  let inverse_transform ~x self =
                     Py.Module.get_function_with_keywords self "inverse_transform"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

                  let partial_fit ?y ~x self =
                     Py.Module.get_function_with_keywords self "partial_fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

                  let transform ~x self =
                     Py.Module.get_function_with_keywords self "transform"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let scale_ self =
  match Py.Object.get_attr_string self "scale_" with
| None -> raise (Wrap_utils.Attribute_not_found "scale_")
| Some x -> Ndarray.of_pyobject x
let max_abs_ self =
  match Py.Object.get_attr_string self "max_abs_" with
| None -> raise (Wrap_utils.Attribute_not_found "max_abs_")
| Some x -> Ndarray.of_pyobject x
let n_samples_seen_ self =
  match Py.Object.get_attr_string self "n_samples_seen_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_samples_seen_")
| Some x -> Py.Int.to_int x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MinMaxScaler = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?feature_range ?copy () =
   Py.Module.get_function_with_keywords ns "MinMaxScaler"
     [||]
     (Wrap_utils.keyword_args [("feature_range", feature_range); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let partial_fit ?y ~x self =
   Py.Module.get_function_with_keywords self "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let min_ self =
  match Py.Object.get_attr_string self "min_" with
| None -> raise (Wrap_utils.Attribute_not_found "min_")
| Some x -> Ndarray.of_pyobject x
let scale_ self =
  match Py.Object.get_attr_string self "scale_" with
| None -> raise (Wrap_utils.Attribute_not_found "scale_")
| Some x -> Ndarray.of_pyobject x
let data_min_ self =
  match Py.Object.get_attr_string self "data_min_" with
| None -> raise (Wrap_utils.Attribute_not_found "data_min_")
| Some x -> Ndarray.of_pyobject x
let data_max_ self =
  match Py.Object.get_attr_string self "data_max_" with
| None -> raise (Wrap_utils.Attribute_not_found "data_max_")
| Some x -> Ndarray.of_pyobject x
let data_range_ self =
  match Py.Object.get_attr_string self "data_range_" with
| None -> raise (Wrap_utils.Attribute_not_found "data_range_")
| Some x -> Ndarray.of_pyobject x
let n_samples_seen_ self =
  match Py.Object.get_attr_string self "n_samples_seen_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_samples_seen_")
| Some x -> Py.Int.to_int x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MultiLabelBinarizer = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?classes ?sparse_output () =
   Py.Module.get_function_with_keywords ns "MultiLabelBinarizer"
     [||]
     (Wrap_utils.keyword_args [("classes", Wrap_utils.Option.map classes Ndarray.to_pyobject); ("sparse_output", Wrap_utils.Option.map sparse_output Py.Bool.of_bool)])

let fit ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Ndarray.List.to_pyobject))])

let fit_transform ~y self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Ndarray.List.to_pyobject))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

                  let inverse_transform ~yt self =
                     Py.Module.get_function_with_keywords self "inverse_transform"
                       [||]
                       (Wrap_utils.keyword_args [("yt", Some(yt |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~y self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y ))])
     |> Ndarray.of_pyobject
let classes_ self =
  match Py.Object.get_attr_string self "classes_" with
| None -> raise (Wrap_utils.Attribute_not_found "classes_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Normalizer = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?norm ?copy () =
                     Py.Module.get_function_with_keywords ns "Normalizer"
                       [||]
                       (Wrap_utils.keyword_args [("norm", Wrap_utils.Option.map norm (function
| `L1 -> Py.String.of_string "l1"
| `L2 -> Py.String.of_string "l2"
| `Max -> Py.String.of_string "max"
| `PyObject x -> Wrap_utils.id x
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

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

                  let transform ?copy ~x self =
                     Py.Module.get_function_with_keywords self "transform"
                       [||]
                       (Wrap_utils.keyword_args [("copy", copy); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module OneHotEncoder = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?categories ?drop ?sparse ?dtype ?handle_unknown () =
                     Py.Module.get_function_with_keywords ns "OneHotEncoder"
                       [||]
                       (Wrap_utils.keyword_args [("categories", Wrap_utils.Option.map categories (function
| `Auto -> Py.String.of_string "auto"
| `PyObject x -> Wrap_utils.id x
)); ("drop", Wrap_utils.Option.map drop (function
| `First -> Py.String.of_string "first"
| `PyObject x -> Wrap_utils.id x
)); ("sparse", Wrap_utils.Option.map sparse Py.Bool.of_bool); ("dtype", dtype); ("handle_unknown", Wrap_utils.Option.map handle_unknown (function
| `Error -> Py.String.of_string "error"
| `Ignore -> Py.String.of_string "ignore"
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
let get_feature_names ?input_features self =
   Py.Module.get_function_with_keywords self "get_feature_names"
     [||]
     (Wrap_utils.keyword_args [("input_features", Wrap_utils.Option.map input_features (Py.List.of_list_map Py.String.of_string))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

                  let inverse_transform ~x self =
                     Py.Module.get_function_with_keywords self "inverse_transform"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let categories_ self =
  match Py.Object.get_attr_string self "categories_" with
| None -> raise (Wrap_utils.Attribute_not_found "categories_")
| Some x -> Wrap_utils.id x
let drop_idx_ self =
  match Py.Object.get_attr_string self "drop_idx_" with
| None -> raise (Wrap_utils.Attribute_not_found "drop_idx_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module OrdinalEncoder = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?categories ?dtype () =
                     Py.Module.get_function_with_keywords ns "OrdinalEncoder"
                       [||]
                       (Wrap_utils.keyword_args [("categories", Wrap_utils.Option.map categories (function
| `Auto -> Py.String.of_string "auto"
| `PyObject x -> Wrap_utils.id x
)); ("dtype", dtype)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

                  let inverse_transform ~x self =
                     Py.Module.get_function_with_keywords self "inverse_transform"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let categories_ self =
  match Py.Object.get_attr_string self "categories_" with
| None -> raise (Wrap_utils.Attribute_not_found "categories_")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module PolynomialFeatures = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?degree ?interaction_only ?include_bias ?order () =
                     Py.Module.get_function_with_keywords ns "PolynomialFeatures"
                       [||]
                       (Wrap_utils.keyword_args [("degree", Wrap_utils.Option.map degree Py.Int.of_int); ("interaction_only", Wrap_utils.Option.map interaction_only Py.Bool.of_bool); ("include_bias", Wrap_utils.Option.map include_bias Py.Bool.of_bool); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
))])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Ndarray.of_pyobject
let get_feature_names ?input_features self =
   Py.Module.get_function_with_keywords self "get_feature_names"
     [||]
     (Wrap_utils.keyword_args [("input_features", Wrap_utils.Option.map input_features (Py.List.of_list_map Py.String.of_string))])
     |> (Py.List.to_list_map Py.String.to_string)
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
let powers_ self =
  match Py.Object.get_attr_string self "powers_" with
| None -> raise (Wrap_utils.Attribute_not_found "powers_")
| Some x -> Ndarray.of_pyobject x
let n_input_features_ self =
  match Py.Object.get_attr_string self "n_input_features_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_input_features_")
| Some x -> Py.Int.to_int x
let n_output_features_ self =
  match Py.Object.get_attr_string self "n_output_features_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_output_features_")
| Some x -> Py.Int.to_int x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module PowerTransformer = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?method_ ?standardize ?copy () =
   Py.Module.get_function_with_keywords ns "PowerTransformer"
     [||]
     (Wrap_utils.keyword_args [("method", Wrap_utils.Option.map method_ Py.String.of_string); ("standardize", Wrap_utils.Option.map standardize Py.Bool.of_bool); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x ))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let lambdas_ self =
  match Py.Object.get_attr_string self "lambdas_" with
| None -> raise (Wrap_utils.Attribute_not_found "lambdas_")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module QuantileTransformer = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?n_quantiles ?output_distribution ?ignore_implicit_zeros ?subsample ?random_state ?copy () =
                     Py.Module.get_function_with_keywords ns "QuantileTransformer"
                       [||]
                       (Wrap_utils.keyword_args [("n_quantiles", Wrap_utils.Option.map n_quantiles Py.Int.of_int); ("output_distribution", Wrap_utils.Option.map output_distribution Py.String.of_string); ("ignore_implicit_zeros", Wrap_utils.Option.map ignore_implicit_zeros Py.Bool.of_bool); ("subsample", Wrap_utils.Option.map subsample Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

                  let fit ?y ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
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

                  let inverse_transform ~x self =
                     Py.Module.get_function_with_keywords self "inverse_transform"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

                  let transform ~x self =
                     Py.Module.get_function_with_keywords self "transform"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let n_quantiles_ self =
  match Py.Object.get_attr_string self "n_quantiles_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_quantiles_")
| Some x -> Py.Int.to_int x
let quantiles_ self =
  match Py.Object.get_attr_string self "quantiles_" with
| None -> raise (Wrap_utils.Attribute_not_found "quantiles_")
| Some x -> Ndarray.of_pyobject x
let references_ self =
  match Py.Object.get_attr_string self "references_" with
| None -> raise (Wrap_utils.Attribute_not_found "references_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RobustScaler = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?with_centering ?with_scaling ?quantile_range ?copy () =
   Py.Module.get_function_with_keywords ns "RobustScaler"
     [||]
     (Wrap_utils.keyword_args [("with_centering", Wrap_utils.Option.map with_centering Py.Bool.of_bool); ("with_scaling", Wrap_utils.Option.map with_scaling Py.Bool.of_bool); ("quantile_range", quantile_range); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

                  let transform ~x self =
                     Py.Module.get_function_with_keywords self "transform"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let center_ self =
  match Py.Object.get_attr_string self "center_" with
| None -> raise (Wrap_utils.Attribute_not_found "center_")
| Some x -> Ndarray.of_pyobject x
let scale_ self =
  match Py.Object.get_attr_string self "scale_" with
| None -> raise (Wrap_utils.Attribute_not_found "scale_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module StandardScaler = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?copy ?with_mean ?with_std () =
   Py.Module.get_function_with_keywords ns "StandardScaler"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("with_mean", Wrap_utils.Option.map with_mean Py.Bool.of_bool); ("with_std", Wrap_utils.Option.map with_std Py.Bool.of_bool)])

                  let fit ?y ~x self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
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

let inverse_transform ?copy ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
                  let partial_fit ?y ~x self =
                     Py.Module.get_function_with_keywords self "partial_fit"
                       [||]
                       (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ?copy ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let scale_ self =
  match Py.Object.get_attr_string self "scale_" with
| None -> raise (Wrap_utils.Attribute_not_found "scale_")
| Some x -> Wrap_utils.id x
let mean_ self =
  match Py.Object.get_attr_string self "mean_" with
| None -> raise (Wrap_utils.Attribute_not_found "mean_")
| Some x -> Wrap_utils.id x
let var_ self =
  match Py.Object.get_attr_string self "var_" with
| None -> raise (Wrap_utils.Attribute_not_found "var_")
| Some x -> Wrap_utils.id x
let n_samples_seen_ self =
  match Py.Object.get_attr_string self "n_samples_seen_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_samples_seen_")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let add_dummy_feature ?value ~x () =
                     Py.Module.get_function_with_keywords ns "add_dummy_feature"
                       [||]
                       (Wrap_utils.keyword_args [("value", Wrap_utils.Option.map value Py.Float.of_float); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

                  let binarize ?threshold ?copy ~x () =
                     Py.Module.get_function_with_keywords ns "binarize"
                       [||]
                       (Wrap_utils.keyword_args [("threshold", Wrap_utils.Option.map threshold (function
| `Float x -> Py.Float.of_float x
| `PyObject x -> Wrap_utils.id x
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

let label_binarize ?neg_label ?pos_label ?sparse_output ~y ~classes () =
   Py.Module.get_function_with_keywords ns "label_binarize"
     [||]
     (Wrap_utils.keyword_args [("neg_label", Wrap_utils.Option.map neg_label Py.Int.of_int); ("pos_label", Wrap_utils.Option.map pos_label Py.Int.of_int); ("sparse_output", Wrap_utils.Option.map sparse_output Py.Bool.of_bool); ("y", Some(y |> Ndarray.to_pyobject)); ("classes", Some(classes |> Ndarray.to_pyobject))])

let maxabs_scale ?axis ?copy ~x () =
   Py.Module.get_function_with_keywords ns "maxabs_scale"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Ndarray.to_pyobject))])

let minmax_scale ?feature_range ?axis ?copy ~x () =
   Py.Module.get_function_with_keywords ns "minmax_scale"
     [||]
     (Wrap_utils.keyword_args [("feature_range", feature_range); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Ndarray.to_pyobject))])

                  let normalize ?norm ?axis ?copy ?return_norm ~x () =
                     Py.Module.get_function_with_keywords ns "normalize"
                       [||]
                       (Wrap_utils.keyword_args [("norm", Wrap_utils.Option.map norm (function
| `L1 -> Py.String.of_string "l1"
| `L2 -> Py.String.of_string "l2"
| `Max -> Py.String.of_string "max"
| `PyObject x -> Wrap_utils.id x
)); ("axis", axis); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("return_norm", Wrap_utils.Option.map return_norm Py.Bool.of_bool); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let power_transform ?method_ ?standardize ?copy ~x () =
   Py.Module.get_function_with_keywords ns "power_transform"
     [||]
     (Wrap_utils.keyword_args [("method", Wrap_utils.Option.map method_ Py.String.of_string); ("standardize", Wrap_utils.Option.map standardize Py.Bool.of_bool); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
                  let quantile_transform ?axis ?n_quantiles ?output_distribution ?ignore_implicit_zeros ?subsample ?random_state ?copy ~x () =
                     Py.Module.get_function_with_keywords ns "quantile_transform"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("n_quantiles", Wrap_utils.Option.map n_quantiles Py.Int.of_int); ("output_distribution", Wrap_utils.Option.map output_distribution Py.String.of_string); ("ignore_implicit_zeros", Wrap_utils.Option.map ignore_implicit_zeros Py.Bool.of_bool); ("subsample", Wrap_utils.Option.map subsample Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

let robust_scale ?axis ?with_centering ?with_scaling ?quantile_range ?copy ~x () =
   Py.Module.get_function_with_keywords ns "robust_scale"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("with_centering", Wrap_utils.Option.map with_centering Py.Bool.of_bool); ("with_scaling", Wrap_utils.Option.map with_scaling Py.Bool.of_bool); ("quantile_range", quantile_range); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Ndarray.to_pyobject))])

                  let scale ?axis ?with_mean ?with_std ?copy ~x () =
                     Py.Module.get_function_with_keywords ns "scale"
                       [||]
                       (Wrap_utils.keyword_args [("axis", axis); ("with_mean", Wrap_utils.Option.map with_mean Py.Bool.of_bool); ("with_std", Wrap_utils.Option.map with_std Py.Bool.of_bool); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

