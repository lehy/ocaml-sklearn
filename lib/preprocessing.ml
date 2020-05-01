let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.preprocessing"

let get_py name = Py.Module.get ns name
module Binarizer = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?threshold ?copy () =
                     Py.Module.get_function_with_keywords ns "Binarizer"
                       [||]
                       (Wrap_utils.keyword_args [("threshold", Wrap_utils.Option.map threshold (function
| `F x -> Py.Float.of_float x
| `T_0_0_by x -> Wrap_utils.id x
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

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

let transform ?copy ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
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
     (Wrap_utils.keyword_args [("func", func); ("inverse_func", inverse_func); ("validate", Wrap_utils.Option.map validate Py.Bool.of_bool); ("accept_sparse", Wrap_utils.Option.map accept_sparse Py.Bool.of_bool); ("check_inverse", Wrap_utils.Option.map check_inverse Py.Bool.of_bool); ("kw_args", Wrap_utils.Option.map kw_args Dict.to_pyobject); ("inv_kw_args", Wrap_utils.Option.map inv_kw_args Dict.to_pyobject)])

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
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
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
| `I x -> Py.Int.of_int x
| `Arr x -> Arr.to_pyobject x
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
let inverse_transform ~xt self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("Xt", Some(xt |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let n_bins_opt self =
  match Py.Object.get_attr_string self "n_bins_" with
  | None -> failwith "attribute n_bins_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let n_bins_ self = match n_bins_opt self with
  | None -> raise Not_found
  | Some x -> x

let bin_edges_opt self =
  match Py.Object.get_attr_string self "bin_edges_" with
  | None -> failwith "attribute bin_edges_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let bin_edges_ self = match bin_edges_opt self with
  | None -> raise Not_found
  | Some x -> x
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
     (Wrap_utils.keyword_args [("y", y); ("K", Some(k |> Arr.to_pyobject))])

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

let transform ?copy ~k self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("K", Some(k |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let k_fit_rows_opt self =
  match Py.Object.get_attr_string self "K_fit_rows_" with
  | None -> failwith "attribute K_fit_rows_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let k_fit_rows_ self = match k_fit_rows_opt self with
  | None -> raise Not_found
  | Some x -> x

let k_fit_all_opt self =
  match Py.Object.get_attr_string self "K_fit_all_" with
  | None -> failwith "attribute K_fit_all_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let k_fit_all_ self = match k_fit_all_opt self with
  | None -> raise Not_found
  | Some x -> x
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
     (Wrap_utils.keyword_args [("y", Some(y |> Arr.to_pyobject))])

let fit_transform ~y self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let inverse_transform ?threshold ~y self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("threshold", Wrap_utils.Option.map threshold Py.Float.of_float); ("Y", Some(y |> Arr.to_pyobject))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~y self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let classes_opt self =
  match Py.Object.get_attr_string self "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_type_opt self =
  match Py.Object.get_attr_string self "y_type_" with
  | None -> failwith "attribute y_type_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.String.to_string x)

let y_type_ self = match y_type_opt self with
  | None -> raise Not_found
  | Some x -> x

let sparse_input_opt self =
  match Py.Object.get_attr_string self "sparse_input_" with
  | None -> failwith "attribute sparse_input_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Bool.to_bool x)

let sparse_input_ self = match sparse_input_opt self with
  | None -> raise Not_found
  | Some x -> x
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
     (Wrap_utils.keyword_args [("y", Some(y |> Arr.to_pyobject))])

let fit_transform ~y self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let inverse_transform ~y self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~y self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let classes_opt self =
  match Py.Object.get_attr_string self "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x
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
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])

let partial_fit ?y ~x self =
   Py.Module.get_function_with_keywords self "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let scale_opt self =
  match Py.Object.get_attr_string self "scale_" with
  | None -> failwith "attribute scale_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let scale_ self = match scale_opt self with
  | None -> raise Not_found
  | Some x -> x

let max_abs_opt self =
  match Py.Object.get_attr_string self "max_abs_" with
  | None -> failwith "attribute max_abs_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let max_abs_ self = match max_abs_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_samples_seen_opt self =
  match Py.Object.get_attr_string self "n_samples_seen_" with
  | None -> failwith "attribute n_samples_seen_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_samples_seen_ self = match n_samples_seen_opt self with
  | None -> raise Not_found
  | Some x -> x
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
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let partial_fit ?y ~x self =
   Py.Module.get_function_with_keywords self "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let min_opt self =
  match Py.Object.get_attr_string self "min_" with
  | None -> failwith "attribute min_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let min_ self = match min_opt self with
  | None -> raise Not_found
  | Some x -> x

let scale_opt self =
  match Py.Object.get_attr_string self "scale_" with
  | None -> failwith "attribute scale_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let scale_ self = match scale_opt self with
  | None -> raise Not_found
  | Some x -> x

let data_min_opt self =
  match Py.Object.get_attr_string self "data_min_" with
  | None -> failwith "attribute data_min_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let data_min_ self = match data_min_opt self with
  | None -> raise Not_found
  | Some x -> x

let data_max_opt self =
  match Py.Object.get_attr_string self "data_max_" with
  | None -> failwith "attribute data_max_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let data_max_ self = match data_max_opt self with
  | None -> raise Not_found
  | Some x -> x

let data_range_opt self =
  match Py.Object.get_attr_string self "data_range_" with
  | None -> failwith "attribute data_range_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let data_range_ self = match data_range_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_samples_seen_opt self =
  match Py.Object.get_attr_string self "n_samples_seen_" with
  | None -> failwith "attribute n_samples_seen_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_samples_seen_ self = match n_samples_seen_opt self with
  | None -> raise Not_found
  | Some x -> x
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
     (Wrap_utils.keyword_args [("classes", Wrap_utils.Option.map classes Arr.to_pyobject); ("sparse_output", Wrap_utils.Option.map sparse_output Py.Bool.of_bool)])

let fit ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Arr.List.to_pyobject))])

let fit_transform ~y self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Arr.List.to_pyobject))])
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let inverse_transform ~yt self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("yt", Some(yt |> Arr.to_pyobject))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~y self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Arr.List.to_pyobject))])
     |> Arr.of_pyobject

let classes_opt self =
  match Py.Object.get_attr_string self "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x
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
| `T_l2_by x -> Wrap_utils.id x
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

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

let transform ?copy ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
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
| `A_list_of_array_like x -> Wrap_utils.id x
)); ("drop", Wrap_utils.Option.map drop (function
| `First -> Py.String.of_string "first"
| `A_array_like x -> Wrap_utils.id x
)); ("sparse", Wrap_utils.Option.map sparse Py.Bool.of_bool); ("dtype", dtype); ("handle_unknown", Wrap_utils.Option.map handle_unknown (function
| `Error -> Py.String.of_string "error"
| `Ignore -> Py.String.of_string "ignore"
))])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])

let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let get_feature_names ?input_features self =
   Py.Module.get_function_with_keywords self "get_feature_names"
     [||]
     (Wrap_utils.keyword_args [("input_features", Wrap_utils.Option.map input_features (Py.List.of_list_map Py.String.of_string))])
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let categories_opt self =
  match Py.Object.get_attr_string self "categories_" with
  | None -> failwith "attribute categories_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.List.of_pyobject x)

let categories_ self = match categories_opt self with
  | None -> raise Not_found
  | Some x -> x

let drop_idx_opt self =
  match Py.Object.get_attr_string self "drop_idx_" with
  | None -> failwith "attribute drop_idx_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let drop_idx_ self = match drop_idx_opt self with
  | None -> raise Not_found
  | Some x -> x
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
| `A_list_of_array_like x -> Wrap_utils.id x
)); ("dtype", dtype)])

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
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let categories_opt self =
  match Py.Object.get_attr_string self "categories_" with
  | None -> failwith "attribute categories_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.List.of_pyobject x)

let categories_ self = match categories_opt self with
  | None -> raise Not_found
  | Some x -> x
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
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Arr.of_pyobject
let get_feature_names ?input_features self =
   Py.Module.get_function_with_keywords self "get_feature_names"
     [||]
     (Wrap_utils.keyword_args [("input_features", Wrap_utils.Option.map input_features (Py.List.of_list_map Py.String.of_string))])
     |> (Py.List.to_list_map Py.String.to_string)
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

let powers_opt self =
  match Py.Object.get_attr_string self "powers_" with
  | None -> failwith "attribute powers_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let powers_ self = match powers_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_input_features_opt self =
  match Py.Object.get_attr_string self "n_input_features_" with
  | None -> failwith "attribute n_input_features_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_input_features_ self = match n_input_features_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_output_features_opt self =
  match Py.Object.get_attr_string self "n_output_features_" with
  | None -> failwith "attribute n_output_features_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_output_features_ self = match n_output_features_opt self with
  | None -> raise Not_found
  | Some x -> x
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
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])

let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let lambdas_opt self =
  match Py.Object.get_attr_string self "lambdas_" with
  | None -> failwith "attribute lambdas_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let lambdas_ self = match lambdas_opt self with
  | None -> raise Not_found
  | Some x -> x
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
     (Wrap_utils.keyword_args [("n_quantiles", Wrap_utils.Option.map n_quantiles Py.Int.of_int); ("output_distribution", Wrap_utils.Option.map output_distribution Py.String.of_string); ("ignore_implicit_zeros", Wrap_utils.Option.map ignore_implicit_zeros Py.Bool.of_bool); ("subsample", Wrap_utils.Option.map subsample Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

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
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let n_quantiles_opt self =
  match Py.Object.get_attr_string self "n_quantiles_" with
  | None -> failwith "attribute n_quantiles_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_quantiles_ self = match n_quantiles_opt self with
  | None -> raise Not_found
  | Some x -> x

let quantiles_opt self =
  match Py.Object.get_attr_string self "quantiles_" with
  | None -> failwith "attribute quantiles_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let quantiles_ self = match quantiles_opt self with
  | None -> raise Not_found
  | Some x -> x

let references_opt self =
  match Py.Object.get_attr_string self "references_" with
  | None -> failwith "attribute references_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let references_ self = match references_opt self with
  | None -> raise Not_found
  | Some x -> x
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
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let center_opt self =
  match Py.Object.get_attr_string self "center_" with
  | None -> failwith "attribute center_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let center_ self = match center_opt self with
  | None -> raise Not_found
  | Some x -> x

let scale_opt self =
  match Py.Object.get_attr_string self "scale_" with
  | None -> failwith "attribute scale_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let scale_ self = match scale_opt self with
  | None -> raise Not_found
  | Some x -> x
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
let inverse_transform ?copy ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let partial_fit ?y ~x self =
   Py.Module.get_function_with_keywords self "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ?copy ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let scale_opt self =
  match Py.Object.get_attr_string self "scale_" with
  | None -> failwith "attribute scale_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let scale_ self = match scale_opt self with
  | None -> raise Not_found
  | Some x -> x

let mean_opt self =
  match Py.Object.get_attr_string self "mean_" with
  | None -> failwith "attribute mean_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let mean_ self = match mean_opt self with
  | None -> raise Not_found
  | Some x -> x

let var_opt self =
  match Py.Object.get_attr_string self "var_" with
  | None -> failwith "attribute var_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let var_ self = match var_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_samples_seen_opt self =
  match Py.Object.get_attr_string self "n_samples_seen_" with
  | None -> failwith "attribute n_samples_seen_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun x -> if Py.Int.check x then `I (Py.Int.to_int x) else if (fun x -> (Wrap_utils.isinstance Wrap_utils.ndarray x) || (Wrap_utils.isinstance Wrap_utils.csr_matrix x)) x then `Arr (Arr.of_pyobject x) else failwith "could not identify type from Python value") x)

let n_samples_seen_ self = match n_samples_seen_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let add_dummy_feature ?value ~x () =
   Py.Module.get_function_with_keywords ns "add_dummy_feature"
     [||]
     (Wrap_utils.keyword_args [("value", Wrap_utils.Option.map value Py.Float.of_float); ("X", Some(x |> Arr.to_pyobject))])

                  let binarize ?threshold ?copy ~x () =
                     Py.Module.get_function_with_keywords ns "binarize"
                       [||]
                       (Wrap_utils.keyword_args [("threshold", Wrap_utils.Option.map threshold (function
| `F x -> Py.Float.of_float x
| `T_0_0_by x -> Wrap_utils.id x
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])

let label_binarize ?neg_label ?pos_label ?sparse_output ~y ~classes () =
   Py.Module.get_function_with_keywords ns "label_binarize"
     [||]
     (Wrap_utils.keyword_args [("neg_label", Wrap_utils.Option.map neg_label Py.Int.of_int); ("pos_label", Wrap_utils.Option.map pos_label Py.Int.of_int); ("sparse_output", Wrap_utils.Option.map sparse_output Py.Bool.of_bool); ("y", Some(y |> Arr.to_pyobject)); ("classes", Some(classes |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let maxabs_scale ?axis ?copy ~x () =
   Py.Module.get_function_with_keywords ns "maxabs_scale"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])

let minmax_scale ?feature_range ?axis ?copy ~x () =
   Py.Module.get_function_with_keywords ns "minmax_scale"
     [||]
     (Wrap_utils.keyword_args [("feature_range", feature_range); ("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])

                  let normalize ?norm ?axis ?copy ?return_norm ~x () =
                     Py.Module.get_function_with_keywords ns "normalize"
                       [||]
                       (Wrap_utils.keyword_args [("norm", Wrap_utils.Option.map norm (function
| `L1 -> Py.String.of_string "l1"
| `L2 -> Py.String.of_string "l2"
| `Max -> Py.String.of_string "max"
| `T_l2_by x -> Wrap_utils.id x
)); ("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `T_1_by x -> Wrap_utils.id x
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("return_norm", Wrap_utils.Option.map return_norm Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])
                       |> (fun x -> ((Arr.of_pyobject (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let power_transform ?method_ ?standardize ?copy ~x () =
                     Py.Module.get_function_with_keywords ns "power_transform"
                       [||]
                       (Wrap_utils.keyword_args [("method", Wrap_utils.Option.map method_ (function
| `Yeo_johnson -> Py.String.of_string "yeo-johnson"
| `Box_cox -> Py.String.of_string "box-cox"
)); ("standardize", Wrap_utils.Option.map standardize Py.Bool.of_bool); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])
                       |> Arr.of_pyobject
                  let quantile_transform ?axis ?n_quantiles ?output_distribution ?ignore_implicit_zeros ?subsample ?random_state ?copy ~x () =
                     Py.Module.get_function_with_keywords ns "quantile_transform"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("n_quantiles", Wrap_utils.Option.map n_quantiles Py.Int.of_int); ("output_distribution", Wrap_utils.Option.map output_distribution (function
| `Uniform -> Py.String.of_string "uniform"
| `Normal -> Py.String.of_string "normal"
)); ("ignore_implicit_zeros", Wrap_utils.Option.map ignore_implicit_zeros Py.Bool.of_bool); ("subsample", Wrap_utils.Option.map subsample Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])
                       |> Arr.of_pyobject
let robust_scale ?axis ?with_centering ?with_scaling ?quantile_range ?copy ~x () =
   Py.Module.get_function_with_keywords ns "robust_scale"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("with_centering", Wrap_utils.Option.map with_centering Py.Bool.of_bool); ("with_scaling", Wrap_utils.Option.map with_scaling Py.Bool.of_bool); ("quantile_range", quantile_range); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])

let scale ?axis ?with_mean ?with_std ?copy ~x () =
   Py.Module.get_function_with_keywords ns "scale"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("with_mean", Wrap_utils.Option.map with_mean Py.Bool.of_bool); ("with_std", Wrap_utils.Option.map with_std Py.Bool.of_bool); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])

