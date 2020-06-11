let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.metrics"

let get_py name = Py.Module.get __wrap_namespace name
module ConfusionMatrixDisplay = struct
type tag = [`ConfusionMatrixDisplay]
type t = [`ConfusionMatrixDisplay | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ~confusion_matrix ~display_labels () =
   Py.Module.get_function_with_keywords __wrap_namespace "ConfusionMatrixDisplay"
     [||]
     (Wrap_utils.keyword_args [("confusion_matrix", Some(confusion_matrix |> Np.Obj.to_pyobject)); ("display_labels", Some(display_labels |> Np.Obj.to_pyobject))])
     |> of_pyobject
                  let plot ?include_values ?cmap ?xticks_rotation ?values_format ?ax self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "plot"
                       [||]
                       (Wrap_utils.keyword_args [("include_values", Wrap_utils.Option.map include_values Py.Bool.of_bool); ("cmap", Wrap_utils.Option.map cmap (function
| `S x -> Py.String.of_string x
| `Matplotlib_Colormap x -> Wrap_utils.id x
)); ("xticks_rotation", Wrap_utils.Option.map xticks_rotation (function
| `Horizontal -> Py.String.of_string "horizontal"
| `F x -> Py.Float.of_float x
| `Vertical -> Py.String.of_string "vertical"
)); ("values_format", Wrap_utils.Option.map values_format Py.String.of_string); ("ax", ax)])


let im_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "im_" with
  | None -> failwith "attribute im_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let im_ self = match im_opt self with
  | None -> raise Not_found
  | Some x -> x

let text_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "text_" with
  | None -> failwith "attribute text_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let text_ self = match text_opt self with
  | None -> raise Not_found
  | Some x -> x

let ax_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "ax_" with
  | None -> failwith "attribute ax_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let ax_ self = match ax_opt self with
  | None -> raise Not_found
  | Some x -> x

let figure_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "figure_" with
  | None -> failwith "attribute figure_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let figure_ self = match figure_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module PrecisionRecallDisplay = struct
type tag = [`PrecisionRecallDisplay]
type t = [`Object | `PrecisionRecallDisplay] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ~precision ~recall ~average_precision ~estimator_name () =
   Py.Module.get_function_with_keywords __wrap_namespace "PrecisionRecallDisplay"
     [||]
     (Wrap_utils.keyword_args [("precision", Some(precision |> Np.Obj.to_pyobject)); ("recall", Some(recall |> Np.Obj.to_pyobject)); ("average_precision", Some(average_precision |> Py.Float.of_float)); ("estimator_name", Some(estimator_name |> Py.String.of_string))])
     |> of_pyobject
let plot ?ax ?name ?kwargs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "plot"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("ax", ax); ("name", Wrap_utils.Option.map name Py.String.of_string)]) (match kwargs with None -> [] | Some x -> x))


let line_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "line_" with
  | None -> failwith "attribute line_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let line_ self = match line_opt self with
  | None -> raise Not_found
  | Some x -> x

let ax_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "ax_" with
  | None -> failwith "attribute ax_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let ax_ self = match ax_opt self with
  | None -> raise Not_found
  | Some x -> x

let figure_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "figure_" with
  | None -> failwith "attribute figure_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let figure_ self = match figure_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RocCurveDisplay = struct
type tag = [`RocCurveDisplay]
type t = [`Object | `RocCurveDisplay] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ~fpr ~tpr ~roc_auc ~estimator_name () =
   Py.Module.get_function_with_keywords __wrap_namespace "RocCurveDisplay"
     [||]
     (Wrap_utils.keyword_args [("fpr", Some(fpr |> Np.Obj.to_pyobject)); ("tpr", Some(tpr |> Np.Obj.to_pyobject)); ("roc_auc", Some(roc_auc |> Py.Float.of_float)); ("estimator_name", Some(estimator_name |> Py.String.of_string))])
     |> of_pyobject
let plot ?ax ?name ?kwargs self =
   Py.Module.get_function_with_keywords (to_pyobject self) "plot"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("ax", ax); ("name", Wrap_utils.Option.map name Py.String.of_string)]) (match kwargs with None -> [] | Some x -> x))


let line_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "line_" with
  | None -> failwith "attribute line_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let line_ self = match line_opt self with
  | None -> raise Not_found
  | Some x -> x

let ax_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "ax_" with
  | None -> failwith "attribute ax_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let ax_ self = match ax_opt self with
  | None -> raise Not_found
  | Some x -> x

let figure_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "figure_" with
  | None -> failwith "attribute figure_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let figure_ self = match figure_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Cluster = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.metrics.cluster"

let get_py name = Py.Module.get __wrap_namespace name
let adjusted_mutual_info_score ?average_method ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "adjusted_mutual_info_score"
     [||]
     (Wrap_utils.keyword_args [("average_method", Wrap_utils.Option.map average_method Py.String.of_string); ("labels_true", Some(labels_true |> Np.Obj.to_pyobject)); ("labels_pred", Some(labels_pred |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let adjusted_rand_score ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "adjusted_rand_score"
     [||]
     (Wrap_utils.keyword_args [("labels_true", Some(labels_true |> Np.Obj.to_pyobject)); ("labels_pred", Some(labels_pred |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let calinski_harabasz_score ~x ~labels () =
   Py.Module.get_function_with_keywords __wrap_namespace "calinski_harabasz_score"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("labels", Some(labels |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let calinski_harabaz_score ~x ~labels () =
   Py.Module.get_function_with_keywords __wrap_namespace "calinski_harabaz_score"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("labels", Some(labels ))])

let completeness_score ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "completeness_score"
     [||]
     (Wrap_utils.keyword_args [("labels_true", Some(labels_true |> Np.Obj.to_pyobject)); ("labels_pred", Some(labels_pred |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
                  let consensus_score ?similarity ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "consensus_score"
                       [||]
                       (Wrap_utils.keyword_args [("similarity", Wrap_utils.Option.map similarity (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("a", Some(a )); ("b", Some(b ))])

let contingency_matrix ?eps ?sparse ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "contingency_matrix"
     [||]
     (Wrap_utils.keyword_args [("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("sparse", Wrap_utils.Option.map sparse Py.Bool.of_bool); ("labels_true", Some(labels_true |> Np.Obj.to_pyobject)); ("labels_pred", Some(labels_pred |> Np.Obj.to_pyobject))])

let davies_bouldin_score ~x ~labels () =
   Py.Module.get_function_with_keywords __wrap_namespace "davies_bouldin_score"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("labels", Some(labels |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let entropy labels =
   Py.Module.get_function_with_keywords __wrap_namespace "entropy"
     [||]
     (Wrap_utils.keyword_args [("labels", Some(labels |> Np.Obj.to_pyobject))])

let fowlkes_mallows_score ?sparse ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "fowlkes_mallows_score"
     [||]
     (Wrap_utils.keyword_args [("sparse", Wrap_utils.Option.map sparse Py.Bool.of_bool); ("labels_true", Some(labels_true |> Np.Obj.to_pyobject)); ("labels_pred", Some(labels_pred |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let homogeneity_completeness_v_measure ?beta ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "homogeneity_completeness_v_measure"
     [||]
     (Wrap_utils.keyword_args [("beta", Wrap_utils.Option.map beta Py.Float.of_float); ("labels_true", Some(labels_true |> Np.Obj.to_pyobject)); ("labels_pred", Some(labels_pred |> Np.Obj.to_pyobject))])
     |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let homogeneity_score ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "homogeneity_score"
     [||]
     (Wrap_utils.keyword_args [("labels_true", Some(labels_true |> Np.Obj.to_pyobject)); ("labels_pred", Some(labels_pred |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let mutual_info_score ?contingency ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "mutual_info_score"
     [||]
     (Wrap_utils.keyword_args [("contingency", Wrap_utils.Option.map contingency Np.Obj.to_pyobject); ("labels_true", Some(labels_true |> Np.Obj.to_pyobject)); ("labels_pred", Some(labels_pred |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let normalized_mutual_info_score ?average_method ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "normalized_mutual_info_score"
     [||]
     (Wrap_utils.keyword_args [("average_method", Wrap_utils.Option.map average_method Py.String.of_string); ("labels_true", Some(labels_true |> Np.Obj.to_pyobject)); ("labels_pred", Some(labels_pred |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
                  let silhouette_samples ?metric ?kwds ~x ~labels () =
                     Py.Module.get_function_with_keywords __wrap_namespace "silhouette_samples"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("X", Some(x |> (function
| `Otherwise x -> Wrap_utils.id x
| `Arr x -> Np.Obj.to_pyobject x
))); ("labels", Some(labels |> Np.Obj.to_pyobject))]) (match kwds with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let silhouette_score ?metric ?sample_size ?random_state ?kwds ~x ~labels () =
                     Py.Module.get_function_with_keywords __wrap_namespace "silhouette_score"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("sample_size", Wrap_utils.Option.map sample_size Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("X", Some(x |> (function
| `Otherwise x -> Wrap_utils.id x
| `Arr x -> Np.Obj.to_pyobject x
))); ("labels", Some(labels |> Np.Obj.to_pyobject))]) (match kwds with None -> [] | Some x -> x))
                       |> Py.Float.to_float
let v_measure_score ?beta ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "v_measure_score"
     [||]
     (Wrap_utils.keyword_args [("beta", Wrap_utils.Option.map beta Py.Float.of_float); ("labels_true", Some(labels_true |> Np.Obj.to_pyobject)); ("labels_pred", Some(labels_pred |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float

end
module Pairwise = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.metrics.pairwise"

let get_py name = Py.Module.get __wrap_namespace name
module Csr_matrix = struct
type tag = [`Csr_matrix]
type t = [`ArrayLike | `Csr_matrix | `IndexMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_index x = (x :> [`IndexMixin] Obj.t)
let create ?shape ?dtype ?copy ~arg1 () =
   Py.Module.get_function_with_keywords __wrap_namespace "csr_matrix"
     [||]
     (Wrap_utils.keyword_args [("shape", Wrap_utils.Option.map shape (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("dtype", dtype); ("copy", copy); ("arg1", Some(arg1 ))])
     |> of_pyobject
let get_item ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key ))])

let iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []
     |> (fun py -> Py.Iter.to_seq py |> Seq.map Dict.of_pyobject)
let __setitem__ ~key ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key )); ("x", Some(x ))])

let arcsin self =
   Py.Module.get_function_with_keywords (to_pyobject self) "arcsin"
     [||]
     []

let arcsinh self =
   Py.Module.get_function_with_keywords (to_pyobject self) "arcsinh"
     [||]
     []

let arctan self =
   Py.Module.get_function_with_keywords (to_pyobject self) "arctan"
     [||]
     []

let arctanh self =
   Py.Module.get_function_with_keywords (to_pyobject self) "arctanh"
     [||]
     []

                  let argmax ?axis ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "argmax"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

                  let argmin ?axis ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "argmin"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

                  let asformat ?copy ~format self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "asformat"
                       [||]
                       (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("format", Some(format |> (function
| `S x -> Py.String.of_string x
| `None -> Py.none
)))])

let asfptype self =
   Py.Module.get_function_with_keywords (to_pyobject self) "asfptype"
     [||]
     []

                  let astype ?casting ?copy ~dtype self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "astype"
                       [||]
                       (Wrap_utils.keyword_args [("casting", Wrap_utils.Option.map casting (function
| `No -> Py.String.of_string "no"
| `Equiv -> Py.String.of_string "equiv"
| `Safe -> Py.String.of_string "safe"
| `Same_kind -> Py.String.of_string "same_kind"
| `Unsafe -> Py.String.of_string "unsafe"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("dtype", Some(dtype |> (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)))])

let ceil self =
   Py.Module.get_function_with_keywords (to_pyobject self) "ceil"
     [||]
     []

let check_format ?full_check self =
   Py.Module.get_function_with_keywords (to_pyobject self) "check_format"
     [||]
     (Wrap_utils.keyword_args [("full_check", Wrap_utils.Option.map full_check Py.Bool.of_bool)])

let conj ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conj"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let conjugate ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conjugate"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "copy"
     [||]
     []

let count_nonzero self =
   Py.Module.get_function_with_keywords (to_pyobject self) "count_nonzero"
     [||]
     []

let deg2rad self =
   Py.Module.get_function_with_keywords (to_pyobject self) "deg2rad"
     [||]
     []

let diagonal ?k self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diagonal"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int)])

let dot ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let eliminate_zeros self =
   Py.Module.get_function_with_keywords (to_pyobject self) "eliminate_zeros"
     [||]
     []

let expm1 self =
   Py.Module.get_function_with_keywords (to_pyobject self) "expm1"
     [||]
     []

let floor self =
   Py.Module.get_function_with_keywords (to_pyobject self) "floor"
     [||]
     []

let getH self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getH"
     [||]
     []

let get_shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_shape"
     [||]
     []

let getcol ~i self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getcol"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i ))])

let getformat self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getformat"
     [||]
     []

let getmaxprint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getmaxprint"
     [||]
     []

                  let getnnz ?axis self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "getnnz"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
))])

let getrow ~i self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getrow"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i ))])

let log1p self =
   Py.Module.get_function_with_keywords (to_pyobject self) "log1p"
     [||]
     []

                  let max ?axis ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "max"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

let maximum ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "maximum"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

                  let mean ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "mean"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let min ?axis ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "min"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("out", out)])

let minimum ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "minimum"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let multiply ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "multiply"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let nonzero self =
   Py.Module.get_function_with_keywords (to_pyobject self) "nonzero"
     [||]
     []

let power ?dtype ~n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "power"
     [||]
     (Wrap_utils.keyword_args [("dtype", dtype); ("n", Some(n ))])

let prune self =
   Py.Module.get_function_with_keywords (to_pyobject self) "prune"
     [||]
     []

let rad2deg self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rad2deg"
     [||]
     []

let reshape ?kwargs args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "reshape"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t))
let resize shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "resize"
     (Array.of_list @@ List.concat [(List.map Py.Int.of_int shape)])
     []

let rint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rint"
     [||]
     []

let set_shape ~shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_shape"
     [||]
     (Wrap_utils.keyword_args [("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])

let setdiag ?k ~values self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setdiag"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("values", Some(values |> Np.Obj.to_pyobject))])

let sign self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sign"
     [||]
     []

let sin self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sin"
     [||]
     []

let sinh self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sinh"
     [||]
     []

let sort_indices self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sort_indices"
     [||]
     []

let sorted_indices self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sorted_indices"
     [||]
     []

let sqrt self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sqrt"
     [||]
     []

                  let sum ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "sum"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let sum_duplicates self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sum_duplicates"
     [||]
     []

let tan self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tan"
     [||]
     []

let tanh self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tanh"
     [||]
     []

                  let toarray ?order ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "toarray"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let tobsr ?blocksize ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tobsr"
     [||]
     (Wrap_utils.keyword_args [("blocksize", blocksize); ("copy", copy)])

let tocoo ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocoo"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tocsc ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocsc"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tocsr ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tocsr"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

                  let todense ?order ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "todense"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let todia ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "todia"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let todok ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "todok"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tolil ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "tolil"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let transpose ?axes ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transpose"
     [||]
     (Wrap_utils.keyword_args [("axes", axes); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let trunc self =
   Py.Module.get_function_with_keywords (to_pyobject self) "trunc"
     [||]
     []


let dtype_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "dtype" with
  | None -> failwith "attribute dtype not found"
  | Some x -> if Py.is_none x then None else Some (Np.Dtype.of_pyobject x)

let dtype self = match dtype_opt self with
  | None -> raise Not_found
  | Some x -> x

let shape_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "shape" with
  | None -> failwith "attribute shape not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> Py.List.to_list_map (Py.Int.to_int) py) x)

let shape self = match shape_opt self with
  | None -> raise Not_found
  | Some x -> x

let ndim_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "ndim" with
  | None -> failwith "attribute ndim not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let ndim self = match ndim_opt self with
  | None -> raise Not_found
  | Some x -> x

let nnz_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "nnz" with
  | None -> failwith "attribute nnz not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let nnz self = match nnz_opt self with
  | None -> raise Not_found
  | Some x -> x

let data_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "data" with
  | None -> failwith "attribute data not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let data self = match data_opt self with
  | None -> raise Not_found
  | Some x -> x

let indices_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "indices" with
  | None -> failwith "attribute indices not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let indices self = match indices_opt self with
  | None -> raise Not_found
  | Some x -> x

let indptr_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "indptr" with
  | None -> failwith "attribute indptr not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let indptr self = match indptr_opt self with
  | None -> raise Not_found
  | Some x -> x

let has_sorted_indices_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "has_sorted_indices" with
  | None -> failwith "attribute has_sorted_indices not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let has_sorted_indices self = match has_sorted_indices_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Partial = struct
type tag = [`Partial]
type t = [`Object | `Partial] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?keywords ~func args =
   Py.Module.get_function_with_keywords __wrap_namespace "partial"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("func", Some(func ))]) (match keywords with None -> [] | Some x -> x))
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let additive_chi2_kernel ?y ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "additive_chi2_kernel"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let check_array ?accept_sparse ?accept_large_sparse ?dtype ?order ?copy ?force_all_finite ?ensure_2d ?allow_nd ?ensure_min_samples ?ensure_min_features ?warn_on_dtype ?estimator ~array () =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_array"
                       [||]
                       (Wrap_utils.keyword_args [("accept_sparse", Wrap_utils.Option.map accept_sparse (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `S x -> Py.String.of_string x
| `Bool x -> Py.Bool.of_bool x
)); ("accept_large_sparse", Wrap_utils.Option.map accept_large_sparse Py.Bool.of_bool); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
| `Dtypes x -> (fun ml -> Py.List.of_list_map Np.Dtype.to_pyobject ml) x
| `None -> Py.none
)); ("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("force_all_finite", Wrap_utils.Option.map force_all_finite (function
| `Allow_nan -> Py.String.of_string "allow-nan"
| `Bool x -> Py.Bool.of_bool x
)); ("ensure_2d", Wrap_utils.Option.map ensure_2d Py.Bool.of_bool); ("allow_nd", Wrap_utils.Option.map allow_nd Py.Bool.of_bool); ("ensure_min_samples", Wrap_utils.Option.map ensure_min_samples Py.Int.of_int); ("ensure_min_features", Wrap_utils.Option.map ensure_min_features Py.Int.of_int); ("warn_on_dtype", Wrap_utils.Option.map warn_on_dtype Py.Bool.of_bool); ("estimator", Wrap_utils.Option.map estimator Np.Obj.to_pyobject); ("array", Some(array ))])

let check_non_negative ~x ~whom () =
   Py.Module.get_function_with_keywords __wrap_namespace "check_non_negative"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("whom", Some(whom |> Py.String.of_string))])

let check_paired_arrays ~x ~y () =
   Py.Module.get_function_with_keywords __wrap_namespace "check_paired_arrays"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("Y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let check_pairwise_arrays ?precomputed ?dtype ?accept_sparse ?force_all_finite ?copy ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_pairwise_arrays"
                       [||]
                       (Wrap_utils.keyword_args [("precomputed", Wrap_utils.Option.map precomputed Py.Bool.of_bool); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
| `Dtypes x -> (fun ml -> Py.List.of_list_map Np.Dtype.to_pyobject ml) x
)); ("accept_sparse", Wrap_utils.Option.map accept_sparse (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `S x -> Py.String.of_string x
| `Bool x -> Py.Bool.of_bool x
)); ("force_all_finite", Wrap_utils.Option.map force_all_finite (function
| `Allow_nan -> Py.String.of_string "allow-nan"
| `Bool x -> Py.Bool.of_bool x
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject)); ("Y", Some(y |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
let chi2_kernel ?y ?gamma ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "chi2_kernel"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("gamma", Wrap_utils.Option.map gamma Py.Float.of_float); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let cosine_distances ?y ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "cosine_distances"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])

let cosine_similarity ?y ?dense_output ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "cosine_similarity"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("dense_output", Wrap_utils.Option.map dense_output Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])

let delayed ?check_pickle ~function_ () =
   Py.Module.get_function_with_keywords __wrap_namespace "delayed"
     [||]
     (Wrap_utils.keyword_args [("check_pickle", check_pickle); ("function", Some(function_ ))])

let distance_metrics () =
   Py.Module.get_function_with_keywords __wrap_namespace "distance_metrics"
     [||]
     []

let effective_n_jobs ?n_jobs () =
   Py.Module.get_function_with_keywords __wrap_namespace "effective_n_jobs"
     [||]
     (Wrap_utils.keyword_args [("n_jobs", n_jobs)])

let euclidean_distances ?y ?y_norm_squared ?squared ?x_norm_squared ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "euclidean_distances"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("Y_norm_squared", Wrap_utils.Option.map y_norm_squared Np.Obj.to_pyobject); ("squared", Wrap_utils.Option.map squared Py.Bool.of_bool); ("X_norm_squared", Wrap_utils.Option.map x_norm_squared Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let gen_batches ?min_batch_size ~n ~batch_size () =
   Py.Module.get_function_with_keywords __wrap_namespace "gen_batches"
     [||]
     (Wrap_utils.keyword_args [("min_batch_size", Wrap_utils.Option.map min_batch_size Py.Int.of_int); ("n", Some(n |> Py.Int.of_int)); ("batch_size", Some(batch_size ))])

let gen_even_slices ?n_samples ~n ~n_packs () =
   Py.Module.get_function_with_keywords __wrap_namespace "gen_even_slices"
     [||]
     (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("n", Some(n |> Py.Int.of_int)); ("n_packs", Some(n_packs ))])

                  let get_chunk_n_rows ?max_n_rows ?working_memory ~row_bytes () =
                     Py.Module.get_function_with_keywords __wrap_namespace "get_chunk_n_rows"
                       [||]
                       (Wrap_utils.keyword_args [("max_n_rows", Wrap_utils.Option.map max_n_rows Py.Int.of_int); ("working_memory", Wrap_utils.Option.map working_memory (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("row_bytes", Some(row_bytes |> Py.Int.of_int))])

let haversine_distances ?y ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "haversine_distances"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let is_scalar_nan x =
   Py.Module.get_function_with_keywords __wrap_namespace "is_scalar_nan"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let issparse x =
   Py.Module.get_function_with_keywords __wrap_namespace "issparse"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let kernel_metrics () =
   Py.Module.get_function_with_keywords __wrap_namespace "kernel_metrics"
     [||]
     []

let laplacian_kernel ?y ?gamma ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "laplacian_kernel"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("gamma", Wrap_utils.Option.map gamma Py.Float.of_float); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let linear_kernel ?y ?dense_output ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "linear_kernel"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("dense_output", Wrap_utils.Option.map dense_output Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])

let manhattan_distances ?y ?sum_over_features ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "manhattan_distances"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("sum_over_features", Wrap_utils.Option.map sum_over_features Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let nan_euclidean_distances ?y ?squared ?missing_values ?copy ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "nan_euclidean_distances"
                       [||]
                       (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("squared", Wrap_utils.Option.map squared Py.Bool.of_bool); ("missing_values", Wrap_utils.Option.map missing_values (function
| `Np_nan x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let normalize ?norm ?axis ?copy ?return_norm ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "normalize"
                       [||]
                       (Wrap_utils.keyword_args [("norm", Wrap_utils.Option.map norm (function
| `L1 -> Py.String.of_string "l1"
| `L2 -> Py.String.of_string "l2"
| `Max -> Py.String.of_string "max"
)); ("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("return_norm", Wrap_utils.Option.map return_norm Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
let paired_cosine_distances ~x ~y () =
   Py.Module.get_function_with_keywords __wrap_namespace "paired_cosine_distances"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("Y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let paired_distances ?metric ?kwds ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "paired_distances"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("X", Some(x |> Np.Obj.to_pyobject)); ("Y", Some(y |> Np.Obj.to_pyobject))]) (match kwds with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let paired_euclidean_distances ~x ~y () =
   Py.Module.get_function_with_keywords __wrap_namespace "paired_euclidean_distances"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("Y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let paired_manhattan_distances ~x ~y () =
   Py.Module.get_function_with_keywords __wrap_namespace "paired_manhattan_distances"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("Y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let pairwise_distances ?y ?metric ?n_jobs ?force_all_finite ?kwds ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "pairwise_distances"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("force_all_finite", Wrap_utils.Option.map force_all_finite (function
| `Allow_nan -> Py.String.of_string "allow-nan"
| `Bool x -> Py.Bool.of_bool x
)); ("X", Some(x |> (function
| `Otherwise x -> Wrap_utils.id x
| `Arr x -> Np.Obj.to_pyobject x
)))]) (match kwds with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let pairwise_distances_argmin ?axis ?metric ?metric_kwargs ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "pairwise_distances_argmin"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("metric_kwargs", Wrap_utils.Option.map metric_kwargs Dict.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("Y", Some(y |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let pairwise_distances_argmin_min ?axis ?metric ?metric_kwargs ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "pairwise_distances_argmin_min"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("metric_kwargs", Wrap_utils.Option.map metric_kwargs Dict.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("Y", Some(y |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let pairwise_distances_chunked ?y ?reduce_func ?metric ?n_jobs ?working_memory ?kwds ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "pairwise_distances_chunked"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("reduce_func", reduce_func); ("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("working_memory", Wrap_utils.Option.map working_memory Py.Int.of_int); ("X", Some(x |> Np.Obj.to_pyobject))]) (match kwds with None -> [] | Some x -> x))
                       |> (fun py -> Py.Iter.to_seq py |> Seq.map (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)))
                  let pairwise_kernels ?y ?metric ?filter_params ?n_jobs ?kwds ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "pairwise_kernels"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("filter_params", Wrap_utils.Option.map filter_params Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("X", Some(x |> (function
| `Otherwise x -> Wrap_utils.id x
| `Arr x -> Np.Obj.to_pyobject x
)))]) (match kwds with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let polynomial_kernel ?y ?degree ?gamma ?coef0 ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "polynomial_kernel"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("degree", Wrap_utils.Option.map degree Py.Int.of_int); ("gamma", Wrap_utils.Option.map gamma Py.Float.of_float); ("coef0", Wrap_utils.Option.map coef0 Py.Float.of_float); ("X", Some(x |> Np.Obj.to_pyobject))])

let rbf_kernel ?y ?gamma ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "rbf_kernel"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("gamma", Wrap_utils.Option.map gamma Py.Float.of_float); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let row_norms ?squared ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "row_norms"
     [||]
     (Wrap_utils.keyword_args [("squared", Wrap_utils.Option.map squared Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])

let safe_sparse_dot ?dense_output ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "safe_sparse_dot"
     [||]
     (Wrap_utils.keyword_args [("dense_output", dense_output); ("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let sigmoid_kernel ?y ?gamma ?coef0 ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "sigmoid_kernel"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("gamma", Wrap_utils.Option.map gamma Py.Float.of_float); ("coef0", Wrap_utils.Option.map coef0 Py.Float.of_float); ("X", Some(x |> Np.Obj.to_pyobject))])


end
let accuracy_score ?normalize ?sample_weight ~y_true ~y_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "accuracy_score"
     [||]
     (Wrap_utils.keyword_args [("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_pred", Some(y_pred |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let adjusted_mutual_info_score ?average_method ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "adjusted_mutual_info_score"
     [||]
     (Wrap_utils.keyword_args [("average_method", Wrap_utils.Option.map average_method Py.String.of_string); ("labels_true", Some(labels_true |> Np.Obj.to_pyobject)); ("labels_pred", Some(labels_pred |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let adjusted_rand_score ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "adjusted_rand_score"
     [||]
     (Wrap_utils.keyword_args [("labels_true", Some(labels_true |> Np.Obj.to_pyobject)); ("labels_pred", Some(labels_pred |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let auc ~x ~y () =
   Py.Module.get_function_with_keywords __wrap_namespace "auc"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
                  let average_precision_score ?average ?pos_label ?sample_weight ~y_true ~y_score () =
                     Py.Module.get_function_with_keywords __wrap_namespace "average_precision_score"
                       [||]
                       (Wrap_utils.keyword_args [("average", Wrap_utils.Option.map average (function
| `Micro -> Py.String.of_string "micro"
| `Macro -> Py.String.of_string "macro"
| `Samples -> Py.String.of_string "samples"
| `Weighted -> Py.String.of_string "weighted"
| `None -> Py.none
)); ("pos_label", Wrap_utils.Option.map pos_label (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_score", Some(y_score |> Np.Obj.to_pyobject))])
                       |> Py.Float.to_float
let balanced_accuracy_score ?sample_weight ?adjusted ~y_true ~y_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "balanced_accuracy_score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("adjusted", Wrap_utils.Option.map adjusted Py.Bool.of_bool); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_pred", Some(y_pred |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
                  let brier_score_loss ?sample_weight ?pos_label ~y_true ~y_prob () =
                     Py.Module.get_function_with_keywords __wrap_namespace "brier_score_loss"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("pos_label", Wrap_utils.Option.map pos_label (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_prob", Some(y_prob |> Np.Obj.to_pyobject))])
                       |> Py.Float.to_float
let calinski_harabasz_score ~x ~labels () =
   Py.Module.get_function_with_keywords __wrap_namespace "calinski_harabasz_score"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("labels", Some(labels |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let calinski_harabaz_score ~x ~labels () =
   Py.Module.get_function_with_keywords __wrap_namespace "calinski_harabaz_score"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("labels", Some(labels ))])

                  let check_scoring ?scoring ?allow_none ~estimator () =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_scoring"
                       [||]
                       (Wrap_utils.keyword_args [("scoring", Wrap_utils.Option.map scoring (function
| `Callable x -> Wrap_utils.id x
| `Score x -> (function
| `Explained_variance -> Py.String.of_string "explained_variance"
| `R2 -> Py.String.of_string "r2"
| `Max_error -> Py.String.of_string "max_error"
| `Neg_median_absolute_error -> Py.String.of_string "neg_median_absolute_error"
| `Neg_mean_absolute_error -> Py.String.of_string "neg_mean_absolute_error"
| `Neg_mean_squared_error -> Py.String.of_string "neg_mean_squared_error"
| `Neg_mean_squared_log_error -> Py.String.of_string "neg_mean_squared_log_error"
| `Neg_root_mean_squared_error -> Py.String.of_string "neg_root_mean_squared_error"
| `Neg_mean_poisson_deviance -> Py.String.of_string "neg_mean_poisson_deviance"
| `Neg_mean_gamma_deviance -> Py.String.of_string "neg_mean_gamma_deviance"
| `Accuracy -> Py.String.of_string "accuracy"
| `Roc_auc -> Py.String.of_string "roc_auc"
| `Roc_auc_ovr -> Py.String.of_string "roc_auc_ovr"
| `Roc_auc_ovo -> Py.String.of_string "roc_auc_ovo"
| `Roc_auc_ovr_weighted -> Py.String.of_string "roc_auc_ovr_weighted"
| `Roc_auc_ovo_weighted -> Py.String.of_string "roc_auc_ovo_weighted"
| `Balanced_accuracy -> Py.String.of_string "balanced_accuracy"
| `Average_precision -> Py.String.of_string "average_precision"
| `Neg_log_loss -> Py.String.of_string "neg_log_loss"
| `Neg_brier_score -> Py.String.of_string "neg_brier_score"
| `Adjusted_rand_score -> Py.String.of_string "adjusted_rand_score"
| `Homogeneity_score -> Py.String.of_string "homogeneity_score"
| `Completeness_score -> Py.String.of_string "completeness_score"
| `V_measure_score -> Py.String.of_string "v_measure_score"
| `Mutual_info_score -> Py.String.of_string "mutual_info_score"
| `Adjusted_mutual_info_score -> Py.String.of_string "adjusted_mutual_info_score"
| `Normalized_mutual_info_score -> Py.String.of_string "normalized_mutual_info_score"
| `Fowlkes_mallows_score -> Py.String.of_string "fowlkes_mallows_score"
| `Precision -> Py.String.of_string "precision"
| `Precision_macro -> Py.String.of_string "precision_macro"
| `Precision_micro -> Py.String.of_string "precision_micro"
| `Precision_samples -> Py.String.of_string "precision_samples"
| `Precision_weighted -> Py.String.of_string "precision_weighted"
| `Recall -> Py.String.of_string "recall"
| `Recall_macro -> Py.String.of_string "recall_macro"
| `Recall_micro -> Py.String.of_string "recall_micro"
| `Recall_samples -> Py.String.of_string "recall_samples"
| `Recall_weighted -> Py.String.of_string "recall_weighted"
| `F1 -> Py.String.of_string "f1"
| `F1_macro -> Py.String.of_string "f1_macro"
| `F1_micro -> Py.String.of_string "f1_micro"
| `F1_samples -> Py.String.of_string "f1_samples"
| `F1_weighted -> Py.String.of_string "f1_weighted"
| `Jaccard -> Py.String.of_string "jaccard"
| `Jaccard_macro -> Py.String.of_string "jaccard_macro"
| `Jaccard_micro -> Py.String.of_string "jaccard_micro"
| `Jaccard_samples -> Py.String.of_string "jaccard_samples"
| `Jaccard_weighted -> Py.String.of_string "jaccard_weighted"
) x
)); ("allow_none", Wrap_utils.Option.map allow_none Py.Bool.of_bool); ("estimator", Some(estimator |> Np.Obj.to_pyobject))])

                  let classification_report ?labels ?target_names ?sample_weight ?digits ?output_dict ?zero_division ~y_true ~y_pred () =
                     Py.Module.get_function_with_keywords __wrap_namespace "classification_report"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("target_names", Wrap_utils.Option.map target_names Np.Obj.to_pyobject); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("digits", Wrap_utils.Option.map digits Py.Int.of_int); ("output_dict", Wrap_utils.Option.map output_dict Py.Bool.of_bool); ("zero_division", Wrap_utils.Option.map zero_division (function
| `Warn -> Py.String.of_string "warn"
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
)); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_pred", Some(y_pred |> Np.Obj.to_pyobject))])
                       |> (fun x -> if Py.String.check x then `S (Py.String.to_string x) else if Py.Dict.check x then `Dict ((fun py -> Py.Dict.fold (fun kpy vpy acc -> ((Py.String.to_string kpy), object
      method precision = Py.Dict.get_item_string vpy "precision" |> Wrap_utils.Option.get |> Py.Float.to_float
      method recall = Py.Dict.get_item_string vpy "recall" |> Wrap_utils.Option.get |> Py.Float.to_float
      method f1_score = Py.Dict.get_item_string vpy "f1-score" |> Wrap_utils.Option.get |> Py.Float.to_float
      method support = Py.Dict.get_item_string vpy "support" |> Wrap_utils.Option.get |> Py.Float.to_float
    end)::acc) py [])
     x) else failwith (Printf.sprintf "Sklearn: could not identify type from Python value %s (%s)"
                                                  (Py.Object.to_string x) (Wrap_utils.type_string x)))
let cohen_kappa_score ?labels ?weights ?sample_weight ~y1 ~y2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "cohen_kappa_score"
     [||]
     (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("weights", Wrap_utils.Option.map weights Py.String.of_string); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("y1", Some(y1 |> Np.Obj.to_pyobject)); ("y2", Some(y2 |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let completeness_score ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "completeness_score"
     [||]
     (Wrap_utils.keyword_args [("labels_true", Some(labels_true |> Np.Obj.to_pyobject)); ("labels_pred", Some(labels_pred |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
                  let confusion_matrix ?labels ?sample_weight ?normalize ~y_true ~y_pred () =
                     Py.Module.get_function_with_keywords __wrap_namespace "confusion_matrix"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("normalize", Wrap_utils.Option.map normalize (function
| `True -> Py.String.of_string "true"
| `Pred -> Py.String.of_string "pred"
| `All -> Py.String.of_string "all"
)); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_pred", Some(y_pred |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let consensus_score ?similarity ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "consensus_score"
                       [||]
                       (Wrap_utils.keyword_args [("similarity", Wrap_utils.Option.map similarity (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("a", Some(a )); ("b", Some(b ))])

let coverage_error ?sample_weight ~y_true ~y_score () =
   Py.Module.get_function_with_keywords __wrap_namespace "coverage_error"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_score", Some(y_score |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let davies_bouldin_score ~x ~labels () =
   Py.Module.get_function_with_keywords __wrap_namespace "davies_bouldin_score"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("labels", Some(labels |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let dcg_score ?k ?log_base ?sample_weight ?ignore_ties ~y_true ~y_score () =
   Py.Module.get_function_with_keywords __wrap_namespace "dcg_score"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("log_base", Wrap_utils.Option.map log_base Py.Float.of_float); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("ignore_ties", Wrap_utils.Option.map ignore_ties Py.Bool.of_bool); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_score", Some(y_score |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let euclidean_distances ?y ?y_norm_squared ?squared ?x_norm_squared ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "euclidean_distances"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("Y_norm_squared", Wrap_utils.Option.map y_norm_squared Np.Obj.to_pyobject); ("squared", Wrap_utils.Option.map squared Py.Bool.of_bool); ("X_norm_squared", Wrap_utils.Option.map x_norm_squared Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let explained_variance_score ?sample_weight ?multioutput ~y_true ~y_pred () =
                     Py.Module.get_function_with_keywords __wrap_namespace "explained_variance_score"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("multioutput", Wrap_utils.Option.map multioutput (function
| `Variance_weighted -> Py.String.of_string "variance_weighted"
| `Arr x -> Np.Obj.to_pyobject x
| `Raw_values -> Py.String.of_string "raw_values"
| `Uniform_average -> Py.String.of_string "uniform_average"
)); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_pred", Some(y_pred |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let f1_score ?labels ?pos_label ?average ?sample_weight ?zero_division ~y_true ~y_pred () =
                     Py.Module.get_function_with_keywords __wrap_namespace "f1_score"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("pos_label", Wrap_utils.Option.map pos_label (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("average", Wrap_utils.Option.map average (function
| `Micro -> Py.String.of_string "micro"
| `Binary -> Py.String.of_string "binary"
| `Samples -> Py.String.of_string "samples"
| `Weighted -> Py.String.of_string "weighted"
| `Macro -> Py.String.of_string "macro"
| `None -> Py.none
)); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("zero_division", Wrap_utils.Option.map zero_division (function
| `Warn -> Py.String.of_string "warn"
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
)); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_pred", Some(y_pred |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let fbeta_score ?labels ?pos_label ?average ?sample_weight ?zero_division ~y_true ~y_pred ~beta () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fbeta_score"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("pos_label", Wrap_utils.Option.map pos_label (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("average", Wrap_utils.Option.map average (function
| `Micro -> Py.String.of_string "micro"
| `Binary -> Py.String.of_string "binary"
| `Samples -> Py.String.of_string "samples"
| `Weighted -> Py.String.of_string "weighted"
| `Macro -> Py.String.of_string "macro"
| `None -> Py.none
)); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("zero_division", Wrap_utils.Option.map zero_division (function
| `Warn -> Py.String.of_string "warn"
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
)); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_pred", Some(y_pred |> Np.Obj.to_pyobject)); ("beta", Some(beta |> Py.Float.of_float))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let fowlkes_mallows_score ?sparse ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "fowlkes_mallows_score"
     [||]
     (Wrap_utils.keyword_args [("sparse", Wrap_utils.Option.map sparse Py.Bool.of_bool); ("labels_true", Some(labels_true |> Np.Obj.to_pyobject)); ("labels_pred", Some(labels_pred |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
                  let get_scorer scoring =
                     Py.Module.get_function_with_keywords __wrap_namespace "get_scorer"
                       [||]
                       (Wrap_utils.keyword_args [("scoring", Some(scoring |> (function
| `Callable x -> Wrap_utils.id x
| `Score x -> (function
| `Explained_variance -> Py.String.of_string "explained_variance"
| `R2 -> Py.String.of_string "r2"
| `Max_error -> Py.String.of_string "max_error"
| `Neg_median_absolute_error -> Py.String.of_string "neg_median_absolute_error"
| `Neg_mean_absolute_error -> Py.String.of_string "neg_mean_absolute_error"
| `Neg_mean_squared_error -> Py.String.of_string "neg_mean_squared_error"
| `Neg_mean_squared_log_error -> Py.String.of_string "neg_mean_squared_log_error"
| `Neg_root_mean_squared_error -> Py.String.of_string "neg_root_mean_squared_error"
| `Neg_mean_poisson_deviance -> Py.String.of_string "neg_mean_poisson_deviance"
| `Neg_mean_gamma_deviance -> Py.String.of_string "neg_mean_gamma_deviance"
| `Accuracy -> Py.String.of_string "accuracy"
| `Roc_auc -> Py.String.of_string "roc_auc"
| `Roc_auc_ovr -> Py.String.of_string "roc_auc_ovr"
| `Roc_auc_ovo -> Py.String.of_string "roc_auc_ovo"
| `Roc_auc_ovr_weighted -> Py.String.of_string "roc_auc_ovr_weighted"
| `Roc_auc_ovo_weighted -> Py.String.of_string "roc_auc_ovo_weighted"
| `Balanced_accuracy -> Py.String.of_string "balanced_accuracy"
| `Average_precision -> Py.String.of_string "average_precision"
| `Neg_log_loss -> Py.String.of_string "neg_log_loss"
| `Neg_brier_score -> Py.String.of_string "neg_brier_score"
| `Adjusted_rand_score -> Py.String.of_string "adjusted_rand_score"
| `Homogeneity_score -> Py.String.of_string "homogeneity_score"
| `Completeness_score -> Py.String.of_string "completeness_score"
| `V_measure_score -> Py.String.of_string "v_measure_score"
| `Mutual_info_score -> Py.String.of_string "mutual_info_score"
| `Adjusted_mutual_info_score -> Py.String.of_string "adjusted_mutual_info_score"
| `Normalized_mutual_info_score -> Py.String.of_string "normalized_mutual_info_score"
| `Fowlkes_mallows_score -> Py.String.of_string "fowlkes_mallows_score"
| `Precision -> Py.String.of_string "precision"
| `Precision_macro -> Py.String.of_string "precision_macro"
| `Precision_micro -> Py.String.of_string "precision_micro"
| `Precision_samples -> Py.String.of_string "precision_samples"
| `Precision_weighted -> Py.String.of_string "precision_weighted"
| `Recall -> Py.String.of_string "recall"
| `Recall_macro -> Py.String.of_string "recall_macro"
| `Recall_micro -> Py.String.of_string "recall_micro"
| `Recall_samples -> Py.String.of_string "recall_samples"
| `Recall_weighted -> Py.String.of_string "recall_weighted"
| `F1 -> Py.String.of_string "f1"
| `F1_macro -> Py.String.of_string "f1_macro"
| `F1_micro -> Py.String.of_string "f1_micro"
| `F1_samples -> Py.String.of_string "f1_samples"
| `F1_weighted -> Py.String.of_string "f1_weighted"
| `Jaccard -> Py.String.of_string "jaccard"
| `Jaccard_macro -> Py.String.of_string "jaccard_macro"
| `Jaccard_micro -> Py.String.of_string "jaccard_micro"
| `Jaccard_samples -> Py.String.of_string "jaccard_samples"
| `Jaccard_weighted -> Py.String.of_string "jaccard_weighted"
) x
)))])

let hamming_loss ?labels ?sample_weight ~y_true ~y_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "hamming_loss"
     [||]
     (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_pred", Some(y_pred |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let hinge_loss ?labels ?sample_weight ~y_true ~pred_decision () =
   Py.Module.get_function_with_keywords __wrap_namespace "hinge_loss"
     [||]
     (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("pred_decision", Some(pred_decision |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let homogeneity_completeness_v_measure ?beta ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "homogeneity_completeness_v_measure"
     [||]
     (Wrap_utils.keyword_args [("beta", Wrap_utils.Option.map beta Py.Float.of_float); ("labels_true", Some(labels_true |> Np.Obj.to_pyobject)); ("labels_pred", Some(labels_pred |> Np.Obj.to_pyobject))])
     |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let homogeneity_score ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "homogeneity_score"
     [||]
     (Wrap_utils.keyword_args [("labels_true", Some(labels_true |> Np.Obj.to_pyobject)); ("labels_pred", Some(labels_pred |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
                  let jaccard_score ?labels ?pos_label ?average ?sample_weight ~y_true ~y_pred () =
                     Py.Module.get_function_with_keywords __wrap_namespace "jaccard_score"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("pos_label", Wrap_utils.Option.map pos_label (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("average", Wrap_utils.Option.map average (function
| `Micro -> Py.String.of_string "micro"
| `Binary -> Py.String.of_string "binary"
| `Samples -> Py.String.of_string "samples"
| `Weighted -> Py.String.of_string "weighted"
| `Macro -> Py.String.of_string "macro"
| `None -> Py.none
)); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_pred", Some(y_pred |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let jaccard_similarity_score ?normalize ?sample_weight ~y_true ~y_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "jaccard_similarity_score"
     [||]
     (Wrap_utils.keyword_args [("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_pred", Some(y_pred |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let label_ranking_average_precision_score ?sample_weight ~y_true ~y_score () =
   Py.Module.get_function_with_keywords __wrap_namespace "label_ranking_average_precision_score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_score", Some(y_score |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let label_ranking_loss ?sample_weight ~y_true ~y_score () =
   Py.Module.get_function_with_keywords __wrap_namespace "label_ranking_loss"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_score", Some(y_score |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let log_loss ?eps ?normalize ?sample_weight ?labels ~y_true ~y_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "log_loss"
     [||]
     (Wrap_utils.keyword_args [("eps", Wrap_utils.Option.map eps Py.Float.of_float); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_pred", Some(y_pred |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let make_scorer ?greater_is_better ?needs_proba ?needs_threshold ?kwargs ~score_func () =
   Py.Module.get_function_with_keywords __wrap_namespace "make_scorer"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("greater_is_better", Wrap_utils.Option.map greater_is_better Py.Bool.of_bool); ("needs_proba", Wrap_utils.Option.map needs_proba Py.Bool.of_bool); ("needs_threshold", Wrap_utils.Option.map needs_threshold Py.Bool.of_bool); ("score_func", Some(score_func ))]) (match kwargs with None -> [] | Some x -> x))

let matthews_corrcoef ?sample_weight ~y_true ~y_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "matthews_corrcoef"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_pred", Some(y_pred |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let max_error ~y_true ~y_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "max_error"
     [||]
     (Wrap_utils.keyword_args [("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_pred", Some(y_pred |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
                  let mean_absolute_error ?sample_weight ?multioutput ~y_true ~y_pred () =
                     Py.Module.get_function_with_keywords __wrap_namespace "mean_absolute_error"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("multioutput", Wrap_utils.Option.map multioutput (function
| `Raw_values -> Py.String.of_string "raw_values"
| `Uniform_average -> Py.String.of_string "uniform_average"
| `Arr x -> Np.Obj.to_pyobject x
)); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_pred", Some(y_pred |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let mean_gamma_deviance ?sample_weight ~y_true ~y_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "mean_gamma_deviance"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_pred", Some(y_pred |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let mean_poisson_deviance ?sample_weight ~y_true ~y_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "mean_poisson_deviance"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_pred", Some(y_pred |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
                  let mean_squared_error ?sample_weight ?multioutput ?squared ~y_true ~y_pred () =
                     Py.Module.get_function_with_keywords __wrap_namespace "mean_squared_error"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("multioutput", Wrap_utils.Option.map multioutput (function
| `Raw_values -> Py.String.of_string "raw_values"
| `Uniform_average -> Py.String.of_string "uniform_average"
| `Arr x -> Np.Obj.to_pyobject x
)); ("squared", Wrap_utils.Option.map squared Py.Bool.of_bool); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_pred", Some(y_pred |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let mean_squared_log_error ?sample_weight ?multioutput ~y_true ~y_pred () =
                     Py.Module.get_function_with_keywords __wrap_namespace "mean_squared_log_error"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("multioutput", Wrap_utils.Option.map multioutput (function
| `Raw_values -> Py.String.of_string "raw_values"
| `Uniform_average -> Py.String.of_string "uniform_average"
| `Arr x -> Np.Obj.to_pyobject x
)); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_pred", Some(y_pred |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let mean_tweedie_deviance ?sample_weight ?power ~y_true ~y_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "mean_tweedie_deviance"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("power", Wrap_utils.Option.map power Py.Float.of_float); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_pred", Some(y_pred |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
                  let median_absolute_error ?multioutput ~y_true ~y_pred () =
                     Py.Module.get_function_with_keywords __wrap_namespace "median_absolute_error"
                       [||]
                       (Wrap_utils.keyword_args [("multioutput", Wrap_utils.Option.map multioutput (function
| `Arr x -> Np.Obj.to_pyobject x
| `Raw_values -> Py.String.of_string "raw_values"
| `Uniform_average -> Py.String.of_string "uniform_average"
)); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_pred", Some(y_pred |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let multilabel_confusion_matrix ?sample_weight ?labels ?samplewise ~y_true ~y_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "multilabel_confusion_matrix"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("samplewise", Wrap_utils.Option.map samplewise Py.Bool.of_bool); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_pred", Some(y_pred |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let mutual_info_score ?contingency ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "mutual_info_score"
     [||]
     (Wrap_utils.keyword_args [("contingency", Wrap_utils.Option.map contingency Np.Obj.to_pyobject); ("labels_true", Some(labels_true |> Np.Obj.to_pyobject)); ("labels_pred", Some(labels_pred |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
                  let nan_euclidean_distances ?y ?squared ?missing_values ?copy ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "nan_euclidean_distances"
                       [||]
                       (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("squared", Wrap_utils.Option.map squared Py.Bool.of_bool); ("missing_values", Wrap_utils.Option.map missing_values (function
| `Np_nan x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let ndcg_score ?k ?sample_weight ?ignore_ties ~y_true ~y_score () =
   Py.Module.get_function_with_keywords __wrap_namespace "ndcg_score"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("ignore_ties", Wrap_utils.Option.map ignore_ties Py.Bool.of_bool); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_score", Some(y_score |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let normalized_mutual_info_score ?average_method ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "normalized_mutual_info_score"
     [||]
     (Wrap_utils.keyword_args [("average_method", Wrap_utils.Option.map average_method Py.String.of_string); ("labels_true", Some(labels_true |> Np.Obj.to_pyobject)); ("labels_pred", Some(labels_pred |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
                  let pairwise_distances ?y ?metric ?n_jobs ?force_all_finite ?kwds ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "pairwise_distances"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("force_all_finite", Wrap_utils.Option.map force_all_finite (function
| `Allow_nan -> Py.String.of_string "allow-nan"
| `Bool x -> Py.Bool.of_bool x
)); ("X", Some(x |> (function
| `Otherwise x -> Wrap_utils.id x
| `Arr x -> Np.Obj.to_pyobject x
)))]) (match kwds with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let pairwise_distances_argmin ?axis ?metric ?metric_kwargs ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "pairwise_distances_argmin"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("metric_kwargs", Wrap_utils.Option.map metric_kwargs Dict.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("Y", Some(y |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let pairwise_distances_argmin_min ?axis ?metric ?metric_kwargs ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "pairwise_distances_argmin_min"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("metric_kwargs", Wrap_utils.Option.map metric_kwargs Dict.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("Y", Some(y |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let pairwise_distances_chunked ?y ?reduce_func ?metric ?n_jobs ?working_memory ?kwds ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "pairwise_distances_chunked"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("reduce_func", reduce_func); ("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("working_memory", Wrap_utils.Option.map working_memory Py.Int.of_int); ("X", Some(x |> Np.Obj.to_pyobject))]) (match kwds with None -> [] | Some x -> x))
                       |> (fun py -> Py.Iter.to_seq py |> Seq.map (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)))
                  let pairwise_kernels ?y ?metric ?filter_params ?n_jobs ?kwds ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "pairwise_kernels"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("filter_params", Wrap_utils.Option.map filter_params Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("X", Some(x |> (function
| `Otherwise x -> Wrap_utils.id x
| `Arr x -> Np.Obj.to_pyobject x
)))]) (match kwds with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let plot_confusion_matrix ?labels ?sample_weight ?normalize ?display_labels ?include_values ?xticks_rotation ?values_format ?cmap ?ax ~estimator ~x ~y_true () =
                     Py.Module.get_function_with_keywords __wrap_namespace "plot_confusion_matrix"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("normalize", Wrap_utils.Option.map normalize (function
| `True -> Py.String.of_string "true"
| `Pred -> Py.String.of_string "pred"
| `All -> Py.String.of_string "all"
)); ("display_labels", Wrap_utils.Option.map display_labels Np.Obj.to_pyobject); ("include_values", Wrap_utils.Option.map include_values Py.Bool.of_bool); ("xticks_rotation", Wrap_utils.Option.map xticks_rotation (function
| `Horizontal -> Py.String.of_string "horizontal"
| `F x -> Py.Float.of_float x
| `Vertical -> Py.String.of_string "vertical"
)); ("values_format", Wrap_utils.Option.map values_format Py.String.of_string); ("cmap", Wrap_utils.Option.map cmap (function
| `S x -> Py.String.of_string x
| `Matplotlib_Colormap x -> Wrap_utils.id x
)); ("ax", ax); ("estimator", Some(estimator |> Np.Obj.to_pyobject)); ("X", Some(x |> Np.Obj.to_pyobject)); ("y_true", Some(y_true ))])

                  let plot_precision_recall_curve ?sample_weight ?response_method ?name ?ax ?kwargs ~estimator ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "plot_precision_recall_curve"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("response_method", Wrap_utils.Option.map response_method (function
| `Predict_proba -> Py.String.of_string "predict_proba"
| `Decision_function -> Py.String.of_string "decision_function"
| `Auto -> Py.String.of_string "auto"
)); ("name", Wrap_utils.Option.map name Py.String.of_string); ("ax", ax); ("estimator", Some(estimator |> Np.Obj.to_pyobject)); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))]) (match kwargs with None -> [] | Some x -> x))

                  let plot_roc_curve ?sample_weight ?drop_intermediate ?response_method ?name ?ax ?kwargs ~estimator ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "plot_roc_curve"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("drop_intermediate", Wrap_utils.Option.map drop_intermediate Py.Bool.of_bool); ("response_method", Wrap_utils.Option.map response_method (function
| `Predict_proba -> Py.String.of_string "predict_proba"
| `Decision_function -> Py.String.of_string "decision_function"
| `Auto -> Py.String.of_string "auto"
)); ("name", Wrap_utils.Option.map name Py.String.of_string); ("ax", ax); ("estimator", Some(estimator |> Np.Obj.to_pyobject)); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))]) (match kwargs with None -> [] | Some x -> x))

                  let precision_recall_curve ?pos_label ?sample_weight ~y_true ~probas_pred () =
                     Py.Module.get_function_with_keywords __wrap_namespace "precision_recall_curve"
                       [||]
                       (Wrap_utils.keyword_args [("pos_label", Wrap_utils.Option.map pos_label (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("probas_pred", Some(probas_pred |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 2))))
                  let precision_recall_fscore_support ?beta ?labels ?pos_label ?average ?warn_for ?sample_weight ?zero_division ~y_true ~y_pred () =
                     Py.Module.get_function_with_keywords __wrap_namespace "precision_recall_fscore_support"
                       [||]
                       (Wrap_utils.keyword_args [("beta", Wrap_utils.Option.map beta Py.Float.of_float); ("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("pos_label", Wrap_utils.Option.map pos_label (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("average", Wrap_utils.Option.map average (function
| `Micro -> Py.String.of_string "micro"
| `Binary -> Py.String.of_string "binary"
| `Samples -> Py.String.of_string "samples"
| `Weighted -> Py.String.of_string "weighted"
| `Macro -> Py.String.of_string "macro"
)); ("warn_for", warn_for); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("zero_division", Wrap_utils.Option.map zero_division (function
| `Warn -> Py.String.of_string "warn"
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
)); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_pred", Some(y_pred |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 2)), ((fun py -> if Py.is_none py then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) py)) (Py.Tuple.get x 3))))
                  let precision_score ?labels ?pos_label ?average ?sample_weight ?zero_division ~y_true ~y_pred () =
                     Py.Module.get_function_with_keywords __wrap_namespace "precision_score"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("pos_label", Wrap_utils.Option.map pos_label (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("average", Wrap_utils.Option.map average (function
| `Micro -> Py.String.of_string "micro"
| `Binary -> Py.String.of_string "binary"
| `Samples -> Py.String.of_string "samples"
| `Weighted -> Py.String.of_string "weighted"
| `Macro -> Py.String.of_string "macro"
| `None -> Py.none
)); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("zero_division", Wrap_utils.Option.map zero_division (function
| `Warn -> Py.String.of_string "warn"
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
)); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_pred", Some(y_pred |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let r2_score ?sample_weight ?multioutput ~y_true ~y_pred () =
                     Py.Module.get_function_with_keywords __wrap_namespace "r2_score"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("multioutput", Wrap_utils.Option.map multioutput (function
| `Variance_weighted -> Py.String.of_string "variance_weighted"
| `Arr x -> Np.Obj.to_pyobject x
| `Raw_values -> Py.String.of_string "raw_values"
| `Uniform_average -> Py.String.of_string "uniform_average"
| `None -> Py.none
)); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_pred", Some(y_pred |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let recall_score ?labels ?pos_label ?average ?sample_weight ?zero_division ~y_true ~y_pred () =
                     Py.Module.get_function_with_keywords __wrap_namespace "recall_score"
                       [||]
                       (Wrap_utils.keyword_args [("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("pos_label", Wrap_utils.Option.map pos_label (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("average", Wrap_utils.Option.map average (function
| `Micro -> Py.String.of_string "micro"
| `Binary -> Py.String.of_string "binary"
| `Samples -> Py.String.of_string "samples"
| `Weighted -> Py.String.of_string "weighted"
| `Macro -> Py.String.of_string "macro"
| `None -> Py.none
)); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("zero_division", Wrap_utils.Option.map zero_division (function
| `Warn -> Py.String.of_string "warn"
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
)); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_pred", Some(y_pred |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let roc_auc_score ?average ?sample_weight ?max_fpr ?multi_class ?labels ~y_true ~y_score () =
                     Py.Module.get_function_with_keywords __wrap_namespace "roc_auc_score"
                       [||]
                       (Wrap_utils.keyword_args [("average", Wrap_utils.Option.map average (function
| `Micro -> Py.String.of_string "micro"
| `Macro -> Py.String.of_string "macro"
| `Samples -> Py.String.of_string "samples"
| `Weighted -> Py.String.of_string "weighted"
| `None -> Py.none
)); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("max_fpr", max_fpr); ("multi_class", Wrap_utils.Option.map multi_class (function
| `Raise -> Py.String.of_string "raise"
| `Ovr -> Py.String.of_string "ovr"
| `Ovo -> Py.String.of_string "ovo"
)); ("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_score", Some(y_score |> Np.Obj.to_pyobject))])
                       |> Py.Float.to_float
                  let roc_curve ?pos_label ?sample_weight ?drop_intermediate ~y_true ~y_score () =
                     Py.Module.get_function_with_keywords __wrap_namespace "roc_curve"
                       [||]
                       (Wrap_utils.keyword_args [("pos_label", Wrap_utils.Option.map pos_label (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("drop_intermediate", Wrap_utils.Option.map drop_intermediate Py.Bool.of_bool); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_score", Some(y_score |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 2))))
                  let silhouette_samples ?metric ?kwds ~x ~labels () =
                     Py.Module.get_function_with_keywords __wrap_namespace "silhouette_samples"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("X", Some(x |> (function
| `Otherwise x -> Wrap_utils.id x
| `Arr x -> Np.Obj.to_pyobject x
))); ("labels", Some(labels |> Np.Obj.to_pyobject))]) (match kwds with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let silhouette_score ?metric ?sample_size ?random_state ?kwds ~x ~labels () =
                     Py.Module.get_function_with_keywords __wrap_namespace "silhouette_score"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("sample_size", Wrap_utils.Option.map sample_size Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("X", Some(x |> (function
| `Otherwise x -> Wrap_utils.id x
| `Arr x -> Np.Obj.to_pyobject x
))); ("labels", Some(labels |> Np.Obj.to_pyobject))]) (match kwds with None -> [] | Some x -> x))
                       |> Py.Float.to_float
let v_measure_score ?beta ~labels_true ~labels_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "v_measure_score"
     [||]
     (Wrap_utils.keyword_args [("beta", Wrap_utils.Option.map beta Py.Float.of_float); ("labels_true", Some(labels_true |> Np.Obj.to_pyobject)); ("labels_pred", Some(labels_pred |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let zero_one_loss ?normalize ?sample_weight ~y_true ~y_pred () =
   Py.Module.get_function_with_keywords __wrap_namespace "zero_one_loss"
     [||]
     (Wrap_utils.keyword_args [("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_pred", Some(y_pred |> Np.Obj.to_pyobject))])
     |> (fun x -> if Wrap_utils.check_float x then `F (Py.Float.to_float x) else if Wrap_utils.check_int x then `I (Py.Int.to_int x) else failwith (Printf.sprintf "Sklearn: could not identify type from Python value %s (%s)"
                                (Py.Object.to_string x) (Wrap_utils.type_string x)))
