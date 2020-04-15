let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.calibration"

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
module CalibratedClassifierCV = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?base_estimator ?method_ ?cv () =
                     Py.Module.get_function_with_keywords ns "CalibratedClassifierCV"
                       [||]
                       (Wrap_utils.keyword_args [("base_estimator", base_estimator); ("method", Wrap_utils.Option.map method_ (function
| `Sigmoid -> Py.String.of_string "sigmoid"
| `Isotonic -> Py.String.of_string "isotonic"
)); ("cv", Wrap_utils.Option.map cv (function
| `Int x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Ndarray x -> Ndarray.to_pyobject x
| `Prefit -> Py.String.of_string "prefit"
))])

let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let predict_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let classes_ self =
  match Py.Object.get_attr_string self "classes_" with
| None -> raise (Wrap_utils.Attribute_not_found "classes_")
| Some x -> Ndarray.of_pyobject x
let calibrated_classifiers_ self =
  match Py.Object.get_attr_string self "calibrated_classifiers_" with
| None -> raise (Wrap_utils.Attribute_not_found "calibrated_classifiers_")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ClassifierMixin = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create () =
   Py.Module.get_function_with_keywords ns "ClassifierMixin"
     [||]
     []

let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module IsotonicRegression = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?y_min ?y_max ?increasing ?out_of_bounds () =
                     Py.Module.get_function_with_keywords ns "IsotonicRegression"
                       [||]
                       (Wrap_utils.keyword_args [("y_min", y_min); ("y_max", y_max); ("increasing", Wrap_utils.Option.map increasing (function
| `Bool x -> Py.Bool.of_bool x
| `String x -> Py.String.of_string x
)); ("out_of_bounds", Wrap_utils.Option.map out_of_bounds Py.String.of_string)])

let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let predict ~t self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("T", Some(t |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~t self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("T", Some(t |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let x_min_ self =
  match Py.Object.get_attr_string self "X_min_" with
| None -> raise (Wrap_utils.Attribute_not_found "X_min_")
| Some x -> Py.Float.to_float x
let x_max_ self =
  match Py.Object.get_attr_string self "X_max_" with
| None -> raise (Wrap_utils.Attribute_not_found "X_max_")
| Some x -> Py.Float.to_float x
let f_ self =
  match Py.Object.get_attr_string self "f_" with
| None -> raise (Wrap_utils.Attribute_not_found "f_")
| Some x -> Wrap_utils.id x
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
module LinearSVC = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?penalty ?loss ?dual ?tol ?c ?multi_class ?fit_intercept ?intercept_scaling ?class_weight ?verbose ?random_state ?max_iter () =
                     Py.Module.get_function_with_keywords ns "LinearSVC"
                       [||]
                       (Wrap_utils.keyword_args [("penalty", Wrap_utils.Option.map penalty (function
| `L1 -> Py.String.of_string "l1"
| `L2 -> Py.String.of_string "l2"
)); ("loss", Wrap_utils.Option.map loss (function
| `Hinge -> Py.String.of_string "hinge"
| `Squared_hinge -> Py.String.of_string "squared_hinge"
)); ("dual", Wrap_utils.Option.map dual Py.Bool.of_bool); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("C", Wrap_utils.Option.map c Py.Float.of_float); ("multi_class", Wrap_utils.Option.map multi_class (function
| `Ovr -> Py.String.of_string "ovr"
| `Crammer_singer -> Py.String.of_string "crammer_singer"
)); ("fit_intercept", Wrap_utils.Option.map fit_intercept Py.Bool.of_bool); ("intercept_scaling", Wrap_utils.Option.map intercept_scaling Py.Float.of_float); ("class_weight", Wrap_utils.Option.map class_weight (function
| `DictIntToFloat x -> (Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float) x
| `Balanced -> Py.String.of_string "balanced"
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int)])

                  let decision_function ~x self =
                     Py.Module.get_function_with_keywords self "decision_function"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let densify self =
   Py.Module.get_function_with_keywords self "densify"
     [||]
     []

                  let fit ?sample_weight ~x ~y self =
                     Py.Module.get_function_with_keywords self "fit"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
))); ("y", Some(y |> Ndarray.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

                  let predict ~x self =
                     Py.Module.get_function_with_keywords self "predict"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> Ndarray.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let sparsify self =
   Py.Module.get_function_with_keywords self "sparsify"
     [||]
     []

let coef_ self =
  match Py.Object.get_attr_string self "coef_" with
| None -> raise (Wrap_utils.Attribute_not_found "coef_")
| Some x -> Ndarray.of_pyobject x
let intercept_ self =
  match Py.Object.get_attr_string self "intercept_" with
| None -> raise (Wrap_utils.Attribute_not_found "intercept_")
| Some x -> Ndarray.of_pyobject x
let classes_ self =
  match Py.Object.get_attr_string self "classes_" with
| None -> raise (Wrap_utils.Attribute_not_found "classes_")
| Some x -> Ndarray.of_pyobject x
let n_iter_ self =
  match Py.Object.get_attr_string self "n_iter_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_iter_")
| Some x -> Py.Int.to_int x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MetaEstimatorMixin = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create () =
   Py.Module.get_function_with_keywords ns "MetaEstimatorMixin"
     [||]
     []

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RegressorMixin = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create () =
   Py.Module.get_function_with_keywords ns "RegressorMixin"
     [||]
     []

let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])
     |> Py.Float.to_float
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let calibration_curve ?normalize ?n_bins ?strategy ~y_true ~y_prob () =
                     Py.Module.get_function_with_keywords ns "calibration_curve"
                       [||]
                       (Wrap_utils.keyword_args [("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("n_bins", Wrap_utils.Option.map n_bins Py.Int.of_int); ("strategy", Wrap_utils.Option.map strategy (function
| `Uniform -> Py.String.of_string "uniform"
| `Quantile -> Py.String.of_string "quantile"
)); ("y_true", Some(y_true |> Ndarray.to_pyobject)); ("y_prob", Some(y_prob |> Ndarray.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let check_X_y ?accept_sparse ?accept_large_sparse ?dtype ?order ?copy ?force_all_finite ?ensure_2d ?allow_nd ?multi_output ?ensure_min_samples ?ensure_min_features ?y_numeric ?warn_on_dtype ?estimator ~x ~y () =
                     Py.Module.get_function_with_keywords ns "check_X_y"
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
)); ("ensure_2d", Wrap_utils.Option.map ensure_2d Py.Bool.of_bool); ("allow_nd", Wrap_utils.Option.map allow_nd Py.Bool.of_bool); ("multi_output", Wrap_utils.Option.map multi_output Py.Bool.of_bool); ("ensure_min_samples", Wrap_utils.Option.map ensure_min_samples Py.Int.of_int); ("ensure_min_features", Wrap_utils.Option.map ensure_min_features Py.Int.of_int); ("y_numeric", Wrap_utils.Option.map y_numeric Py.Bool.of_bool); ("warn_on_dtype", Wrap_utils.Option.map warn_on_dtype (function
| `Bool x -> Py.Bool.of_bool x
| `None -> Py.String.of_string "None"
)); ("estimator", Wrap_utils.Option.map estimator (function
| `String x -> Py.String.of_string x
| `Estimator x -> Wrap_utils.id x
)); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `ArrayLike x -> Wrap_utils.id x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
))); ("y", Some(y |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `ArrayLike x -> Wrap_utils.id x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
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

let check_consistent_length arrays =
   Py.Module.get_function_with_keywords ns "check_consistent_length"
     (Wrap_utils.pos_arg Wrap_utils.id arrays)
     []

                  let check_cv ?cv ?y ?classifier () =
                     Py.Module.get_function_with_keywords ns "check_cv"
                       [||]
                       (Wrap_utils.keyword_args [("cv", Wrap_utils.Option.map cv (function
| `Int x -> Py.Int.of_int x
| `CrossValGenerator x -> Wrap_utils.id x
| `Ndarray x -> Ndarray.to_pyobject x
)); ("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("classifier", Wrap_utils.Option.map classifier Py.Bool.of_bool)])

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

                  let clone ?safe ~estimator () =
                     Py.Module.get_function_with_keywords ns "clone"
                       [||]
                       (Wrap_utils.keyword_args [("safe", Wrap_utils.Option.map safe Py.Bool.of_bool); ("estimator", Some(estimator |> (function
| `Estimator x -> Wrap_utils.id x
| `ArrayLike x -> Wrap_utils.id x
| `PyObject x -> Wrap_utils.id x
)))])

let column_or_1d ?warn ~y () =
   Py.Module.get_function_with_keywords ns "column_or_1d"
     [||]
     (Wrap_utils.keyword_args [("warn", Wrap_utils.Option.map warn Py.Bool.of_bool); ("y", Some(y |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let fmin_bfgs ?fprime ?args ?gtol ?norm ?epsilon ?maxiter ?full_output ?disp ?retall ?callback ~f ~x0 () =
   Py.Module.get_function_with_keywords ns "fmin_bfgs"
     [||]
     (Wrap_utils.keyword_args [("fprime", fprime); ("args", args); ("gtol", gtol); ("norm", norm); ("epsilon", epsilon); ("maxiter", maxiter); ("full_output", full_output); ("disp", disp); ("retall", retall); ("callback", callback); ("f", Some(f )); ("x0", Some(x0 ))])
     |> Ndarray.of_pyobject
let indexable iterables =
   Py.Module.get_function_with_keywords ns "indexable"
     (Wrap_utils.pos_arg Wrap_utils.id iterables)
     []

let label_binarize ?neg_label ?pos_label ?sparse_output ~y ~classes () =
   Py.Module.get_function_with_keywords ns "label_binarize"
     [||]
     (Wrap_utils.keyword_args [("neg_label", Wrap_utils.Option.map neg_label Py.Int.of_int); ("pos_label", Wrap_utils.Option.map pos_label Py.Int.of_int); ("sparse_output", Wrap_utils.Option.map sparse_output Py.Bool.of_bool); ("y", Some(y |> Ndarray.to_pyobject)); ("classes", Some(classes |> Ndarray.to_pyobject))])

let signature ?follow_wrapped ~obj () =
   Py.Module.get_function_with_keywords ns "signature"
     [||]
     (Wrap_utils.keyword_args [("follow_wrapped", follow_wrapped); ("obj", Some(obj ))])

