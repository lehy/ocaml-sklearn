let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.calibration"

let get_py name = Py.Module.get __wrap_namespace name
module CalibratedClassifierCV = struct
type tag = [`CalibratedClassifierCV]
type t = [`BaseEstimator | `CalibratedClassifierCV | `ClassifierMixin | `MetaEstimatorMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_meta_estimator x = (x :> [`MetaEstimatorMixin] Obj.t)
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
                  let create ?base_estimator ?method_ ?cv () =
                     Py.Module.get_function_with_keywords __wrap_namespace "CalibratedClassifierCV"
                       [||]
                       (Wrap_utils.keyword_args [("base_estimator", Wrap_utils.Option.map base_estimator Np.Obj.to_pyobject); ("method", Wrap_utils.Option.map method_ (function
| `Sigmoid -> Py.String.of_string "sigmoid"
| `Isotonic -> Py.String.of_string "isotonic"
)); ("cv", Wrap_utils.Option.map cv (function
| `BaseCrossValidator x -> Np.Obj.to_pyobject x
| `Prefit -> Py.String.of_string "prefit"
| `Arr x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
))])
                       |> of_pyobject
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let predict ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let predict_proba ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject

let classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let calibrated_classifiers_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "calibrated_classifiers_" with
  | None -> failwith "attribute calibrated_classifiers_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let calibrated_classifiers_ self = match calibrated_classifiers_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module IsotonicRegression = struct
type tag = [`IsotonicRegression]
type t = [`BaseEstimator | `IsotonicRegression | `Object | `RegressorMixin | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
                  let create ?y_min ?y_max ?increasing ?out_of_bounds () =
                     Py.Module.get_function_with_keywords __wrap_namespace "IsotonicRegression"
                       [||]
                       (Wrap_utils.keyword_args [("y_min", Wrap_utils.Option.map y_min Py.Float.of_float); ("y_max", Wrap_utils.Option.map y_max Py.Float.of_float); ("increasing", Wrap_utils.Option.map increasing (function
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("out_of_bounds", Wrap_utils.Option.map out_of_bounds Py.String.of_string)])
                       |> of_pyobject
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let predict ~t self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict"
     [||]
     (Wrap_utils.keyword_args [("T", Some(t |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let transform ~t self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("T", Some(t |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

let x_min_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "X_min_" with
  | None -> failwith "attribute X_min_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let x_min_ self = match x_min_opt self with
  | None -> raise Not_found
  | Some x -> x

let x_max_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "X_max_" with
  | None -> failwith "attribute X_max_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let x_max_ self = match x_max_opt self with
  | None -> raise Not_found
  | Some x -> x

let f_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "f_" with
  | None -> failwith "attribute f_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let f_ self = match f_opt self with
  | None -> raise Not_found
  | Some x -> x

let increasing_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "increasing_" with
  | None -> failwith "attribute increasing_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Bool.to_bool x)

let increasing_ self = match increasing_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LabelBinarizer = struct
type tag = [`LabelBinarizer]
type t = [`BaseEstimator | `LabelBinarizer | `Object | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let create ?neg_label ?pos_label ?sparse_output () =
   Py.Module.get_function_with_keywords __wrap_namespace "LabelBinarizer"
     [||]
     (Wrap_utils.keyword_args [("neg_label", Wrap_utils.Option.map neg_label Py.Int.of_int); ("pos_label", Wrap_utils.Option.map pos_label Py.Int.of_int); ("sparse_output", Wrap_utils.Option.map sparse_output Py.Bool.of_bool)])
     |> of_pyobject
let fit ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
let fit_transform ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let inverse_transform ?threshold ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("threshold", Wrap_utils.Option.map threshold Py.Float.of_float); ("Y", Some(y |> Np.Obj.to_pyobject))])

let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let transform ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

let classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_type_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "y_type_" with
  | None -> failwith "attribute y_type_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.String.to_string x)

let y_type_ self = match y_type_opt self with
  | None -> raise Not_found
  | Some x -> x

let sparse_input_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "sparse_input_" with
  | None -> failwith "attribute sparse_input_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Bool.to_bool x)

let sparse_input_ self = match sparse_input_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LabelEncoder = struct
type tag = [`LabelEncoder]
type t = [`BaseEstimator | `LabelEncoder | `Object | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "LabelEncoder"
     [||]
     []
     |> of_pyobject
let fit ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
let fit_transform ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let inverse_transform ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let transform ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

let classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module LinearSVC = struct
type tag = [`LinearSVC]
type t = [`BaseEstimator | `ClassifierMixin | `LinearClassifierMixin | `LinearSVC | `Object | `SparseCoefMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_linear_classifier x = (x :> [`LinearClassifierMixin] Obj.t)
let as_sparse_coef x = (x :> [`SparseCoefMixin] Obj.t)
                  let create ?penalty ?loss ?dual ?tol ?c ?multi_class ?fit_intercept ?intercept_scaling ?class_weight ?verbose ?random_state ?max_iter () =
                     Py.Module.get_function_with_keywords __wrap_namespace "LinearSVC"
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
| `Balanced -> Py.String.of_string "balanced"
| `DictIntToFloat x -> (Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float) x
)); ("verbose", Wrap_utils.Option.map verbose Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("max_iter", Wrap_utils.Option.map max_iter Py.Int.of_int)])
                       |> of_pyobject
let decision_function ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decision_function"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let densify self =
   Py.Module.get_function_with_keywords (to_pyobject self) "densify"
     [||]
     []
     |> of_pyobject
let fit ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let predict ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let sparsify self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sparsify"
     [||]
     []
     |> of_pyobject

let coef_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "coef_" with
  | None -> failwith "attribute coef_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let coef_ self = match coef_opt self with
  | None -> raise Not_found
  | Some x -> x

let intercept_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "intercept_" with
  | None -> failwith "attribute intercept_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let intercept_ self = match intercept_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_iter_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_iter_" with
  | None -> failwith "attribute n_iter_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_iter_ self = match n_iter_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let calibration_curve ?normalize ?n_bins ?strategy ~y_true ~y_prob () =
                     Py.Module.get_function_with_keywords __wrap_namespace "calibration_curve"
                       [||]
                       (Wrap_utils.keyword_args [("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool); ("n_bins", Wrap_utils.Option.map n_bins Py.Int.of_int); ("strategy", Wrap_utils.Option.map strategy (function
| `Uniform -> Py.String.of_string "uniform"
| `Quantile -> Py.String.of_string "quantile"
)); ("y_true", Some(y_true |> Np.Obj.to_pyobject)); ("y_prob", Some(y_prob |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let check_array ?accept_sparse ?accept_large_sparse ?dtype ?order ?copy ?force_all_finite ?ensure_2d ?allow_nd ?ensure_min_samples ?ensure_min_features ?estimator ~array () =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_array"
                       [||]
                       (Wrap_utils.keyword_args [("accept_sparse", Wrap_utils.Option.map accept_sparse (function
| `S x -> Py.String.of_string x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `Bool x -> Py.Bool.of_bool x
)); ("accept_large_sparse", Wrap_utils.Option.map accept_large_sparse Py.Bool.of_bool); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
| `Dtypes x -> (fun ml -> Py.List.of_list_map Np.Dtype.to_pyobject ml) x
| `None -> Py.none
)); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("force_all_finite", Wrap_utils.Option.map force_all_finite (function
| `Allow_nan -> Py.String.of_string "allow-nan"
| `Bool x -> Py.Bool.of_bool x
)); ("ensure_2d", Wrap_utils.Option.map ensure_2d Py.Bool.of_bool); ("allow_nd", Wrap_utils.Option.map allow_nd Py.Bool.of_bool); ("ensure_min_samples", Wrap_utils.Option.map ensure_min_samples Py.Int.of_int); ("ensure_min_features", Wrap_utils.Option.map ensure_min_features Py.Int.of_int); ("estimator", Wrap_utils.Option.map estimator Np.Obj.to_pyobject); ("array", Some(array ))])

let check_consistent_length arrays =
   Py.Module.get_function_with_keywords __wrap_namespace "check_consistent_length"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arrays)])
     []

                  let check_cv ?cv ?y ?classifier () =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_cv"
                       [||]
                       (Wrap_utils.keyword_args [("cv", Wrap_utils.Option.map cv (function
| `BaseCrossValidator x -> Np.Obj.to_pyobject x
| `Arr x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("classifier", Wrap_utils.Option.map classifier Py.Bool.of_bool)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`BaseCrossValidator|`Object] Np.Obj.t))
                  let check_is_fitted ?attributes ?msg ?all_or_any ~estimator () =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_is_fitted"
                       [||]
                       (Wrap_utils.keyword_args [("attributes", Wrap_utils.Option.map attributes (function
| `S x -> Py.String.of_string x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `Arr x -> Np.Obj.to_pyobject x
)); ("msg", Wrap_utils.Option.map msg Py.String.of_string); ("all_or_any", Wrap_utils.Option.map all_or_any (function
| `Callable x -> Wrap_utils.id x
| `PyObject x -> Wrap_utils.id x
)); ("estimator", Some(estimator |> Np.Obj.to_pyobject))])

let clone ?safe ~estimator () =
   Py.Module.get_function_with_keywords __wrap_namespace "clone"
     [||]
     (Wrap_utils.keyword_args [("safe", Wrap_utils.Option.map safe Py.Bool.of_bool); ("estimator", Some(estimator |> Np.Obj.to_pyobject))])

let column_or_1d ?warn ~y () =
   Py.Module.get_function_with_keywords __wrap_namespace "column_or_1d"
     [||]
     (Wrap_utils.keyword_args [("warn", Wrap_utils.Option.map warn Py.Bool.of_bool); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let expit ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "expit"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let fmin_bfgs ?fprime ?args ?gtol ?norm ?epsilon ?maxiter ?full_output ?disp ?retall ?callback ~f ~x0 () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fmin_bfgs"
                       [||]
                       (Wrap_utils.keyword_args [("fprime", fprime); ("args", args); ("gtol", Wrap_utils.Option.map gtol Py.Float.of_float); ("norm", Wrap_utils.Option.map norm Py.Float.of_float); ("epsilon", Wrap_utils.Option.map epsilon (function
| `Arr x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("full_output", Wrap_utils.Option.map full_output Py.Bool.of_bool); ("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("retall", Wrap_utils.Option.map retall Py.Bool.of_bool); ("callback", callback); ("f", Some(f )); ("x0", Some(x0 |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 2)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 3)), (Py.Int.to_int (Py.Tuple.get x 4)), (Py.Int.to_int (Py.Tuple.get x 5)), (Py.Int.to_int (Py.Tuple.get x 6)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 7))))
let indexable iterables =
   Py.Module.get_function_with_keywords __wrap_namespace "indexable"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id iterables)])
     []

let label_binarize ?neg_label ?pos_label ?sparse_output ~y ~classes () =
   Py.Module.get_function_with_keywords __wrap_namespace "label_binarize"
     [||]
     (Wrap_utils.keyword_args [("neg_label", Wrap_utils.Option.map neg_label Py.Int.of_int); ("pos_label", Wrap_utils.Option.map pos_label Py.Int.of_int); ("sparse_output", Wrap_utils.Option.map sparse_output Py.Bool.of_bool); ("y", Some(y |> Np.Obj.to_pyobject)); ("classes", Some(classes |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let signature ?follow_wrapped ~obj () =
   Py.Module.get_function_with_keywords __wrap_namespace "signature"
     [||]
     (Wrap_utils.keyword_args [("follow_wrapped", follow_wrapped); ("obj", Some(obj ))])

let xlogy ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "xlogy"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
