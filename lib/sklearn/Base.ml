let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.base"

let get_py name = Py.Module.get __wrap_namespace name
module BaseEstimator = struct
type tag = [`BaseEstimator]
type t = [`BaseEstimator | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "BaseEstimator"
     [||]
     []
     |> of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module BiclusterMixin = struct
type tag = [`BiclusterMixin]
type t = [`BiclusterMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "BiclusterMixin"
     [||]
     []
     |> of_pyobject
let get_indices ~i self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_indices"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i |> Py.Int.of_int))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let get_shape ~i self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_shape"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i |> Py.Int.of_int))])

let get_submatrix ~i ~data self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_submatrix"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i |> Py.Int.of_int)); ("data", Some(data |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ClassifierMixin = struct
type tag = [`ClassifierMixin]
type t = [`ClassifierMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "ClassifierMixin"
     [||]
     []
     |> of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ClusterMixin = struct
type tag = [`ClusterMixin]
type t = [`ClusterMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "ClusterMixin"
     [||]
     []
     |> of_pyobject
let fit_predict ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_predict"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module DensityMixin = struct
type tag = [`DensityMixin]
type t = [`DensityMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "DensityMixin"
     [||]
     []
     |> of_pyobject
let score ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MetaEstimatorMixin = struct
type tag = [`MetaEstimatorMixin]
type t = [`MetaEstimatorMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "MetaEstimatorMixin"
     [||]
     []
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MultiOutputMixin = struct
type tag = [`MultiOutputMixin]
type t = [`MultiOutputMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "MultiOutputMixin"
     [||]
     []
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module OutlierMixin = struct
type tag = [`OutlierMixin]
type t = [`Object | `OutlierMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "OutlierMixin"
     [||]
     []
     |> of_pyobject
let fit_predict ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_predict"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RegressorMixin = struct
type tag = [`RegressorMixin]
type t = [`Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "RegressorMixin"
     [||]
     []
     |> of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> Py.Float.to_float
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module TransformerMixin = struct
type tag = [`TransformerMixin]
type t = [`Object | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "TransformerMixin"
     [||]
     []
     |> of_pyobject
let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Defaultdict = struct
type tag = [`Defaultdict]
type t = [`Defaultdict | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let get_item ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y ))])

let iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []
     |> (fun py -> Py.Iter.to_seq py |> Seq.map Dict.of_pyobject)
let __setitem__ ~key ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     (Array.of_list @@ List.concat [[key ];[value ]])
     []

let fromkeys ?value ~iterable self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fromkeys"
     (Array.of_list @@ List.concat [(match value with None -> [] | Some x -> [x ]);[iterable ]])
     []

let get ?default ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get"
     (Array.of_list @@ List.concat [(match default with None -> [] | Some x -> [x ]);[key ]])
     []

let pop ?d ~k self =
   Py.Module.get_function_with_keywords (to_pyobject self) "pop"
     [||]
     (Wrap_utils.keyword_args [("d", d); ("k", Some(k ))])

let popitem self =
   Py.Module.get_function_with_keywords (to_pyobject self) "popitem"
     [||]
     []

let setdefault ?default ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setdefault"
     (Array.of_list @@ List.concat [(match default with None -> [] | Some x -> [x ]);[key ]])
     []

let update ?e ?f self =
   Py.Module.get_function_with_keywords (to_pyobject self) "update"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("E", e)]) (match f with None -> [] | Some x -> x))

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let check_X_y ?accept_sparse ?accept_large_sparse ?dtype ?order ?copy ?force_all_finite ?ensure_2d ?allow_nd ?multi_output ?ensure_min_samples ?ensure_min_features ?y_numeric ?estimator ~x ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_X_y"
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
)); ("ensure_2d", Wrap_utils.Option.map ensure_2d Py.Bool.of_bool); ("allow_nd", Wrap_utils.Option.map allow_nd Py.Bool.of_bool); ("multi_output", Wrap_utils.Option.map multi_output Py.Bool.of_bool); ("ensure_min_samples", Wrap_utils.Option.map ensure_min_samples Py.Int.of_int); ("ensure_min_features", Wrap_utils.Option.map ensure_min_features Py.Int.of_int); ("y_numeric", Wrap_utils.Option.map y_numeric Py.Bool.of_bool); ("estimator", Wrap_utils.Option.map estimator Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
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

let clone ?safe ~estimator () =
   Py.Module.get_function_with_keywords __wrap_namespace "clone"
     [||]
     (Wrap_utils.keyword_args [("safe", Wrap_utils.Option.map safe Py.Bool.of_bool); ("estimator", Some(estimator |> Np.Obj.to_pyobject))])

let estimator_html_repr estimator =
   Py.Module.get_function_with_keywords __wrap_namespace "estimator_html_repr"
     [||]
     (Wrap_utils.keyword_args [("estimator", Some(estimator |> Np.Obj.to_pyobject))])
     |> Py.String.to_string
let get_config () =
   Py.Module.get_function_with_keywords __wrap_namespace "get_config"
     [||]
     []
     |> Dict.of_pyobject
let is_classifier estimator =
   Py.Module.get_function_with_keywords __wrap_namespace "is_classifier"
     [||]
     (Wrap_utils.keyword_args [("estimator", Some(estimator |> Np.Obj.to_pyobject))])
     |> Py.Bool.to_bool
let is_outlier_detector estimator =
   Py.Module.get_function_with_keywords __wrap_namespace "is_outlier_detector"
     [||]
     (Wrap_utils.keyword_args [("estimator", Some(estimator |> Np.Obj.to_pyobject))])
     |> Py.Bool.to_bool
let is_regressor estimator =
   Py.Module.get_function_with_keywords __wrap_namespace "is_regressor"
     [||]
     (Wrap_utils.keyword_args [("estimator", Some(estimator |> Np.Obj.to_pyobject))])
     |> Py.Bool.to_bool
