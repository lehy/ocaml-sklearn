let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.gaussian_process"

let get_py name = Py.Module.get __wrap_namespace name
module GaussianProcessClassifier = struct
type tag = [`GaussianProcessClassifier]
type t = [`BaseEstimator | `ClassifierMixin | `GaussianProcessClassifier | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_classifier x = (x :> [`ClassifierMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
                  let create ?kernel ?optimizer ?n_restarts_optimizer ?max_iter_predict ?warm_start ?copy_X_train ?random_state ?multi_class ?n_jobs () =
                     Py.Module.get_function_with_keywords __wrap_namespace "GaussianProcessClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("kernel", kernel); ("optimizer", Wrap_utils.Option.map optimizer (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("n_restarts_optimizer", Wrap_utils.Option.map n_restarts_optimizer Py.Int.of_int); ("max_iter_predict", Wrap_utils.Option.map max_iter_predict Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("copy_X_train", Wrap_utils.Option.map copy_X_train Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("multi_class", Wrap_utils.Option.map multi_class Py.String.of_string); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int)])
                       |> of_pyobject
let fit ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let log_marginal_likelihood ?theta ?eval_gradient ?clone_kernel self =
   Py.Module.get_function_with_keywords (to_pyobject self) "log_marginal_likelihood"
     [||]
     (Wrap_utils.keyword_args [("theta", Wrap_utils.Option.map theta Np.Obj.to_pyobject); ("eval_gradient", Wrap_utils.Option.map eval_gradient Py.Bool.of_bool); ("clone_kernel", Wrap_utils.Option.map clone_kernel Py.Bool.of_bool)])
     |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
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

let kernel_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "kernel_" with
  | None -> failwith "attribute kernel_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let kernel_ self = match kernel_opt self with
  | None -> raise Not_found
  | Some x -> x

let log_marginal_likelihood_value_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "log_marginal_likelihood_value_" with
  | None -> failwith "attribute log_marginal_likelihood_value_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let log_marginal_likelihood_value_ self = match log_marginal_likelihood_value_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_classes_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_classes_" with
  | None -> failwith "attribute n_classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_classes_ self = match n_classes_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module GaussianProcessRegressor = struct
type tag = [`GaussianProcessRegressor]
type t = [`BaseEstimator | `GaussianProcessRegressor | `MultiOutputMixin | `Object | `RegressorMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_regressor x = (x :> [`RegressorMixin] Obj.t)
let as_multi_output x = (x :> [`MultiOutputMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
                  let create ?kernel ?alpha ?optimizer ?n_restarts_optimizer ?normalize_y ?copy_X_train ?random_state () =
                     Py.Module.get_function_with_keywords __wrap_namespace "GaussianProcessRegressor"
                       [||]
                       (Wrap_utils.keyword_args [("kernel", kernel); ("alpha", Wrap_utils.Option.map alpha Np.Obj.to_pyobject); ("optimizer", Wrap_utils.Option.map optimizer (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("n_restarts_optimizer", Wrap_utils.Option.map n_restarts_optimizer Py.Int.of_int); ("normalize_y", Wrap_utils.Option.map normalize_y Py.Bool.of_bool); ("copy_X_train", Wrap_utils.Option.map copy_X_train Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
                       |> of_pyobject
let fit ~x ~y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let log_marginal_likelihood ?theta ?eval_gradient ?clone_kernel self =
   Py.Module.get_function_with_keywords (to_pyobject self) "log_marginal_likelihood"
     [||]
     (Wrap_utils.keyword_args [("theta", Wrap_utils.Option.map theta Np.Obj.to_pyobject); ("eval_gradient", Wrap_utils.Option.map eval_gradient Py.Bool.of_bool); ("clone_kernel", Wrap_utils.Option.map clone_kernel Py.Bool.of_bool)])
     |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
let predict ?return_std ?return_cov ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "predict"
     [||]
     (Wrap_utils.keyword_args [("return_std", Wrap_utils.Option.map return_std Py.Bool.of_bool); ("return_cov", Wrap_utils.Option.map return_cov Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let sample_y ?n_samples ?random_state ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "sample_y"
     [||]
     (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("X", Some(x ))])
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

let x_train_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "X_train_" with
  | None -> failwith "attribute X_train_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let x_train_ self = match x_train_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_train_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "y_train_" with
  | None -> failwith "attribute y_train_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let y_train_ self = match y_train_opt self with
  | None -> raise Not_found
  | Some x -> x

let kernel_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "kernel_" with
  | None -> failwith "attribute kernel_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let kernel_ self = match kernel_opt self with
  | None -> raise Not_found
  | Some x -> x

let l_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "L_" with
  | None -> failwith "attribute L_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let l_ self = match l_opt self with
  | None -> raise Not_found
  | Some x -> x

let alpha_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "alpha_" with
  | None -> failwith "attribute alpha_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let alpha_ self = match alpha_opt self with
  | None -> raise Not_found
  | Some x -> x

let log_marginal_likelihood_value_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "log_marginal_likelihood_value_" with
  | None -> failwith "attribute log_marginal_likelihood_value_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let log_marginal_likelihood_value_ self = match log_marginal_likelihood_value_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Kernels = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.gaussian_process.kernels"

let get_py name = Py.Module.get __wrap_namespace name
module CompoundKernel = struct
type tag = [`CompoundKernel]
type t = [`CompoundKernel | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create kernels =
   Py.Module.get_function_with_keywords __wrap_namespace "CompoundKernel"
     [||]
     (Wrap_utils.keyword_args [("kernels", Some(kernels ))])
     |> of_pyobject
let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords (to_pyobject self) "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Np.Obj.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let is_stationary self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_stationary"
     [||]
     []

let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ConstantKernel = struct
type tag = [`ConstantKernel]
type t = [`ConstantKernel | `GenericKernelMixin | `Object | `StationaryKernelMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_stationary_kernel x = (x :> [`StationaryKernelMixin] Obj.t)
let as_generic_kernel x = (x :> [`GenericKernelMixin] Obj.t)
let create ?constant_value ?constant_value_bounds () =
   Py.Module.get_function_with_keywords __wrap_namespace "ConstantKernel"
     [||]
     (Wrap_utils.keyword_args [("constant_value", Wrap_utils.Option.map constant_value Py.Float.of_float); ("constant_value_bounds", Wrap_utils.Option.map constant_value_bounds (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.Float.of_float ml_0); (Py.Float.of_float ml_1)]))])
     |> of_pyobject
let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords (to_pyobject self) "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Np.Obj.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let is_stationary self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_stationary"
     [||]
     []

let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module DotProduct = struct
type tag = [`DotProduct]
type t = [`DotProduct | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?sigma_0 ?sigma_0_bounds () =
   Py.Module.get_function_with_keywords __wrap_namespace "DotProduct"
     [||]
     (Wrap_utils.keyword_args [("sigma_0", Wrap_utils.Option.map sigma_0 Py.Float.of_float); ("sigma_0_bounds", Wrap_utils.Option.map sigma_0_bounds (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.Float.of_float ml_0); (Py.Float.of_float ml_1)]))])
     |> of_pyobject
let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords (to_pyobject self) "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Np.Obj.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let is_stationary self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_stationary"
     [||]
     []

let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ExpSineSquared = struct
type tag = [`ExpSineSquared]
type t = [`ExpSineSquared | `NormalizedKernelMixin | `Object | `StationaryKernelMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_stationary_kernel x = (x :> [`StationaryKernelMixin] Obj.t)
let as_normalized_kernel x = (x :> [`NormalizedKernelMixin] Obj.t)
let create ?length_scale ?periodicity ?length_scale_bounds ?periodicity_bounds () =
   Py.Module.get_function_with_keywords __wrap_namespace "ExpSineSquared"
     [||]
     (Wrap_utils.keyword_args [("length_scale", Wrap_utils.Option.map length_scale Py.Float.of_float); ("periodicity", Wrap_utils.Option.map periodicity Py.Float.of_float); ("length_scale_bounds", Wrap_utils.Option.map length_scale_bounds (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.Float.of_float ml_0); (Py.Float.of_float ml_1)])); ("periodicity_bounds", Wrap_utils.Option.map periodicity_bounds (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.Float.of_float ml_0); (Py.Float.of_float ml_1)]))])
     |> of_pyobject
let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords (to_pyobject self) "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Np.Obj.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let is_stationary self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_stationary"
     [||]
     []

let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Exponentiation = struct
type tag = [`Exponentiation]
type t = [`Exponentiation | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ~kernel ~exponent () =
   Py.Module.get_function_with_keywords __wrap_namespace "Exponentiation"
     [||]
     (Wrap_utils.keyword_args [("kernel", Some(kernel )); ("exponent", Some(exponent |> Py.Float.of_float))])
     |> of_pyobject
let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords (to_pyobject self) "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Np.Obj.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let is_stationary self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_stationary"
     [||]
     []

let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module GenericKernelMixin = struct
type tag = [`GenericKernelMixin]
type t = [`GenericKernelMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "GenericKernelMixin"
     [||]
     []
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Hyperparameter = struct
type tag = [`Hyperparameter]
type t = [`Hyperparameter | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?n_elements ?fixed ~name ~value_type ~bounds () =
   Py.Module.get_function_with_keywords __wrap_namespace "Hyperparameter"
     [||]
     (Wrap_utils.keyword_args [("n_elements", n_elements); ("fixed", fixed); ("name", Some(name )); ("value_type", Some(value_type )); ("bounds", Some(bounds ))])
     |> of_pyobject
let get_item ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []
     |> (fun py -> Py.Iter.to_seq py |> Seq.map Dict.of_pyobject)
let count ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "count"
     (Array.of_list @@ List.concat [[value ]])
     []

let index ?start ?stop ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "index"
     (Array.of_list @@ List.concat [(match start with None -> [] | Some x -> [x ]);(match stop with None -> [] | Some x -> [x ]);[value ]])
     []


let name_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "name" with
  | None -> failwith "attribute name not found"
  | Some x -> if Py.is_none x then None else Some (Py.String.to_string x)

let name self = match name_opt self with
  | None -> raise Not_found
  | Some x -> x

let value_type_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "value_type" with
  | None -> failwith "attribute value_type not found"
  | Some x -> if Py.is_none x then None else Some (Py.String.to_string x)

let value_type self = match value_type_opt self with
  | None -> raise Not_found
  | Some x -> x

let bounds_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "bounds" with
  | None -> failwith "attribute bounds not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let bounds self = match bounds_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_elements_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "n_elements" with
  | None -> failwith "attribute n_elements not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_elements self = match n_elements_opt self with
  | None -> raise Not_found
  | Some x -> x

let fixed_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "fixed" with
  | None -> failwith "attribute fixed not found"
  | Some x -> if Py.is_none x then None else Some (Py.Bool.to_bool x)

let fixed self = match fixed_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Kernel = struct
type tag = [`Kernel]
type t = [`Kernel | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords (to_pyobject self) "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Np.Obj.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let is_stationary self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_stationary"
     [||]
     []

let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module KernelOperator = struct
type tag = [`KernelOperator]
type t = [`KernelOperator | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords (to_pyobject self) "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Np.Obj.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let is_stationary self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_stationary"
     [||]
     []

let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Matern = struct
type tag = [`Matern]
type t = [`Matern | `NormalizedKernelMixin | `Object | `StationaryKernelMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_stationary_kernel x = (x :> [`StationaryKernelMixin] Obj.t)
let as_normalized_kernel x = (x :> [`NormalizedKernelMixin] Obj.t)
let create ?length_scale ?length_scale_bounds ?nu () =
   Py.Module.get_function_with_keywords __wrap_namespace "Matern"
     [||]
     (Wrap_utils.keyword_args [("length_scale", Wrap_utils.Option.map length_scale Np.Obj.to_pyobject); ("length_scale_bounds", Wrap_utils.Option.map length_scale_bounds (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.Float.of_float ml_0); (Py.Float.of_float ml_1)])); ("nu", Wrap_utils.Option.map nu Py.Float.of_float)])
     |> of_pyobject
let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords (to_pyobject self) "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Np.Obj.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let is_stationary self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_stationary"
     [||]
     []

let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module NormalizedKernelMixin = struct
type tag = [`NormalizedKernelMixin]
type t = [`NormalizedKernelMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "NormalizedKernelMixin"
     [||]
     []
     |> of_pyobject
let diag ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module PairwiseKernel = struct
type tag = [`PairwiseKernel]
type t = [`Object | `PairwiseKernel] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
                  let create ?gamma ?gamma_bounds ?metric ?pairwise_kernels_kwargs () =
                     Py.Module.get_function_with_keywords __wrap_namespace "PairwiseKernel"
                       [||]
                       (Wrap_utils.keyword_args [("gamma", Wrap_utils.Option.map gamma Py.Float.of_float); ("gamma_bounds", Wrap_utils.Option.map gamma_bounds (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.Float.of_float ml_0); (Py.Float.of_float ml_1)])); ("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("pairwise_kernels_kwargs", Wrap_utils.Option.map pairwise_kernels_kwargs Dict.to_pyobject)])
                       |> of_pyobject
let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords (to_pyobject self) "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Np.Obj.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let is_stationary self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_stationary"
     [||]
     []

let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Product = struct
type tag = [`Product]
type t = [`Object | `Product] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ~k1 ~k2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "Product"
     [||]
     (Wrap_utils.keyword_args [("k1", Some(k1 )); ("k2", Some(k2 ))])
     |> of_pyobject
let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords (to_pyobject self) "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Np.Obj.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let is_stationary self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_stationary"
     [||]
     []

let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RBF = struct
type tag = [`RBF]
type t = [`NormalizedKernelMixin | `Object | `RBF | `StationaryKernelMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_stationary_kernel x = (x :> [`StationaryKernelMixin] Obj.t)
let as_normalized_kernel x = (x :> [`NormalizedKernelMixin] Obj.t)
let create ?length_scale ?length_scale_bounds () =
   Py.Module.get_function_with_keywords __wrap_namespace "RBF"
     [||]
     (Wrap_utils.keyword_args [("length_scale", Wrap_utils.Option.map length_scale Np.Obj.to_pyobject); ("length_scale_bounds", Wrap_utils.Option.map length_scale_bounds (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.Float.of_float ml_0); (Py.Float.of_float ml_1)]))])
     |> of_pyobject
let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords (to_pyobject self) "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Np.Obj.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let is_stationary self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_stationary"
     [||]
     []

let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RationalQuadratic = struct
type tag = [`RationalQuadratic]
type t = [`NormalizedKernelMixin | `Object | `RationalQuadratic | `StationaryKernelMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_stationary_kernel x = (x :> [`StationaryKernelMixin] Obj.t)
let as_normalized_kernel x = (x :> [`NormalizedKernelMixin] Obj.t)
let create ?length_scale ?alpha ?length_scale_bounds ?alpha_bounds () =
   Py.Module.get_function_with_keywords __wrap_namespace "RationalQuadratic"
     [||]
     (Wrap_utils.keyword_args [("length_scale", Wrap_utils.Option.map length_scale Py.Float.of_float); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("length_scale_bounds", Wrap_utils.Option.map length_scale_bounds (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.Float.of_float ml_0); (Py.Float.of_float ml_1)])); ("alpha_bounds", Wrap_utils.Option.map alpha_bounds (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.Float.of_float ml_0); (Py.Float.of_float ml_1)]))])
     |> of_pyobject
let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords (to_pyobject self) "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Np.Obj.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let is_stationary self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_stationary"
     [||]
     []

let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module StationaryKernelMixin = struct
type tag = [`StationaryKernelMixin]
type t = [`Object | `StationaryKernelMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "StationaryKernelMixin"
     [||]
     []
     |> of_pyobject
let is_stationary self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_stationary"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Sum = struct
type tag = [`Sum]
type t = [`Object | `Sum] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ~k1 ~k2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "Sum"
     [||]
     (Wrap_utils.keyword_args [("k1", Some(k1 )); ("k2", Some(k2 ))])
     |> of_pyobject
let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords (to_pyobject self) "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Np.Obj.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let is_stationary self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_stationary"
     [||]
     []

let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module WhiteKernel = struct
type tag = [`WhiteKernel]
type t = [`GenericKernelMixin | `Object | `StationaryKernelMixin | `WhiteKernel] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_stationary_kernel x = (x :> [`StationaryKernelMixin] Obj.t)
let as_generic_kernel x = (x :> [`GenericKernelMixin] Obj.t)
let create ?noise_level ?noise_level_bounds () =
   Py.Module.get_function_with_keywords __wrap_namespace "WhiteKernel"
     [||]
     (Wrap_utils.keyword_args [("noise_level", Wrap_utils.Option.map noise_level Py.Float.of_float); ("noise_level_bounds", Wrap_utils.Option.map noise_level_bounds (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.Float.of_float ml_0); (Py.Float.of_float ml_1)]))])
     |> of_pyobject
let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords (to_pyobject self) "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Np.Obj.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let is_stationary self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_stationary"
     [||]
     []

let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let abstractmethod funcobj =
   Py.Module.get_function_with_keywords __wrap_namespace "abstractmethod"
     [||]
     (Wrap_utils.keyword_args [("funcobj", Some(funcobj ))])

                  let cdist ?metric ?kwargs ~xa ~xb args =
                     Py.Module.get_function_with_keywords __wrap_namespace "cdist"
                       (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
                       (List.rev_append (Wrap_utils.keyword_args [("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("XA", Some(xa |> Np.Obj.to_pyobject)); ("XB", Some(xb |> Np.Obj.to_pyobject))]) (match kwargs with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let clone ?safe ~estimator () =
   Py.Module.get_function_with_keywords __wrap_namespace "clone"
     [||]
     (Wrap_utils.keyword_args [("safe", Wrap_utils.Option.map safe Py.Bool.of_bool); ("estimator", Some(estimator |> Np.Obj.to_pyobject))])

let gamma ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "gamma"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let kv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "kv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let namedtuple ?rename ?defaults ?module_ ~typename ~field_names () =
   Py.Module.get_function_with_keywords __wrap_namespace "namedtuple"
     [||]
     (Wrap_utils.keyword_args [("rename", rename); ("defaults", defaults); ("module", module_); ("typename", Some(typename )); ("field_names", Some(field_names ))])

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
                  let pdist ?metric ?kwargs ~x args =
                     Py.Module.get_function_with_keywords __wrap_namespace "pdist"
                       (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
                       (List.rev_append (Wrap_utils.keyword_args [("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("X", Some(x |> Np.Obj.to_pyobject))]) (match kwargs with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let signature ?follow_wrapped ~obj () =
   Py.Module.get_function_with_keywords __wrap_namespace "signature"
     [||]
     (Wrap_utils.keyword_args [("follow_wrapped", follow_wrapped); ("obj", Some(obj ))])

let squareform ?force ?checks ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "squareform"
     [||]
     (Wrap_utils.keyword_args [("force", Wrap_utils.Option.map force Py.String.of_string); ("checks", Wrap_utils.Option.map checks Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

end
