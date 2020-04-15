let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.gaussian_process"

module GaussianProcessClassifier = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?kernel ?optimizer ?n_restarts_optimizer ?max_iter_predict ?warm_start ?copy_X_train ?random_state ?multi_class ?n_jobs () =
                     Py.Module.get_function_with_keywords ns "GaussianProcessClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("kernel", kernel); ("optimizer", Wrap_utils.Option.map optimizer (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("n_restarts_optimizer", Wrap_utils.Option.map n_restarts_optimizer Py.Int.of_int); ("max_iter_predict", Wrap_utils.Option.map max_iter_predict Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("copy_X_train", Wrap_utils.Option.map copy_X_train Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("multi_class", Wrap_utils.Option.map multi_class Py.String.of_string); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
))])

let fit ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("y", Some(y |> Ndarray.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

                  let log_marginal_likelihood ?theta ?eval_gradient ?clone_kernel self =
                     Py.Module.get_function_with_keywords self "log_marginal_likelihood"
                       [||]
                       (Wrap_utils.keyword_args [("theta", Wrap_utils.Option.map theta (function
| `Ndarray x -> Ndarray.to_pyobject x
| `None -> Py.String.of_string "None"
)); ("eval_gradient", Wrap_utils.Option.map eval_gradient Py.Bool.of_bool); ("clone_kernel", Wrap_utils.Option.map clone_kernel Py.Bool.of_bool)])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> Ndarray.of_pyobject
let predict_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
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

let kernel_ self =
  match Py.Object.get_attr_string self "kernel_" with
| None -> raise (Wrap_utils.Attribute_not_found "kernel_")
| Some x -> Wrap_utils.id x
let log_marginal_likelihood_value_ self =
  match Py.Object.get_attr_string self "log_marginal_likelihood_value_" with
| None -> raise (Wrap_utils.Attribute_not_found "log_marginal_likelihood_value_")
| Some x -> Py.Float.to_float x
let classes_ self =
  match Py.Object.get_attr_string self "classes_" with
| None -> raise (Wrap_utils.Attribute_not_found "classes_")
| Some x -> Ndarray.of_pyobject x
let n_classes_ self =
  match Py.Object.get_attr_string self "n_classes_" with
| None -> raise (Wrap_utils.Attribute_not_found "n_classes_")
| Some x -> Py.Int.to_int x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module GaussianProcessRegressor = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?kernel ?alpha ?optimizer ?n_restarts_optimizer ?normalize_y ?copy_X_train ?random_state () =
                     Py.Module.get_function_with_keywords ns "GaussianProcessRegressor"
                       [||]
                       (Wrap_utils.keyword_args [("kernel", kernel); ("alpha", Wrap_utils.Option.map alpha (function
| `Float x -> Py.Float.of_float x
| `Ndarray x -> Ndarray.to_pyobject x
)); ("optimizer", Wrap_utils.Option.map optimizer (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("n_restarts_optimizer", Wrap_utils.Option.map n_restarts_optimizer Py.Int.of_int); ("normalize_y", Wrap_utils.Option.map normalize_y Py.Bool.of_bool); ("copy_X_train", Wrap_utils.Option.map copy_X_train Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])

let fit ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("y", Some(y |> Ndarray.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

                  let log_marginal_likelihood ?theta ?eval_gradient ?clone_kernel self =
                     Py.Module.get_function_with_keywords self "log_marginal_likelihood"
                       [||]
                       (Wrap_utils.keyword_args [("theta", Wrap_utils.Option.map theta (function
| `Ndarray x -> Ndarray.to_pyobject x
| `None -> Py.String.of_string "None"
)); ("eval_gradient", Wrap_utils.Option.map eval_gradient Py.Bool.of_bool); ("clone_kernel", Wrap_utils.Option.map clone_kernel Py.Bool.of_bool)])
                       |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
let predict ?return_std ?return_cov ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("return_std", Wrap_utils.Option.map return_std Py.Bool.of_bool); ("return_cov", Wrap_utils.Option.map return_cov Py.Bool.of_bool); ("X", Some(x ))])
     |> Ndarray.of_pyobject
                  let sample_y ?n_samples ?random_state ~x self =
                     Py.Module.get_function_with_keywords self "sample_y"
                       [||]
                       (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("X", Some(x ))])
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

let x_train_ self =
  match Py.Object.get_attr_string self "X_train_" with
| None -> raise (Wrap_utils.Attribute_not_found "X_train_")
| Some x -> Wrap_utils.id x
let y_train_ self =
  match Py.Object.get_attr_string self "y_train_" with
| None -> raise (Wrap_utils.Attribute_not_found "y_train_")
| Some x -> Ndarray.of_pyobject x
let kernel_ self =
  match Py.Object.get_attr_string self "kernel_" with
| None -> raise (Wrap_utils.Attribute_not_found "kernel_")
| Some x -> Wrap_utils.id x
let l_ self =
  match Py.Object.get_attr_string self "L_" with
| None -> raise (Wrap_utils.Attribute_not_found "L_")
| Some x -> Ndarray.of_pyobject x
let alpha_ self =
  match Py.Object.get_attr_string self "alpha_" with
| None -> raise (Wrap_utils.Attribute_not_found "alpha_")
| Some x -> Ndarray.of_pyobject x
let log_marginal_likelihood_value_ self =
  match Py.Object.get_attr_string self "log_marginal_likelihood_value_" with
| None -> raise (Wrap_utils.Attribute_not_found "log_marginal_likelihood_value_")
| Some x -> Py.Float.to_float x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Kernels = struct
let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.gaussian_process.kernels"

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
module CompoundKernel = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ~kernels () =
   Py.Module.get_function_with_keywords ns "CompoundKernel"
     [||]
     (Wrap_utils.keyword_args [("kernels", Some(kernels ))])

let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords self "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Ndarray.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords self "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let is_stationary self =
   Py.Module.get_function_with_keywords self "is_stationary"
     [||]
     []

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ConstantKernel = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?constant_value ?constant_value_bounds () =
   Py.Module.get_function_with_keywords ns "ConstantKernel"
     [||]
     (Wrap_utils.keyword_args [("constant_value", Wrap_utils.Option.map constant_value Py.Float.of_float); ("constant_value_bounds", constant_value_bounds)])

let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords self "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Ndarray.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords self "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let is_stationary self =
   Py.Module.get_function_with_keywords self "is_stationary"
     [||]
     []

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module DotProduct = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?sigma_0 ?sigma_0_bounds () =
   Py.Module.get_function_with_keywords ns "DotProduct"
     [||]
     (Wrap_utils.keyword_args [("sigma_0", Wrap_utils.Option.map sigma_0 Py.Float.of_float); ("sigma_0_bounds", sigma_0_bounds)])

let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords self "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Ndarray.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords self "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let is_stationary self =
   Py.Module.get_function_with_keywords self "is_stationary"
     [||]
     []

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ExpSineSquared = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?length_scale ?periodicity ?length_scale_bounds ?periodicity_bounds () =
   Py.Module.get_function_with_keywords ns "ExpSineSquared"
     [||]
     (Wrap_utils.keyword_args [("length_scale", Wrap_utils.Option.map length_scale Py.Float.of_float); ("periodicity", Wrap_utils.Option.map periodicity Py.Float.of_float); ("length_scale_bounds", length_scale_bounds); ("periodicity_bounds", periodicity_bounds)])

let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords self "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Ndarray.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords self "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let is_stationary self =
   Py.Module.get_function_with_keywords self "is_stationary"
     [||]
     []

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Exponentiation = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ~kernel ~exponent () =
   Py.Module.get_function_with_keywords ns "Exponentiation"
     [||]
     (Wrap_utils.keyword_args [("kernel", Some(kernel )); ("exponent", Some(exponent |> Py.Float.of_float))])

let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords self "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Ndarray.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords self "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let is_stationary self =
   Py.Module.get_function_with_keywords self "is_stationary"
     [||]
     []

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module GenericKernelMixin = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create () =
   Py.Module.get_function_with_keywords ns "GenericKernelMixin"
     [||]
     []

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Hyperparameter = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?n_elements ?fixed ~name ~value_type ~bounds () =
   Py.Module.get_function_with_keywords ns "Hyperparameter"
     [||]
     (Wrap_utils.keyword_args [("n_elements", n_elements); ("fixed", fixed); ("name", Some(name )); ("value_type", Some(value_type )); ("bounds", Some(bounds ))])

let get_item ~key self =
   Py.Module.get_function_with_keywords self "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key ))])

let count ~value self =
   Py.Module.get_function_with_keywords self "count"
     [||]
     (Wrap_utils.keyword_args [("value", Some(value ))])

let index ?start ?stop ~value self =
   Py.Module.get_function_with_keywords self "index"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("stop", stop); ("value", Some(value ))])

let name self =
  match Py.Object.get_attr_string self "name" with
| None -> raise (Wrap_utils.Attribute_not_found "name")
| Some x -> Py.String.to_string x
let value_type self =
  match Py.Object.get_attr_string self "value_type" with
| None -> raise (Wrap_utils.Attribute_not_found "value_type")
| Some x -> Py.String.to_string x
let bounds self =
  match Py.Object.get_attr_string self "bounds" with
| None -> raise (Wrap_utils.Attribute_not_found "bounds")
| Some x -> Wrap_utils.id x
let n_elements self =
  match Py.Object.get_attr_string self "n_elements" with
| None -> raise (Wrap_utils.Attribute_not_found "n_elements")
| Some x -> Py.Int.to_int x
let fixed self =
  match Py.Object.get_attr_string self "fixed" with
| None -> raise (Wrap_utils.Attribute_not_found "fixed")
| Some x -> Py.Bool.to_bool x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Matern = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?length_scale ?length_scale_bounds ?nu () =
                     Py.Module.get_function_with_keywords ns "Matern"
                       [||]
                       (Wrap_utils.keyword_args [("length_scale", Wrap_utils.Option.map length_scale (function
| `Float x -> Py.Float.of_float x
| `PyObject x -> Wrap_utils.id x
)); ("length_scale_bounds", length_scale_bounds); ("nu", Wrap_utils.Option.map nu Py.Float.of_float)])

let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords self "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Ndarray.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords self "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let is_stationary self =
   Py.Module.get_function_with_keywords self "is_stationary"
     [||]
     []

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module NormalizedKernelMixin = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create () =
   Py.Module.get_function_with_keywords ns "NormalizedKernelMixin"
     [||]
     []

let diag ~x self =
   Py.Module.get_function_with_keywords self "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> Ndarray.of_pyobject
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module PairwiseKernel = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?gamma ?gamma_bounds ?metric ?pairwise_kernels_kwargs () =
                     Py.Module.get_function_with_keywords ns "PairwiseKernel"
                       [||]
                       (Wrap_utils.keyword_args [("gamma", Wrap_utils.Option.map gamma Py.Float.of_float); ("gamma_bounds", gamma_bounds); ("metric", Wrap_utils.Option.map metric (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("pairwise_kernels_kwargs", pairwise_kernels_kwargs)])

let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords self "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Ndarray.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords self "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let is_stationary self =
   Py.Module.get_function_with_keywords self "is_stationary"
     [||]
     []

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Product = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ~k1 ~k2 () =
   Py.Module.get_function_with_keywords ns "Product"
     [||]
     (Wrap_utils.keyword_args [("k1", Some(k1 )); ("k2", Some(k2 ))])

let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords self "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Ndarray.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords self "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let is_stationary self =
   Py.Module.get_function_with_keywords self "is_stationary"
     [||]
     []

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RBF = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?length_scale ?length_scale_bounds () =
                     Py.Module.get_function_with_keywords ns "RBF"
                       [||]
                       (Wrap_utils.keyword_args [("length_scale", Wrap_utils.Option.map length_scale (function
| `Float x -> Py.Float.of_float x
| `PyObject x -> Wrap_utils.id x
)); ("length_scale_bounds", length_scale_bounds)])

let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords self "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Ndarray.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords self "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let is_stationary self =
   Py.Module.get_function_with_keywords self "is_stationary"
     [||]
     []

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RationalQuadratic = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?length_scale ?alpha ?length_scale_bounds ?alpha_bounds () =
   Py.Module.get_function_with_keywords ns "RationalQuadratic"
     [||]
     (Wrap_utils.keyword_args [("length_scale", Wrap_utils.Option.map length_scale Py.Float.of_float); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("length_scale_bounds", length_scale_bounds); ("alpha_bounds", alpha_bounds)])

let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords self "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Ndarray.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords self "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let is_stationary self =
   Py.Module.get_function_with_keywords self "is_stationary"
     [||]
     []

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module StationaryKernelMixin = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create () =
   Py.Module.get_function_with_keywords ns "StationaryKernelMixin"
     [||]
     []

let is_stationary self =
   Py.Module.get_function_with_keywords self "is_stationary"
     [||]
     []

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Sum = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ~k1 ~k2 () =
   Py.Module.get_function_with_keywords ns "Sum"
     [||]
     (Wrap_utils.keyword_args [("k1", Some(k1 )); ("k2", Some(k2 ))])

let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords self "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Ndarray.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords self "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let is_stationary self =
   Py.Module.get_function_with_keywords self "is_stationary"
     [||]
     []

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module WhiteKernel = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?noise_level ?noise_level_bounds () =
   Py.Module.get_function_with_keywords ns "WhiteKernel"
     [||]
     (Wrap_utils.keyword_args [("noise_level", Wrap_utils.Option.map noise_level Py.Float.of_float); ("noise_level_bounds", noise_level_bounds)])

let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords self "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Ndarray.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords self "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let is_stationary self =
   Py.Module.get_function_with_keywords self "is_stationary"
     [||]
     []

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let abstractmethod ~funcobj () =
   Py.Module.get_function_with_keywords ns "abstractmethod"
     [||]
     (Wrap_utils.keyword_args [("funcobj", Some(funcobj ))])

let cdist ?metric ?kwargs ~xa ~xb args =
   Py.Module.get_function_with_keywords ns "cdist"
     (Wrap_utils.pos_arg Wrap_utils.id args)
     (List.rev_append (Wrap_utils.keyword_args [("metric", metric); ("XA", Some(xa |> Ndarray.to_pyobject)); ("XB", Some(xb ))]) (match kwargs with None -> [] | Some x -> x))
     |> Ndarray.of_pyobject
                  let clone ?safe ~estimator () =
                     Py.Module.get_function_with_keywords ns "clone"
                       [||]
                       (Wrap_utils.keyword_args [("safe", Wrap_utils.Option.map safe Py.Bool.of_bool); ("estimator", Some(estimator |> (function
| `Estimator x -> Wrap_utils.id x
| `ArrayLike x -> Wrap_utils.id x
| `PyObject x -> Wrap_utils.id x
)))])

let namedtuple ?rename ?defaults ?module_ ~typename ~field_names () =
   Py.Module.get_function_with_keywords ns "namedtuple"
     [||]
     (Wrap_utils.keyword_args [("rename", rename); ("defaults", defaults); ("module", module_); ("typename", Some(typename )); ("field_names", Some(field_names ))])

                  let pairwise_kernels ?y ?metric ?filter_params ?n_jobs ?kwds ~x () =
                     Py.Module.get_function_with_keywords ns "pairwise_kernels"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("metric", Wrap_utils.Option.map metric (function
| `String x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("filter_params", Wrap_utils.Option.map filter_params Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))]) (match kwds with None -> [] | Some x -> x))

let pdist ?metric ?kwargs ~x args =
   Py.Module.get_function_with_keywords ns "pdist"
     (Wrap_utils.pos_arg Wrap_utils.id args)
     (List.rev_append (Wrap_utils.keyword_args [("metric", metric); ("X", Some(x |> Ndarray.to_pyobject))]) (match kwargs with None -> [] | Some x -> x))
     |> Ndarray.of_pyobject
let signature ?follow_wrapped ~obj () =
   Py.Module.get_function_with_keywords ns "signature"
     [||]
     (Wrap_utils.keyword_args [("follow_wrapped", follow_wrapped); ("obj", Some(obj ))])

let squareform ?force ?checks ~x () =
   Py.Module.get_function_with_keywords ns "squareform"
     [||]
     (Wrap_utils.keyword_args [("force", force); ("checks", checks); ("X", Some(x |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject

end
