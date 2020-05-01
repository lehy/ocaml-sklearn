let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.gaussian_process"

let get_py name = Py.Module.get ns name
module GaussianProcessClassifier = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?kernel ?optimizer ?n_restarts_optimizer ?max_iter_predict ?warm_start ?copy_X_train ?random_state ?multi_class ?n_jobs () =
                     Py.Module.get_function_with_keywords ns "GaussianProcessClassifier"
                       [||]
                       (Wrap_utils.keyword_args [("kernel", kernel); ("optimizer", Wrap_utils.Option.map optimizer (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("n_restarts_optimizer", Wrap_utils.Option.map n_restarts_optimizer Py.Int.of_int); ("max_iter_predict", Wrap_utils.Option.map max_iter_predict Py.Int.of_int); ("warm_start", Wrap_utils.Option.map warm_start Py.Bool.of_bool); ("copy_X_train", Wrap_utils.Option.map copy_X_train Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("multi_class", Wrap_utils.Option.map multi_class Py.String.of_string); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int)])

let fit ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("y", Some(y |> Arr.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let log_marginal_likelihood ?theta ?eval_gradient ?clone_kernel self =
   Py.Module.get_function_with_keywords self "log_marginal_likelihood"
     [||]
     (Wrap_utils.keyword_args [("theta", Wrap_utils.Option.map theta Arr.to_pyobject); ("eval_gradient", Wrap_utils.Option.map eval_gradient Py.Bool.of_bool); ("clone_kernel", Wrap_utils.Option.map clone_kernel Py.Bool.of_bool)])
     |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Arr.of_pyobject (Py.Tuple.get x 1))))
let predict ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let predict_proba ~x self =
   Py.Module.get_function_with_keywords self "predict_proba"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)


let kernel_opt self =
  match Py.Object.get_attr_string self "kernel_" with
  | None -> failwith "attribute kernel_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let kernel_ self = match kernel_opt self with
  | None -> raise Not_found
  | Some x -> x

let log_marginal_likelihood_value_opt self =
  match Py.Object.get_attr_string self "log_marginal_likelihood_value_" with
  | None -> failwith "attribute log_marginal_likelihood_value_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let log_marginal_likelihood_value_ self = match log_marginal_likelihood_value_opt self with
  | None -> raise Not_found
  | Some x -> x

let classes_opt self =
  match Py.Object.get_attr_string self "classes_" with
  | None -> failwith "attribute classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let classes_ self = match classes_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_classes_opt self =
  match Py.Object.get_attr_string self "n_classes_" with
  | None -> failwith "attribute n_classes_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_classes_ self = match n_classes_opt self with
  | None -> raise Not_found
  | Some x -> x
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
| `F x -> Py.Float.of_float x
| `Arr x -> Arr.to_pyobject x
)); ("optimizer", Wrap_utils.Option.map optimizer (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("n_restarts_optimizer", Wrap_utils.Option.map n_restarts_optimizer Py.Int.of_int); ("normalize_y", Wrap_utils.Option.map normalize_y Py.Bool.of_bool); ("copy_X_train", Wrap_utils.Option.map copy_X_train Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])

let fit ~x ~y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("y", Some(y |> Arr.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let log_marginal_likelihood ?theta ?eval_gradient ?clone_kernel self =
   Py.Module.get_function_with_keywords self "log_marginal_likelihood"
     [||]
     (Wrap_utils.keyword_args [("theta", Wrap_utils.Option.map theta Arr.to_pyobject); ("eval_gradient", Wrap_utils.Option.map eval_gradient Py.Bool.of_bool); ("clone_kernel", Wrap_utils.Option.map clone_kernel Py.Bool.of_bool)])
     |> (fun x -> ((Py.Float.to_float (Py.Tuple.get x 0)), (Arr.of_pyobject (Py.Tuple.get x 1))))
let predict ?return_std ?return_cov ~x self =
   Py.Module.get_function_with_keywords self "predict"
     [||]
     (Wrap_utils.keyword_args [("return_std", Wrap_utils.Option.map return_std Py.Bool.of_bool); ("return_cov", Wrap_utils.Option.map return_cov Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let sample_y ?n_samples ?random_state ~x self =
   Py.Module.get_function_with_keywords self "sample_y"
     [||]
     (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("X", Some(x ))])
     |> Arr.of_pyobject
let score ?sample_weight ~x ~y self =
   Py.Module.get_function_with_keywords self "score"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject)); ("y", Some(y |> Arr.to_pyobject))])
     |> Py.Float.to_float
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)


let x_train_opt self =
  match Py.Object.get_attr_string self "X_train_" with
  | None -> failwith "attribute X_train_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let x_train_ self = match x_train_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_train_opt self =
  match Py.Object.get_attr_string self "y_train_" with
  | None -> failwith "attribute y_train_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let y_train_ self = match y_train_opt self with
  | None -> raise Not_found
  | Some x -> x

let kernel_opt self =
  match Py.Object.get_attr_string self "kernel_" with
  | None -> failwith "attribute kernel_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let kernel_ self = match kernel_opt self with
  | None -> raise Not_found
  | Some x -> x

let l_opt self =
  match Py.Object.get_attr_string self "L_" with
  | None -> failwith "attribute L_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let l_ self = match l_opt self with
  | None -> raise Not_found
  | Some x -> x

let alpha_opt self =
  match Py.Object.get_attr_string self "alpha_" with
  | None -> failwith "attribute alpha_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let alpha_ self = match alpha_opt self with
  | None -> raise Not_found
  | Some x -> x

let log_marginal_likelihood_value_opt self =
  match Py.Object.get_attr_string self "log_marginal_likelihood_value_" with
  | None -> failwith "attribute log_marginal_likelihood_value_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let log_marginal_likelihood_value_ self = match log_marginal_likelihood_value_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Kernels = struct
let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.gaussian_process.kernels"

let get_py name = Py.Module.get ns name
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
     (Wrap_utils.keyword_args [("theta", Some(theta |> Arr.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords self "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
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
     (Wrap_utils.keyword_args [("theta", Some(theta |> Arr.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords self "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
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
     (Wrap_utils.keyword_args [("theta", Some(theta |> Arr.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords self "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
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
     (Wrap_utils.keyword_args [("theta", Some(theta |> Arr.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords self "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
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
     (Wrap_utils.keyword_args [("theta", Some(theta |> Arr.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords self "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
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


let name_opt self =
  match Py.Object.get_attr_string self "name" with
  | None -> failwith "attribute name not found"
  | Some x -> if Py.is_none x then None else Some (Py.String.to_string x)

let name self = match name_opt self with
  | None -> raise Not_found
  | Some x -> x

let value_type_opt self =
  match Py.Object.get_attr_string self "value_type" with
  | None -> failwith "attribute value_type not found"
  | Some x -> if Py.is_none x then None else Some (Py.String.to_string x)

let value_type self = match value_type_opt self with
  | None -> raise Not_found
  | Some x -> x

let bounds_opt self =
  match Py.Object.get_attr_string self "bounds" with
  | None -> failwith "attribute bounds not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let bounds self = match bounds_opt self with
  | None -> raise Not_found
  | Some x -> x

let n_elements_opt self =
  match Py.Object.get_attr_string self "n_elements" with
  | None -> failwith "attribute n_elements not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let n_elements self = match n_elements_opt self with
  | None -> raise Not_found
  | Some x -> x

let fixed_opt self =
  match Py.Object.get_attr_string self "fixed" with
  | None -> failwith "attribute fixed not found"
  | Some x -> if Py.is_none x then None else Some (Py.Bool.to_bool x)

let fixed self = match fixed_opt self with
  | None -> raise Not_found
  | Some x -> x
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
| `F x -> Py.Float.of_float x
| `Array_with x -> Wrap_utils.id x
)); ("length_scale_bounds", length_scale_bounds); ("nu", Wrap_utils.Option.map nu Py.Float.of_float)])

let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords self "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Arr.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords self "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
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
     |> Arr.of_pyobject
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
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("pairwise_kernels_kwargs", Wrap_utils.Option.map pairwise_kernels_kwargs Dict.to_pyobject)])

let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords self "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Arr.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords self "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
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
     (Wrap_utils.keyword_args [("theta", Some(theta |> Arr.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords self "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
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
| `F x -> Py.Float.of_float x
| `Array_with x -> Wrap_utils.id x
)); ("length_scale_bounds", length_scale_bounds)])

let clone_with_theta ~theta self =
   Py.Module.get_function_with_keywords self "clone_with_theta"
     [||]
     (Wrap_utils.keyword_args [("theta", Some(theta |> Arr.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords self "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
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
     (Wrap_utils.keyword_args [("theta", Some(theta |> Arr.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords self "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
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
     (Wrap_utils.keyword_args [("theta", Some(theta |> Arr.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords self "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
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
     (Wrap_utils.keyword_args [("theta", Some(theta |> Arr.to_pyobject))])

let diag ~x self =
   Py.Module.get_function_with_keywords self "diag"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
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
     (List.rev_append (Wrap_utils.keyword_args [("metric", metric); ("XA", Some(xa |> Arr.to_pyobject)); ("XB", Some(xb ))]) (match kwargs with None -> [] | Some x -> x))
     |> Arr.of_pyobject
                  let clone ?safe ~estimator () =
                     Py.Module.get_function_with_keywords ns "clone"
                       [||]
                       (Wrap_utils.keyword_args [("safe", Wrap_utils.Option.map safe Py.Bool.of_bool); ("estimator", Some(estimator |> (function
| `Estimator x -> Wrap_utils.id x
| `Arr x -> Arr.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])

let namedtuple ?rename ?defaults ?module_ ~typename ~field_names () =
   Py.Module.get_function_with_keywords ns "namedtuple"
     [||]
     (Wrap_utils.keyword_args [("rename", rename); ("defaults", defaults); ("module", module_); ("typename", Some(typename )); ("field_names", Some(field_names ))])

                  let pairwise_kernels ?y ?metric ?filter_params ?n_jobs ?kwds ~x () =
                     Py.Module.get_function_with_keywords ns "pairwise_kernels"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Arr.to_pyobject); ("metric", Wrap_utils.Option.map metric (function
| `S x -> Py.String.of_string x
| `Callable x -> Wrap_utils.id x
)); ("filter_params", Wrap_utils.Option.map filter_params Py.Bool.of_bool); ("n_jobs", Wrap_utils.Option.map n_jobs Py.Int.of_int); ("X", Some(x |> (function
| `Arr x -> Arr.to_pyobject x
| `Otherwise x -> Wrap_utils.id x
)))]) (match kwds with None -> [] | Some x -> x))

let pdist ?metric ?kwargs ~x args =
   Py.Module.get_function_with_keywords ns "pdist"
     (Wrap_utils.pos_arg Wrap_utils.id args)
     (List.rev_append (Wrap_utils.keyword_args [("metric", metric); ("X", Some(x |> Arr.to_pyobject))]) (match kwargs with None -> [] | Some x -> x))
     |> Arr.of_pyobject
let signature ?follow_wrapped ~obj () =
   Py.Module.get_function_with_keywords ns "signature"
     [||]
     (Wrap_utils.keyword_args [("follow_wrapped", follow_wrapped); ("obj", Some(obj ))])

let squareform ?force ?checks ~x () =
   Py.Module.get_function_with_keywords ns "squareform"
     [||]
     (Wrap_utils.keyword_args [("force", force); ("checks", checks); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

end
