let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.utils"

module Bunch = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?kwargs () =
   Py.Module.get_function_with_keywords ns "Bunch"
     [||]
     (match kwargs with None -> [] | Some x -> x)

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Path = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?kwargs args =
   Py.Module.get_function_with_keywords ns "Path"
     (Wrap_utils.pos_arg Wrap_utils.id args)
     (match kwargs with None -> [] | Some x -> x)

let absolute self =
   Py.Module.get_function_with_keywords self "absolute"
     [||]
     []

let as_posix self =
   Py.Module.get_function_with_keywords self "as_posix"
     [||]
     []

let as_uri self =
   Py.Module.get_function_with_keywords self "as_uri"
     [||]
     []

let chmod ~mode self =
   Py.Module.get_function_with_keywords self "chmod"
     [||]
     (Wrap_utils.keyword_args [("mode", Some(mode ))])

let cwd self =
   Py.Module.get_function_with_keywords self "cwd"
     [||]
     []

let exists self =
   Py.Module.get_function_with_keywords self "exists"
     [||]
     []

let expanduser self =
   Py.Module.get_function_with_keywords self "expanduser"
     [||]
     []

let glob ~pattern self =
   Py.Module.get_function_with_keywords self "glob"
     [||]
     (Wrap_utils.keyword_args [("pattern", Some(pattern ))])

let group self =
   Py.Module.get_function_with_keywords self "group"
     [||]
     []

let home self =
   Py.Module.get_function_with_keywords self "home"
     [||]
     []

let is_absolute self =
   Py.Module.get_function_with_keywords self "is_absolute"
     [||]
     []

let is_block_device self =
   Py.Module.get_function_with_keywords self "is_block_device"
     [||]
     []

let is_char_device self =
   Py.Module.get_function_with_keywords self "is_char_device"
     [||]
     []

let is_dir self =
   Py.Module.get_function_with_keywords self "is_dir"
     [||]
     []

let is_fifo self =
   Py.Module.get_function_with_keywords self "is_fifo"
     [||]
     []

let is_file self =
   Py.Module.get_function_with_keywords self "is_file"
     [||]
     []

let is_mount self =
   Py.Module.get_function_with_keywords self "is_mount"
     [||]
     []

let is_reserved self =
   Py.Module.get_function_with_keywords self "is_reserved"
     [||]
     []

let is_socket self =
   Py.Module.get_function_with_keywords self "is_socket"
     [||]
     []

let is_symlink self =
   Py.Module.get_function_with_keywords self "is_symlink"
     [||]
     []

let iterdir self =
   Py.Module.get_function_with_keywords self "iterdir"
     [||]
     []

let joinpath args self =
   Py.Module.get_function_with_keywords self "joinpath"
     (Wrap_utils.pos_arg Wrap_utils.id args)
     []

let lchmod ~mode self =
   Py.Module.get_function_with_keywords self "lchmod"
     [||]
     (Wrap_utils.keyword_args [("mode", Some(mode ))])

let lstat self =
   Py.Module.get_function_with_keywords self "lstat"
     [||]
     []

let match_ ~path_pattern self =
   Py.Module.get_function_with_keywords self "match"
     [||]
     (Wrap_utils.keyword_args [("path_pattern", Some(path_pattern ))])

let mkdir ?mode ?parents ?exist_ok self =
   Py.Module.get_function_with_keywords self "mkdir"
     [||]
     (Wrap_utils.keyword_args [("mode", mode); ("parents", parents); ("exist_ok", exist_ok)])

let open_ ?mode ?buffering ?encoding ?errors ?newline self =
   Py.Module.get_function_with_keywords self "open"
     [||]
     (Wrap_utils.keyword_args [("mode", mode); ("buffering", buffering); ("encoding", encoding); ("errors", errors); ("newline", newline)])

let owner self =
   Py.Module.get_function_with_keywords self "owner"
     [||]
     []

let read_bytes self =
   Py.Module.get_function_with_keywords self "read_bytes"
     [||]
     []

let read_text ?encoding ?errors self =
   Py.Module.get_function_with_keywords self "read_text"
     [||]
     (Wrap_utils.keyword_args [("encoding", encoding); ("errors", errors)])

let relative_to other self =
   Py.Module.get_function_with_keywords self "relative_to"
     (Wrap_utils.pos_arg Wrap_utils.id other)
     []

let rename ~target self =
   Py.Module.get_function_with_keywords self "rename"
     [||]
     (Wrap_utils.keyword_args [("target", Some(target ))])

let replace ~target self =
   Py.Module.get_function_with_keywords self "replace"
     [||]
     (Wrap_utils.keyword_args [("target", Some(target ))])

let resolve ?strict self =
   Py.Module.get_function_with_keywords self "resolve"
     [||]
     (Wrap_utils.keyword_args [("strict", strict)])

let rglob ~pattern self =
   Py.Module.get_function_with_keywords self "rglob"
     [||]
     (Wrap_utils.keyword_args [("pattern", Some(pattern ))])

let rmdir self =
   Py.Module.get_function_with_keywords self "rmdir"
     [||]
     []

let samefile ~other_path self =
   Py.Module.get_function_with_keywords self "samefile"
     [||]
     (Wrap_utils.keyword_args [("other_path", Some(other_path ))])

let stat self =
   Py.Module.get_function_with_keywords self "stat"
     [||]
     []

let symlink_to ?target_is_directory ~target self =
   Py.Module.get_function_with_keywords self "symlink_to"
     [||]
     (Wrap_utils.keyword_args [("target_is_directory", target_is_directory); ("target", Some(target ))])

let touch ?mode ?exist_ok self =
   Py.Module.get_function_with_keywords self "touch"
     [||]
     (Wrap_utils.keyword_args [("mode", mode); ("exist_ok", exist_ok)])

let unlink self =
   Py.Module.get_function_with_keywords self "unlink"
     [||]
     []

let with_name ~name self =
   Py.Module.get_function_with_keywords self "with_name"
     [||]
     (Wrap_utils.keyword_args [("name", Some(name ))])

let with_suffix ~suffix self =
   Py.Module.get_function_with_keywords self "with_suffix"
     [||]
     (Wrap_utils.keyword_args [("suffix", Some(suffix ))])

let write_bytes ~data self =
   Py.Module.get_function_with_keywords self "write_bytes"
     [||]
     (Wrap_utils.keyword_args [("data", Some(data ))])

let write_text ?encoding ?errors ~data self =
   Py.Module.get_function_with_keywords self "write_text"
     [||]
     (Wrap_utils.keyword_args [("encoding", encoding); ("errors", errors); ("data", Some(data ))])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let all_estimators ?include_meta_estimators ?include_other ?type_filter ?include_dont_test () =
                     Py.Module.get_function_with_keywords ns "all_estimators"
                       [||]
                       (Wrap_utils.keyword_args [("include_meta_estimators", Wrap_utils.Option.map include_meta_estimators Py.Bool.of_bool); ("include_other", Wrap_utils.Option.map include_other Py.Bool.of_bool); ("type_filter", Wrap_utils.Option.map type_filter (function
| `String x -> Py.String.of_string x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `None -> Py.String.of_string "None"
)); ("include_dont_test", Wrap_utils.Option.map include_dont_test Py.Bool.of_bool)])

module Arrayfuncs = struct
let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.utils.arrayfuncs"

let cholesky_delete ~l ~go_out () =
   Py.Module.get_function_with_keywords ns "cholesky_delete"
     [||]
     (Wrap_utils.keyword_args [("L", Some(l )); ("go_out", Some(go_out ))])


end
                  let as_float_array ?copy ?force_all_finite ~x () =
                     Py.Module.get_function_with_keywords ns "as_float_array"
                       [||]
                       (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("force_all_finite", Wrap_utils.Option.map force_all_finite (function
| `Bool x -> Py.Bool.of_bool x
| `Allow_nan -> Py.String.of_string "allow-nan"
)); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

                  let assert_all_finite ?allow_nan ~x () =
                     Py.Module.get_function_with_keywords ns "assert_all_finite"
                       [||]
                       (Wrap_utils.keyword_args [("allow_nan", Wrap_utils.Option.map allow_nan Py.Bool.of_bool); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

                  let axis0_safe_slice ~x ~mask ~len_mask () =
                     Py.Module.get_function_with_keywords ns "axis0_safe_slice"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
))); ("mask", Some(mask |> Ndarray.to_pyobject)); ("len_mask", Some(len_mask |> Py.Int.of_int))])

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

let check_matplotlib_support ~caller_name () =
   Py.Module.get_function_with_keywords ns "check_matplotlib_support"
     [||]
     (Wrap_utils.keyword_args [("caller_name", Some(caller_name |> Py.String.of_string))])

let check_pandas_support ~caller_name () =
   Py.Module.get_function_with_keywords ns "check_pandas_support"
     [||]
     (Wrap_utils.keyword_args [("caller_name", Some(caller_name |> Py.String.of_string))])

                  let check_random_state ~seed () =
                     Py.Module.get_function_with_keywords ns "check_random_state"
                       [||]
                       (Wrap_utils.keyword_args [("seed", Some(seed |> (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)))])

                  let check_scalar ?min_val ?max_val ~x ~name ~target_type () =
                     Py.Module.get_function_with_keywords ns "check_scalar"
                       [||]
                       (Wrap_utils.keyword_args [("min_val", Wrap_utils.Option.map min_val (function
| `Float x -> Py.Float.of_float x
| `Int x -> Py.Int.of_int x
)); ("max_val", Wrap_utils.Option.map max_val (function
| `Float x -> Py.Float.of_float x
| `Int x -> Py.Int.of_int x
)); ("x", Some(x )); ("name", Some(name |> Py.String.of_string)); ("target_type", Some(target_type |> (function
| `Dtype x -> Wrap_utils.id x
| `PyObject x -> Wrap_utils.id x
)))])

                  let check_symmetric ?tol ?raise_warning ?raise_exception ~array () =
                     Py.Module.get_function_with_keywords ns "check_symmetric"
                       [||]
                       (Wrap_utils.keyword_args [("tol", tol); ("raise_warning", raise_warning); ("raise_exception", raise_exception); ("array", Some(array |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

module Class_weight = struct
let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.utils.class_weight"

                  let compute_class_weight ~class_weight ~classes ~y () =
                     Py.Module.get_function_with_keywords ns "compute_class_weight"
                       [||]
                       (Wrap_utils.keyword_args [("class_weight", Some(class_weight |> (function
| `DictIntToFloat x -> (Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float) x
| `Balanced -> Py.String.of_string "balanced"
| `None -> Py.String.of_string "None"
))); ("classes", Some(classes |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])
                       |> Ndarray.of_pyobject
                  let compute_sample_weight ?indices ~class_weight ~y () =
                     Py.Module.get_function_with_keywords ns "compute_sample_weight"
                       [||]
                       (Wrap_utils.keyword_args [("indices", Wrap_utils.Option.map indices (function
| `Ndarray x -> Ndarray.to_pyobject x
| `None -> Py.String.of_string "None"
)); ("class_weight", Some(class_weight |> (function
| `DictIntToFloat x -> (Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float) x
| `Balanced -> Py.String.of_string "balanced"
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
))); ("y", Some(y |> Ndarray.to_pyobject))])
                       |> Ndarray.of_pyobject

end
let column_or_1d ?warn ~y () =
   Py.Module.get_function_with_keywords ns "column_or_1d"
     [||]
     (Wrap_utils.keyword_args [("warn", Wrap_utils.Option.map warn Py.Bool.of_bool); ("y", Some(y |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
                  let compute_class_weight ~class_weight ~classes ~y () =
                     Py.Module.get_function_with_keywords ns "compute_class_weight"
                       [||]
                       (Wrap_utils.keyword_args [("class_weight", Some(class_weight |> (function
| `DictIntToFloat x -> (Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float) x
| `Balanced -> Py.String.of_string "balanced"
| `None -> Py.String.of_string "None"
))); ("classes", Some(classes |> Ndarray.to_pyobject)); ("y", Some(y |> Ndarray.to_pyobject))])
                       |> Ndarray.of_pyobject
                  let compute_sample_weight ?indices ~class_weight ~y () =
                     Py.Module.get_function_with_keywords ns "compute_sample_weight"
                       [||]
                       (Wrap_utils.keyword_args [("indices", Wrap_utils.Option.map indices (function
| `Ndarray x -> Ndarray.to_pyobject x
| `None -> Py.String.of_string "None"
)); ("class_weight", Some(class_weight |> (function
| `DictIntToFloat x -> (Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float) x
| `Balanced -> Py.String.of_string "balanced"
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
))); ("y", Some(y |> Ndarray.to_pyobject))])
                       |> Ndarray.of_pyobject
let contextmanager ~func () =
   Py.Module.get_function_with_keywords ns "contextmanager"
     [||]
     (Wrap_utils.keyword_args [("func", Some(func ))])

let cpu_count () =
   Py.Module.get_function_with_keywords ns "cpu_count"
     [||]
     []

let delayed ?check_pickle ~function_ () =
   Py.Module.get_function_with_keywords ns "delayed"
     [||]
     (Wrap_utils.keyword_args [("check_pickle", check_pickle); ("function", Some(function_ ))])

let deprecate ~obj () =
   Py.Module.get_function_with_keywords ns "deprecate"
     [||]
     (Wrap_utils.keyword_args [("obj", Some(obj ))])

module Deprecated = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?extra () =
   Py.Module.get_function_with_keywords ns "deprecated"
     [||]
     (Wrap_utils.keyword_args [("extra", Wrap_utils.Option.map extra Py.String.of_string)])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Deprecation = struct
let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.utils.deprecation"

module Deprecated = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?extra () =
   Py.Module.get_function_with_keywords ns "deprecated"
     [||]
     (Wrap_utils.keyword_args [("extra", Wrap_utils.Option.map extra Py.String.of_string)])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end

end
let effective_n_jobs ?n_jobs () =
   Py.Module.get_function_with_keywords ns "effective_n_jobs"
     [||]
     (Wrap_utils.keyword_args [("n_jobs", n_jobs)])

module Extmath = struct
let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.utils.extmath"

let cartesian ?out ~arrays () =
   Py.Module.get_function_with_keywords ns "cartesian"
     [||]
     (Wrap_utils.keyword_args [("out", out); ("arrays", Some(arrays ))])
     |> Ndarray.of_pyobject
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

                  let check_random_state ~seed () =
                     Py.Module.get_function_with_keywords ns "check_random_state"
                       [||]
                       (Wrap_utils.keyword_args [("seed", Some(seed |> (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)))])

let density ?kwargs ~w () =
   Py.Module.get_function_with_keywords ns "density"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("w", Some(w |> Ndarray.to_pyobject))]) (match kwargs with None -> [] | Some x -> x))

module Deprecated = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?extra () =
   Py.Module.get_function_with_keywords ns "deprecated"
     [||]
     (Wrap_utils.keyword_args [("extra", Wrap_utils.Option.map extra Py.String.of_string)])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let fast_logdet ~a () =
   Py.Module.get_function_with_keywords ns "fast_logdet"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Ndarray.to_pyobject))])

                  let log_logistic ?out ~x () =
                     Py.Module.get_function_with_keywords ns "log_logistic"
                       [||]
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("X", Some(x |> Ndarray.to_pyobject))])
                       |> Ndarray.of_pyobject
let make_nonnegative ?min_value ~x () =
   Py.Module.get_function_with_keywords ns "make_nonnegative"
     [||]
     (Wrap_utils.keyword_args [("min_value", min_value); ("X", Some(x |> Ndarray.to_pyobject))])

                  let randomized_range_finder ?power_iteration_normalizer ?random_state ~a ~size ~n_iter () =
                     Py.Module.get_function_with_keywords ns "randomized_range_finder"
                       [||]
                       (Wrap_utils.keyword_args [("power_iteration_normalizer", Wrap_utils.Option.map power_iteration_normalizer Py.String.of_string); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("A", Some(a )); ("size", Some(size |> Py.Int.of_int)); ("n_iter", Some(n_iter |> Py.Int.of_int))])

                  let randomized_svd ?n_oversamples ?n_iter ?power_iteration_normalizer ?transpose ?flip_sign ?random_state ~m ~n_components () =
                     Py.Module.get_function_with_keywords ns "randomized_svd"
                       [||]
                       (Wrap_utils.keyword_args [("n_oversamples", n_oversamples); ("n_iter", Wrap_utils.Option.map n_iter (function
| `Int x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)); ("power_iteration_normalizer", Wrap_utils.Option.map power_iteration_normalizer Py.String.of_string); ("transpose", Wrap_utils.Option.map transpose (function
| `Bool x -> Py.Bool.of_bool x
| `Auto -> Py.String.of_string "auto"
)); ("flip_sign", Wrap_utils.Option.map flip_sign (function
| `Bool x -> Py.Bool.of_bool x
| `PyObject x -> Wrap_utils.id x
)); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("M", Some(m |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
))); ("n_components", Some(n_components |> Py.Int.of_int))])

let row_norms ?squared ~x () =
   Py.Module.get_function_with_keywords ns "row_norms"
     [||]
     (Wrap_utils.keyword_args [("squared", squared); ("X", Some(x |> Ndarray.to_pyobject))])

let safe_min ~x () =
   Py.Module.get_function_with_keywords ns "safe_min"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Ndarray.to_pyobject))])

                  let safe_sparse_dot ?dense_output ~a ~b () =
                     Py.Module.get_function_with_keywords ns "safe_sparse_dot"
                       [||]
                       (Wrap_utils.keyword_args [("dense_output", dense_output); ("a", Some(a |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
))); ("b", Some(b ))])

let softmax ?copy ~x () =
   Py.Module.get_function_with_keywords ns "softmax"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x ))])
     |> Ndarray.of_pyobject
let squared_norm ~x () =
   Py.Module.get_function_with_keywords ns "squared_norm"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Ndarray.to_pyobject))])

let stable_cumsum ?axis ?rtol ?atol ~arr () =
   Py.Module.get_function_with_keywords ns "stable_cumsum"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("rtol", rtol); ("atol", atol); ("arr", Some(arr |> Ndarray.to_pyobject))])

let svd_flip ?u_based_decision ~u ~v () =
   Py.Module.get_function_with_keywords ns "svd_flip"
     [||]
     (Wrap_utils.keyword_args [("u_based_decision", Wrap_utils.Option.map u_based_decision Py.Bool.of_bool); ("u", Some(u |> Ndarray.to_pyobject)); ("v", Some(v |> Ndarray.to_pyobject))])

let weighted_mode ?axis ~a ~w () =
   Py.Module.get_function_with_keywords ns "weighted_mode"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("a", Some(a |> Ndarray.to_pyobject)); ("w", Some(w ))])
     |> Ndarray.of_pyobject

end
module Fixes = struct
let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.utils.fixes"

module LooseVersion = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?vstring () =
   Py.Module.get_function_with_keywords ns "LooseVersion"
     [||]
     (Wrap_utils.keyword_args [("vstring", vstring)])

let parse ~vstring self =
   Py.Module.get_function_with_keywords self "parse"
     [||]
     (Wrap_utils.keyword_args [("vstring", Some(vstring ))])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module MaskedArray = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?data ?mask ?dtype ?copy ?subok ?ndmin ?fill_value ?keep_mask ?hard_mask ?shrink ?order ?options () =
   Py.Module.get_function_with_keywords ns "MaskedArray"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("data", Wrap_utils.Option.map data Ndarray.to_pyobject); ("mask", mask); ("dtype", dtype); ("copy", copy); ("subok", subok); ("ndmin", ndmin); ("fill_value", fill_value); ("keep_mask", keep_mask); ("hard_mask", hard_mask); ("shrink", shrink); ("order", order)]) (match options with None -> [] | Some x -> x))

let get_item ~indx self =
   Py.Module.get_function_with_keywords self "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("indx", Some(indx ))])

let all ?axis ?out ?keepdims self =
   Py.Module.get_function_with_keywords self "all"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims)])

let anom ?axis ?dtype self =
   Py.Module.get_function_with_keywords self "anom"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("dtype", dtype)])

let any ?axis ?out ?keepdims self =
   Py.Module.get_function_with_keywords self "any"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("keepdims", keepdims)])

                  let argmax ?axis ?fill_value ?out self =
                     Py.Module.get_function_with_keywords self "argmax"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("fill_value", fill_value); ("out", out)])

                  let argmin ?axis ?fill_value ?out self =
                     Py.Module.get_function_with_keywords self "argmin"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("fill_value", fill_value); ("out", out)])

let argpartition ?kwargs args self =
   Py.Module.get_function_with_keywords self "argpartition"
     (Wrap_utils.pos_arg Wrap_utils.id args)
     (match kwargs with None -> [] | Some x -> x)

                  let argsort ?axis ?kind ?order ?endwith ?fill_value self =
                     Py.Module.get_function_with_keywords self "argsort"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("kind", Wrap_utils.Option.map kind (function
| `Quicksort -> Py.String.of_string "quicksort"
| `Mergesort -> Py.String.of_string "mergesort"
| `Heapsort -> Py.String.of_string "heapsort"
| `Stable -> Py.String.of_string "stable"
)); ("order", order); ("endwith", Wrap_utils.Option.map endwith Py.Bool.of_bool); ("fill_value", fill_value)])

let compress ?axis ?out ~condition self =
   Py.Module.get_function_with_keywords self "compress"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("condition", Some(condition ))])

let compressed self =
   Py.Module.get_function_with_keywords self "compressed"
     [||]
     []
     |> Ndarray.of_pyobject
let copy ?params args self =
   Py.Module.get_function_with_keywords self "copy"
     (Wrap_utils.pos_arg Wrap_utils.id args)
     (match params with None -> [] | Some x -> x)

                  let count ?axis ?keepdims self =
                     Py.Module.get_function_with_keywords self "count"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
)); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool)])

let cumprod ?axis ?dtype ?out self =
   Py.Module.get_function_with_keywords self "cumprod"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out)])

let cumsum ?axis ?dtype ?out self =
   Py.Module.get_function_with_keywords self "cumsum"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out)])

let diagonal ?params args self =
   Py.Module.get_function_with_keywords self "diagonal"
     (Wrap_utils.pos_arg Wrap_utils.id args)
     (match params with None -> [] | Some x -> x)

let dot ?out ?strict ~b self =
   Py.Module.get_function_with_keywords self "dot"
     [||]
     (Wrap_utils.keyword_args [("out", out); ("strict", strict); ("b", Some(b ))])

let filled ?fill_value self =
   Py.Module.get_function_with_keywords self "filled"
     [||]
     (Wrap_utils.keyword_args [("fill_value", Wrap_utils.Option.map fill_value Ndarray.to_pyobject)])
     |> Ndarray.of_pyobject
let flatten ?params args self =
   Py.Module.get_function_with_keywords self "flatten"
     (Wrap_utils.pos_arg Wrap_utils.id args)
     (match params with None -> [] | Some x -> x)
     |> Ndarray.of_pyobject
let get_fill_value self =
   Py.Module.get_function_with_keywords self "get_fill_value"
     [||]
     []

let get_imag self =
   Py.Module.get_function_with_keywords self "get_imag"
     [||]
     []

let get_real self =
   Py.Module.get_function_with_keywords self "get_real"
     [||]
     []

let harden_mask self =
   Py.Module.get_function_with_keywords self "harden_mask"
     [||]
     []

let ids self =
   Py.Module.get_function_with_keywords self "ids"
     [||]
     []

let iscontiguous self =
   Py.Module.get_function_with_keywords self "iscontiguous"
     [||]
     []

                  let max ?axis ?out ?fill_value ?keepdims self =
                     Py.Module.get_function_with_keywords self "max"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("out", out); ("fill_value", fill_value); ("keepdims", keepdims)])
                       |> Ndarray.of_pyobject
let mean ?axis ?dtype ?out ?keepdims self =
   Py.Module.get_function_with_keywords self "mean"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("keepdims", keepdims)])

                  let min ?axis ?out ?fill_value ?keepdims self =
                     Py.Module.get_function_with_keywords self "min"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("out", out); ("fill_value", fill_value); ("keepdims", keepdims)])
                       |> Ndarray.of_pyobject
let mini ?axis self =
   Py.Module.get_function_with_keywords self "mini"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int)])

let nonzero self =
   Py.Module.get_function_with_keywords self "nonzero"
     [||]
     []

let partition ?kwargs args self =
   Py.Module.get_function_with_keywords self "partition"
     (Wrap_utils.pos_arg Wrap_utils.id args)
     (match kwargs with None -> [] | Some x -> x)

let prod ?axis ?dtype ?out ?keepdims self =
   Py.Module.get_function_with_keywords self "prod"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("keepdims", keepdims)])

let product ?axis ?dtype ?out ?keepdims self =
   Py.Module.get_function_with_keywords self "product"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("keepdims", keepdims)])

                  let ptp ?axis ?out ?fill_value ?keepdims self =
                     Py.Module.get_function_with_keywords self "ptp"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("out", out); ("fill_value", fill_value); ("keepdims", keepdims)])
                       |> Ndarray.of_pyobject
let put ?mode ~indices ~values self =
   Py.Module.get_function_with_keywords self "put"
     [||]
     (Wrap_utils.keyword_args [("mode", mode); ("indices", Some(indices )); ("values", Some(values ))])

                  let ravel ?order self =
                     Py.Module.get_function_with_keywords self "ravel"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
| `A -> Py.String.of_string "A"
| `K -> Py.String.of_string "K"
))])

let repeat ?params args self =
   Py.Module.get_function_with_keywords self "repeat"
     (Wrap_utils.pos_arg Wrap_utils.id args)
     (match params with None -> [] | Some x -> x)

let reshape ?kwargs s self =
   Py.Module.get_function_with_keywords self "reshape"
     (Wrap_utils.pos_arg Wrap_utils.id s)
     (match kwargs with None -> [] | Some x -> x)
     |> Ndarray.of_pyobject
let resize ?refcheck ?order ~newshape self =
   Py.Module.get_function_with_keywords self "resize"
     [||]
     (Wrap_utils.keyword_args [("refcheck", refcheck); ("order", order); ("newshape", Some(newshape ))])

let round ?decimals ?out self =
   Py.Module.get_function_with_keywords self "round"
     [||]
     (Wrap_utils.keyword_args [("decimals", decimals); ("out", out)])

let set_fill_value ?value self =
   Py.Module.get_function_with_keywords self "set_fill_value"
     [||]
     (Wrap_utils.keyword_args [("value", value)])

let shrink_mask self =
   Py.Module.get_function_with_keywords self "shrink_mask"
     [||]
     []

let soften_mask self =
   Py.Module.get_function_with_keywords self "soften_mask"
     [||]
     []

let sort ?axis ?kind ?order ?endwith ?fill_value self =
   Py.Module.get_function_with_keywords self "sort"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("kind", kind); ("order", order); ("endwith", endwith); ("fill_value", fill_value)])
     |> Ndarray.of_pyobject
let squeeze ?params args self =
   Py.Module.get_function_with_keywords self "squeeze"
     (Wrap_utils.pos_arg Wrap_utils.id args)
     (match params with None -> [] | Some x -> x)

let std ?axis ?dtype ?out ?ddof ?keepdims self =
   Py.Module.get_function_with_keywords self "std"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("ddof", ddof); ("keepdims", keepdims)])

let sum ?axis ?dtype ?out ?keepdims self =
   Py.Module.get_function_with_keywords self "sum"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", out); ("keepdims", keepdims)])

let swapaxes ?params args self =
   Py.Module.get_function_with_keywords self "swapaxes"
     (Wrap_utils.pos_arg Wrap_utils.id args)
     (match params with None -> [] | Some x -> x)

let take ?axis ?out ?mode ~indices self =
   Py.Module.get_function_with_keywords self "take"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("out", out); ("mode", mode); ("indices", Some(indices ))])

let tobytes ?fill_value ?order self =
   Py.Module.get_function_with_keywords self "tobytes"
     [||]
     (Wrap_utils.keyword_args [("fill_value", fill_value); ("order", order)])

let tofile ?sep ?format ~fid self =
   Py.Module.get_function_with_keywords self "tofile"
     [||]
     (Wrap_utils.keyword_args [("sep", sep); ("format", format); ("fid", Some(fid ))])

let toflex self =
   Py.Module.get_function_with_keywords self "toflex"
     [||]
     []
     |> Ndarray.of_pyobject
let tolist ?fill_value self =
   Py.Module.get_function_with_keywords self "tolist"
     [||]
     (Wrap_utils.keyword_args [("fill_value", fill_value)])

let torecords self =
   Py.Module.get_function_with_keywords self "torecords"
     [||]
     []
     |> Ndarray.of_pyobject
let tostring ?fill_value ?order self =
   Py.Module.get_function_with_keywords self "tostring"
     [||]
     (Wrap_utils.keyword_args [("fill_value", fill_value); ("order", order)])

let trace ?offset ?axis1 ?axis2 ?dtype ?out self =
   Py.Module.get_function_with_keywords self "trace"
     [||]
     (Wrap_utils.keyword_args [("offset", offset); ("axis1", axis1); ("axis2", axis2); ("dtype", dtype); ("out", out)])

let transpose ?params args self =
   Py.Module.get_function_with_keywords self "transpose"
     (Wrap_utils.pos_arg Wrap_utils.id args)
     (match params with None -> [] | Some x -> x)
     |> Ndarray.of_pyobject
let unshare_mask self =
   Py.Module.get_function_with_keywords self "unshare_mask"
     [||]
     []

let var ?axis ?dtype ?out ?ddof ?keepdims self =
   Py.Module.get_function_with_keywords self "var"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("dtype", dtype); ("out", Wrap_utils.Option.map out Ndarray.to_pyobject); ("ddof", Wrap_utils.Option.map ddof Py.Int.of_int); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool)])

let view ?dtype ?type_ ?fill_value self =
   Py.Module.get_function_with_keywords self "view"
     [||]
     (Wrap_utils.keyword_args [("dtype", dtype); ("type", type_); ("fill_value", fill_value)])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let comb ?exact ?repetition ~n ~k () =
                     Py.Module.get_function_with_keywords ns "comb"
                       [||]
                       (Wrap_utils.keyword_args [("exact", exact); ("repetition", repetition); ("N", Some(n |> (function
| `Int x -> Py.Int.of_int x
| `Ndarray x -> Ndarray.to_pyobject x
))); ("k", Some(k ))])

                  let lobpcg ?b ?m ?y ?tol ?maxiter ?largest ?verbosityLevel ?retLambdaHistory ?retResidualNormsHistory ~a ~x () =
                     Py.Module.get_function_with_keywords ns "lobpcg"
                       [||]
                       (Wrap_utils.keyword_args [("B", b); ("M", m); ("Y", y); ("tol", tol); ("maxiter", maxiter); ("largest", largest); ("verbosityLevel", verbosityLevel); ("retLambdaHistory", retLambdaHistory); ("retResidualNormsHistory", retResidualNormsHistory); ("A", Some(a |> (function
| `SparseMatrix x -> Csr_matrix.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
| `PyObject x -> Wrap_utils.id x
))); ("X", Some(x ))])
                       |> Ndarray.of_pyobject
let logsumexp ?axis ?b ?keepdims ?return_sign ~a () =
   Py.Module.get_function_with_keywords ns "logsumexp"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("b", Wrap_utils.Option.map b Ndarray.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("return_sign", Wrap_utils.Option.map return_sign Py.Bool.of_bool); ("a", Some(a |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let loguniform ?kwds args =
   Py.Module.get_function_with_keywords ns "loguniform"
     (Wrap_utils.pos_arg Wrap_utils.id args)
     (match kwds with None -> [] | Some x -> x)

let pinvh ?cond ?rcond ?lower ?return_rank ?check_finite ~a () =
   Py.Module.get_function_with_keywords ns "pinvh"
     [||]
     (Wrap_utils.keyword_args [("cond", cond); ("rcond", rcond); ("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("return_rank", Wrap_utils.Option.map return_rank Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a ))])

                  let sparse_lsqr ?damp ?atol ?btol ?conlim ?iter_lim ?show ?calc_var ?x0 ~a ~b () =
                     Py.Module.get_function_with_keywords ns "sparse_lsqr"
                       [||]
                       (Wrap_utils.keyword_args [("damp", damp); ("atol", atol); ("btol", btol); ("conlim", conlim); ("iter_lim", iter_lim); ("show", show); ("calc_var", calc_var); ("x0", x0); ("A", Some(a |> (function
| `SparseMatrix x -> Csr_matrix.to_pyobject x
| `Ndarray x -> Ndarray.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
))); ("b", Some(b ))])


end
let gen_batches ?min_batch_size ~n ~batch_size () =
   Py.Module.get_function_with_keywords ns "gen_batches"
     [||]
     (Wrap_utils.keyword_args [("min_batch_size", min_batch_size); ("n", Some(n |> Py.Int.of_int)); ("batch_size", Some(batch_size ))])

let gen_even_slices ?n_samples ~n ~n_packs () =
   Py.Module.get_function_with_keywords ns "gen_even_slices"
     [||]
     (Wrap_utils.keyword_args [("n_samples", n_samples); ("n", Some(n |> Py.Int.of_int)); ("n_packs", Some(n_packs ))])

let get_chunk_n_rows ?max_n_rows ?working_memory ~row_bytes () =
   Py.Module.get_function_with_keywords ns "get_chunk_n_rows"
     [||]
     (Wrap_utils.keyword_args [("max_n_rows", max_n_rows); ("working_memory", working_memory); ("row_bytes", Some(row_bytes |> Py.Int.of_int))])

let get_config () =
   Py.Module.get_function_with_keywords ns "get_config"
     [||]
     []

module Graph = struct
let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.utils.graph"

                  let single_source_shortest_path_length ?cutoff ~graph ~source () =
                     Py.Module.get_function_with_keywords ns "single_source_shortest_path_length"
                       [||]
                       (Wrap_utils.keyword_args [("cutoff", cutoff); ("graph", Some(graph |> (function
| `SparseMatrix x -> Csr_matrix.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("source", Some(source ))])


end
module Graph_shortest_path = struct
let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.utils.graph_shortest_path"

module Float64 = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?x () =
   Py.Module.get_function_with_keywords ns "float64"
     [||]
     (Wrap_utils.keyword_args [("x", x)])

let get_item ~key self =
   Py.Module.get_function_with_keywords self "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key ))])

let fromhex ~string self =
   Py.Module.get_function_with_keywords self "fromhex"
     [||]
     (Wrap_utils.keyword_args [("string", Some(string ))])

let hex self =
   Py.Module.get_function_with_keywords self "hex"
     [||]
     []

let is_integer self =
   Py.Module.get_function_with_keywords self "is_integer"
     [||]
     []

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let isspmatrix ~x () =
   Py.Module.get_function_with_keywords ns "isspmatrix"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isspmatrix_csr ~x () =
   Py.Module.get_function_with_keywords ns "isspmatrix_csr"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])


end
                  let hash ?hash_name ?coerce_mmap ~obj () =
                     Py.Module.get_function_with_keywords ns "hash"
                       [||]
                       (Wrap_utils.keyword_args [("hash_name", Wrap_utils.Option.map hash_name (function
| `Md5 -> Py.String.of_string "md5"
| `Sha1 -> Py.String.of_string "sha1"
)); ("coerce_mmap", coerce_mmap); ("obj", Some(obj ))])

let import_module ?package ~name () =
   Py.Module.get_function_with_keywords ns "import_module"
     [||]
     (Wrap_utils.keyword_args [("package", package); ("name", Some(name ))])

let indexable iterables =
   Py.Module.get_function_with_keywords ns "indexable"
     (Wrap_utils.pos_arg Wrap_utils.id iterables)
     []

let indices_to_mask ~indices ~mask_length () =
   Py.Module.get_function_with_keywords ns "indices_to_mask"
     [||]
     (Wrap_utils.keyword_args [("indices", Some(indices )); ("mask_length", Some(mask_length ))])

let is_scalar_nan ~x () =
   Py.Module.get_function_with_keywords ns "is_scalar_nan"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let issparse ~x () =
   Py.Module.get_function_with_keywords ns "issparse"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

module Metaestimators = struct
let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.utils.metaestimators"

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
let abstractmethod ~funcobj () =
   Py.Module.get_function_with_keywords ns "abstractmethod"
     [||]
     (Wrap_utils.keyword_args [("funcobj", Some(funcobj ))])

                  let if_delegate_has_method ~delegate () =
                     Py.Module.get_function_with_keywords ns "if_delegate_has_method"
                       [||]
                       (Wrap_utils.keyword_args [("delegate", Some(delegate |> (function
| `String x -> Py.String.of_string x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `PyObject x -> Wrap_utils.id x
)))])

let update_wrapper ?assigned ?updated ~wrapper ~wrapped () =
   Py.Module.get_function_with_keywords ns "update_wrapper"
     [||]
     (Wrap_utils.keyword_args [("assigned", assigned); ("updated", updated); ("wrapper", Some(wrapper )); ("wrapped", Some(wrapped ))])


end
module Multiclass = struct
let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.utils.multiclass"

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

let check_classification_targets ~y () =
   Py.Module.get_function_with_keywords ns "check_classification_targets"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Ndarray.to_pyobject))])

let class_distribution ?sample_weight ~y () =
   Py.Module.get_function_with_keywords ns "class_distribution"
     [||]
     (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("y", Some(y ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
module Dok_matrix = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?shape ?dtype ?copy ~arg1 () =
   Py.Module.get_function_with_keywords ns "dok_matrix"
     [||]
     (Wrap_utils.keyword_args [("shape", Wrap_utils.Option.map shape (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("dtype", dtype); ("copy", copy); ("arg1", Some(arg1 ))])

let get_item ~key self =
   Py.Module.get_function_with_keywords self "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key ))])

let asformat ?copy ~format self =
   Py.Module.get_function_with_keywords self "asformat"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("format", Some(format |> Py.String.of_string))])

let asfptype self =
   Py.Module.get_function_with_keywords self "asfptype"
     [||]
     []

                  let astype ?casting ?copy ~dtype self =
                     Py.Module.get_function_with_keywords self "astype"
                       [||]
                       (Wrap_utils.keyword_args [("casting", casting); ("copy", copy); ("dtype", Some(dtype |> (function
| `String x -> Py.String.of_string x
| `PyObject x -> Wrap_utils.id x
)))])

let conj ?copy self =
   Py.Module.get_function_with_keywords self "conj"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let conjtransp self =
   Py.Module.get_function_with_keywords self "conjtransp"
     [||]
     []

let conjugate ?copy self =
   Py.Module.get_function_with_keywords self "conjugate"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let copy self =
   Py.Module.get_function_with_keywords self "copy"
     [||]
     []

let count_nonzero self =
   Py.Module.get_function_with_keywords self "count_nonzero"
     [||]
     []

let diagonal ?k self =
   Py.Module.get_function_with_keywords self "diagonal"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int)])

let dot ~other self =
   Py.Module.get_function_with_keywords self "dot"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let fromkeys ?value ~iterable self =
   Py.Module.get_function_with_keywords self "fromkeys"
     [||]
     (Wrap_utils.keyword_args [("value", value); ("iterable", Some(iterable ))])

let get ?default ~key self =
   Py.Module.get_function_with_keywords self "get"
     [||]
     (Wrap_utils.keyword_args [("default", default); ("key", Some(key ))])

let getH self =
   Py.Module.get_function_with_keywords self "getH"
     [||]
     []

let get_shape self =
   Py.Module.get_function_with_keywords self "get_shape"
     [||]
     []

let getcol ~j self =
   Py.Module.get_function_with_keywords self "getcol"
     [||]
     (Wrap_utils.keyword_args [("j", Some(j ))])

let getformat self =
   Py.Module.get_function_with_keywords self "getformat"
     [||]
     []

let getmaxprint self =
   Py.Module.get_function_with_keywords self "getmaxprint"
     [||]
     []

                  let getnnz ?axis self =
                     Py.Module.get_function_with_keywords self "getnnz"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
))])

let getrow ~i self =
   Py.Module.get_function_with_keywords self "getrow"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i ))])

let maximum ~other self =
   Py.Module.get_function_with_keywords self "maximum"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

                  let mean ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords self "mean"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
)); ("dtype", dtype); ("out", out)])

let minimum ~other self =
   Py.Module.get_function_with_keywords self "minimum"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let multiply ~other self =
   Py.Module.get_function_with_keywords self "multiply"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let nonzero self =
   Py.Module.get_function_with_keywords self "nonzero"
     [||]
     []

let power ?dtype ~n self =
   Py.Module.get_function_with_keywords self "power"
     [||]
     (Wrap_utils.keyword_args [("dtype", dtype); ("n", Some(n ))])

let reshape ?kwargs args self =
   Py.Module.get_function_with_keywords self "reshape"
     (Wrap_utils.pos_arg Wrap_utils.id args)
     (match kwargs with None -> [] | Some x -> x)
     |> Csr_matrix.of_pyobject
let resize shape self =
   Py.Module.get_function_with_keywords self "resize"
     (Wrap_utils.pos_arg Py.Int.of_int shape)
     []

let set_shape ~shape self =
   Py.Module.get_function_with_keywords self "set_shape"
     [||]
     (Wrap_utils.keyword_args [("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])

let setdefault ?default ~key self =
   Py.Module.get_function_with_keywords self "setdefault"
     [||]
     (Wrap_utils.keyword_args [("default", default); ("key", Some(key ))])

let setdiag ?k ~values self =
   Py.Module.get_function_with_keywords self "setdiag"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("values", Some(values |> Ndarray.to_pyobject))])

                  let sum ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords self "sum"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
)); ("dtype", dtype); ("out", out)])

                  let toarray ?order ?out self =
                     Py.Module.get_function_with_keywords self "toarray"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("out", Wrap_utils.Option.map out Ndarray.to_pyobject)])
                       |> Ndarray.of_pyobject
let tobsr ?blocksize ?copy self =
   Py.Module.get_function_with_keywords self "tobsr"
     [||]
     (Wrap_utils.keyword_args [("blocksize", blocksize); ("copy", copy)])

let tocoo ?copy self =
   Py.Module.get_function_with_keywords self "tocoo"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tocsc ?copy self =
   Py.Module.get_function_with_keywords self "tocsc"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tocsr ?copy self =
   Py.Module.get_function_with_keywords self "tocsr"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

                  let todense ?order ?out self =
                     Py.Module.get_function_with_keywords self "todense"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("out", Wrap_utils.Option.map out Ndarray.to_pyobject)])

let todia ?copy self =
   Py.Module.get_function_with_keywords self "todia"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let todok ?copy self =
   Py.Module.get_function_with_keywords self "todok"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tolil ?copy self =
   Py.Module.get_function_with_keywords self "tolil"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let transpose ?axes ?copy self =
   Py.Module.get_function_with_keywords self "transpose"
     [||]
     (Wrap_utils.keyword_args [("axes", axes); ("copy", copy)])

let update ~val_ self =
   Py.Module.get_function_with_keywords self "update"
     [||]
     (Wrap_utils.keyword_args [("val", Some(val_ ))])

let dtype self =
  match Py.Object.get_attr_string self "dtype" with
| None -> raise (Wrap_utils.Attribute_not_found "dtype")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let is_multilabel ~y () =
   Py.Module.get_function_with_keywords ns "is_multilabel"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Ndarray.to_pyobject))])
     |> Py.Bool.to_bool
let issparse ~x () =
   Py.Module.get_function_with_keywords ns "issparse"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

module Lil_matrix = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?shape ?dtype ?copy ~arg1 () =
   Py.Module.get_function_with_keywords ns "lil_matrix"
     [||]
     (Wrap_utils.keyword_args [("shape", Wrap_utils.Option.map shape (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("dtype", dtype); ("copy", copy); ("arg1", Some(arg1 ))])

let get_item ~key self =
   Py.Module.get_function_with_keywords self "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key ))])

let asformat ?copy ~format self =
   Py.Module.get_function_with_keywords self "asformat"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("format", Some(format |> Py.String.of_string))])

let asfptype self =
   Py.Module.get_function_with_keywords self "asfptype"
     [||]
     []

                  let astype ?casting ?copy ~dtype self =
                     Py.Module.get_function_with_keywords self "astype"
                       [||]
                       (Wrap_utils.keyword_args [("casting", casting); ("copy", copy); ("dtype", Some(dtype |> (function
| `String x -> Py.String.of_string x
| `PyObject x -> Wrap_utils.id x
)))])

let conj ?copy self =
   Py.Module.get_function_with_keywords self "conj"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let conjugate ?copy self =
   Py.Module.get_function_with_keywords self "conjugate"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let copy self =
   Py.Module.get_function_with_keywords self "copy"
     [||]
     []

let count_nonzero self =
   Py.Module.get_function_with_keywords self "count_nonzero"
     [||]
     []

let diagonal ?k self =
   Py.Module.get_function_with_keywords self "diagonal"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int)])

let dot ~other self =
   Py.Module.get_function_with_keywords self "dot"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let getH self =
   Py.Module.get_function_with_keywords self "getH"
     [||]
     []

let get_shape self =
   Py.Module.get_function_with_keywords self "get_shape"
     [||]
     []

let getcol ~j self =
   Py.Module.get_function_with_keywords self "getcol"
     [||]
     (Wrap_utils.keyword_args [("j", Some(j ))])

let getformat self =
   Py.Module.get_function_with_keywords self "getformat"
     [||]
     []

let getmaxprint self =
   Py.Module.get_function_with_keywords self "getmaxprint"
     [||]
     []

                  let getnnz ?axis self =
                     Py.Module.get_function_with_keywords self "getnnz"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
))])

let getrow ~i self =
   Py.Module.get_function_with_keywords self "getrow"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i ))])

let getrowview ~i self =
   Py.Module.get_function_with_keywords self "getrowview"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i ))])

let maximum ~other self =
   Py.Module.get_function_with_keywords self "maximum"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

                  let mean ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords self "mean"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
)); ("dtype", dtype); ("out", out)])

let minimum ~other self =
   Py.Module.get_function_with_keywords self "minimum"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let multiply ~other self =
   Py.Module.get_function_with_keywords self "multiply"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let nonzero self =
   Py.Module.get_function_with_keywords self "nonzero"
     [||]
     []

let power ?dtype ~n self =
   Py.Module.get_function_with_keywords self "power"
     [||]
     (Wrap_utils.keyword_args [("dtype", dtype); ("n", Some(n ))])

let reshape ?kwargs args self =
   Py.Module.get_function_with_keywords self "reshape"
     (Wrap_utils.pos_arg Wrap_utils.id args)
     (match kwargs with None -> [] | Some x -> x)
     |> Csr_matrix.of_pyobject
let resize shape self =
   Py.Module.get_function_with_keywords self "resize"
     (Wrap_utils.pos_arg Py.Int.of_int shape)
     []

let set_shape ~shape self =
   Py.Module.get_function_with_keywords self "set_shape"
     [||]
     (Wrap_utils.keyword_args [("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])

let setdiag ?k ~values self =
   Py.Module.get_function_with_keywords self "setdiag"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("values", Some(values |> Ndarray.to_pyobject))])

                  let sum ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords self "sum"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
)); ("dtype", dtype); ("out", out)])

                  let toarray ?order ?out self =
                     Py.Module.get_function_with_keywords self "toarray"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("out", Wrap_utils.Option.map out Ndarray.to_pyobject)])
                       |> Ndarray.of_pyobject
let tobsr ?blocksize ?copy self =
   Py.Module.get_function_with_keywords self "tobsr"
     [||]
     (Wrap_utils.keyword_args [("blocksize", blocksize); ("copy", copy)])

let tocoo ?copy self =
   Py.Module.get_function_with_keywords self "tocoo"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tocsc ?copy self =
   Py.Module.get_function_with_keywords self "tocsc"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tocsr ?copy self =
   Py.Module.get_function_with_keywords self "tocsr"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

                  let todense ?order ?out self =
                     Py.Module.get_function_with_keywords self "todense"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("out", Wrap_utils.Option.map out Ndarray.to_pyobject)])

let todia ?copy self =
   Py.Module.get_function_with_keywords self "todia"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let todok ?copy self =
   Py.Module.get_function_with_keywords self "todok"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tolil ?copy self =
   Py.Module.get_function_with_keywords self "tolil"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let transpose ?axes ?copy self =
   Py.Module.get_function_with_keywords self "transpose"
     [||]
     (Wrap_utils.keyword_args [("axes", axes); ("copy", copy)])

let dtype self =
  match Py.Object.get_attr_string self "dtype" with
| None -> raise (Wrap_utils.Attribute_not_found "dtype")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Spmatrix = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?maxprint () =
   Py.Module.get_function_with_keywords ns "spmatrix"
     [||]
     (Wrap_utils.keyword_args [("maxprint", maxprint)])

let asformat ?copy ~format self =
   Py.Module.get_function_with_keywords self "asformat"
     [||]
     (Wrap_utils.keyword_args [("copy", copy); ("format", Some(format |> Py.String.of_string))])

let asfptype self =
   Py.Module.get_function_with_keywords self "asfptype"
     [||]
     []

                  let astype ?casting ?copy ~dtype self =
                     Py.Module.get_function_with_keywords self "astype"
                       [||]
                       (Wrap_utils.keyword_args [("casting", casting); ("copy", copy); ("dtype", Some(dtype |> (function
| `String x -> Py.String.of_string x
| `PyObject x -> Wrap_utils.id x
)))])

let conj ?copy self =
   Py.Module.get_function_with_keywords self "conj"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let conjugate ?copy self =
   Py.Module.get_function_with_keywords self "conjugate"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let copy self =
   Py.Module.get_function_with_keywords self "copy"
     [||]
     []

let count_nonzero self =
   Py.Module.get_function_with_keywords self "count_nonzero"
     [||]
     []

let diagonal ?k self =
   Py.Module.get_function_with_keywords self "diagonal"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int)])

let dot ~other self =
   Py.Module.get_function_with_keywords self "dot"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let getH self =
   Py.Module.get_function_with_keywords self "getH"
     [||]
     []

let get_shape self =
   Py.Module.get_function_with_keywords self "get_shape"
     [||]
     []

let getcol ~j self =
   Py.Module.get_function_with_keywords self "getcol"
     [||]
     (Wrap_utils.keyword_args [("j", Some(j ))])

let getformat self =
   Py.Module.get_function_with_keywords self "getformat"
     [||]
     []

let getmaxprint self =
   Py.Module.get_function_with_keywords self "getmaxprint"
     [||]
     []

                  let getnnz ?axis self =
                     Py.Module.get_function_with_keywords self "getnnz"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
))])

let getrow ~i self =
   Py.Module.get_function_with_keywords self "getrow"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i ))])

let maximum ~other self =
   Py.Module.get_function_with_keywords self "maximum"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

                  let mean ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords self "mean"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
)); ("dtype", dtype); ("out", out)])

let minimum ~other self =
   Py.Module.get_function_with_keywords self "minimum"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let multiply ~other self =
   Py.Module.get_function_with_keywords self "multiply"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let nonzero self =
   Py.Module.get_function_with_keywords self "nonzero"
     [||]
     []

let power ?dtype ~n self =
   Py.Module.get_function_with_keywords self "power"
     [||]
     (Wrap_utils.keyword_args [("dtype", dtype); ("n", Some(n ))])

let reshape ?kwargs args self =
   Py.Module.get_function_with_keywords self "reshape"
     (Wrap_utils.pos_arg Wrap_utils.id args)
     (match kwargs with None -> [] | Some x -> x)
     |> Csr_matrix.of_pyobject
let resize ~shape self =
   Py.Module.get_function_with_keywords self "resize"
     [||]
     (Wrap_utils.keyword_args [("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])

let set_shape ~shape self =
   Py.Module.get_function_with_keywords self "set_shape"
     [||]
     (Wrap_utils.keyword_args [("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])

let setdiag ?k ~values self =
   Py.Module.get_function_with_keywords self "setdiag"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("values", Some(values |> Ndarray.to_pyobject))])

                  let sum ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords self "sum"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
)); ("dtype", dtype); ("out", out)])

                  let toarray ?order ?out self =
                     Py.Module.get_function_with_keywords self "toarray"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("out", Wrap_utils.Option.map out Ndarray.to_pyobject)])
                       |> Ndarray.of_pyobject
let tobsr ?blocksize ?copy self =
   Py.Module.get_function_with_keywords self "tobsr"
     [||]
     (Wrap_utils.keyword_args [("blocksize", blocksize); ("copy", copy)])

let tocoo ?copy self =
   Py.Module.get_function_with_keywords self "tocoo"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tocsc ?copy self =
   Py.Module.get_function_with_keywords self "tocsc"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tocsr ?copy self =
   Py.Module.get_function_with_keywords self "tocsr"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

                  let todense ?order ?out self =
                     Py.Module.get_function_with_keywords self "todense"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("out", Wrap_utils.Option.map out Ndarray.to_pyobject)])

let todia ?copy self =
   Py.Module.get_function_with_keywords self "todia"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let todok ?copy self =
   Py.Module.get_function_with_keywords self "todok"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let tolil ?copy self =
   Py.Module.get_function_with_keywords self "tolil"
     [||]
     (Wrap_utils.keyword_args [("copy", copy)])

let transpose ?axes ?copy self =
   Py.Module.get_function_with_keywords self "transpose"
     [||]
     (Wrap_utils.keyword_args [("axes", axes); ("copy", copy)])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let type_of_target ~y () =
   Py.Module.get_function_with_keywords ns "type_of_target"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Ndarray.to_pyobject))])
     |> Py.String.to_string
let unique_labels ys =
   Py.Module.get_function_with_keywords ns "unique_labels"
     (Wrap_utils.pos_arg Wrap_utils.id ys)
     []
     |> Ndarray.of_pyobject

end
module Murmurhash = struct
(* this module has no callables, skipping init and ns *)

end
module Optimize = struct
let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.utils.optimize"

module Deprecated = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?extra () =
   Py.Module.get_function_with_keywords ns "deprecated"
     [||]
     (Wrap_utils.keyword_args [("extra", Wrap_utils.Option.map extra Py.String.of_string)])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let line_search_wolfe1 ?gfk ?old_fval ?old_old_fval ?args ?c1 ?c2 ?amax ?amin ?xtol ~f ~fprime ~xk ~pk () =
   Py.Module.get_function_with_keywords ns "line_search_wolfe1"
     [||]
     (Wrap_utils.keyword_args [("gfk", Wrap_utils.Option.map gfk Ndarray.to_pyobject); ("old_fval", old_fval); ("old_old_fval", old_old_fval); ("args", args); ("c1", c1); ("c2", c2); ("amax", amax); ("amin", amin); ("xtol", xtol); ("f", Some(f )); ("fprime", Some(fprime )); ("xk", Some(xk )); ("pk", Some(pk ))])

let line_search_wolfe2 ?gfk ?old_fval ?old_old_fval ?args ?c1 ?c2 ?amax ?extra_condition ?maxiter ~f ~myfprime ~xk ~pk () =
   Py.Module.get_function_with_keywords ns "line_search_wolfe2"
     [||]
     (Wrap_utils.keyword_args [("gfk", gfk); ("old_fval", old_fval); ("old_old_fval", old_old_fval); ("args", args); ("c1", c1); ("c2", c2); ("amax", amax); ("extra_condition", extra_condition); ("maxiter", maxiter); ("f", Some(f )); ("myfprime", Some(myfprime )); ("xk", Some(xk )); ("pk", Some(pk ))])

let newton_cg ?args ?tol ?maxiter ?maxinner ?line_search ?warn ~grad_hess ~func ~grad ~x0 () =
   Py.Module.get_function_with_keywords ns "newton_cg"
     [||]
     (Wrap_utils.keyword_args [("args", args); ("tol", tol); ("maxiter", maxiter); ("maxinner", maxinner); ("line_search", line_search); ("warn", warn); ("grad_hess", Some(grad_hess )); ("func", Some(func )); ("grad", Some(grad )); ("x0", Some(x0 ))])


end
module Parallel_backend = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?n_jobs ?inner_max_num_threads ?backend_params ~backend () =
   Py.Module.get_function_with_keywords ns "parallel_backend"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("n_jobs", n_jobs); ("inner_max_num_threads", inner_max_num_threads); ("backend", Some(backend ))]) (match backend_params with None -> [] | Some x -> x))

let unregister self =
   Py.Module.get_function_with_keywords self "unregister"
     [||]
     []

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Random = struct
let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.utils.random"

                  let check_random_state ~seed () =
                     Py.Module.get_function_with_keywords ns "check_random_state"
                       [||]
                       (Wrap_utils.keyword_args [("seed", Some(seed |> (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)))])

module Deprecated = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?extra () =
   Py.Module.get_function_with_keywords ns "deprecated"
     [||]
     (Wrap_utils.keyword_args [("extra", Wrap_utils.Option.map extra Py.String.of_string)])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let random_choice_csc ?class_probability ?random_state ~n_samples ~classes () =
   Py.Module.get_function_with_keywords ns "random_choice_csc"
     [||]
     (Wrap_utils.keyword_args [("class_probability", class_probability); ("random_state", random_state); ("n_samples", Some(n_samples )); ("classes", Some(classes ))])


end
let register_parallel_backend ?make_default ~name ~factory () =
   Py.Module.get_function_with_keywords ns "register_parallel_backend"
     [||]
     (Wrap_utils.keyword_args [("make_default", make_default); ("name", Some(name )); ("factory", Some(factory ))])

let resample ?options arrays =
   Py.Module.get_function_with_keywords ns "resample"
     (Wrap_utils.pos_arg Wrap_utils.id arrays)
     (match options with None -> [] | Some x -> x)

                  let safe_indexing ?axis ~x ~indices () =
                     Py.Module.get_function_with_keywords ns "safe_indexing"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
| `ArrayLike x -> Wrap_utils.id x
| `PyObject x -> Wrap_utils.id x
))); ("indices", Some(indices |> (function
| `Bool x -> Py.Bool.of_bool x
| `Int x -> Py.Int.of_int x
| `String x -> Py.String.of_string x
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])

                  let safe_mask ~x ~mask () =
                     Py.Module.get_function_with_keywords ns "safe_mask"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
))); ("mask", Some(mask |> Ndarray.to_pyobject))])

                  let safe_sqr ?copy ~x () =
                     Py.Module.get_function_with_keywords ns "safe_sqr"
                       [||]
                       (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> (function
| `SparseMatrix x -> Csr_matrix.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])

let shuffle ?options arrays =
   Py.Module.get_function_with_keywords ns "shuffle"
     (Wrap_utils.pos_arg Wrap_utils.id arrays)
     (match options with None -> [] | Some x -> x)

module Sparsefuncs = struct
let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.utils.sparsefuncs"

                  let count_nonzero ?axis ?sample_weight ~x () =
                     Py.Module.get_function_with_keywords ns "count_nonzero"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
)); ("sample_weight", Wrap_utils.Option.map sample_weight Ndarray.to_pyobject); ("X", Some(x ))])

let csc_median_axis_0 ~x () =
   Py.Module.get_function_with_keywords ns "csc_median_axis_0"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> Ndarray.of_pyobject
let incr_mean_variance_axis ~x ~axis ~last_mean ~last_var ~last_n () =
   Py.Module.get_function_with_keywords ns "incr_mean_variance_axis"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("axis", Some(axis )); ("last_mean", Some(last_mean )); ("last_var", Some(last_var )); ("last_n", Some(last_n ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
let inplace_column_scale ~x ~scale () =
   Py.Module.get_function_with_keywords ns "inplace_column_scale"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("scale", Some(scale ))])

let inplace_csr_column_scale ~x ~scale () =
   Py.Module.get_function_with_keywords ns "inplace_csr_column_scale"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("scale", Some(scale ))])

let inplace_csr_row_scale ~x ~scale () =
   Py.Module.get_function_with_keywords ns "inplace_csr_row_scale"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("scale", Some(scale ))])

let inplace_row_scale ~x ~scale () =
   Py.Module.get_function_with_keywords ns "inplace_row_scale"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("scale", Some(scale ))])

let inplace_swap_column ~x ~m ~n () =
   Py.Module.get_function_with_keywords ns "inplace_swap_column"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("m", Some(m |> Py.Int.of_int)); ("n", Some(n |> Py.Int.of_int))])

let inplace_swap_row ~x ~m ~n () =
   Py.Module.get_function_with_keywords ns "inplace_swap_row"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("m", Some(m |> Py.Int.of_int)); ("n", Some(n |> Py.Int.of_int))])

let inplace_swap_row_csc ~x ~m ~n () =
   Py.Module.get_function_with_keywords ns "inplace_swap_row_csc"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("m", Some(m |> Py.Int.of_int)); ("n", Some(n |> Py.Int.of_int))])

let inplace_swap_row_csr ~x ~m ~n () =
   Py.Module.get_function_with_keywords ns "inplace_swap_row_csr"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("m", Some(m |> Py.Int.of_int)); ("n", Some(n |> Py.Int.of_int))])

let mean_variance_axis ~x ~axis () =
   Py.Module.get_function_with_keywords ns "mean_variance_axis"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("axis", Some(axis ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let min_max_axis ?ignore_nan ~x ~axis () =
   Py.Module.get_function_with_keywords ns "min_max_axis"
     [||]
     (Wrap_utils.keyword_args [("ignore_nan", Wrap_utils.Option.map ignore_nan Py.Bool.of_bool); ("X", Some(x )); ("axis", Some(axis ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))

end
module Sparsefuncs_fast = struct
let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.utils.sparsefuncs_fast"

let assign_rows_csr ~x ~x_rows ~out_rows ~out () =
   Py.Module.get_function_with_keywords ns "assign_rows_csr"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("X_rows", Some(x_rows )); ("out_rows", Some(out_rows )); ("out", Some(out ))])


end
module Stats = struct
let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.utils.stats"

let stable_cumsum ?axis ?rtol ?atol ~arr () =
   Py.Module.get_function_with_keywords ns "stable_cumsum"
     [||]
     (Wrap_utils.keyword_args [("axis", axis); ("rtol", rtol); ("atol", atol); ("arr", Some(arr |> Ndarray.to_pyobject))])


end
let tosequence ~x () =
   Py.Module.get_function_with_keywords ns "tosequence"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Ndarray.to_pyobject))])

module Validation = struct
let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.utils.validation"

module LooseVersion = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?vstring () =
   Py.Module.get_function_with_keywords ns "LooseVersion"
     [||]
     (Wrap_utils.keyword_args [("vstring", vstring)])

let parse ~vstring self =
   Py.Module.get_function_with_keywords self "parse"
     [||]
     (Wrap_utils.keyword_args [("vstring", Some(vstring ))])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Parameter = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ~name ~kind ~default ~annotation () =
   Py.Module.get_function_with_keywords ns "Parameter"
     [||]
     (Wrap_utils.keyword_args [("name", Some(name )); ("kind", Some(kind )); ("default", Some(default )); ("annotation", Some(annotation ))])

let replace ?name ?kind ?annotation ?default self =
   Py.Module.get_function_with_keywords self "replace"
     [||]
     (Wrap_utils.keyword_args [("name", name); ("kind", kind); ("annotation", annotation); ("default", default)])

let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let as_float_array ?copy ?force_all_finite ~x () =
                     Py.Module.get_function_with_keywords ns "as_float_array"
                       [||]
                       (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("force_all_finite", Wrap_utils.Option.map force_all_finite (function
| `Bool x -> Py.Bool.of_bool x
| `Allow_nan -> Py.String.of_string "allow-nan"
)); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

                  let assert_all_finite ?allow_nan ~x () =
                     Py.Module.get_function_with_keywords ns "assert_all_finite"
                       [||]
                       (Wrap_utils.keyword_args [("allow_nan", Wrap_utils.Option.map allow_nan Py.Bool.of_bool); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

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

                  let check_memory ~memory () =
                     Py.Module.get_function_with_keywords ns "check_memory"
                       [||]
                       (Wrap_utils.keyword_args [("memory", Some(memory |> (function
| `String x -> Py.String.of_string x
| `JoblibMemory x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)))])

                  let check_non_negative ~x ~whom () =
                     Py.Module.get_function_with_keywords ns "check_non_negative"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
))); ("whom", Some(whom |> Py.String.of_string))])

                  let check_random_state ~seed () =
                     Py.Module.get_function_with_keywords ns "check_random_state"
                       [||]
                       (Wrap_utils.keyword_args [("seed", Some(seed |> (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)))])

                  let check_scalar ?min_val ?max_val ~x ~name ~target_type () =
                     Py.Module.get_function_with_keywords ns "check_scalar"
                       [||]
                       (Wrap_utils.keyword_args [("min_val", Wrap_utils.Option.map min_val (function
| `Float x -> Py.Float.of_float x
| `Int x -> Py.Int.of_int x
)); ("max_val", Wrap_utils.Option.map max_val (function
| `Float x -> Py.Float.of_float x
| `Int x -> Py.Int.of_int x
)); ("x", Some(x )); ("name", Some(name |> Py.String.of_string)); ("target_type", Some(target_type |> (function
| `Dtype x -> Wrap_utils.id x
| `PyObject x -> Wrap_utils.id x
)))])

                  let check_symmetric ?tol ?raise_warning ?raise_exception ~array () =
                     Py.Module.get_function_with_keywords ns "check_symmetric"
                       [||]
                       (Wrap_utils.keyword_args [("tol", tol); ("raise_warning", raise_warning); ("raise_exception", raise_exception); ("array", Some(array |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

let column_or_1d ?warn ~y () =
   Py.Module.get_function_with_keywords ns "column_or_1d"
     [||]
     (Wrap_utils.keyword_args [("warn", Wrap_utils.Option.map warn Py.Bool.of_bool); ("y", Some(y |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let has_fit_parameter ~estimator ~parameter () =
   Py.Module.get_function_with_keywords ns "has_fit_parameter"
     [||]
     (Wrap_utils.keyword_args [("estimator", Some(estimator )); ("parameter", Some(parameter |> Py.String.of_string))])
     |> Py.Bool.to_bool
let indexable iterables =
   Py.Module.get_function_with_keywords ns "indexable"
     (Wrap_utils.pos_arg Wrap_utils.id iterables)
     []

let isclass ~object_ () =
   Py.Module.get_function_with_keywords ns "isclass"
     [||]
     (Wrap_utils.keyword_args [("object", Some(object_ ))])

let signature ?follow_wrapped ~obj () =
   Py.Module.get_function_with_keywords ns "signature"
     [||]
     (Wrap_utils.keyword_args [("follow_wrapped", follow_wrapped); ("obj", Some(obj ))])

let wraps ?assigned ?updated ~wrapped () =
   Py.Module.get_function_with_keywords ns "wraps"
     [||]
     (Wrap_utils.keyword_args [("assigned", assigned); ("updated", updated); ("wrapped", Some(wrapped ))])


end
