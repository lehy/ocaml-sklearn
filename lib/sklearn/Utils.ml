let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.utils"

let get_py name = Py.Module.get __wrap_namespace name
module Bunch = struct
type tag = [`Bunch]
type t = [`Bunch | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs () =
   Py.Module.get_function_with_keywords __wrap_namespace "Bunch"
     [||]
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module DataConversionWarning = struct
type tag = [`DataConversionWarning]
type t = [`BaseException | `DataConversionWarning | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_exception x = (x :> [`BaseException] Obj.t)
let with_traceback ~tb self =
   Py.Module.get_function_with_keywords (to_pyobject self) "with_traceback"
     [||]
     (Wrap_utils.keyword_args [("tb", Some(tb ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Path = struct
type tag = [`Path]
type t = [`Object | `Path] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create args =
   Py.Module.get_function_with_keywords __wrap_namespace "Path"
     [||]
     (Wrap_utils.keyword_args [("args", Some(args ))])
     |> of_pyobject
let absolute self =
   Py.Module.get_function_with_keywords (to_pyobject self) "absolute"
     [||]
     []

let as_posix self =
   Py.Module.get_function_with_keywords (to_pyobject self) "as_posix"
     [||]
     []

let as_uri self =
   Py.Module.get_function_with_keywords (to_pyobject self) "as_uri"
     [||]
     []

let chmod ~mode self =
   Py.Module.get_function_with_keywords (to_pyobject self) "chmod"
     [||]
     (Wrap_utils.keyword_args [("mode", Some(mode ))])

let cwd self =
   Py.Module.get_function_with_keywords (to_pyobject self) "cwd"
     [||]
     []

let exists self =
   Py.Module.get_function_with_keywords (to_pyobject self) "exists"
     [||]
     []

let expanduser self =
   Py.Module.get_function_with_keywords (to_pyobject self) "expanduser"
     [||]
     []

let glob ~pattern self =
   Py.Module.get_function_with_keywords (to_pyobject self) "glob"
     [||]
     (Wrap_utils.keyword_args [("pattern", Some(pattern ))])

let group self =
   Py.Module.get_function_with_keywords (to_pyobject self) "group"
     [||]
     []

let home self =
   Py.Module.get_function_with_keywords (to_pyobject self) "home"
     [||]
     []

let is_absolute self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_absolute"
     [||]
     []

let is_block_device self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_block_device"
     [||]
     []

let is_char_device self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_char_device"
     [||]
     []

let is_dir self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_dir"
     [||]
     []

let is_fifo self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_fifo"
     [||]
     []

let is_file self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_file"
     [||]
     []

let is_mount self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_mount"
     [||]
     []

let is_reserved self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_reserved"
     [||]
     []

let is_socket self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_socket"
     [||]
     []

let is_symlink self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_symlink"
     [||]
     []

let iterdir self =
   Py.Module.get_function_with_keywords (to_pyobject self) "iterdir"
     [||]
     []

let joinpath args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "joinpath"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     []

let lchmod ~mode self =
   Py.Module.get_function_with_keywords (to_pyobject self) "lchmod"
     [||]
     (Wrap_utils.keyword_args [("mode", Some(mode ))])

let link_to ~target self =
   Py.Module.get_function_with_keywords (to_pyobject self) "link_to"
     [||]
     (Wrap_utils.keyword_args [("target", Some(target ))])

let lstat self =
   Py.Module.get_function_with_keywords (to_pyobject self) "lstat"
     [||]
     []

let match_ ~path_pattern self =
   Py.Module.get_function_with_keywords (to_pyobject self) "match"
     [||]
     (Wrap_utils.keyword_args [("path_pattern", Some(path_pattern ))])

let mkdir ?mode ?parents ?exist_ok self =
   Py.Module.get_function_with_keywords (to_pyobject self) "mkdir"
     [||]
     (Wrap_utils.keyword_args [("mode", mode); ("parents", parents); ("exist_ok", exist_ok)])

let open_ ?mode ?buffering ?encoding ?errors ?newline self =
   Py.Module.get_function_with_keywords (to_pyobject self) "open"
     [||]
     (Wrap_utils.keyword_args [("mode", mode); ("buffering", buffering); ("encoding", encoding); ("errors", errors); ("newline", newline)])

let owner self =
   Py.Module.get_function_with_keywords (to_pyobject self) "owner"
     [||]
     []

let read_bytes self =
   Py.Module.get_function_with_keywords (to_pyobject self) "read_bytes"
     [||]
     []

let read_text ?encoding ?errors self =
   Py.Module.get_function_with_keywords (to_pyobject self) "read_text"
     [||]
     (Wrap_utils.keyword_args [("encoding", encoding); ("errors", errors)])

let relative_to other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "relative_to"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id other)])
     []

let rename ~target self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rename"
     [||]
     (Wrap_utils.keyword_args [("target", Some(target ))])

let replace ~target self =
   Py.Module.get_function_with_keywords (to_pyobject self) "replace"
     [||]
     (Wrap_utils.keyword_args [("target", Some(target ))])

let resolve ?strict self =
   Py.Module.get_function_with_keywords (to_pyobject self) "resolve"
     [||]
     (Wrap_utils.keyword_args [("strict", strict)])

let rglob ~pattern self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rglob"
     [||]
     (Wrap_utils.keyword_args [("pattern", Some(pattern ))])

let rmdir self =
   Py.Module.get_function_with_keywords (to_pyobject self) "rmdir"
     [||]
     []

let samefile ~other_path self =
   Py.Module.get_function_with_keywords (to_pyobject self) "samefile"
     [||]
     (Wrap_utils.keyword_args [("other_path", Some(other_path ))])

let stat self =
   Py.Module.get_function_with_keywords (to_pyobject self) "stat"
     [||]
     []

let symlink_to ?target_is_directory ~target self =
   Py.Module.get_function_with_keywords (to_pyobject self) "symlink_to"
     [||]
     (Wrap_utils.keyword_args [("target_is_directory", target_is_directory); ("target", Some(target ))])

let touch ?mode ?exist_ok self =
   Py.Module.get_function_with_keywords (to_pyobject self) "touch"
     [||]
     (Wrap_utils.keyword_args [("mode", mode); ("exist_ok", exist_ok)])

let unlink ?missing_ok self =
   Py.Module.get_function_with_keywords (to_pyobject self) "unlink"
     [||]
     (Wrap_utils.keyword_args [("missing_ok", missing_ok)])

let with_name ~name self =
   Py.Module.get_function_with_keywords (to_pyobject self) "with_name"
     [||]
     (Wrap_utils.keyword_args [("name", Some(name ))])

let with_suffix ~suffix self =
   Py.Module.get_function_with_keywords (to_pyobject self) "with_suffix"
     [||]
     (Wrap_utils.keyword_args [("suffix", Some(suffix ))])

let write_bytes ~data self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write_bytes"
     [||]
     (Wrap_utils.keyword_args [("data", Some(data ))])

let write_text ?encoding ?errors ~data self =
   Py.Module.get_function_with_keywords (to_pyobject self) "write_text"
     [||]
     (Wrap_utils.keyword_args [("encoding", encoding); ("errors", errors); ("data", Some(data ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Sequence = struct
type tag = [`Sequence]
type t = [`Object | `Sequence] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let get_item ~index self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index ))])

let iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []
     |> (fun py -> Py.Iter.to_seq py |> Seq.map Dict.of_pyobject)
let count ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "count"
     [||]
     (Wrap_utils.keyword_args [("value", Some(value ))])

let index ?start ?stop ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "index"
     [||]
     (Wrap_utils.keyword_args [("start", start); ("stop", stop); ("value", Some(value ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Compress = struct
type tag = [`Compress]
type t = [`Compress | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ~data ~selectors () =
   Py.Module.get_function_with_keywords __wrap_namespace "compress"
     [||]
     (Wrap_utils.keyword_args [("data", Some(data )); ("selectors", Some(selectors ))])
     |> of_pyobject
let iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []
     |> (fun py -> Py.Iter.to_seq py |> Seq.map Dict.of_pyobject)
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Islice = struct
type tag = [`Islice]
type t = [`Islice | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ~iterable ~stop () =
   Py.Module.get_function_with_keywords __wrap_namespace "islice"
     [||]
     (Wrap_utils.keyword_args [("iterable", Some(iterable )); ("stop", Some(stop ))])
     |> of_pyobject
let iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []
     |> (fun py -> Py.Iter.to_seq py |> Seq.map Dict.of_pyobject)
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Itemgetter = struct
type tag = [`Itemgetter]
type t = [`Itemgetter | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Parallel_backend = struct
type tag = [`Parallel_backend]
type t = [`Object | `Parallel_backend] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?n_jobs ?inner_max_num_threads ?backend_params ~backend () =
   Py.Module.get_function_with_keywords __wrap_namespace "parallel_backend"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("n_jobs", n_jobs); ("inner_max_num_threads", inner_max_num_threads); ("backend", Some(backend ))]) (match backend_params with None -> [] | Some x -> x))
     |> of_pyobject
let unregister self =
   Py.Module.get_function_with_keywords (to_pyobject self) "unregister"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Arrayfuncs = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.utils.arrayfuncs"

let get_py name = Py.Module.get __wrap_namespace name
let cholesky_delete ~l ~go_out () =
   Py.Module.get_function_with_keywords __wrap_namespace "cholesky_delete"
     [||]
     (Wrap_utils.keyword_args [("L", Some(l )); ("go_out", Some(go_out ))])


end
module Class_weight = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.utils.class_weight"

let get_py name = Py.Module.get __wrap_namespace name
                  let compute_class_weight ~class_weight ~classes ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "compute_class_weight"
                       [||]
                       (Wrap_utils.keyword_args [("class_weight", Some(class_weight |> (function
| `Balanced -> Py.String.of_string "balanced"
| `DictIntToFloat x -> (Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float) x
| `None -> Py.none
))); ("classes", Some(classes |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let compute_sample_weight ?indices ~class_weight ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "compute_sample_weight"
                       [||]
                       (Wrap_utils.keyword_args [("indices", Wrap_utils.Option.map indices Np.Obj.to_pyobject); ("class_weight", Some(class_weight |> (function
| `List_of_dicts x -> Wrap_utils.id x
| `Balanced -> Py.String.of_string "balanced"
| `DictIntToFloat x -> (Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float) x
| `None -> Py.none
))); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

end
module Deprecation = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.utils.deprecation"

let get_py name = Py.Module.get __wrap_namespace name

end
module Extmath = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.utils.extmath"

let get_py name = Py.Module.get __wrap_namespace name
let cartesian ?out ~arrays () =
   Py.Module.get_function_with_keywords __wrap_namespace "cartesian"
     [||]
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("arrays", Some(arrays |> Np.Numpy.Ndarray.List.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
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

                  let check_random_state seed =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_random_state"
                       [||]
                       (Wrap_utils.keyword_args [("seed", Some(seed |> (function
| `Optional x -> (function
| `I x -> Py.Int.of_int x
| `None -> Py.none
) x
| `RandomState x -> Wrap_utils.id x
)))])

let density ?kwargs ~w () =
   Py.Module.get_function_with_keywords __wrap_namespace "density"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("w", Some(w |> Np.Obj.to_pyobject))]) (match kwargs with None -> [] | Some x -> x))

let fast_logdet a =
   Py.Module.get_function_with_keywords __wrap_namespace "fast_logdet"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject))])

                  let log_logistic ?out ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "log_logistic"
                       [||]
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Arr x -> Np.Obj.to_pyobject x
| `T_ x -> Wrap_utils.id x
)); ("X", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let make_nonnegative ?min_value ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "make_nonnegative"
     [||]
     (Wrap_utils.keyword_args [("min_value", Wrap_utils.Option.map min_value Py.Float.of_float); ("X", Some(x |> Np.Obj.to_pyobject))])

                  let randomized_range_finder ?power_iteration_normalizer ?random_state ~a ~size ~n_iter () =
                     Py.Module.get_function_with_keywords __wrap_namespace "randomized_range_finder"
                       [||]
                       (Wrap_utils.keyword_args [("power_iteration_normalizer", Wrap_utils.Option.map power_iteration_normalizer (function
| `Auto -> Py.String.of_string "auto"
| `QR -> Py.String.of_string "QR"
| `LU -> Py.String.of_string "LU"
| `None -> Py.String.of_string "none"
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("A", Some(a |> Np.Obj.to_pyobject)); ("size", Some(size |> Py.Int.of_int)); ("n_iter", Some(n_iter |> Py.Int.of_int))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let randomized_svd ?n_oversamples ?n_iter ?power_iteration_normalizer ?transpose ?flip_sign ?random_state ~m ~n_components () =
                     Py.Module.get_function_with_keywords __wrap_namespace "randomized_svd"
                       [||]
                       (Wrap_utils.keyword_args [("n_oversamples", n_oversamples); ("n_iter", n_iter); ("power_iteration_normalizer", Wrap_utils.Option.map power_iteration_normalizer (function
| `Auto -> Py.String.of_string "auto"
| `QR -> Py.String.of_string "QR"
| `LU -> Py.String.of_string "LU"
| `None -> Py.String.of_string "none"
)); ("transpose", Wrap_utils.Option.map transpose (function
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("flip_sign", Wrap_utils.Option.map flip_sign Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("M", Some(m |> Np.Obj.to_pyobject)); ("n_components", Some(n_components |> Py.Int.of_int))])

let row_norms ?squared ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "row_norms"
     [||]
     (Wrap_utils.keyword_args [("squared", Wrap_utils.Option.map squared Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])

let safe_min x =
   Py.Module.get_function_with_keywords __wrap_namespace "safe_min"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])

let safe_sparse_dot ?dense_output ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "safe_sparse_dot"
     [||]
     (Wrap_utils.keyword_args [("dense_output", dense_output); ("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let softmax ?copy ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "softmax"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let squared_norm x =
   Py.Module.get_function_with_keywords __wrap_namespace "squared_norm"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject))])

let stable_cumsum ?axis ?rtol ?atol ~arr () =
   Py.Module.get_function_with_keywords __wrap_namespace "stable_cumsum"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("rtol", Wrap_utils.Option.map rtol Py.Float.of_float); ("atol", Wrap_utils.Option.map atol Py.Float.of_float); ("arr", Some(arr |> Np.Obj.to_pyobject))])

let svd_flip ?u_based_decision ~u ~v () =
   Py.Module.get_function_with_keywords __wrap_namespace "svd_flip"
     [||]
     (Wrap_utils.keyword_args [("u_based_decision", Wrap_utils.Option.map u_based_decision Py.Bool.of_bool); ("u", Some(u |> Np.Obj.to_pyobject)); ("v", Some(v |> Np.Obj.to_pyobject))])

let weighted_mode ?axis ~a ~w () =
   Py.Module.get_function_with_keywords __wrap_namespace "weighted_mode"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("a", Some(a |> Np.Obj.to_pyobject)); ("w", Some(w |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))

end
module Fixes = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.utils.fixes"

let get_py name = Py.Module.get __wrap_namespace name
module LooseVersion = struct
type tag = [`LooseVersion]
type t = [`LooseVersion | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?vstring () =
   Py.Module.get_function_with_keywords __wrap_namespace "LooseVersion"
     [||]
     (Wrap_utils.keyword_args [("vstring", vstring)])
     |> of_pyobject
let parse ~vstring self =
   Py.Module.get_function_with_keywords (to_pyobject self) "parse"
     [||]
     (Wrap_utils.keyword_args [("vstring", Some(vstring ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let lobpcg ?b ?m ?y ?tol ?maxiter ?largest ?verbosityLevel ?retLambdaHistory ?retResidualNormsHistory ~a ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "lobpcg"
                       [||]
                       (Wrap_utils.keyword_args [("B", Wrap_utils.Option.map b (function
| `PyObject x -> Wrap_utils.id x
| `Spmatrix x -> Np.Obj.to_pyobject x
)); ("M", Wrap_utils.Option.map m (function
| `PyObject x -> Wrap_utils.id x
| `Spmatrix x -> Np.Obj.to_pyobject x
)); ("Y", Wrap_utils.Option.map y (function
| `PyObject x -> Wrap_utils.id x
| `Arr x -> Np.Obj.to_pyobject x
)); ("tol", Wrap_utils.Option.map tol (function
| `S x -> Py.String.of_string x
| `Bool x -> Py.Bool.of_bool x
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("largest", Wrap_utils.Option.map largest Py.Bool.of_bool); ("verbosityLevel", Wrap_utils.Option.map verbosityLevel Py.Int.of_int); ("retLambdaHistory", Wrap_utils.Option.map retLambdaHistory Py.Bool.of_bool); ("retResidualNormsHistory", Wrap_utils.Option.map retResidualNormsHistory Py.Bool.of_bool); ("A", Some(a |> (function
| `Spmatrix x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("X", Some(x |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)), (Np.Numpy.Ndarray.List.of_pyobject (Py.Tuple.get x 2)), (Np.Numpy.Ndarray.List.of_pyobject (Py.Tuple.get x 3))))
let loguniform ?loc ?scale ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "loguniform"
     [||]
     (Wrap_utils.keyword_args [("loc", loc); ("scale", scale); ("a", Some(a )); ("b", Some(b ))])

let parse_version v =
   Py.Module.get_function_with_keywords __wrap_namespace "parse_version"
     [||]
     (Wrap_utils.keyword_args [("v", Some(v ))])

                  let sparse_lsqr ?damp ?atol ?btol ?conlim ?iter_lim ?show ?calc_var ?x0 ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "sparse_lsqr"
                       [||]
                       (Wrap_utils.keyword_args [("damp", Wrap_utils.Option.map damp Py.Float.of_float); ("atol", atol); ("btol", btol); ("conlim", Wrap_utils.Option.map conlim Py.Float.of_float); ("iter_lim", Wrap_utils.Option.map iter_lim Py.Int.of_int); ("show", Wrap_utils.Option.map show Py.Bool.of_bool); ("calc_var", Wrap_utils.Option.map calc_var Py.Bool.of_bool); ("x0", Wrap_utils.Option.map x0 Np.Obj.to_pyobject); ("A", Some(a |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `LinearOperator x -> Wrap_utils.id x
))); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), (Py.Float.to_float (Py.Tuple.get x 3)), (Py.Float.to_float (Py.Tuple.get x 4)), (Py.Float.to_float (Py.Tuple.get x 5)), (Py.Float.to_float (Py.Tuple.get x 6)), (Py.Float.to_float (Py.Tuple.get x 7)), (Py.Float.to_float (Py.Tuple.get x 8)), (Wrap_utils.id (Py.Tuple.get x 9))))

end
module Graph = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.utils.graph"

let get_py name = Py.Module.get __wrap_namespace name
let single_source_shortest_path_length ?cutoff ~graph ~source () =
   Py.Module.get_function_with_keywords __wrap_namespace "single_source_shortest_path_length"
     [||]
     (Wrap_utils.keyword_args [("cutoff", Wrap_utils.Option.map cutoff Py.Int.of_int); ("graph", Some(graph |> Np.Obj.to_pyobject)); ("source", Some(source |> Py.Int.of_int))])


end
module Graph_shortest_path = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.utils.graph_shortest_path"

let get_py name = Py.Module.get __wrap_namespace name
module DTYPE = struct
type tag = [`Float64]
type t = [`Float64 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?x () =
   Py.Module.get_function_with_keywords __wrap_namespace "DTYPE"
     (Array.of_list @@ List.concat [(match x with None -> [] | Some x -> [x ])])
     []
     |> of_pyobject
let get_item ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let fromhex ~string self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fromhex"
     (Array.of_list @@ List.concat [[string ]])
     []

let hex self =
   Py.Module.get_function_with_keywords (to_pyobject self) "hex"
     [||]
     []

let is_integer self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_integer"
     [||]
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Np.Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ITYPE = struct
type tag = [`Int32]
type t = [`Int32 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let get_item ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let newbyteorder ?new_order self =
   Py.Module.get_function_with_keywords (to_pyobject self) "newbyteorder"
     [||]
     (Wrap_utils.keyword_args [("new_order", Wrap_utils.Option.map new_order Py.String.of_string)])
     |> Np.Dtype.of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let isspmatrix x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let isspmatrix_csr x =
   Py.Module.get_function_with_keywords __wrap_namespace "isspmatrix_csr"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])


end
module Metaestimators = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.utils.metaestimators"

let get_py name = Py.Module.get __wrap_namespace name
module Attrgetter = struct
type tag = [`Attrgetter]
type t = [`Attrgetter | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let any ?kwds args =
   Py.Module.get_function_with_keywords __wrap_namespace "Any"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwds with None -> [] | Some x -> x)

let list ?kwargs args =
   Py.Module.get_function_with_keywords __wrap_namespace "List"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)

let abstractmethod funcobj =
   Py.Module.get_function_with_keywords __wrap_namespace "abstractmethod"
     [||]
     (Wrap_utils.keyword_args [("funcobj", Some(funcobj ))])

                  let if_delegate_has_method delegate =
                     Py.Module.get_function_with_keywords __wrap_namespace "if_delegate_has_method"
                       [||]
                       (Wrap_utils.keyword_args [("delegate", Some(delegate |> (function
| `S x -> Py.String.of_string x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
)))])

let update_wrapper ?assigned ?updated ~wrapper ~wrapped () =
   Py.Module.get_function_with_keywords __wrap_namespace "update_wrapper"
     [||]
     (Wrap_utils.keyword_args [("assigned", assigned); ("updated", updated); ("wrapper", Some(wrapper )); ("wrapped", Some(wrapped ))])


end
module Multiclass = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.utils.multiclass"

let get_py name = Py.Module.get __wrap_namespace name
module Chain = struct
type tag = [`Chain]
type t = [`Chain | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create iterables =
   Py.Module.get_function_with_keywords __wrap_namespace "chain"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id iterables)])
     []
     |> of_pyobject
let iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []
     |> (fun py -> Py.Iter.to_seq py |> Seq.map Dict.of_pyobject)
let from_iterable ~iterable self =
   Py.Module.get_function_with_keywords (to_pyobject self) "from_iterable"
     (Array.of_list @@ List.concat [[iterable ]])
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Dok_matrix = struct
type tag = [`Dok_matrix]
type t = [`ArrayLike | `Dok_matrix | `IndexMixin | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_index x = (x :> [`IndexMixin] Obj.t)
let create ?shape ?dtype ?copy ~arg1 () =
   Py.Module.get_function_with_keywords __wrap_namespace "dok_matrix"
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

let clear self =
   Py.Module.get_function_with_keywords (to_pyobject self) "clear"
     [||]
     []

let conj ?copy self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conj"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool)])

let conjtransp self =
   Py.Module.get_function_with_keywords (to_pyobject self) "conjtransp"
     [||]
     []

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

let diagonal ?k self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diagonal"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int)])

let dot ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let fromkeys ?value ~iterable self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fromkeys"
     (Array.of_list @@ List.concat [(match value with None -> [] | Some x -> [x ]);[iterable ]])
     []

let get ?default ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get"
     [||]
     (Wrap_utils.keyword_args [("default", default); ("key", Some(key ))])

let getH self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getH"
     [||]
     []

let get_shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_shape"
     [||]
     []

let getcol ~j self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getcol"
     [||]
     (Wrap_utils.keyword_args [("j", Some(j ))])

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

let items self =
   Py.Module.get_function_with_keywords (to_pyobject self) "items"
     [||]
     []

let keys self =
   Py.Module.get_function_with_keywords (to_pyobject self) "keys"
     [||]
     []

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

let pop ?d ~k self =
   Py.Module.get_function_with_keywords (to_pyobject self) "pop"
     [||]
     (Wrap_utils.keyword_args [("d", d); ("k", Some(k ))])

let popitem self =
   Py.Module.get_function_with_keywords (to_pyobject self) "popitem"
     [||]
     []

let power ?dtype ~n self =
   Py.Module.get_function_with_keywords (to_pyobject self) "power"
     [||]
     (Wrap_utils.keyword_args [("dtype", dtype); ("n", Some(n ))])

let reshape ?kwargs args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "reshape"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t))
let resize shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "resize"
     (Array.of_list @@ List.concat [(List.map Py.Int.of_int shape)])
     []

let set_shape ~shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_shape"
     [||]
     (Wrap_utils.keyword_args [("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])

let setdefault ?default ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setdefault"
     (Array.of_list @@ List.concat [(match default with None -> [] | Some x -> [x ]);[key ]])
     []

let setdiag ?k ~values self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setdiag"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("values", Some(values |> Np.Obj.to_pyobject))])

                  let sum ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "sum"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let toarray ?order ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "toarray"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("out", Wrap_utils.Option.map out (function
| `T2_D x -> Wrap_utils.id x
| `Arr x -> Np.Obj.to_pyobject x
))])

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
)); ("out", Wrap_utils.Option.map out (function
| `T2_D x -> Wrap_utils.id x
| `Arr x -> Np.Obj.to_pyobject x
))])

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

let update ~val_ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "update"
     [||]
     (Wrap_utils.keyword_args [("val", Some(val_ ))])

let values self =
   Py.Module.get_function_with_keywords (to_pyobject self) "values"
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
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Lil_matrix = struct
type tag = [`Lil_matrix]
type t = [`ArrayLike | `IndexMixin | `Lil_matrix | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_index x = (x :> [`IndexMixin] Obj.t)
let create ?shape ?dtype ?copy ~arg1 () =
   Py.Module.get_function_with_keywords __wrap_namespace "lil_matrix"
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

let diagonal ?k self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diagonal"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int)])

let dot ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let getH self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getH"
     [||]
     []

let get_shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_shape"
     [||]
     []

let getcol ~j self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getcol"
     [||]
     (Wrap_utils.keyword_args [("j", Some(j ))])

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

let getrowview ~i self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getrowview"
     [||]
     (Wrap_utils.keyword_args [("i", Some(i ))])

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

let reshape ?kwargs args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "reshape"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t))
let resize shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "resize"
     (Array.of_list @@ List.concat [(List.map Py.Int.of_int shape)])
     []

let set_shape ~shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_shape"
     [||]
     (Wrap_utils.keyword_args [("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])

let setdiag ?k ~values self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setdiag"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("values", Some(values |> Np.Obj.to_pyobject))])

                  let sum ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "sum"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let toarray ?order ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "toarray"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("out", Wrap_utils.Option.map out (function
| `T2_D x -> Wrap_utils.id x
| `Arr x -> Np.Obj.to_pyobject x
))])

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
)); ("out", Wrap_utils.Option.map out (function
| `T2_D x -> Wrap_utils.id x
| `Arr x -> Np.Obj.to_pyobject x
))])

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

let rows_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "rows" with
  | None -> failwith "attribute rows not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let rows self = match rows_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Spmatrix = struct
type tag = [`Spmatrix]
type t = [`ArrayLike | `Object | `Spmatrix] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?maxprint () =
   Py.Module.get_function_with_keywords __wrap_namespace "spmatrix"
     [||]
     (Wrap_utils.keyword_args [("maxprint", maxprint)])
     |> of_pyobject
let iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []
     |> (fun py -> Py.Iter.to_seq py |> Seq.map Dict.of_pyobject)
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

let diagonal ?k self =
   Py.Module.get_function_with_keywords (to_pyobject self) "diagonal"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int)])

let dot ~other self =
   Py.Module.get_function_with_keywords (to_pyobject self) "dot"
     [||]
     (Wrap_utils.keyword_args [("other", Some(other ))])

let getH self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getH"
     [||]
     []

let get_shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_shape"
     [||]
     []

let getcol ~j self =
   Py.Module.get_function_with_keywords (to_pyobject self) "getcol"
     [||]
     (Wrap_utils.keyword_args [("j", Some(j ))])

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

let reshape ?kwargs args self =
   Py.Module.get_function_with_keywords (to_pyobject self) "reshape"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwargs with None -> [] | Some x -> x)
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t))
let resize ~shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "resize"
     [||]
     (Wrap_utils.keyword_args [("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])

let set_shape ~shape self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_shape"
     [||]
     (Wrap_utils.keyword_args [("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml)))])

let setdiag ?k ~values self =
   Py.Module.get_function_with_keywords (to_pyobject self) "setdiag"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("values", Some(values |> Np.Obj.to_pyobject))])

                  let sum ?axis ?dtype ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "sum"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let toarray ?order ?out self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "toarray"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("out", Wrap_utils.Option.map out (function
| `T2_D x -> Wrap_utils.id x
| `Arr x -> Np.Obj.to_pyobject x
))])

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
)); ("out", Wrap_utils.Option.map out (function
| `T2_D x -> Wrap_utils.id x
| `Arr x -> Np.Obj.to_pyobject x
))])

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

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
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

let check_classification_targets y =
   Py.Module.get_function_with_keywords __wrap_namespace "check_classification_targets"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Np.Obj.to_pyobject))])

                  let class_distribution ?sample_weight ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "class_distribution"
                       [||]
                       (Wrap_utils.keyword_args [("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("y", Some(y |> (function
| `Sparse_matrix_of_size x -> Wrap_utils.id x
| `Arr x -> Np.Obj.to_pyobject x
)))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
let is_multilabel y =
   Py.Module.get_function_with_keywords __wrap_namespace "is_multilabel"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Np.Obj.to_pyobject))])
     |> Py.Bool.to_bool
let issparse x =
   Py.Module.get_function_with_keywords __wrap_namespace "issparse"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let type_of_target y =
   Py.Module.get_function_with_keywords __wrap_namespace "type_of_target"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Np.Obj.to_pyobject))])
     |> Py.String.to_string
let unique_labels ys =
   Py.Module.get_function_with_keywords __wrap_namespace "unique_labels"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id ys)])
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

end
module Murmurhash = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.utils.murmurhash"

let get_py name = Py.Module.get __wrap_namespace name

end
module Optimize = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.utils.optimize"

let get_py name = Py.Module.get __wrap_namespace name
let line_search_wolfe1 ?gfk ?old_fval ?old_old_fval ?args ?c1 ?c2 ?amax ?amin ?xtol ~f ~fprime ~xk ~pk () =
   Py.Module.get_function_with_keywords __wrap_namespace "line_search_wolfe1"
     [||]
     (Wrap_utils.keyword_args [("gfk", Wrap_utils.Option.map gfk Np.Obj.to_pyobject); ("old_fval", Wrap_utils.Option.map old_fval Py.Float.of_float); ("old_old_fval", Wrap_utils.Option.map old_old_fval Py.Float.of_float); ("args", args); ("c1", c1); ("c2", c2); ("amax", amax); ("amin", amin); ("xtol", xtol); ("f", Some(f )); ("fprime", Some(fprime )); ("xk", Some(xk |> Np.Obj.to_pyobject)); ("pk", Some(pk |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let line_search_wolfe2 ?gfk ?old_fval ?old_old_fval ?args ?c1 ?c2 ?amax ?extra_condition ?maxiter ~f ~myfprime ~xk ~pk () =
   Py.Module.get_function_with_keywords __wrap_namespace "line_search_wolfe2"
     [||]
     (Wrap_utils.keyword_args [("gfk", Wrap_utils.Option.map gfk Np.Obj.to_pyobject); ("old_fval", Wrap_utils.Option.map old_fval Py.Float.of_float); ("old_old_fval", Wrap_utils.Option.map old_old_fval Py.Float.of_float); ("args", args); ("c1", Wrap_utils.Option.map c1 Py.Float.of_float); ("c2", Wrap_utils.Option.map c2 Py.Float.of_float); ("amax", Wrap_utils.Option.map amax Py.Float.of_float); ("extra_condition", extra_condition); ("maxiter", Wrap_utils.Option.map maxiter Py.Int.of_int); ("f", Some(f )); ("myfprime", Some(myfprime )); ("xk", Some(xk |> Np.Obj.to_pyobject)); ("pk", Some(pk |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> if Py.is_none py then None else Some (Py.Float.to_float py)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), ((fun py -> if Py.is_none py then None else Some (Py.Float.to_float py)) (Py.Tuple.get x 3)), (Py.Float.to_float (Py.Tuple.get x 4)), ((fun py -> if Py.is_none py then None else Some (Py.Float.to_float py)) (Py.Tuple.get x 5))))
let newton_cg ?args ?tol ?maxiter ?maxinner ?line_search ?warn ~grad_hess ~func ~grad ~x0 () =
   Py.Module.get_function_with_keywords __wrap_namespace "newton_cg"
     [||]
     (Wrap_utils.keyword_args [("args", args); ("tol", tol); ("maxiter", maxiter); ("maxinner", maxinner); ("line_search", line_search); ("warn", warn); ("grad_hess", Some(grad_hess )); ("func", Some(func )); ("grad", Some(grad )); ("x0", Some(x0 ))])


end
module Random = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.utils.random"

let get_py name = Py.Module.get __wrap_namespace name
                  let check_random_state seed =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_random_state"
                       [||]
                       (Wrap_utils.keyword_args [("seed", Some(seed |> (function
| `Optional x -> (function
| `I x -> Py.Int.of_int x
| `None -> Py.none
) x
| `RandomState x -> Wrap_utils.id x
)))])

let random_choice_csc ?class_probability ?random_state ~n_samples ~classes () =
   Py.Module.get_function_with_keywords __wrap_namespace "random_choice_csc"
     [||]
     (Wrap_utils.keyword_args [("class_probability", class_probability); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("n_samples", Some(n_samples )); ("classes", Some(classes ))])


end
module Sparsefuncs = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.utils.sparsefuncs"

let get_py name = Py.Module.get __wrap_namespace name
                  let count_nonzero ?axis ?sample_weight ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "count_nonzero"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
)); ("sample_weight", Wrap_utils.Option.map sample_weight Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))])

let csc_median_axis_0 x =
   Py.Module.get_function_with_keywords __wrap_namespace "csc_median_axis_0"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let incr_mean_variance_axis ~x ~axis ~last_mean ~last_var ~last_n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "incr_mean_variance_axis"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Csc_matrix x -> Np.Obj.to_pyobject x
| `Csr_matrix x -> Np.Obj.to_pyobject x
))); ("axis", Some(axis |> Py.Int.of_int)); ("last_mean", Some(last_mean )); ("last_var", Some(last_var )); ("last_n", Some(last_n |> Py.Int.of_int))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2))))
                  let inplace_column_scale ~x ~scale () =
                     Py.Module.get_function_with_keywords __wrap_namespace "inplace_column_scale"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Csc_matrix x -> Np.Obj.to_pyobject x
| `Csr_matrix x -> Np.Obj.to_pyobject x
))); ("scale", Some(scale ))])

let inplace_csr_column_scale ~x ~scale () =
   Py.Module.get_function_with_keywords __wrap_namespace "inplace_csr_column_scale"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("scale", Some(scale ))])

let inplace_csr_row_scale ~x ~scale () =
   Py.Module.get_function_with_keywords __wrap_namespace "inplace_csr_row_scale"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("scale", Some(scale ))])

                  let inplace_row_scale ~x ~scale () =
                     Py.Module.get_function_with_keywords __wrap_namespace "inplace_row_scale"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Csc_matrix x -> Np.Obj.to_pyobject x
| `Csr_matrix x -> Np.Obj.to_pyobject x
))); ("scale", Some(scale ))])

                  let inplace_swap_column ~x ~m ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "inplace_swap_column"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Csc_matrix x -> Np.Obj.to_pyobject x
| `Csr_matrix x -> Np.Obj.to_pyobject x
))); ("m", Some(m |> Py.Int.of_int)); ("n", Some(n |> Py.Int.of_int))])

                  let inplace_swap_row ~x ~m ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "inplace_swap_row"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Csc_matrix x -> Np.Obj.to_pyobject x
| `Csr_matrix x -> Np.Obj.to_pyobject x
))); ("m", Some(m |> Py.Int.of_int)); ("n", Some(n |> Py.Int.of_int))])

let inplace_swap_row_csc ~x ~m ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "inplace_swap_row_csc"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("m", Some(m |> Py.Int.of_int)); ("n", Some(n |> Py.Int.of_int))])

let inplace_swap_row_csr ~x ~m ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "inplace_swap_row_csr"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("m", Some(m |> Py.Int.of_int)); ("n", Some(n |> Py.Int.of_int))])

                  let mean_variance_axis ~x ~axis () =
                     Py.Module.get_function_with_keywords __wrap_namespace "mean_variance_axis"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Csc_matrix x -> Np.Obj.to_pyobject x
| `Csr_matrix x -> Np.Obj.to_pyobject x
))); ("axis", Some(axis |> Py.Int.of_int))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let min_max_axis ?ignore_nan ~x ~axis () =
                     Py.Module.get_function_with_keywords __wrap_namespace "min_max_axis"
                       [||]
                       (Wrap_utils.keyword_args [("ignore_nan", Wrap_utils.Option.map ignore_nan Py.Bool.of_bool); ("X", Some(x |> (function
| `Csc_matrix x -> Np.Obj.to_pyobject x
| `Csr_matrix x -> Np.Obj.to_pyobject x
))); ("axis", Some(axis |> Py.Int.of_int))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))

end
module Sparsefuncs_fast = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.utils.sparsefuncs_fast"

let get_py name = Py.Module.get __wrap_namespace name
let assign_rows_csr ~x ~x_rows ~out_rows ~out () =
   Py.Module.get_function_with_keywords __wrap_namespace "assign_rows_csr"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x )); ("X_rows", Some(x_rows )); ("out_rows", Some(out_rows )); ("out", Some(out ))])


end
module Stats = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.utils.stats"

let get_py name = Py.Module.get __wrap_namespace name
let stable_cumsum ?axis ?rtol ?atol ~arr () =
   Py.Module.get_function_with_keywords __wrap_namespace "stable_cumsum"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("rtol", Wrap_utils.Option.map rtol Py.Float.of_float); ("atol", Wrap_utils.Option.map atol Py.Float.of_float); ("arr", Some(arr |> Np.Obj.to_pyobject))])


end
module Validation = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.utils.validation"

let get_py name = Py.Module.get __wrap_namespace name
module ComplexWarning = struct
type tag = [`ComplexWarning]
type t = [`BaseException | `ComplexWarning | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_exception x = (x :> [`BaseException] Obj.t)
let with_traceback ~tb self =
   Py.Module.get_function_with_keywords (to_pyobject self) "with_traceback"
     [||]
     (Wrap_utils.keyword_args [("tb", Some(tb ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Parameter = struct
type tag = [`Parameter]
type t = [`Object | `Parameter] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ~name ~kind ~default ~annotation () =
   Py.Module.get_function_with_keywords __wrap_namespace "Parameter"
     [||]
     (Wrap_utils.keyword_args [("name", Some(name )); ("kind", Some(kind )); ("default", Some(default )); ("annotation", Some(annotation ))])
     |> of_pyobject
let replace ?name ?kind ?annotation ?default self =
   Py.Module.get_function_with_keywords (to_pyobject self) "replace"
     [||]
     (Wrap_utils.keyword_args [("name", name); ("kind", kind); ("annotation", annotation); ("default", default)])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Suppress = struct
type tag = [`Suppress]
type t = [`Object | `Suppress] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create exceptions =
   Py.Module.get_function_with_keywords __wrap_namespace "suppress"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id exceptions)])
     []
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let as_float_array ?copy ?force_all_finite ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "as_float_array"
                       [||]
                       (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("force_all_finite", Wrap_utils.Option.map force_all_finite (function
| `Allow_nan -> Py.String.of_string "allow-nan"
| `Bool x -> Py.Bool.of_bool x
)); ("X", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let assert_all_finite ?allow_nan ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "assert_all_finite"
     [||]
     (Wrap_utils.keyword_args [("allow_nan", Wrap_utils.Option.map allow_nan Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])

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

let check_consistent_length arrays =
   Py.Module.get_function_with_keywords __wrap_namespace "check_consistent_length"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arrays)])
     []

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

                  let check_memory memory =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_memory"
                       [||]
                       (Wrap_utils.keyword_args [("memory", Some(memory |> (function
| `S x -> Py.String.of_string x
| `Object_with_the_joblib_Memory_interface x -> Wrap_utils.id x
| `None -> Py.none
)))])

let check_non_negative ~x ~whom () =
   Py.Module.get_function_with_keywords __wrap_namespace "check_non_negative"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("whom", Some(whom |> Py.String.of_string))])

                  let check_random_state seed =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_random_state"
                       [||]
                       (Wrap_utils.keyword_args [("seed", Some(seed |> (function
| `Optional x -> (function
| `I x -> Py.Int.of_int x
| `None -> Py.none
) x
| `RandomState x -> Wrap_utils.id x
)))])

                  let check_scalar ?min_val ?max_val ~x ~name ~target_type () =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_scalar"
                       [||]
                       (Wrap_utils.keyword_args [("min_val", Wrap_utils.Option.map min_val (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("max_val", Wrap_utils.Option.map max_val (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("x", Some(x )); ("name", Some(name |> Py.String.of_string)); ("target_type", Some(target_type |> (function
| `Dtype x -> Np.Dtype.to_pyobject x
| `Tuple x -> Wrap_utils.id x
)))])

let check_symmetric ?tol ?raise_warning ?raise_exception ~array () =
   Py.Module.get_function_with_keywords __wrap_namespace "check_symmetric"
     [||]
     (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("raise_warning", Wrap_utils.Option.map raise_warning Py.Bool.of_bool); ("raise_exception", Wrap_utils.Option.map raise_exception Py.Bool.of_bool); ("array", Some(array |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let column_or_1d ?warn ~y () =
   Py.Module.get_function_with_keywords __wrap_namespace "column_or_1d"
     [||]
     (Wrap_utils.keyword_args [("warn", Wrap_utils.Option.map warn Py.Bool.of_bool); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let has_fit_parameter ~estimator ~parameter () =
   Py.Module.get_function_with_keywords __wrap_namespace "has_fit_parameter"
     [||]
     (Wrap_utils.keyword_args [("estimator", Some(estimator |> Np.Obj.to_pyobject)); ("parameter", Some(parameter |> Py.String.of_string))])
     |> Py.Bool.to_bool
let indexable iterables =
   Py.Module.get_function_with_keywords __wrap_namespace "indexable"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id iterables)])
     []

let isclass object_ =
   Py.Module.get_function_with_keywords __wrap_namespace "isclass"
     [||]
     (Wrap_utils.keyword_args [("object", Some(object_ ))])

let parse_version v =
   Py.Module.get_function_with_keywords __wrap_namespace "parse_version"
     [||]
     (Wrap_utils.keyword_args [("v", Some(v ))])

let signature ?follow_wrapped ~obj () =
   Py.Module.get_function_with_keywords __wrap_namespace "signature"
     [||]
     (Wrap_utils.keyword_args [("follow_wrapped", follow_wrapped); ("obj", Some(obj ))])

let wraps ?assigned ?updated ~wrapped () =
   Py.Module.get_function_with_keywords __wrap_namespace "wraps"
     [||]
     (Wrap_utils.keyword_args [("assigned", assigned); ("updated", updated); ("wrapped", Some(wrapped ))])


end
                  let all_estimators ?type_filter () =
                     Py.Module.get_function_with_keywords __wrap_namespace "all_estimators"
                       [||]
                       (Wrap_utils.keyword_args [("type_filter", Wrap_utils.Option.map type_filter (function
| `S x -> Py.String.of_string x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
))])

                  let as_float_array ?copy ?force_all_finite ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "as_float_array"
                       [||]
                       (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("force_all_finite", Wrap_utils.Option.map force_all_finite (function
| `Allow_nan -> Py.String.of_string "allow-nan"
| `Bool x -> Py.Bool.of_bool x
)); ("X", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let assert_all_finite ?allow_nan ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "assert_all_finite"
     [||]
     (Wrap_utils.keyword_args [("allow_nan", Wrap_utils.Option.map allow_nan Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])

let axis0_safe_slice ~x ~mask ~len_mask () =
   Py.Module.get_function_with_keywords __wrap_namespace "axis0_safe_slice"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("mask", Some(mask |> Np.Obj.to_pyobject)); ("len_mask", Some(len_mask |> Py.Int.of_int))])

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

let check_consistent_length arrays =
   Py.Module.get_function_with_keywords __wrap_namespace "check_consistent_length"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arrays)])
     []

let check_matplotlib_support caller_name =
   Py.Module.get_function_with_keywords __wrap_namespace "check_matplotlib_support"
     [||]
     (Wrap_utils.keyword_args [("caller_name", Some(caller_name |> Py.String.of_string))])

let check_pandas_support caller_name =
   Py.Module.get_function_with_keywords __wrap_namespace "check_pandas_support"
     [||]
     (Wrap_utils.keyword_args [("caller_name", Some(caller_name |> Py.String.of_string))])

                  let check_random_state seed =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_random_state"
                       [||]
                       (Wrap_utils.keyword_args [("seed", Some(seed |> (function
| `Optional x -> (function
| `I x -> Py.Int.of_int x
| `None -> Py.none
) x
| `RandomState x -> Wrap_utils.id x
)))])

                  let check_scalar ?min_val ?max_val ~x ~name ~target_type () =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_scalar"
                       [||]
                       (Wrap_utils.keyword_args [("min_val", Wrap_utils.Option.map min_val (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("max_val", Wrap_utils.Option.map max_val (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("x", Some(x )); ("name", Some(name |> Py.String.of_string)); ("target_type", Some(target_type |> (function
| `Dtype x -> Np.Dtype.to_pyobject x
| `Tuple x -> Wrap_utils.id x
)))])

let check_symmetric ?tol ?raise_warning ?raise_exception ~array () =
   Py.Module.get_function_with_keywords __wrap_namespace "check_symmetric"
     [||]
     (Wrap_utils.keyword_args [("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("raise_warning", Wrap_utils.Option.map raise_warning Py.Bool.of_bool); ("raise_exception", Wrap_utils.Option.map raise_exception Py.Bool.of_bool); ("array", Some(array |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let column_or_1d ?warn ~y () =
   Py.Module.get_function_with_keywords __wrap_namespace "column_or_1d"
     [||]
     (Wrap_utils.keyword_args [("warn", Wrap_utils.Option.map warn Py.Bool.of_bool); ("y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let compute_class_weight ~class_weight ~classes ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "compute_class_weight"
                       [||]
                       (Wrap_utils.keyword_args [("class_weight", Some(class_weight |> (function
| `Balanced -> Py.String.of_string "balanced"
| `DictIntToFloat x -> (Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float) x
| `None -> Py.none
))); ("classes", Some(classes |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let compute_sample_weight ?indices ~class_weight ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "compute_sample_weight"
                       [||]
                       (Wrap_utils.keyword_args [("indices", Wrap_utils.Option.map indices Np.Obj.to_pyobject); ("class_weight", Some(class_weight |> (function
| `List_of_dicts x -> Wrap_utils.id x
| `Balanced -> Py.String.of_string "balanced"
| `DictIntToFloat x -> (Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float) x
| `None -> Py.none
))); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let contextmanager func =
   Py.Module.get_function_with_keywords __wrap_namespace "contextmanager"
     [||]
     (Wrap_utils.keyword_args [("func", Some(func ))])

let estimator_html_repr estimator =
   Py.Module.get_function_with_keywords __wrap_namespace "estimator_html_repr"
     [||]
     (Wrap_utils.keyword_args [("estimator", Some(estimator |> Np.Obj.to_pyobject))])
     |> Py.String.to_string
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
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("row_bytes", Some(row_bytes |> Py.Int.of_int))])

let get_config () =
   Py.Module.get_function_with_keywords __wrap_namespace "get_config"
     [||]
     []
     |> Dict.of_pyobject
let import_module ?package ~name () =
   Py.Module.get_function_with_keywords __wrap_namespace "import_module"
     [||]
     (Wrap_utils.keyword_args [("package", package); ("name", Some(name ))])

let indexable iterables =
   Py.Module.get_function_with_keywords __wrap_namespace "indexable"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id iterables)])
     []

let indices_to_mask ~indices ~mask_length () =
   Py.Module.get_function_with_keywords __wrap_namespace "indices_to_mask"
     [||]
     (Wrap_utils.keyword_args [("indices", Some(indices |> Np.Obj.to_pyobject)); ("mask_length", Some(mask_length |> Py.Int.of_int))])

let is_scalar_nan x =
   Py.Module.get_function_with_keywords __wrap_namespace "is_scalar_nan"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let issparse x =
   Py.Module.get_function_with_keywords __wrap_namespace "issparse"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])

let parse_version v =
   Py.Module.get_function_with_keywords __wrap_namespace "parse_version"
     [||]
     (Wrap_utils.keyword_args [("v", Some(v ))])

let register_parallel_backend ?make_default ~name ~factory () =
   Py.Module.get_function_with_keywords __wrap_namespace "register_parallel_backend"
     [||]
     (Wrap_utils.keyword_args [("make_default", make_default); ("name", Some(name )); ("factory", Some(factory ))])

let resample ?options arrays =
   Py.Module.get_function_with_keywords __wrap_namespace "resample"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arrays)])
     (match options with None -> [] | Some x -> x)

                  let safe_indexing ?axis ~x ~indices () =
                     Py.Module.get_function_with_keywords __wrap_namespace "safe_indexing"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis Py.Int.of_int); ("X", Some(x |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("indices", Some(indices |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `Bool x -> Py.Bool.of_bool x
| `Slice x -> Np.Wrap_utils.Slice.to_pyobject x
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
)))])

let safe_mask ~x ~mask () =
   Py.Module.get_function_with_keywords __wrap_namespace "safe_mask"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject)); ("mask", Some(mask |> Np.Obj.to_pyobject))])

let safe_sqr ?copy ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "safe_sqr"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])

let shuffle ?random_state ?n_samples arrays =
   Py.Module.get_function_with_keywords __wrap_namespace "shuffle"
     (Array.of_list @@ List.concat [(List.map Np.Obj.to_pyobject arrays)])
     (Wrap_utils.keyword_args [("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int)])
     |> (fun py -> Py.List.to_list_map ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))) py)
let tosequence x =
   Py.Module.get_function_with_keywords __wrap_namespace "tosequence"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject))])

