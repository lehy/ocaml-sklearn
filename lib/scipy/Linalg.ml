let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.linalg"

let get_py name = Py.Module.get __wrap_namespace name
module LinAlgError = struct
type tag = [`LinAlgError]
type t = [`BaseException | `LinAlgError | `Object] Obj.t
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
module LinAlgWarning = struct
type tag = [`LinAlgWarning]
type t = [`BaseException | `LinAlgWarning | `Object] Obj.t
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
module Basic = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.linalg.basic"

let get_py name = Py.Module.get __wrap_namespace name
let atleast_1d arys =
   Py.Module.get_function_with_keywords __wrap_namespace "atleast_1d"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arys)])
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let atleast_2d arys =
   Py.Module.get_function_with_keywords __wrap_namespace "atleast_2d"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arys)])
     []

let det ?overwrite_a ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "det"
     [||]
     (Wrap_utils.keyword_args [("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])

let get_flinalg_funcs ?arrays ?debug ~names () =
   Py.Module.get_function_with_keywords __wrap_namespace "get_flinalg_funcs"
     [||]
     (Wrap_utils.keyword_args [("arrays", arrays); ("debug", debug); ("names", Some(names ))])

                  let get_lapack_funcs ?arrays ?dtype ~names () =
                     Py.Module.get_function_with_keywords __wrap_namespace "get_lapack_funcs"
                       [||]
                       (Wrap_utils.keyword_args [("arrays", Wrap_utils.Option.map arrays (fun ml -> Py.List.of_list_map Np.Obj.to_pyobject ml)); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("names", Some(names |> (function
| `Sequence_of_str x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let inv ?overwrite_a ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "inv"
     [||]
     (Wrap_utils.keyword_args [("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let levinson ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "levinson"
                       [||]
                       (Wrap_utils.keyword_args [("a", Some(a |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("b", Some(b |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let lstsq ?cond ?overwrite_a ?overwrite_b ?check_finite ?lapack_driver ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "lstsq"
     [||]
     (Wrap_utils.keyword_args [("cond", Wrap_utils.Option.map cond Py.Float.of_float); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("lapack_driver", Wrap_utils.Option.map lapack_driver Py.String.of_string); ("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), ((fun py -> if Py.is_none py then None else Some (Wrap_utils.id py)) (Py.Tuple.get x 3))))
let matrix_balance ?permute ?scale ?separate ?overwrite_a ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "matrix_balance"
     [||]
     (Wrap_utils.keyword_args [("permute", Wrap_utils.Option.map permute Py.Bool.of_bool); ("scale", Wrap_utils.Option.map scale Py.Float.of_float); ("separate", Wrap_utils.Option.map separate Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("A", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let pinv ?cond ?rcond ?return_rank ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "pinv"
     [||]
     (Wrap_utils.keyword_args [("cond", cond); ("rcond", rcond); ("return_rank", Wrap_utils.Option.map return_rank Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
let pinv2 ?cond ?rcond ?return_rank ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "pinv2"
     [||]
     (Wrap_utils.keyword_args [("cond", cond); ("rcond", rcond); ("return_rank", Wrap_utils.Option.map return_rank Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
let pinvh ?cond ?rcond ?lower ?return_rank ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "pinvh"
     [||]
     (Wrap_utils.keyword_args [("cond", cond); ("rcond", rcond); ("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("return_rank", Wrap_utils.Option.map return_rank Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
let solve ?sym_pos ?lower ?overwrite_a ?overwrite_b ?debug ?check_finite ?assume_a ?transposed ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "solve"
     [||]
     (Wrap_utils.keyword_args [("sym_pos", Wrap_utils.Option.map sym_pos Py.Bool.of_bool); ("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("debug", debug); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("assume_a", Wrap_utils.Option.map assume_a Py.String.of_string); ("transposed", Wrap_utils.Option.map transposed Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let solve_banded ?overwrite_ab ?overwrite_b ?debug ?check_finite ~l_and_u ~ab ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "solve_banded"
     [||]
     (Wrap_utils.keyword_args [("overwrite_ab", Wrap_utils.Option.map overwrite_ab Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("debug", debug); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("l_and_u", Some(l_and_u )); ("ab", Some(ab )); ("b", Some(b ))])

let solve_circulant ?singular ?tol ?caxis ?baxis ?outaxis ~c ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "solve_circulant"
     [||]
     (Wrap_utils.keyword_args [("singular", Wrap_utils.Option.map singular Py.String.of_string); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("caxis", Wrap_utils.Option.map caxis Py.Int.of_int); ("baxis", Wrap_utils.Option.map baxis Py.Int.of_int); ("outaxis", Wrap_utils.Option.map outaxis Py.Int.of_int); ("c", Some(c |> Np.Obj.to_pyobject)); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let solve_toeplitz ?check_finite ~c_or_cr ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "solve_toeplitz"
                       [||]
                       (Wrap_utils.keyword_args [("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("c_or_cr", Some(c_or_cr |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_array_like_array_like_ x -> Wrap_utils.id x
))); ("b", Some(b ))])

                  let solve_triangular ?trans ?lower ?unit_diagonal ?overwrite_b ?debug ?check_finite ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "solve_triangular"
                       [||]
                       (Wrap_utils.keyword_args [("trans", Wrap_utils.Option.map trans (function
| `C -> Py.String.of_string "C"
| `Two -> Py.Int.of_int 2
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `T -> Py.String.of_string "T"
| `N -> Py.String.of_string "N"
)); ("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("unit_diagonal", Wrap_utils.Option.map unit_diagonal Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("debug", debug); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b ))])

let solveh_banded ?overwrite_ab ?overwrite_b ?lower ?check_finite ~ab ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "solveh_banded"
     [||]
     (Wrap_utils.keyword_args [("overwrite_ab", Wrap_utils.Option.map overwrite_ab Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("ab", Some(ab )); ("b", Some(b ))])

let warn ?category ?stacklevel ?source ~message () =
   Py.Module.get_function_with_keywords __wrap_namespace "warn"
     [||]
     (Wrap_utils.keyword_args [("category", category); ("stacklevel", stacklevel); ("source", source); ("message", Some(message ))])


end
module Blas = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.linalg.blas"

let get_py name = Py.Module.get __wrap_namespace name
let crotg ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "crotg"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a )); ("b", Some(b ))])

let drotg ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "drotg"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a )); ("b", Some(b ))])
     |> Py.Float.to_float
                  let find_best_blas_type ?arrays ?dtype () =
                     Py.Module.get_function_with_keywords __wrap_namespace "find_best_blas_type"
                       [||]
                       (Wrap_utils.keyword_args [("arrays", Wrap_utils.Option.map arrays (fun ml -> Py.List.of_list_map Np.Obj.to_pyobject ml)); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
))])
                       |> (fun x -> ((Py.String.to_string (Py.Tuple.get x 0)), (Np.Dtype.of_pyobject (Py.Tuple.get x 1)), (Py.Bool.to_bool (Py.Tuple.get x 2))))
                  let get_blas_funcs ?arrays ?dtype ~names () =
                     Py.Module.get_function_with_keywords __wrap_namespace "get_blas_funcs"
                       [||]
                       (Wrap_utils.keyword_args [("arrays", Wrap_utils.Option.map arrays (fun ml -> Py.List.of_list_map Np.Obj.to_pyobject ml)); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("names", Some(names |> (function
| `Sequence_of_str x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let srotg ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "srotg"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a )); ("b", Some(b ))])
     |> Py.Float.to_float
let zrotg ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "zrotg"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a )); ("b", Some(b ))])


end
module Cython_blas = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.linalg.cython_blas"

let get_py name = Py.Module.get __wrap_namespace name

end
module Cython_lapack = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.linalg.cython_lapack"

let get_py name = Py.Module.get __wrap_namespace name

end
module Decomp = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.linalg.decomp"

let get_py name = Py.Module.get __wrap_namespace name
module Inexact = struct
type tag = [`Inexact]
type t = [`Inexact | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create () =
   Py.Module.get_function_with_keywords __wrap_namespace "inexact"
     [||]
     []
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let argsort ?axis ?kind ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "argsort"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (function
| `I x -> Py.Int.of_int x
| `None -> Py.none
)); ("kind", Wrap_utils.Option.map kind (function
| `Quicksort -> Py.String.of_string "quicksort"
| `Heapsort -> Py.String.of_string "heapsort"
| `Stable -> Py.String.of_string "stable"
| `Mergesort -> Py.String.of_string "mergesort"
)); ("order", Wrap_utils.Option.map order (function
| `S x -> Py.String.of_string x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let array ?dtype ?copy ?order ?subok ?ndmin ~object_ () =
                     Py.Module.get_function_with_keywords __wrap_namespace "array"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("order", Wrap_utils.Option.map order (function
| `K -> Py.String.of_string "K"
| `A -> Py.String.of_string "A"
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("ndmin", Wrap_utils.Option.map ndmin Py.Int.of_int); ("object", Some(object_ |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let asarray ?dtype ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let cdf2rdf ~w ~v () =
   Py.Module.get_function_with_keywords __wrap_namespace "cdf2rdf"
     [||]
     (Wrap_utils.keyword_args [("w", Some(w )); ("v", Some(v ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let conj ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "conj"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eig ?b ?left ?right ?overwrite_a ?overwrite_b ?check_finite ?homogeneous_eigvals ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "eig"
     [||]
     (Wrap_utils.keyword_args [("b", Wrap_utils.Option.map b Np.Obj.to_pyobject); ("left", Wrap_utils.Option.map left Py.Bool.of_bool); ("right", Wrap_utils.Option.map right Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("homogeneous_eigvals", Wrap_utils.Option.map homogeneous_eigvals Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
                  let eig_banded ?lower ?eigvals_only ?overwrite_a_band ?select ?select_range ?max_ev ?check_finite ~a_band () =
                     Py.Module.get_function_with_keywords __wrap_namespace "eig_banded"
                       [||]
                       (Wrap_utils.keyword_args [("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("eigvals_only", Wrap_utils.Option.map eigvals_only Py.Bool.of_bool); ("overwrite_a_band", Wrap_utils.Option.map overwrite_a_band Py.Bool.of_bool); ("select", Wrap_utils.Option.map select (function
| `A -> Py.String.of_string "a"
| `V -> Py.String.of_string "v"
| `I -> Py.String.of_string "i"
)); ("select_range", select_range); ("max_ev", Wrap_utils.Option.map max_ev Py.Int.of_int); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a_band", Some(a_band ))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let eigh ?b ?lower ?eigvals_only ?overwrite_a ?overwrite_b ?turbo ?eigvals ?type_ ?check_finite ?subset_by_index ?subset_by_value ?driver ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "eigh"
     [||]
     (Wrap_utils.keyword_args [("b", Wrap_utils.Option.map b Np.Obj.to_pyobject); ("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("eigvals_only", Wrap_utils.Option.map eigvals_only Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("turbo", Wrap_utils.Option.map turbo Py.Bool.of_bool); ("eigvals", eigvals); ("type", Wrap_utils.Option.map type_ Py.Int.of_int); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("subset_by_index", Wrap_utils.Option.map subset_by_index Np.Obj.to_pyobject); ("subset_by_value", Wrap_utils.Option.map subset_by_value Np.Obj.to_pyobject); ("driver", Wrap_utils.Option.map driver Py.String.of_string); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let eigh_tridiagonal ?eigvals_only ?select ?select_range ?check_finite ?tol ?lapack_driver ~d ~e () =
                     Py.Module.get_function_with_keywords __wrap_namespace "eigh_tridiagonal"
                       [||]
                       (Wrap_utils.keyword_args [("eigvals_only", eigvals_only); ("select", Wrap_utils.Option.map select (function
| `A -> Py.String.of_string "a"
| `V -> Py.String.of_string "v"
| `I -> Py.String.of_string "i"
)); ("select_range", select_range); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("lapack_driver", Wrap_utils.Option.map lapack_driver Py.String.of_string); ("d", Some(d |> Np.Obj.to_pyobject)); ("e", Some(e |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let eigvals ?b ?overwrite_a ?check_finite ?homogeneous_eigvals ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "eigvals"
     [||]
     (Wrap_utils.keyword_args [("b", Wrap_utils.Option.map b Np.Obj.to_pyobject); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("homogeneous_eigvals", Wrap_utils.Option.map homogeneous_eigvals Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])

                  let eigvals_banded ?lower ?overwrite_a_band ?select ?select_range ?check_finite ~a_band () =
                     Py.Module.get_function_with_keywords __wrap_namespace "eigvals_banded"
                       [||]
                       (Wrap_utils.keyword_args [("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("overwrite_a_band", Wrap_utils.Option.map overwrite_a_band Py.Bool.of_bool); ("select", Wrap_utils.Option.map select (function
| `A -> Py.String.of_string "a"
| `V -> Py.String.of_string "v"
| `I -> Py.String.of_string "i"
)); ("select_range", select_range); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a_band", Some(a_band ))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eigvalsh ?b ?lower ?overwrite_a ?overwrite_b ?turbo ?eigvals ?type_ ?check_finite ?subset_by_index ?subset_by_value ?driver ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "eigvalsh"
     [||]
     (Wrap_utils.keyword_args [("b", Wrap_utils.Option.map b Np.Obj.to_pyobject); ("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("turbo", Wrap_utils.Option.map turbo Py.Bool.of_bool); ("eigvals", eigvals); ("type", Wrap_utils.Option.map type_ Py.Int.of_int); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("subset_by_index", Wrap_utils.Option.map subset_by_index Np.Obj.to_pyobject); ("subset_by_value", Wrap_utils.Option.map subset_by_value Np.Obj.to_pyobject); ("driver", Wrap_utils.Option.map driver Py.String.of_string); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let eigvalsh_tridiagonal ?select ?select_range ?check_finite ?tol ?lapack_driver ~d ~e () =
                     Py.Module.get_function_with_keywords __wrap_namespace "eigvalsh_tridiagonal"
                       [||]
                       (Wrap_utils.keyword_args [("select", Wrap_utils.Option.map select (function
| `A -> Py.String.of_string "a"
| `V -> Py.String.of_string "v"
| `I -> Py.String.of_string "i"
)); ("select_range", select_range); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("lapack_driver", Wrap_utils.Option.map lapack_driver Py.String.of_string); ("d", Some(d |> Np.Obj.to_pyobject)); ("e", Some(e |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let einsum ?out ?optimize ?kwargs operands =
                     Py.Module.get_function_with_keywords __wrap_namespace "einsum"
                       (Array.of_list @@ List.concat [(List.map Wrap_utils.id operands)])
                       (List.rev_append (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("optimize", Wrap_utils.Option.map optimize (function
| `Optimal -> Py.String.of_string "optimal"
| `Greedy -> Py.String.of_string "greedy"
| `Bool x -> Py.Bool.of_bool x
))]) (match kwargs with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let empty ?dtype ?order ~shape () =
                     Py.Module.get_function_with_keywords __wrap_namespace "empty"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (function
| `I x -> Py.Int.of_int x
| `Tuple_of_int x -> Wrap_utils.id x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let eye ?m ?k ?dtype ?order ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "eye"
                       [||]
                       (Wrap_utils.keyword_args [("M", Wrap_utils.Option.map m Py.Int.of_int); ("k", Wrap_utils.Option.map k Py.Int.of_int); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("N", Some(n |> Py.Int.of_int))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let flatnonzero a =
   Py.Module.get_function_with_keywords __wrap_namespace "flatnonzero"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let get_lapack_funcs ?arrays ?dtype ~names () =
                     Py.Module.get_function_with_keywords __wrap_namespace "get_lapack_funcs"
                       [||]
                       (Wrap_utils.keyword_args [("arrays", Wrap_utils.Option.map arrays (fun ml -> Py.List.of_list_map Np.Obj.to_pyobject ml)); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("names", Some(names |> (function
| `Sequence_of_str x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let hessenberg ?calc_q ?overwrite_a ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "hessenberg"
     [||]
     (Wrap_utils.keyword_args [("calc_q", Wrap_utils.Option.map calc_q Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let iscomplex x =
   Py.Module.get_function_with_keywords __wrap_namespace "iscomplex"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject))])

let iscomplexobj x =
   Py.Module.get_function_with_keywords __wrap_namespace "iscomplexobj"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x ))])
     |> Py.Bool.to_bool
                  let isfinite ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "isfinite"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let nonzero a =
   Py.Module.get_function_with_keywords __wrap_namespace "nonzero"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Np.Obj.to_pyobject))])

                  let norm ?ord ?axis ?keepdims ?check_finite ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "norm"
                       [||]
                       (Wrap_utils.keyword_args [("ord", Wrap_utils.Option.map ord (function
| `PyObject x -> Wrap_utils.id x
| `Fro -> Py.String.of_string "fro"
)); ("axis", Wrap_utils.Option.map axis (function
| `T2_tuple_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a ))])

                  let zeros ?dtype ?order ~shape () =
                     Py.Module.get_function_with_keywords __wrap_namespace "zeros"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
module Decomp_cholesky = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.linalg.decomp_cholesky"

let get_py name = Py.Module.get __wrap_namespace name
                  let asarray ?dtype ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let asarray_chkfinite ?dtype ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray_chkfinite"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let atleast_2d arys =
   Py.Module.get_function_with_keywords __wrap_namespace "atleast_2d"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arys)])
     []

let cho_factor ?lower ?overwrite_a ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "cho_factor"
     [||]
     (Wrap_utils.keyword_args [("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Bool.to_bool (Py.Tuple.get x 1))))
let cho_solve ?overwrite_b ?check_finite ~c_and_lower ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "cho_solve"
     [||]
     (Wrap_utils.keyword_args [("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("c_and_lower", Some(c_and_lower )); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let cho_solve_banded ?overwrite_b ?check_finite ~cb_and_lower ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "cho_solve_banded"
     [||]
     (Wrap_utils.keyword_args [("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("cb_and_lower", Some(cb_and_lower )); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let cholesky ?lower ?overwrite_a ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "cholesky"
     [||]
     (Wrap_utils.keyword_args [("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let cholesky_banded ?overwrite_ab ?lower ?check_finite ~ab () =
   Py.Module.get_function_with_keywords __wrap_namespace "cholesky_banded"
     [||]
     (Wrap_utils.keyword_args [("overwrite_ab", Wrap_utils.Option.map overwrite_ab Py.Bool.of_bool); ("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("ab", Some(ab ))])

                  let get_lapack_funcs ?arrays ?dtype ~names () =
                     Py.Module.get_function_with_keywords __wrap_namespace "get_lapack_funcs"
                       [||]
                       (Wrap_utils.keyword_args [("arrays", Wrap_utils.Option.map arrays (fun ml -> Py.List.of_list_map Np.Obj.to_pyobject ml)); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("names", Some(names |> (function
| `Sequence_of_str x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
module Decomp_lu = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.linalg.decomp_lu"

let get_py name = Py.Module.get __wrap_namespace name
                  let asarray ?dtype ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let asarray_chkfinite ?dtype ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray_chkfinite"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let get_flinalg_funcs ?arrays ?debug ~names () =
   Py.Module.get_function_with_keywords __wrap_namespace "get_flinalg_funcs"
     [||]
     (Wrap_utils.keyword_args [("arrays", arrays); ("debug", debug); ("names", Some(names ))])

                  let get_lapack_funcs ?arrays ?dtype ~names () =
                     Py.Module.get_function_with_keywords __wrap_namespace "get_lapack_funcs"
                       [||]
                       (Wrap_utils.keyword_args [("arrays", Wrap_utils.Option.map arrays (fun ml -> Py.List.of_list_map Np.Obj.to_pyobject ml)); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("names", Some(names |> (function
| `Sequence_of_str x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let lu ?permute_l ?overwrite_a ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "lu"
     [||]
     (Wrap_utils.keyword_args [("permute_l", Wrap_utils.Option.map permute_l Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3))))
let lu_factor ?overwrite_a ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "lu_factor"
     [||]
     (Wrap_utils.keyword_args [("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let lu_solve ?trans ?overwrite_b ?check_finite ~lu_and_piv ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "lu_solve"
                       [||]
                       (Wrap_utils.keyword_args [("trans", Wrap_utils.Option.map trans (function
| `Two -> Py.Int.of_int 2
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
)); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("lu_and_piv", Some(lu_and_piv )); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let warn ?category ?stacklevel ?source ~message () =
   Py.Module.get_function_with_keywords __wrap_namespace "warn"
     [||]
     (Wrap_utils.keyword_args [("category", category); ("stacklevel", stacklevel); ("source", source); ("message", Some(message ))])


end
module Decomp_qr = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.linalg.decomp_qr"

let get_py name = Py.Module.get __wrap_namespace name
                  let get_lapack_funcs ?arrays ?dtype ~names () =
                     Py.Module.get_function_with_keywords __wrap_namespace "get_lapack_funcs"
                       [||]
                       (Wrap_utils.keyword_args [("arrays", Wrap_utils.Option.map arrays (fun ml -> Py.List.of_list_map Np.Obj.to_pyobject ml)); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("names", Some(names |> (function
| `Sequence_of_str x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let qr ?overwrite_a ?lwork ?mode ?pivoting ?check_finite ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "qr"
                       [||]
                       (Wrap_utils.keyword_args [("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("lwork", Wrap_utils.Option.map lwork Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Full -> Py.String.of_string "full"
| `R -> Py.String.of_string "r"
| `Economic -> Py.String.of_string "economic"
| `Raw -> Py.String.of_string "raw"
)); ("pivoting", Wrap_utils.Option.map pivoting Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
                  let qr_multiply ?mode ?pivoting ?conjugate ?overwrite_a ?overwrite_c ~a ~c () =
                     Py.Module.get_function_with_keywords __wrap_namespace "qr_multiply"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Left -> Py.String.of_string "left"
| `Right -> Py.String.of_string "right"
)); ("pivoting", Wrap_utils.Option.map pivoting Py.Bool.of_bool); ("conjugate", Wrap_utils.Option.map conjugate Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("overwrite_c", Wrap_utils.Option.map overwrite_c Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject)); ("c", Some(c |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
                  let rq ?overwrite_a ?lwork ?mode ?check_finite ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "rq"
                       [||]
                       (Wrap_utils.keyword_args [("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("lwork", Wrap_utils.Option.map lwork Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Full -> Py.String.of_string "full"
| `R -> Py.String.of_string "r"
| `Economic -> Py.String.of_string "economic"
)); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let safecall ?kwargs ~f ~name args =
   Py.Module.get_function_with_keywords __wrap_namespace "safecall"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("f", Some(f )); ("name", Some(name ))]) (match kwargs with None -> [] | Some x -> x))


end
module Decomp_schur = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.linalg.decomp_schur"

let get_py name = Py.Module.get __wrap_namespace name
module Single = struct
type tag = [`Float32]
type t = [`Float32 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
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
                  let array ?dtype ?copy ?order ?subok ?ndmin ~object_ () =
                     Py.Module.get_function_with_keywords __wrap_namespace "array"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("order", Wrap_utils.Option.map order (function
| `K -> Py.String.of_string "K"
| `A -> Py.String.of_string "A"
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("ndmin", Wrap_utils.Option.map ndmin Py.Int.of_int); ("object", Some(object_ |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let asarray ?dtype ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let asarray_chkfinite ?dtype ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "asarray_chkfinite"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eigvals ?b ?overwrite_a ?check_finite ?homogeneous_eigvals ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "eigvals"
     [||]
     (Wrap_utils.keyword_args [("b", Wrap_utils.Option.map b Np.Obj.to_pyobject); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("homogeneous_eigvals", Wrap_utils.Option.map homogeneous_eigvals Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])

                  let get_lapack_funcs ?arrays ?dtype ~names () =
                     Py.Module.get_function_with_keywords __wrap_namespace "get_lapack_funcs"
                       [||]
                       (Wrap_utils.keyword_args [("arrays", Wrap_utils.Option.map arrays (fun ml -> Py.List.of_list_map Np.Obj.to_pyobject ml)); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("names", Some(names |> (function
| `Sequence_of_str x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let norm ?ord ?axis ?keepdims ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "norm"
                       [||]
                       (Wrap_utils.keyword_args [("ord", Wrap_utils.Option.map ord (function
| `PyObject x -> Wrap_utils.id x
| `Nuc -> Py.String.of_string "nuc"
| `Fro -> Py.String.of_string "fro"
)); ("axis", Wrap_utils.Option.map axis (function
| `T2_tuple_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])

let rsf2csf ?check_finite ~t ~z () =
   Py.Module.get_function_with_keywords __wrap_namespace "rsf2csf"
     [||]
     (Wrap_utils.keyword_args [("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("T", Some(t |> Np.Obj.to_pyobject)); ("Z", Some(z |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let schur ?output ?lwork ?overwrite_a ?sort ?check_finite ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "schur"
                       [||]
                       (Wrap_utils.keyword_args [("output", Wrap_utils.Option.map output (function
| `Real -> Py.String.of_string "real"
| `Complex -> Py.String.of_string "complex"
)); ("lwork", Wrap_utils.Option.map lwork Py.Int.of_int); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("sort", Wrap_utils.Option.map sort (function
| `Rhp -> Py.String.of_string "rhp"
| `Iuc -> Py.String.of_string "iuc"
| `Lhp -> Py.String.of_string "lhp"
| `Ouc -> Py.String.of_string "ouc"
| `Callable x -> Wrap_utils.id x
)); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2))))

end
module Decomp_svd = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.linalg.decomp_svd"

let get_py name = Py.Module.get __wrap_namespace name
                  let arccos ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "arccos"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let arcsin ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "arcsin"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let clip ?out ?kwargs ~a ~a_min ~a_max () =
                     Py.Module.get_function_with_keywords __wrap_namespace "clip"
                       [||]
                       (List.rev_append (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("a", Some(a |> Np.Obj.to_pyobject)); ("a_min", Some(a_min |> (function
| `Bool x -> Py.Bool.of_bool x
| `F x -> Py.Float.of_float x
| `S x -> Py.String.of_string x
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `None -> Py.none
))); ("a_max", Some(a_max |> (function
| `Bool x -> Py.Bool.of_bool x
| `F x -> Py.Float.of_float x
| `S x -> Py.String.of_string x
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
| `None -> Py.none
)))]) (match kwargs with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let diag ?k ~v () =
   Py.Module.get_function_with_keywords __wrap_namespace "diag"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("v", Some(v |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let diagsvd ~s ~m ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "diagsvd"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s )); ("M", Some(m |> Py.Int.of_int)); ("N", Some(n |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let dot ?out ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "dot"
     [||]
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let get_lapack_funcs ?arrays ?dtype ~names () =
                     Py.Module.get_function_with_keywords __wrap_namespace "get_lapack_funcs"
                       [||]
                       (Wrap_utils.keyword_args [("arrays", Wrap_utils.Option.map arrays (fun ml -> Py.List.of_list_map Np.Obj.to_pyobject ml)); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("names", Some(names |> (function
| `Sequence_of_str x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let null_space ?rcond ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "null_space"
     [||]
     (Wrap_utils.keyword_args [("rcond", Wrap_utils.Option.map rcond Py.Float.of_float); ("A", Some(a |> Np.Obj.to_pyobject))])

let orth ?rcond ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "orth"
     [||]
     (Wrap_utils.keyword_args [("rcond", Wrap_utils.Option.map rcond Py.Float.of_float); ("A", Some(a |> Np.Obj.to_pyobject))])

let subspace_angles ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "subspace_angles"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject)); ("B", Some(b ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let svd ?full_matrices ?compute_uv ?overwrite_a ?check_finite ?lapack_driver ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "svd"
                       [||]
                       (Wrap_utils.keyword_args [("full_matrices", Wrap_utils.Option.map full_matrices Py.Bool.of_bool); ("compute_uv", Wrap_utils.Option.map compute_uv Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("lapack_driver", Wrap_utils.Option.map lapack_driver (function
| `Gesdd -> Py.String.of_string "gesdd"
| `Gesvd -> Py.String.of_string "gesvd"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
let svdvals ?overwrite_a ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "svdvals"
     [||]
     (Wrap_utils.keyword_args [("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])

                  let where ?x ?y ~condition () =
                     Py.Module.get_function_with_keywords __wrap_namespace "where"
                       [||]
                       (Wrap_utils.keyword_args [("x", x); ("y", y); ("condition", Some(condition |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Bool x -> Py.Bool.of_bool x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let zeros ?dtype ?order ~shape () =
                     Py.Module.get_function_with_keywords __wrap_namespace "zeros"
                       [||]
                       (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
)); ("shape", Some(shape |> (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
module Flinalg = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.linalg.flinalg"

let get_py name = Py.Module.get __wrap_namespace name
let get_flinalg_funcs ?arrays ?debug ~names () =
   Py.Module.get_function_with_keywords __wrap_namespace "get_flinalg_funcs"
     [||]
     (Wrap_utils.keyword_args [("arrays", arrays); ("debug", debug); ("names", Some(names ))])

let has_column_major_storage arr =
   Py.Module.get_function_with_keywords __wrap_namespace "has_column_major_storage"
     [||]
     (Wrap_utils.keyword_args [("arr", Some(arr ))])


end
module Lapack = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.linalg.lapack"

let get_py name = Py.Module.get __wrap_namespace name
let cgegv ?kwds args =
   Py.Module.get_function_with_keywords __wrap_namespace "cgegv"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwds with None -> [] | Some x -> x)

let dgegv ?kwds args =
   Py.Module.get_function_with_keywords __wrap_namespace "dgegv"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwds with None -> [] | Some x -> x)

                  let get_lapack_funcs ?arrays ?dtype ~names () =
                     Py.Module.get_function_with_keywords __wrap_namespace "get_lapack_funcs"
                       [||]
                       (Wrap_utils.keyword_args [("arrays", Wrap_utils.Option.map arrays (fun ml -> Py.List.of_list_map Np.Obj.to_pyobject ml)); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("names", Some(names |> (function
| `Sequence_of_str x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let sgegv ?kwds args =
   Py.Module.get_function_with_keywords __wrap_namespace "sgegv"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwds with None -> [] | Some x -> x)

let zgegv ?kwds args =
   Py.Module.get_function_with_keywords __wrap_namespace "zgegv"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwds with None -> [] | Some x -> x)


end
module Matfuncs = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.linalg.matfuncs"

let get_py name = Py.Module.get __wrap_namespace name
module Single = struct
type tag = [`Float32]
type t = [`Float32 | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
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
                  let absolute ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "absolute"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let amax ?axis ?out ?keepdims ?initial ?where ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "amax"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("initial", Wrap_utils.Option.map initial (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("where", where); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let conjugate ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "conjugate"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let coshm a =
   Py.Module.get_function_with_keywords __wrap_namespace "coshm"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let cosm a =
   Py.Module.get_function_with_keywords __wrap_namespace "cosm"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let diag ?k ~v () =
   Py.Module.get_function_with_keywords __wrap_namespace "diag"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("v", Some(v |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let dot ?out ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "dot"
     [||]
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let expm a =
   Py.Module.get_function_with_keywords __wrap_namespace "expm"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let expm_cond ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "expm_cond"
     [||]
     (Wrap_utils.keyword_args [("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("A", Some(a ))])
     |> Py.Float.to_float
let expm_frechet ?method_ ?compute_expm ?check_finite ~a ~e () =
   Py.Module.get_function_with_keywords __wrap_namespace "expm_frechet"
     [||]
     (Wrap_utils.keyword_args [("method", Wrap_utils.Option.map method_ Py.String.of_string); ("compute_expm", Wrap_utils.Option.map compute_expm Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("A", Some(a |> Np.Obj.to_pyobject)); ("E", Some(e |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let fractional_matrix_power ~a ~t () =
   Py.Module.get_function_with_keywords __wrap_namespace "fractional_matrix_power"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject)); ("t", Some(t |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let funm ?disp ~a ~func () =
   Py.Module.get_function_with_keywords __wrap_namespace "funm"
     [||]
     (Wrap_utils.keyword_args [("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("A", Some(a |> Np.Obj.to_pyobject)); ("func", Some(func ))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))
let inv ?overwrite_a ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "inv"
     [||]
     (Wrap_utils.keyword_args [("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let isfinite ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "isfinite"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let khatri_rao ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "khatri_rao"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a )); ("b", Some(b ))])

                  let logical_not ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "logical_not"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])

let logm ?disp ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "logm"
     [||]
     (Wrap_utils.keyword_args [("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("A", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))
                  let norm ?ord ?axis ?keepdims ?check_finite ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "norm"
                       [||]
                       (Wrap_utils.keyword_args [("ord", Wrap_utils.Option.map ord (function
| `PyObject x -> Wrap_utils.id x
| `Fro -> Py.String.of_string "fro"
)); ("axis", Wrap_utils.Option.map axis (function
| `T2_tuple_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a ))])

                  let prod ?axis ?dtype ?out ?keepdims ?initial ?where ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "prod"
                       [||]
                       (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("initial", Wrap_utils.Option.map initial (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("where", where); ("a", Some(a |> Np.Obj.to_pyobject))])

                  let ravel ?order ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ravel"
                       [||]
                       (Wrap_utils.keyword_args [("order", Wrap_utils.Option.map order (function
| `C -> Py.String.of_string "C"
| `F -> Py.String.of_string "F"
| `A -> Py.String.of_string "A"
| `K -> Py.String.of_string "K"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let rsf2csf ?check_finite ~t ~z () =
   Py.Module.get_function_with_keywords __wrap_namespace "rsf2csf"
     [||]
     (Wrap_utils.keyword_args [("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("T", Some(t |> Np.Obj.to_pyobject)); ("Z", Some(z |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let schur ?output ?lwork ?overwrite_a ?sort ?check_finite ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "schur"
                       [||]
                       (Wrap_utils.keyword_args [("output", Wrap_utils.Option.map output (function
| `Real -> Py.String.of_string "real"
| `Complex -> Py.String.of_string "complex"
)); ("lwork", Wrap_utils.Option.map lwork Py.Int.of_int); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("sort", Wrap_utils.Option.map sort (function
| `Rhp -> Py.String.of_string "rhp"
| `Iuc -> Py.String.of_string "iuc"
| `Lhp -> Py.String.of_string "lhp"
| `Ouc -> Py.String.of_string "ouc"
| `Callable x -> Wrap_utils.id x
)); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2))))
                  let sign ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "sign"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let signm ?disp ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "signm"
     [||]
     (Wrap_utils.keyword_args [("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("A", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))
let sinhm a =
   Py.Module.get_function_with_keywords __wrap_namespace "sinhm"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let sinm a =
   Py.Module.get_function_with_keywords __wrap_namespace "sinm"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let solve ?sym_pos ?lower ?overwrite_a ?overwrite_b ?debug ?check_finite ?assume_a ?transposed ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "solve"
     [||]
     (Wrap_utils.keyword_args [("sym_pos", Wrap_utils.Option.map sym_pos Py.Bool.of_bool); ("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("debug", debug); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("assume_a", Wrap_utils.Option.map assume_a Py.String.of_string); ("transposed", Wrap_utils.Option.map transposed Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let sqrtm ?disp ?blocksize ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "sqrtm"
     [||]
     (Wrap_utils.keyword_args [("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("blocksize", Wrap_utils.Option.map blocksize Py.Int.of_int); ("A", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))
                  let svd ?full_matrices ?compute_uv ?overwrite_a ?check_finite ?lapack_driver ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "svd"
                       [||]
                       (Wrap_utils.keyword_args [("full_matrices", Wrap_utils.Option.map full_matrices Py.Bool.of_bool); ("compute_uv", Wrap_utils.Option.map compute_uv Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("lapack_driver", Wrap_utils.Option.map lapack_driver (function
| `Gesdd -> Py.String.of_string "gesdd"
| `Gesvd -> Py.String.of_string "gesvd"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
let tanhm a =
   Py.Module.get_function_with_keywords __wrap_namespace "tanhm"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let tanm a =
   Py.Module.get_function_with_keywords __wrap_namespace "tanm"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let transpose ?axes ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "transpose"
     [||]
     (Wrap_utils.keyword_args [("axes", axes); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let triu ?k ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "triu"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("m", Some(m |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
module Misc = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.linalg.misc"

let get_py name = Py.Module.get __wrap_namespace name
                  let get_blas_funcs ?arrays ?dtype ~names () =
                     Py.Module.get_function_with_keywords __wrap_namespace "get_blas_funcs"
                       [||]
                       (Wrap_utils.keyword_args [("arrays", Wrap_utils.Option.map arrays (fun ml -> Py.List.of_list_map Np.Obj.to_pyobject ml)); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("names", Some(names |> (function
| `Sequence_of_str x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let get_lapack_funcs ?arrays ?dtype ~names () =
                     Py.Module.get_function_with_keywords __wrap_namespace "get_lapack_funcs"
                       [||]
                       (Wrap_utils.keyword_args [("arrays", Wrap_utils.Option.map arrays (fun ml -> Py.List.of_list_map Np.Obj.to_pyobject ml)); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("names", Some(names |> (function
| `Sequence_of_str x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let norm ?ord ?axis ?keepdims ?check_finite ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "norm"
                       [||]
                       (Wrap_utils.keyword_args [("ord", Wrap_utils.Option.map ord (function
| `PyObject x -> Wrap_utils.id x
| `Fro -> Py.String.of_string "fro"
)); ("axis", Wrap_utils.Option.map axis (function
| `T2_tuple_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a ))])


end
module Special_matrices = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.linalg.special_matrices"

let get_py name = Py.Module.get __wrap_namespace name
let as_strided ?shape ?strides ?subok ?writeable ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "as_strided"
     [||]
     (Wrap_utils.keyword_args [("shape", shape); ("strides", strides); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("writeable", Wrap_utils.Option.map writeable Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let block_diag arrs =
   Py.Module.get_function_with_keywords __wrap_namespace "block_diag"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arrs)])
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let circulant c =
   Py.Module.get_function_with_keywords __wrap_namespace "circulant"
     [||]
     (Wrap_utils.keyword_args [("c", Some(c |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let companion a =
   Py.Module.get_function_with_keywords __wrap_namespace "companion"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Np.Obj.to_pyobject))])

let convolution_matrix ?mode ~a ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "convolution_matrix"
     [||]
     (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode Py.String.of_string); ("a", Some(a |> Np.Obj.to_pyobject)); ("n", Some(n |> Py.Int.of_int))])

let dft ?scale ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "dft"
     [||]
     (Wrap_utils.keyword_args [("scale", Wrap_utils.Option.map scale Py.Float.of_float); ("n", Some(n |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let fiedler a =
   Py.Module.get_function_with_keywords __wrap_namespace "fiedler"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let fiedler_companion a =
   Py.Module.get_function_with_keywords __wrap_namespace "fiedler_companion"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Np.Obj.to_pyobject))])

let hadamard ?dtype ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "hadamard"
     [||]
     (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("n", Some(n |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let hankel ?r ~c () =
   Py.Module.get_function_with_keywords __wrap_namespace "hankel"
     [||]
     (Wrap_utils.keyword_args [("r", Wrap_utils.Option.map r Np.Obj.to_pyobject); ("c", Some(c |> Np.Obj.to_pyobject))])

let helmert ?full ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "helmert"
     [||]
     (Wrap_utils.keyword_args [("full", Wrap_utils.Option.map full Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let hilbert n =
   Py.Module.get_function_with_keywords __wrap_namespace "hilbert"
     [||]
     (Wrap_utils.keyword_args [("n", Some(n |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let invhilbert ?exact ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "invhilbert"
     [||]
     (Wrap_utils.keyword_args [("exact", Wrap_utils.Option.map exact Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let invpascal ?kind ?exact ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "invpascal"
     [||]
     (Wrap_utils.keyword_args [("kind", Wrap_utils.Option.map kind Py.String.of_string); ("exact", Wrap_utils.Option.map exact Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let kron ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "kron"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b ))])

let leslie ~f ~s () =
   Py.Module.get_function_with_keywords __wrap_namespace "leslie"
     [||]
     (Wrap_utils.keyword_args [("f", Some(f |> Np.Obj.to_pyobject)); ("s", Some(s ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let pascal ?kind ?exact ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "pascal"
     [||]
     (Wrap_utils.keyword_args [("kind", Wrap_utils.Option.map kind Py.String.of_string); ("exact", Wrap_utils.Option.map exact Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let toeplitz ?r ~c () =
   Py.Module.get_function_with_keywords __wrap_namespace "toeplitz"
     [||]
     (Wrap_utils.keyword_args [("r", Wrap_utils.Option.map r Np.Obj.to_pyobject); ("c", Some(c |> Np.Obj.to_pyobject))])

let tri ?m ?k ?dtype ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "tri"
     [||]
     (Wrap_utils.keyword_args [("M", Wrap_utils.Option.map m Py.Int.of_int); ("k", Wrap_utils.Option.map k Py.Int.of_int); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("N", Some(n |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let tril ?k ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "tril"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("m", Some(m |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let triu ?k ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "triu"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("m", Some(m |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
let block_diag arrs =
   Py.Module.get_function_with_keywords __wrap_namespace "block_diag"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id arrs)])
     []
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let cdf2rdf ~w ~v () =
   Py.Module.get_function_with_keywords __wrap_namespace "cdf2rdf"
     [||]
     (Wrap_utils.keyword_args [("w", Some(w )); ("v", Some(v ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let cho_factor ?lower ?overwrite_a ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "cho_factor"
     [||]
     (Wrap_utils.keyword_args [("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Bool.to_bool (Py.Tuple.get x 1))))
let cho_solve ?overwrite_b ?check_finite ~c_and_lower ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "cho_solve"
     [||]
     (Wrap_utils.keyword_args [("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("c_and_lower", Some(c_and_lower )); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let cho_solve_banded ?overwrite_b ?check_finite ~cb_and_lower ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "cho_solve_banded"
     [||]
     (Wrap_utils.keyword_args [("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("cb_and_lower", Some(cb_and_lower )); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let cholesky ?lower ?overwrite_a ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "cholesky"
     [||]
     (Wrap_utils.keyword_args [("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let cholesky_banded ?overwrite_ab ?lower ?check_finite ~ab () =
   Py.Module.get_function_with_keywords __wrap_namespace "cholesky_banded"
     [||]
     (Wrap_utils.keyword_args [("overwrite_ab", Wrap_utils.Option.map overwrite_ab Py.Bool.of_bool); ("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("ab", Some(ab ))])

let circulant c =
   Py.Module.get_function_with_keywords __wrap_namespace "circulant"
     [||]
     (Wrap_utils.keyword_args [("c", Some(c |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let clarkson_woodruff_transform ?seed ~input_matrix ~sketch_size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "clarkson_woodruff_transform"
                       [||]
                       (Wrap_utils.keyword_args [("seed", Wrap_utils.Option.map seed (function
| `T_numpy_random_RandomState_instance x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("input_matrix", Some(input_matrix |> Np.Obj.to_pyobject)); ("sketch_size", Some(sketch_size |> Py.Int.of_int))])

let companion a =
   Py.Module.get_function_with_keywords __wrap_namespace "companion"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Np.Obj.to_pyobject))])

let convolution_matrix ?mode ~a ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "convolution_matrix"
     [||]
     (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode Py.String.of_string); ("a", Some(a |> Np.Obj.to_pyobject)); ("n", Some(n |> Py.Int.of_int))])

let coshm a =
   Py.Module.get_function_with_keywords __wrap_namespace "coshm"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let cosm a =
   Py.Module.get_function_with_keywords __wrap_namespace "cosm"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let cossin ?p ?q ?separate ?swap_sign ?compute_u ?compute_vh ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "cossin"
     [||]
     (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p Py.Int.of_int); ("q", Wrap_utils.Option.map q Py.Int.of_int); ("separate", Wrap_utils.Option.map separate Py.Bool.of_bool); ("swap_sign", Wrap_utils.Option.map swap_sign Py.Bool.of_bool); ("compute_u", Wrap_utils.Option.map compute_u Py.Bool.of_bool); ("compute_vh", Wrap_utils.Option.map compute_vh Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
let det ?overwrite_a ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "det"
     [||]
     (Wrap_utils.keyword_args [("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])

let dft ?scale ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "dft"
     [||]
     (Wrap_utils.keyword_args [("scale", Wrap_utils.Option.map scale Py.Float.of_float); ("n", Some(n |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let diagsvd ~s ~m ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "diagsvd"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s )); ("M", Some(m |> Py.Int.of_int)); ("N", Some(n |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eig ?b ?left ?right ?overwrite_a ?overwrite_b ?check_finite ?homogeneous_eigvals ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "eig"
     [||]
     (Wrap_utils.keyword_args [("b", Wrap_utils.Option.map b Np.Obj.to_pyobject); ("left", Wrap_utils.Option.map left Py.Bool.of_bool); ("right", Wrap_utils.Option.map right Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("homogeneous_eigvals", Wrap_utils.Option.map homogeneous_eigvals Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
                  let eig_banded ?lower ?eigvals_only ?overwrite_a_band ?select ?select_range ?max_ev ?check_finite ~a_band () =
                     Py.Module.get_function_with_keywords __wrap_namespace "eig_banded"
                       [||]
                       (Wrap_utils.keyword_args [("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("eigvals_only", Wrap_utils.Option.map eigvals_only Py.Bool.of_bool); ("overwrite_a_band", Wrap_utils.Option.map overwrite_a_band Py.Bool.of_bool); ("select", Wrap_utils.Option.map select (function
| `A -> Py.String.of_string "a"
| `V -> Py.String.of_string "v"
| `I -> Py.String.of_string "i"
)); ("select_range", select_range); ("max_ev", Wrap_utils.Option.map max_ev Py.Int.of_int); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a_band", Some(a_band ))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let eigh ?b ?lower ?eigvals_only ?overwrite_a ?overwrite_b ?turbo ?eigvals ?type_ ?check_finite ?subset_by_index ?subset_by_value ?driver ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "eigh"
     [||]
     (Wrap_utils.keyword_args [("b", Wrap_utils.Option.map b Np.Obj.to_pyobject); ("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("eigvals_only", Wrap_utils.Option.map eigvals_only Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("turbo", Wrap_utils.Option.map turbo Py.Bool.of_bool); ("eigvals", eigvals); ("type", Wrap_utils.Option.map type_ Py.Int.of_int); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("subset_by_index", Wrap_utils.Option.map subset_by_index Np.Obj.to_pyobject); ("subset_by_value", Wrap_utils.Option.map subset_by_value Np.Obj.to_pyobject); ("driver", Wrap_utils.Option.map driver Py.String.of_string); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let eigh_tridiagonal ?eigvals_only ?select ?select_range ?check_finite ?tol ?lapack_driver ~d ~e () =
                     Py.Module.get_function_with_keywords __wrap_namespace "eigh_tridiagonal"
                       [||]
                       (Wrap_utils.keyword_args [("eigvals_only", eigvals_only); ("select", Wrap_utils.Option.map select (function
| `A -> Py.String.of_string "a"
| `V -> Py.String.of_string "v"
| `I -> Py.String.of_string "i"
)); ("select_range", select_range); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("lapack_driver", Wrap_utils.Option.map lapack_driver Py.String.of_string); ("d", Some(d |> Np.Obj.to_pyobject)); ("e", Some(e |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let eigvals ?b ?overwrite_a ?check_finite ?homogeneous_eigvals ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "eigvals"
     [||]
     (Wrap_utils.keyword_args [("b", Wrap_utils.Option.map b Np.Obj.to_pyobject); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("homogeneous_eigvals", Wrap_utils.Option.map homogeneous_eigvals Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])

                  let eigvals_banded ?lower ?overwrite_a_band ?select ?select_range ?check_finite ~a_band () =
                     Py.Module.get_function_with_keywords __wrap_namespace "eigvals_banded"
                       [||]
                       (Wrap_utils.keyword_args [("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("overwrite_a_band", Wrap_utils.Option.map overwrite_a_band Py.Bool.of_bool); ("select", Wrap_utils.Option.map select (function
| `A -> Py.String.of_string "a"
| `V -> Py.String.of_string "v"
| `I -> Py.String.of_string "i"
)); ("select_range", select_range); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a_band", Some(a_band ))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eigvalsh ?b ?lower ?overwrite_a ?overwrite_b ?turbo ?eigvals ?type_ ?check_finite ?subset_by_index ?subset_by_value ?driver ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "eigvalsh"
     [||]
     (Wrap_utils.keyword_args [("b", Wrap_utils.Option.map b Np.Obj.to_pyobject); ("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("turbo", Wrap_utils.Option.map turbo Py.Bool.of_bool); ("eigvals", eigvals); ("type", Wrap_utils.Option.map type_ Py.Int.of_int); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("subset_by_index", Wrap_utils.Option.map subset_by_index Np.Obj.to_pyobject); ("subset_by_value", Wrap_utils.Option.map subset_by_value Np.Obj.to_pyobject); ("driver", Wrap_utils.Option.map driver Py.String.of_string); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let eigvalsh_tridiagonal ?select ?select_range ?check_finite ?tol ?lapack_driver ~d ~e () =
                     Py.Module.get_function_with_keywords __wrap_namespace "eigvalsh_tridiagonal"
                       [||]
                       (Wrap_utils.keyword_args [("select", Wrap_utils.Option.map select (function
| `A -> Py.String.of_string "a"
| `V -> Py.String.of_string "v"
| `I -> Py.String.of_string "i"
)); ("select_range", select_range); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("lapack_driver", Wrap_utils.Option.map lapack_driver Py.String.of_string); ("d", Some(d |> Np.Obj.to_pyobject)); ("e", Some(e |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let expm a =
   Py.Module.get_function_with_keywords __wrap_namespace "expm"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let expm_cond ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "expm_cond"
     [||]
     (Wrap_utils.keyword_args [("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("A", Some(a ))])
     |> Py.Float.to_float
let expm_frechet ?method_ ?compute_expm ?check_finite ~a ~e () =
   Py.Module.get_function_with_keywords __wrap_namespace "expm_frechet"
     [||]
     (Wrap_utils.keyword_args [("method", Wrap_utils.Option.map method_ Py.String.of_string); ("compute_expm", Wrap_utils.Option.map compute_expm Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("A", Some(a |> Np.Obj.to_pyobject)); ("E", Some(e |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let fiedler a =
   Py.Module.get_function_with_keywords __wrap_namespace "fiedler"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let fiedler_companion a =
   Py.Module.get_function_with_keywords __wrap_namespace "fiedler_companion"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Np.Obj.to_pyobject))])

                  let find_best_blas_type ?arrays ?dtype () =
                     Py.Module.get_function_with_keywords __wrap_namespace "find_best_blas_type"
                       [||]
                       (Wrap_utils.keyword_args [("arrays", Wrap_utils.Option.map arrays (fun ml -> Py.List.of_list_map Np.Obj.to_pyobject ml)); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
))])
                       |> (fun x -> ((Py.String.to_string (Py.Tuple.get x 0)), (Np.Dtype.of_pyobject (Py.Tuple.get x 1)), (Py.Bool.to_bool (Py.Tuple.get x 2))))
let fractional_matrix_power ~a ~t () =
   Py.Module.get_function_with_keywords __wrap_namespace "fractional_matrix_power"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject)); ("t", Some(t |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let funm ?disp ~a ~func () =
   Py.Module.get_function_with_keywords __wrap_namespace "funm"
     [||]
     (Wrap_utils.keyword_args [("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("A", Some(a |> Np.Obj.to_pyobject)); ("func", Some(func ))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))
                  let get_blas_funcs ?arrays ?dtype ~names () =
                     Py.Module.get_function_with_keywords __wrap_namespace "get_blas_funcs"
                       [||]
                       (Wrap_utils.keyword_args [("arrays", Wrap_utils.Option.map arrays (fun ml -> Py.List.of_list_map Np.Obj.to_pyobject ml)); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("names", Some(names |> (function
| `Sequence_of_str x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let get_lapack_funcs ?arrays ?dtype ~names () =
                     Py.Module.get_function_with_keywords __wrap_namespace "get_lapack_funcs"
                       [||]
                       (Wrap_utils.keyword_args [("arrays", Wrap_utils.Option.map arrays (fun ml -> Py.List.of_list_map Np.Obj.to_pyobject ml)); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
)); ("names", Some(names |> (function
| `Sequence_of_str x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let hadamard ?dtype ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "hadamard"
     [||]
     (Wrap_utils.keyword_args [("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("n", Some(n |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let hankel ?r ~c () =
   Py.Module.get_function_with_keywords __wrap_namespace "hankel"
     [||]
     (Wrap_utils.keyword_args [("r", Wrap_utils.Option.map r Np.Obj.to_pyobject); ("c", Some(c |> Np.Obj.to_pyobject))])

let helmert ?full ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "helmert"
     [||]
     (Wrap_utils.keyword_args [("full", Wrap_utils.Option.map full Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let hessenberg ?calc_q ?overwrite_a ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "hessenberg"
     [||]
     (Wrap_utils.keyword_args [("calc_q", Wrap_utils.Option.map calc_q Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let hilbert n =
   Py.Module.get_function_with_keywords __wrap_namespace "hilbert"
     [||]
     (Wrap_utils.keyword_args [("n", Some(n |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let inv ?overwrite_a ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "inv"
     [||]
     (Wrap_utils.keyword_args [("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let invhilbert ?exact ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "invhilbert"
     [||]
     (Wrap_utils.keyword_args [("exact", Wrap_utils.Option.map exact Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let invpascal ?kind ?exact ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "invpascal"
     [||]
     (Wrap_utils.keyword_args [("kind", Wrap_utils.Option.map kind Py.String.of_string); ("exact", Wrap_utils.Option.map exact Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let khatri_rao ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "khatri_rao"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a )); ("b", Some(b ))])

let kron ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "kron"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b ))])

let ldl ?lower ?hermitian ?overwrite_a ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "ldl"
     [||]
     (Wrap_utils.keyword_args [("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("hermitian", Wrap_utils.Option.map hermitian Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("A", Some(a ))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
let leslie ~f ~s () =
   Py.Module.get_function_with_keywords __wrap_namespace "leslie"
     [||]
     (Wrap_utils.keyword_args [("f", Some(f |> Np.Obj.to_pyobject)); ("s", Some(s ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let logm ?disp ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "logm"
     [||]
     (Wrap_utils.keyword_args [("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("A", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))
let lstsq ?cond ?overwrite_a ?overwrite_b ?check_finite ?lapack_driver ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "lstsq"
     [||]
     (Wrap_utils.keyword_args [("cond", Wrap_utils.Option.map cond Py.Float.of_float); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("lapack_driver", Wrap_utils.Option.map lapack_driver Py.String.of_string); ("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2)), ((fun py -> if Py.is_none py then None else Some (Wrap_utils.id py)) (Py.Tuple.get x 3))))
let lu ?permute_l ?overwrite_a ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "lu"
     [||]
     (Wrap_utils.keyword_args [("permute_l", Wrap_utils.Option.map permute_l Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3))))
let lu_factor ?overwrite_a ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "lu_factor"
     [||]
     (Wrap_utils.keyword_args [("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let lu_solve ?trans ?overwrite_b ?check_finite ~lu_and_piv ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "lu_solve"
                       [||]
                       (Wrap_utils.keyword_args [("trans", Wrap_utils.Option.map trans (function
| `Two -> Py.Int.of_int 2
| `One -> Py.Int.of_int 1
| `Zero -> Py.Int.of_int 0
)); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("lu_and_piv", Some(lu_and_piv )); ("b", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let matrix_balance ?permute ?scale ?separate ?overwrite_a ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "matrix_balance"
     [||]
     (Wrap_utils.keyword_args [("permute", Wrap_utils.Option.map permute Py.Bool.of_bool); ("scale", Wrap_utils.Option.map scale Py.Float.of_float); ("separate", Wrap_utils.Option.map separate Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("A", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let norm ?ord ?axis ?keepdims ?check_finite ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "norm"
                       [||]
                       (Wrap_utils.keyword_args [("ord", Wrap_utils.Option.map ord (function
| `PyObject x -> Wrap_utils.id x
| `Fro -> Py.String.of_string "fro"
)); ("axis", Wrap_utils.Option.map axis (function
| `T2_tuple_of_ints x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a ))])

let null_space ?rcond ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "null_space"
     [||]
     (Wrap_utils.keyword_args [("rcond", Wrap_utils.Option.map rcond Py.Float.of_float); ("A", Some(a |> Np.Obj.to_pyobject))])

                  let ordqz ?sort ?output ?overwrite_a ?overwrite_b ?check_finite ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ordqz"
                       [||]
                       (Wrap_utils.keyword_args [("sort", Wrap_utils.Option.map sort (function
| `Rhp -> Py.String.of_string "rhp"
| `Iuc -> Py.String.of_string "iuc"
| `Lhp -> Py.String.of_string "lhp"
| `Ouc -> Py.String.of_string "ouc"
| `Callable x -> Wrap_utils.id x
)); ("output", Wrap_utils.Option.map output (function
| `Real -> Py.String.of_string "real"
| `Complex -> Py.String.of_string "complex"
)); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("A", Some(a |> Np.Obj.to_pyobject)); ("B", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 3)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 4)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 5))))
let orth ?rcond ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "orth"
     [||]
     (Wrap_utils.keyword_args [("rcond", Wrap_utils.Option.map rcond Py.Float.of_float); ("A", Some(a |> Np.Obj.to_pyobject))])

let orthogonal_procrustes ?check_finite ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "orthogonal_procrustes"
     [||]
     (Wrap_utils.keyword_args [("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("A", Some(a |> Np.Obj.to_pyobject)); ("B", Some(b |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))
let pascal ?kind ?exact ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "pascal"
     [||]
     (Wrap_utils.keyword_args [("kind", Wrap_utils.Option.map kind Py.String.of_string); ("exact", Wrap_utils.Option.map exact Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let pinv ?cond ?rcond ?return_rank ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "pinv"
     [||]
     (Wrap_utils.keyword_args [("cond", cond); ("rcond", rcond); ("return_rank", Wrap_utils.Option.map return_rank Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
let pinv2 ?cond ?rcond ?return_rank ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "pinv2"
     [||]
     (Wrap_utils.keyword_args [("cond", cond); ("rcond", rcond); ("return_rank", Wrap_utils.Option.map return_rank Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
let pinvh ?cond ?rcond ?lower ?return_rank ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "pinvh"
     [||]
     (Wrap_utils.keyword_args [("cond", cond); ("rcond", rcond); ("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("return_rank", Wrap_utils.Option.map return_rank Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Int.to_int (Py.Tuple.get x 1))))
                  let polar ?side ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "polar"
                       [||]
                       (Wrap_utils.keyword_args [("side", Wrap_utils.Option.map side (function
| `Left -> Py.String.of_string "left"
| `Right -> Py.String.of_string "right"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let qr ?overwrite_a ?lwork ?mode ?pivoting ?check_finite ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "qr"
                       [||]
                       (Wrap_utils.keyword_args [("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("lwork", Wrap_utils.Option.map lwork Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Full -> Py.String.of_string "full"
| `R -> Py.String.of_string "r"
| `Economic -> Py.String.of_string "economic"
| `Raw -> Py.String.of_string "raw"
)); ("pivoting", Wrap_utils.Option.map pivoting Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
                  let qr_insert ?which ?rcond ?overwrite_qru ?check_finite ~q ~r ~u ~k () =
                     Py.Module.get_function_with_keywords __wrap_namespace "qr_insert"
                       [||]
                       (Wrap_utils.keyword_args [("which", Wrap_utils.Option.map which (function
| `Row -> Py.String.of_string "row"
| `Col -> Py.String.of_string "col"
)); ("rcond", Wrap_utils.Option.map rcond Py.Float.of_float); ("overwrite_qru", Wrap_utils.Option.map overwrite_qru Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("Q", Some(q |> Np.Obj.to_pyobject)); ("R", Some(r |> Np.Obj.to_pyobject)); ("u", Some(u )); ("k", Some(k |> Py.Int.of_int))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let qr_multiply ?mode ?pivoting ?conjugate ?overwrite_a ?overwrite_c ~a ~c () =
                     Py.Module.get_function_with_keywords __wrap_namespace "qr_multiply"
                       [||]
                       (Wrap_utils.keyword_args [("mode", Wrap_utils.Option.map mode (function
| `Left -> Py.String.of_string "left"
| `Right -> Py.String.of_string "right"
)); ("pivoting", Wrap_utils.Option.map pivoting Py.Bool.of_bool); ("conjugate", Wrap_utils.Option.map conjugate Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("overwrite_c", Wrap_utils.Option.map overwrite_c Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject)); ("c", Some(c |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
let qr_update ?overwrite_qruv ?check_finite ~q ~r ~u ~v () =
   Py.Module.get_function_with_keywords __wrap_namespace "qr_update"
     [||]
     (Wrap_utils.keyword_args [("overwrite_qruv", Wrap_utils.Option.map overwrite_qruv Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("Q", Some(q )); ("R", Some(r )); ("u", Some(u )); ("v", Some(v ))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let qz ?output ?lwork ?sort ?overwrite_a ?overwrite_b ?check_finite ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "qz"
                       [||]
                       (Wrap_utils.keyword_args [("output", Wrap_utils.Option.map output (function
| `Real -> Py.String.of_string "real"
| `Complex -> Py.String.of_string "complex"
)); ("lwork", Wrap_utils.Option.map lwork Py.Int.of_int); ("sort", Wrap_utils.Option.map sort (function
| `Rhp -> Py.String.of_string "rhp"
| `Iuc -> Py.String.of_string "iuc"
| `Lhp -> Py.String.of_string "lhp"
| `Ouc -> Py.String.of_string "ouc"
| `Callable x -> Wrap_utils.id x
)); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("A", Some(a |> Np.Obj.to_pyobject)); ("B", Some(b |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 3))))
                  let rq ?overwrite_a ?lwork ?mode ?check_finite ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "rq"
                       [||]
                       (Wrap_utils.keyword_args [("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("lwork", Wrap_utils.Option.map lwork Py.Int.of_int); ("mode", Wrap_utils.Option.map mode (function
| `Full -> Py.String.of_string "full"
| `R -> Py.String.of_string "r"
| `Economic -> Py.String.of_string "economic"
)); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let rsf2csf ?check_finite ~t ~z () =
   Py.Module.get_function_with_keywords __wrap_namespace "rsf2csf"
     [||]
     (Wrap_utils.keyword_args [("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("T", Some(t |> Np.Obj.to_pyobject)); ("Z", Some(z |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let schur ?output ?lwork ?overwrite_a ?sort ?check_finite ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "schur"
                       [||]
                       (Wrap_utils.keyword_args [("output", Wrap_utils.Option.map output (function
| `Real -> Py.String.of_string "real"
| `Complex -> Py.String.of_string "complex"
)); ("lwork", Wrap_utils.Option.map lwork Py.Int.of_int); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("sort", Wrap_utils.Option.map sort (function
| `Rhp -> Py.String.of_string "rhp"
| `Iuc -> Py.String.of_string "iuc"
| `Lhp -> Py.String.of_string "lhp"
| `Ouc -> Py.String.of_string "ouc"
| `Callable x -> Wrap_utils.id x
)); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Int.to_int (Py.Tuple.get x 2))))
let signm ?disp ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "signm"
     [||]
     (Wrap_utils.keyword_args [("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("A", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))
let sinhm a =
   Py.Module.get_function_with_keywords __wrap_namespace "sinhm"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let sinm a =
   Py.Module.get_function_with_keywords __wrap_namespace "sinm"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let solve ?sym_pos ?lower ?overwrite_a ?overwrite_b ?debug ?check_finite ?assume_a ?transposed ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "solve"
     [||]
     (Wrap_utils.keyword_args [("sym_pos", Wrap_utils.Option.map sym_pos Py.Bool.of_bool); ("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("debug", debug); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("assume_a", Wrap_utils.Option.map assume_a Py.String.of_string); ("transposed", Wrap_utils.Option.map transposed Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let solve_banded ?overwrite_ab ?overwrite_b ?debug ?check_finite ~l_and_u ~ab ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "solve_banded"
     [||]
     (Wrap_utils.keyword_args [("overwrite_ab", Wrap_utils.Option.map overwrite_ab Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("debug", debug); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("l_and_u", Some(l_and_u )); ("ab", Some(ab )); ("b", Some(b ))])

let solve_circulant ?singular ?tol ?caxis ?baxis ?outaxis ~c ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "solve_circulant"
     [||]
     (Wrap_utils.keyword_args [("singular", Wrap_utils.Option.map singular Py.String.of_string); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("caxis", Wrap_utils.Option.map caxis Py.Int.of_int); ("baxis", Wrap_utils.Option.map baxis Py.Int.of_int); ("outaxis", Wrap_utils.Option.map outaxis Py.Int.of_int); ("c", Some(c |> Np.Obj.to_pyobject)); ("b", Some(b |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let solve_continuous_are ?e ?s ?balanced ~a ~b ~q ~r () =
   Py.Module.get_function_with_keywords __wrap_namespace "solve_continuous_are"
     [||]
     (Wrap_utils.keyword_args [("e", Wrap_utils.Option.map e Np.Obj.to_pyobject); ("s", Wrap_utils.Option.map s Np.Obj.to_pyobject); ("balanced", Wrap_utils.Option.map balanced Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b |> Np.Obj.to_pyobject)); ("q", Some(q |> Np.Obj.to_pyobject)); ("r", Some(r |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let solve_continuous_lyapunov ~a ~q () =
   Py.Module.get_function_with_keywords __wrap_namespace "solve_continuous_lyapunov"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Np.Obj.to_pyobject)); ("q", Some(q |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let solve_discrete_are ?e ?s ?balanced ~a ~b ~q ~r () =
   Py.Module.get_function_with_keywords __wrap_namespace "solve_discrete_are"
     [||]
     (Wrap_utils.keyword_args [("e", Wrap_utils.Option.map e Np.Obj.to_pyobject); ("s", Wrap_utils.Option.map s Np.Obj.to_pyobject); ("balanced", Wrap_utils.Option.map balanced Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b |> Np.Obj.to_pyobject)); ("q", Some(q |> Np.Obj.to_pyobject)); ("r", Some(r |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let solve_discrete_lyapunov ?method_ ~a ~q () =
                     Py.Module.get_function_with_keywords __wrap_namespace "solve_discrete_lyapunov"
                       [||]
                       (Wrap_utils.keyword_args [("method", Wrap_utils.Option.map method_ (function
| `Bilinear -> Py.String.of_string "bilinear"
| `Direct -> Py.String.of_string "direct"
)); ("a", Some(a )); ("q", Some(q ))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let solve_lyapunov ~a ~q () =
   Py.Module.get_function_with_keywords __wrap_namespace "solve_lyapunov"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Np.Obj.to_pyobject)); ("q", Some(q |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let solve_sylvester ~a ~b ~q () =
   Py.Module.get_function_with_keywords __wrap_namespace "solve_sylvester"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b |> Np.Obj.to_pyobject)); ("q", Some(q |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let solve_toeplitz ?check_finite ~c_or_cr ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "solve_toeplitz"
                       [||]
                       (Wrap_utils.keyword_args [("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("c_or_cr", Some(c_or_cr |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_array_like_array_like_ x -> Wrap_utils.id x
))); ("b", Some(b ))])

                  let solve_triangular ?trans ?lower ?unit_diagonal ?overwrite_b ?debug ?check_finite ~a ~b () =
                     Py.Module.get_function_with_keywords __wrap_namespace "solve_triangular"
                       [||]
                       (Wrap_utils.keyword_args [("trans", Wrap_utils.Option.map trans (function
| `C -> Py.String.of_string "C"
| `Two -> Py.Int.of_int 2
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `T -> Py.String.of_string "T"
| `N -> Py.String.of_string "N"
)); ("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("unit_diagonal", Wrap_utils.Option.map unit_diagonal Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("debug", debug); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject)); ("b", Some(b ))])

let solveh_banded ?overwrite_ab ?overwrite_b ?lower ?check_finite ~ab ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "solveh_banded"
     [||]
     (Wrap_utils.keyword_args [("overwrite_ab", Wrap_utils.Option.map overwrite_ab Py.Bool.of_bool); ("overwrite_b", Wrap_utils.Option.map overwrite_b Py.Bool.of_bool); ("lower", Wrap_utils.Option.map lower Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("ab", Some(ab )); ("b", Some(b ))])

let sqrtm ?disp ?blocksize ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "sqrtm"
     [||]
     (Wrap_utils.keyword_args [("disp", Wrap_utils.Option.map disp Py.Bool.of_bool); ("blocksize", Wrap_utils.Option.map blocksize Py.Int.of_int); ("A", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))
let subspace_angles ~a ~b () =
   Py.Module.get_function_with_keywords __wrap_namespace "subspace_angles"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject)); ("B", Some(b ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let svd ?full_matrices ?compute_uv ?overwrite_a ?check_finite ?lapack_driver ~a () =
                     Py.Module.get_function_with_keywords __wrap_namespace "svd"
                       [||]
                       (Wrap_utils.keyword_args [("full_matrices", Wrap_utils.Option.map full_matrices Py.Bool.of_bool); ("compute_uv", Wrap_utils.Option.map compute_uv Py.Bool.of_bool); ("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("lapack_driver", Wrap_utils.Option.map lapack_driver (function
| `Gesdd -> Py.String.of_string "gesdd"
| `Gesvd -> Py.String.of_string "gesvd"
)); ("a", Some(a |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2))))
let svdvals ?overwrite_a ?check_finite ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "svdvals"
     [||]
     (Wrap_utils.keyword_args [("overwrite_a", Wrap_utils.Option.map overwrite_a Py.Bool.of_bool); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])

let tanhm a =
   Py.Module.get_function_with_keywords __wrap_namespace "tanhm"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let tanm a =
   Py.Module.get_function_with_keywords __wrap_namespace "tanm"
     [||]
     (Wrap_utils.keyword_args [("A", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let toeplitz ?r ~c () =
   Py.Module.get_function_with_keywords __wrap_namespace "toeplitz"
     [||]
     (Wrap_utils.keyword_args [("r", Wrap_utils.Option.map r Np.Obj.to_pyobject); ("c", Some(c |> Np.Obj.to_pyobject))])

let tri ?m ?k ?dtype ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "tri"
     [||]
     (Wrap_utils.keyword_args [("M", Wrap_utils.Option.map m Py.Int.of_int); ("k", Wrap_utils.Option.map k Py.Int.of_int); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("N", Some(n |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let tril ?k ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "tril"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("m", Some(m |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let triu ?k ~m () =
   Py.Module.get_function_with_keywords __wrap_namespace "triu"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("m", Some(m |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
