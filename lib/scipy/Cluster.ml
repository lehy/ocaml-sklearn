let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.cluster"

let get_py name = Py.Module.get __wrap_namespace name
module Hierarchy = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.cluster.hierarchy"

let get_py name = Py.Module.get __wrap_namespace name
module ClusterNode = struct
type tag = [`ClusterNode]
type t = [`ClusterNode | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?left ?right ?dist ?count ~id () =
   Py.Module.get_function_with_keywords __wrap_namespace "ClusterNode"
     [||]
     (Wrap_utils.keyword_args [("left", left); ("right", right); ("dist", Wrap_utils.Option.map dist Py.Float.of_float); ("count", Wrap_utils.Option.map count Py.Int.of_int); ("id", Some(id |> Py.Int.of_int))])
     |> of_pyobject
let get_count self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_count"
     [||]
     []
     |> Py.Int.to_int
let get_id self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_id"
     [||]
     []
     |> Py.Int.to_int
let get_left self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_left"
     [||]
     []

let get_right self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_right"
     [||]
     []

let is_leaf self =
   Py.Module.get_function_with_keywords (to_pyobject self) "is_leaf"
     [||]
     []
     |> Py.Bool.to_bool
let pre_order ?func self =
   Py.Module.get_function_with_keywords (to_pyobject self) "pre_order"
     [||]
     (Wrap_utils.keyword_args [("func", func)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ClusterWarning = struct
type tag = [`ClusterWarning]
type t = [`BaseException | `ClusterWarning | `Object] Obj.t
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
module Deque = struct
type tag = [`Deque]
type t = [`Deque | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let __setitem__ ~key ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     (Array.of_list @@ List.concat [[key ];[value ]])
     []

let count ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "count"
     [||]
     (Wrap_utils.keyword_args [("value", Some(value ))])

let insert ~index ~object_ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "insert"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index )); ("object", Some(object_ ))])

let remove ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "remove"
     [||]
     (Wrap_utils.keyword_args [("value", Some(value ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Xrange = struct
type tag = [`Range]
type t = [`Object | `Range] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create stop =
   Py.Module.get_function_with_keywords __wrap_namespace "xrange"
     [||]
     (Wrap_utils.keyword_args [("stop", Some(stop ))])
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let count ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "count"
     [||]
     (Wrap_utils.keyword_args [("value", Some(value ))])

let index ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "index"
     [||]
     (Wrap_utils.keyword_args [("value", Some(value ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let average y =
   Py.Module.get_function_with_keywords __wrap_namespace "average"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let centroid y =
   Py.Module.get_function_with_keywords __wrap_namespace "centroid"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let complete y =
   Py.Module.get_function_with_keywords __wrap_namespace "complete"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let cophenet ?y ~z () =
   Py.Module.get_function_with_keywords __wrap_namespace "cophenet"
     [||]
     (Wrap_utils.keyword_args [("Y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("Z", Some(z |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let correspond ~z ~y () =
   Py.Module.get_function_with_keywords __wrap_namespace "correspond"
     [||]
     (Wrap_utils.keyword_args [("Z", Some(z |> Np.Obj.to_pyobject)); ("Y", Some(y |> Np.Obj.to_pyobject))])
     |> Py.Bool.to_bool
let cut_tree ?n_clusters ?height ~z () =
   Py.Module.get_function_with_keywords __wrap_namespace "cut_tree"
     [||]
     (Wrap_utils.keyword_args [("n_clusters", Wrap_utils.Option.map n_clusters Np.Obj.to_pyobject); ("height", Wrap_utils.Option.map height Np.Obj.to_pyobject); ("Z", Some(z ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let dendrogram ?p ?truncate_mode ?color_threshold ?get_leaves ?orientation ?labels ?count_sort ?distance_sort ?show_leaf_counts ?no_plot ?no_labels ?leaf_font_size ?leaf_rotation ?leaf_label_func ?show_contracted ?link_color_func ?ax ?above_threshold_color ~z () =
                     Py.Module.get_function_with_keywords __wrap_namespace "dendrogram"
                       [||]
                       (Wrap_utils.keyword_args [("p", Wrap_utils.Option.map p Py.Int.of_int); ("truncate_mode", Wrap_utils.Option.map truncate_mode Py.String.of_string); ("color_threshold", Wrap_utils.Option.map color_threshold Py.Float.of_float); ("get_leaves", Wrap_utils.Option.map get_leaves Py.Bool.of_bool); ("orientation", Wrap_utils.Option.map orientation Py.String.of_string); ("labels", Wrap_utils.Option.map labels Np.Obj.to_pyobject); ("count_sort", Wrap_utils.Option.map count_sort (function
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("distance_sort", Wrap_utils.Option.map distance_sort (function
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)); ("show_leaf_counts", Wrap_utils.Option.map show_leaf_counts Py.Bool.of_bool); ("no_plot", Wrap_utils.Option.map no_plot Py.Bool.of_bool); ("no_labels", Wrap_utils.Option.map no_labels Py.Bool.of_bool); ("leaf_font_size", Wrap_utils.Option.map leaf_font_size Py.Int.of_int); ("leaf_rotation", Wrap_utils.Option.map leaf_rotation Py.Float.of_float); ("leaf_label_func", Wrap_utils.Option.map leaf_label_func (function
| `Lambda x -> Wrap_utils.id x
| `Callable x -> Wrap_utils.id x
)); ("show_contracted", Wrap_utils.Option.map show_contracted Py.Bool.of_bool); ("link_color_func", link_color_func); ("ax", ax); ("above_threshold_color", Wrap_utils.Option.map above_threshold_color Py.String.of_string); ("Z", Some(z |> Np.Obj.to_pyobject))])

                  let fcluster ?criterion ?depth ?r ?monocrit ~z ~t () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fcluster"
                       [||]
                       (Wrap_utils.keyword_args [("criterion", Wrap_utils.Option.map criterion Py.String.of_string); ("depth", Wrap_utils.Option.map depth Py.Int.of_int); ("R", Wrap_utils.Option.map r Np.Obj.to_pyobject); ("monocrit", Wrap_utils.Option.map monocrit Np.Obj.to_pyobject); ("Z", Some(z |> Np.Obj.to_pyobject)); ("t", Some(t |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let fclusterdata ?criterion ?metric ?depth ?method_ ?r ~x ~t () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fclusterdata"
                       [||]
                       (Wrap_utils.keyword_args [("criterion", Wrap_utils.Option.map criterion Py.String.of_string); ("metric", Wrap_utils.Option.map metric Py.String.of_string); ("depth", Wrap_utils.Option.map depth Py.Int.of_int); ("method", Wrap_utils.Option.map method_ Py.String.of_string); ("R", Wrap_utils.Option.map r Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject)); ("t", Some(t |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
| `Bool x -> Py.Bool.of_bool x
| `S x -> Py.String.of_string x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let from_mlab_linkage z =
   Py.Module.get_function_with_keywords __wrap_namespace "from_mlab_linkage"
     [||]
     (Wrap_utils.keyword_args [("Z", Some(z |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let inconsistent ?d ~z () =
   Py.Module.get_function_with_keywords __wrap_namespace "inconsistent"
     [||]
     (Wrap_utils.keyword_args [("d", Wrap_utils.Option.map d Py.Int.of_int); ("Z", Some(z |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let is_isomorphic ~t1 ~t2 () =
   Py.Module.get_function_with_keywords __wrap_namespace "is_isomorphic"
     [||]
     (Wrap_utils.keyword_args [("T1", Some(t1 |> Np.Obj.to_pyobject)); ("T2", Some(t2 |> Np.Obj.to_pyobject))])
     |> Py.Bool.to_bool
let is_monotonic z =
   Py.Module.get_function_with_keywords __wrap_namespace "is_monotonic"
     [||]
     (Wrap_utils.keyword_args [("Z", Some(z |> Np.Obj.to_pyobject))])
     |> Py.Bool.to_bool
let is_valid_im ?warning ?throw ?name ~r () =
   Py.Module.get_function_with_keywords __wrap_namespace "is_valid_im"
     [||]
     (Wrap_utils.keyword_args [("warning", Wrap_utils.Option.map warning Py.Bool.of_bool); ("throw", Wrap_utils.Option.map throw Py.Bool.of_bool); ("name", Wrap_utils.Option.map name Py.String.of_string); ("R", Some(r |> Np.Obj.to_pyobject))])
     |> Py.Bool.to_bool
let is_valid_linkage ?warning ?throw ?name ~z () =
   Py.Module.get_function_with_keywords __wrap_namespace "is_valid_linkage"
     [||]
     (Wrap_utils.keyword_args [("warning", Wrap_utils.Option.map warning Py.Bool.of_bool); ("throw", Wrap_utils.Option.map throw Py.Bool.of_bool); ("name", Wrap_utils.Option.map name Py.String.of_string); ("Z", Some(z |> Np.Obj.to_pyobject))])
     |> Py.Bool.to_bool
let leaders ~z ~t () =
   Py.Module.get_function_with_keywords __wrap_namespace "leaders"
     [||]
     (Wrap_utils.keyword_args [("Z", Some(z |> Np.Obj.to_pyobject)); ("T", Some(t |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let leaves_list z =
   Py.Module.get_function_with_keywords __wrap_namespace "leaves_list"
     [||]
     (Wrap_utils.keyword_args [("Z", Some(z |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let linkage ?method_ ?metric ?optimal_ordering ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "linkage"
                       [||]
                       (Wrap_utils.keyword_args [("method", Wrap_utils.Option.map method_ Py.String.of_string); ("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("optimal_ordering", Wrap_utils.Option.map optimal_ordering Py.Bool.of_bool); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let maxRstat ~z ~r ~i () =
   Py.Module.get_function_with_keywords __wrap_namespace "maxRstat"
     [||]
     (Wrap_utils.keyword_args [("Z", Some(z |> Np.Obj.to_pyobject)); ("R", Some(r |> Np.Obj.to_pyobject)); ("i", Some(i |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let maxdists z =
   Py.Module.get_function_with_keywords __wrap_namespace "maxdists"
     [||]
     (Wrap_utils.keyword_args [("Z", Some(z |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let maxinconsts ~z ~r () =
   Py.Module.get_function_with_keywords __wrap_namespace "maxinconsts"
     [||]
     (Wrap_utils.keyword_args [("Z", Some(z |> Np.Obj.to_pyobject)); ("R", Some(r |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let median y =
   Py.Module.get_function_with_keywords __wrap_namespace "median"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let num_obs_linkage z =
   Py.Module.get_function_with_keywords __wrap_namespace "num_obs_linkage"
     [||]
     (Wrap_utils.keyword_args [("Z", Some(z |> Np.Obj.to_pyobject))])
     |> Py.Int.to_int
                  let optimal_leaf_ordering ?metric ~z ~y () =
                     Py.Module.get_function_with_keywords __wrap_namespace "optimal_leaf_ordering"
                       [||]
                       (Wrap_utils.keyword_args [("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("Z", Some(z |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let set_link_color_palette palette =
                     Py.Module.get_function_with_keywords __wrap_namespace "set_link_color_palette"
                       [||]
                       (Wrap_utils.keyword_args [("palette", Some(palette |> (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `None -> Py.none
)))])

let single y =
   Py.Module.get_function_with_keywords __wrap_namespace "single"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let to_mlab_linkage z =
   Py.Module.get_function_with_keywords __wrap_namespace "to_mlab_linkage"
     [||]
     (Wrap_utils.keyword_args [("Z", Some(z |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let to_tree ?rd ~z () =
   Py.Module.get_function_with_keywords __wrap_namespace "to_tree"
     [||]
     (Wrap_utils.keyword_args [("rd", Wrap_utils.Option.map rd Py.Bool.of_bool); ("Z", Some(z |> Np.Obj.to_pyobject))])

let ward y =
   Py.Module.get_function_with_keywords __wrap_namespace "ward"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let weighted y =
   Py.Module.get_function_with_keywords __wrap_namespace "weighted"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
module Vq = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.cluster.vq"

let get_py name = Py.Module.get __wrap_namespace name
module ClusterError = struct
type tag = [`ClusterError]
type t = [`BaseException | `ClusterError | `Object] Obj.t
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
module Deque = struct
type tag = [`Deque]
type t = [`Deque | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let __setitem__ ~key ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     (Array.of_list @@ List.concat [[key ];[value ]])
     []

let count ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "count"
     [||]
     (Wrap_utils.keyword_args [("value", Some(value ))])

let insert ~index ~object_ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "insert"
     [||]
     (Wrap_utils.keyword_args [("index", Some(index )); ("object", Some(object_ ))])

let remove ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "remove"
     [||]
     (Wrap_utils.keyword_args [("value", Some(value ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Xrange = struct
type tag = [`Range]
type t = [`Object | `Range] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create stop =
   Py.Module.get_function_with_keywords __wrap_namespace "xrange"
     [||]
     (Wrap_utils.keyword_args [("stop", Some(stop ))])
     |> of_pyobject
let __getitem__ ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     (Array.of_list @@ List.concat [[key ]])
     []

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let count ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "count"
     [||]
     (Wrap_utils.keyword_args [("value", Some(value ))])

let index ~value self =
   Py.Module.get_function_with_keywords (to_pyobject self) "index"
     [||]
     (Wrap_utils.keyword_args [("value", Some(value ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let cdist ?metric ?kwargs ~xa ~xb args =
                     Py.Module.get_function_with_keywords __wrap_namespace "cdist"
                       (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
                       (List.rev_append (Wrap_utils.keyword_args [("metric", Wrap_utils.Option.map metric (function
| `Callable x -> Wrap_utils.id x
| `S x -> Py.String.of_string x
)); ("XA", Some(xa |> Np.Obj.to_pyobject)); ("XB", Some(xb |> Np.Obj.to_pyobject))]) (match kwargs with None -> [] | Some x -> x))
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let kmeans ?iter ?thresh ?check_finite ~obs ~k_or_guess () =
                     Py.Module.get_function_with_keywords __wrap_namespace "kmeans"
                       [||]
                       (Wrap_utils.keyword_args [("iter", Wrap_utils.Option.map iter Py.Int.of_int); ("thresh", Wrap_utils.Option.map thresh Py.Float.of_float); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("obs", Some(obs |> Np.Obj.to_pyobject)); ("k_or_guess", Some(k_or_guess |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), (Py.Float.to_float (Py.Tuple.get x 1))))
                  let kmeans2 ?iter ?thresh ?minit ?missing ?check_finite ~data ~k () =
                     Py.Module.get_function_with_keywords __wrap_namespace "kmeans2"
                       [||]
                       (Wrap_utils.keyword_args [("iter", Wrap_utils.Option.map iter Py.Int.of_int); ("thresh", Wrap_utils.Option.map thresh Py.Float.of_float); ("minit", Wrap_utils.Option.map minit Py.String.of_string); ("missing", Wrap_utils.Option.map missing Py.String.of_string); ("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("data", Some(data |> Np.Obj.to_pyobject)); ("k", Some(k |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let py_vq ?check_finite ~obs ~code_book () =
   Py.Module.get_function_with_keywords __wrap_namespace "py_vq"
     [||]
     (Wrap_utils.keyword_args [("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("obs", Some(obs |> Np.Obj.to_pyobject)); ("code_book", Some(code_book |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let py_vq2 ?kwds args =
   Py.Module.get_function_with_keywords __wrap_namespace "py_vq2"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (match kwds with None -> [] | Some x -> x)
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let vq ?check_finite ~obs ~code_book () =
   Py.Module.get_function_with_keywords __wrap_namespace "vq"
     [||]
     (Wrap_utils.keyword_args [("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("obs", Some(obs |> Np.Obj.to_pyobject)); ("code_book", Some(code_book |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let whiten ?check_finite ~obs () =
   Py.Module.get_function_with_keywords __wrap_namespace "whiten"
     [||]
     (Wrap_utils.keyword_args [("check_finite", Wrap_utils.Option.map check_finite Py.Bool.of_bool); ("obs", Some(obs |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
