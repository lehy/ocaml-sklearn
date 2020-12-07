let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.datasets"

let get_py name = Py.Module.get __wrap_namespace name
let clear_data_home ?data_home () =
   Py.Module.get_function_with_keywords __wrap_namespace "clear_data_home"
     [||]
     (Wrap_utils.keyword_args [("data_home", Wrap_utils.Option.map data_home Py.String.of_string)])

                  let dump_svmlight_file ?zero_based ?comment ?query_id ?multilabel ~x ~y ~f () =
                     Py.Module.get_function_with_keywords __wrap_namespace "dump_svmlight_file"
                       [||]
                       (Wrap_utils.keyword_args [("zero_based", Wrap_utils.Option.map zero_based Py.Bool.of_bool); ("comment", Wrap_utils.Option.map comment Py.String.of_string); ("query_id", Wrap_utils.Option.map query_id Np.Obj.to_pyobject); ("multilabel", Wrap_utils.Option.map multilabel Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject)); ("y", Some(y |> Np.Obj.to_pyobject)); ("f", Some(f |> (function
| `S x -> Py.String.of_string x
| `File_like_in_binary_mode x -> Wrap_utils.id x
)))])

                  let fetch_20newsgroups ?data_home ?subset ?categories ?shuffle ?random_state ?remove ?download_if_missing () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fetch_20newsgroups"
                       [||]
                       (Wrap_utils.keyword_args [("data_home", data_home); ("subset", Wrap_utils.Option.map subset Py.String.of_string); ("categories", Wrap_utils.Option.map categories (function
| `S x -> Py.String.of_string x
| `Arr x -> Np.Obj.to_pyobject x
)); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("remove", remove); ("download_if_missing", download_if_missing); ("return_X_y", Some(false |> Py.Bool.of_bool))])
                       |> (fun bunch -> object method data = Wrap_utils.id (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method target = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method filenames = Wrap_utils.id (Py.Object.get_attr_string bunch "filenames" |> Wrap_utils.Option.get) method descr = Py.String.to_string (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) method target_names = Wrap_utils.id (Py.Object.get_attr_string bunch "target_names" |> Wrap_utils.Option.get) end)
let fetch_20newsgroups_vectorized ?subset ?remove ?data_home ?download_if_missing ?normalize () =
   Py.Module.get_function_with_keywords __wrap_namespace "fetch_20newsgroups_vectorized"
     [||]
     (Wrap_utils.keyword_args [("subset", Wrap_utils.Option.map subset Py.String.of_string); ("remove", remove); ("data_home", data_home); ("download_if_missing", download_if_missing); ("return_X_y", Some(false |> Py.Bool.of_bool)); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool)])
     |> (fun bunch -> object method data = (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t)) (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method target = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method target_names = Wrap_utils.id (Py.Object.get_attr_string bunch "target_names" |> Wrap_utils.Option.get) method descr = Py.String.to_string (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) end)
let fetch_california_housing ?data_home ?download_if_missing () =
   Py.Module.get_function_with_keywords __wrap_namespace "fetch_california_housing"
     [||]
     (Wrap_utils.keyword_args [("data_home", data_home); ("download_if_missing", download_if_missing); ("return_X_y", Some(false |> Py.Bool.of_bool)); ("as_frame", Some(false |> Py.Bool.of_bool))])
     |> (fun bunch -> object method data = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method target = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method feature_names = (fun py -> Py.List.to_list_map (Py.String.to_string) py) (Py.Object.get_attr_string bunch "feature_names" |> Wrap_utils.Option.get) method descr = Py.String.to_string (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) method frame = Wrap_utils.id (Py.Object.get_attr_string bunch "frame" |> Wrap_utils.Option.get) end)
let fetch_covtype ?data_home ?download_if_missing ?random_state ?shuffle () =
   Py.Module.get_function_with_keywords __wrap_namespace "fetch_covtype"
     [||]
     (Wrap_utils.keyword_args [("data_home", Wrap_utils.Option.map data_home Py.String.of_string); ("download_if_missing", Wrap_utils.Option.map download_if_missing Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("return_X_y", Some(false |> Py.Bool.of_bool))])
     |> (fun bunch -> object method data = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method target = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method descr = Py.String.to_string (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) end)
                  let fetch_kddcup99 ?subset ?data_home ?shuffle ?random_state ?percent10 ?download_if_missing () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fetch_kddcup99"
                       [||]
                       (Wrap_utils.keyword_args [("subset", Wrap_utils.Option.map subset (function
| `Http -> Py.String.of_string "http"
| `Smtp -> Py.String.of_string "smtp"
| `SF -> Py.String.of_string "SF"
| `SA -> Py.String.of_string "SA"
)); ("data_home", Wrap_utils.Option.map data_home Py.String.of_string); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("percent10", Wrap_utils.Option.map percent10 Py.Bool.of_bool); ("download_if_missing", Wrap_utils.Option.map download_if_missing Py.Bool.of_bool); ("return_X_y", Some(false |> Py.Bool.of_bool))])
                       |> (fun bunch -> object method data = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method target = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method descr = Py.String.to_string (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) end)
let fetch_lfw_pairs ?subset ?data_home ?funneled ?resize ?color ?slice_ ?download_if_missing () =
   Py.Module.get_function_with_keywords __wrap_namespace "fetch_lfw_pairs"
     [||]
     (Wrap_utils.keyword_args [("subset", subset); ("data_home", data_home); ("funneled", Wrap_utils.Option.map funneled Py.Bool.of_bool); ("resize", Wrap_utils.Option.map resize Py.Float.of_float); ("color", Wrap_utils.Option.map color Py.Bool.of_bool); ("slice_", slice_); ("download_if_missing", download_if_missing)])
     |> (fun bunch -> object method data = Wrap_utils.id (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method pairs = Wrap_utils.id (Py.Object.get_attr_string bunch "pairs" |> Wrap_utils.Option.get) method target = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method descr = Py.String.to_string (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) end)
let fetch_lfw_people ?data_home ?funneled ?resize ?min_faces_per_person ?color ?slice_ ?download_if_missing () =
   Py.Module.get_function_with_keywords __wrap_namespace "fetch_lfw_people"
     [||]
     (Wrap_utils.keyword_args [("data_home", data_home); ("funneled", Wrap_utils.Option.map funneled Py.Bool.of_bool); ("resize", Wrap_utils.Option.map resize Py.Float.of_float); ("min_faces_per_person", Wrap_utils.Option.map min_faces_per_person Py.Int.of_int); ("color", Wrap_utils.Option.map color Py.Bool.of_bool); ("slice_", slice_); ("download_if_missing", download_if_missing); ("return_X_y", Some(false |> Py.Bool.of_bool))])
     |> (fun bunch -> object method data = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method images = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "images" |> Wrap_utils.Option.get) method target = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method descr = Py.String.to_string (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) end)
let fetch_olivetti_faces ?data_home ?shuffle ?random_state ?download_if_missing () =
   Py.Module.get_function_with_keywords __wrap_namespace "fetch_olivetti_faces"
     [||]
     (Wrap_utils.keyword_args [("data_home", data_home); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("download_if_missing", download_if_missing); ("return_X_y", Some(false |> Py.Bool.of_bool))])
     |> (fun bunch -> object method data = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method images = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "images" |> Wrap_utils.Option.get) method target = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method descr = Py.String.to_string (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) end)
                  let fetch_openml ?name ?version ?data_id ?data_home ?target_column ?cache () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fetch_openml"
                       [||]
                       (Wrap_utils.keyword_args [("name", Wrap_utils.Option.map name Py.String.of_string); ("version", Wrap_utils.Option.map version (function
| `I x -> Py.Int.of_int x
| `Active -> Py.String.of_string "active"
)); ("data_id", Wrap_utils.Option.map data_id Py.Int.of_int); ("data_home", Wrap_utils.Option.map data_home Py.String.of_string); ("target_column", Wrap_utils.Option.map target_column (function
| `S x -> Py.String.of_string x
| `Arr x -> Np.Obj.to_pyobject x
| `None -> Py.none
)); ("cache", Wrap_utils.Option.map cache Py.Bool.of_bool); ("return_X_y", Some(false |> Py.Bool.of_bool)); ("as_frame", Some(false |> Py.Bool.of_bool))])
                       |> (fun bunch -> object method data = Wrap_utils.id (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method target = Wrap_utils.id (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method descr = Py.String.to_string (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) method feature_names = (fun py -> Py.List.to_list_map (Py.String.to_string) py) (Py.Object.get_attr_string bunch "feature_names" |> Wrap_utils.Option.get) method target_names = (fun py -> Py.List.to_list_map (Py.String.to_string) py) (Py.Object.get_attr_string bunch "target_names" |> Wrap_utils.Option.get) method categories = (fun py -> if Py.is_none py then None else Some (Dict.of_pyobject py)) (Py.Object.get_attr_string bunch "categories" |> Wrap_utils.Option.get) method details = Dict.of_pyobject (Py.Object.get_attr_string bunch "details" |> Wrap_utils.Option.get) method frame = Wrap_utils.id (Py.Object.get_attr_string bunch "frame" |> Wrap_utils.Option.get) end)
                  let fetch_rcv1 ?data_home ?subset ?download_if_missing ?random_state ?shuffle () =
                     Py.Module.get_function_with_keywords __wrap_namespace "fetch_rcv1"
                       [||]
                       (Wrap_utils.keyword_args [("data_home", Wrap_utils.Option.map data_home Py.String.of_string); ("subset", Wrap_utils.Option.map subset (function
| `Test -> Py.String.of_string "test"
| `All -> Py.String.of_string "all"
| `Train -> Py.String.of_string "train"
)); ("download_if_missing", Wrap_utils.Option.map download_if_missing Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("return_X_y", Some(false |> Py.Bool.of_bool))])
                       |> (fun bunch -> object method data = Wrap_utils.id (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method target = Wrap_utils.id (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method sample_id = Wrap_utils.id (Py.Object.get_attr_string bunch "sample_id" |> Wrap_utils.Option.get) method target_names = Wrap_utils.id (Py.Object.get_attr_string bunch "target_names" |> Wrap_utils.Option.get) method descr = Py.String.to_string (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) end)
let fetch_species_distributions ?data_home ?download_if_missing () =
   Py.Module.get_function_with_keywords __wrap_namespace "fetch_species_distributions"
     [||]
     (Wrap_utils.keyword_args [("data_home", data_home); ("download_if_missing", download_if_missing)])
     |> (fun bunch -> object method data = Wrap_utils.id (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method coverages = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "coverages" |> Wrap_utils.Option.get) method train = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "train" |> Wrap_utils.Option.get) method test = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "test" |> Wrap_utils.Option.get) method nx = Wrap_utils.id (Py.Object.get_attr_string bunch "Nx" |> Wrap_utils.Option.get) method x_left_lower_corner = Wrap_utils.id (Py.Object.get_attr_string bunch "x_left_lower_corner" |> Wrap_utils.Option.get) method grid_size = Py.Float.to_float (Py.Object.get_attr_string bunch "grid_size" |> Wrap_utils.Option.get) end)
let get_data_home ?data_home () =
   Py.Module.get_function_with_keywords __wrap_namespace "get_data_home"
     [||]
     (Wrap_utils.keyword_args [("data_home", Wrap_utils.Option.map data_home Py.String.of_string)])

let load_boston () =
   Py.Module.get_function_with_keywords __wrap_namespace "load_boston"
     [||]
     (Wrap_utils.keyword_args [("return_X_y", Some(false |> Py.Bool.of_bool))])
     |> (fun bunch -> object method data = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method target = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method feature_names = (fun py -> Py.List.to_list_map (Py.String.to_string) py) (Py.Object.get_attr_string bunch "feature_names" |> Wrap_utils.Option.get) method descr = Py.String.to_string (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) method filename = Py.String.to_string (Py.Object.get_attr_string bunch "filename" |> Wrap_utils.Option.get) end)
let load_breast_cancer () =
   Py.Module.get_function_with_keywords __wrap_namespace "load_breast_cancer"
     [||]
     (Wrap_utils.keyword_args [("return_X_y", Some(false |> Py.Bool.of_bool)); ("as_frame", Some(false |> Py.Bool.of_bool))])
     |> (fun bunch -> object method data = Wrap_utils.id (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method target = Wrap_utils.id (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method feature_names = (fun py -> Py.List.to_list_map (Py.String.to_string) py) (Py.Object.get_attr_string bunch "feature_names" |> Wrap_utils.Option.get) method target_names = (fun py -> Py.List.to_list_map (Py.String.to_string) py) (Py.Object.get_attr_string bunch "target_names" |> Wrap_utils.Option.get) method frame = Wrap_utils.id (Py.Object.get_attr_string bunch "frame" |> Wrap_utils.Option.get) method descr = Py.String.to_string (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) method filename = Py.String.to_string (Py.Object.get_attr_string bunch "filename" |> Wrap_utils.Option.get) method https = Wrap_utils.id (Py.Object.get_attr_string bunch "https" |> Wrap_utils.Option.get) end)
let load_diabetes () =
   Py.Module.get_function_with_keywords __wrap_namespace "load_diabetes"
     [||]
     (Wrap_utils.keyword_args [("return_X_y", Some(false |> Py.Bool.of_bool)); ("as_frame", Some(false |> Py.Bool.of_bool))])
     |> (fun bunch -> object method data = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method target = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method feature_names = (fun py -> Py.List.to_list_map (Py.String.to_string) py) (Py.Object.get_attr_string bunch "feature_names" |> Wrap_utils.Option.get) method descr = Py.String.to_string (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) method data_filename = Py.String.to_string (Py.Object.get_attr_string bunch "data_filename" |> Wrap_utils.Option.get) method target_filename = Py.String.to_string (Py.Object.get_attr_string bunch "target_filename" |> Wrap_utils.Option.get) end)
                  let load_digits ?n_class () =
                     Py.Module.get_function_with_keywords __wrap_namespace "load_digits"
                       [||]
                       (Wrap_utils.keyword_args [("n_class", Wrap_utils.Option.map n_class (function
| `Between_0_and_10 x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("return_X_y", Some(false |> Py.Bool.of_bool)); ("as_frame", Some(false |> Py.Bool.of_bool))])
                       |> (fun bunch -> object method data = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method target = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method target_names = (fun py -> Py.List.to_list_map (Py.String.to_string) py) (Py.Object.get_attr_string bunch "target_names" |> Wrap_utils.Option.get) method feature_names = (fun py -> Py.List.to_list_map (Py.String.to_string) py) (Py.Object.get_attr_string bunch "feature_names" |> Wrap_utils.Option.get) method descr = Py.String.to_string (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) method images = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "images" |> Wrap_utils.Option.get) end)
                  let load_files ?description ?categories ?load_content ?shuffle ?encoding ?decode_error ?random_state ~container_path () =
                     Py.Module.get_function_with_keywords __wrap_namespace "load_files"
                       [||]
                       (Wrap_utils.keyword_args [("description", Wrap_utils.Option.map description Py.String.of_string); ("categories", Wrap_utils.Option.map categories Np.Obj.to_pyobject); ("load_content", Wrap_utils.Option.map load_content Py.Bool.of_bool); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("encoding", encoding); ("decode_error", Wrap_utils.Option.map decode_error (function
| `Strict -> Py.String.of_string "strict"
| `Ignore -> Py.String.of_string "ignore"
| `Replace -> Py.String.of_string "replace"
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("container_path", Some(container_path |> Py.String.of_string))])
                       |> (fun bunch -> object method data = (Py.List.to_list_map Py.String.to_string) (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method target = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method target_names = (fun py -> Py.List.to_list_map (Py.String.to_string) py) (Py.Object.get_attr_string bunch "target_names" |> Wrap_utils.Option.get) method descr = Py.String.to_string (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) method filenames = (Py.List.to_list_map Py.String.to_string) (Py.Object.get_attr_string bunch "filenames" |> Wrap_utils.Option.get) end)
let load_iris () =
   Py.Module.get_function_with_keywords __wrap_namespace "load_iris"
     [||]
     (Wrap_utils.keyword_args [("return_X_y", Some(false |> Py.Bool.of_bool)); ("as_frame", Some(false |> Py.Bool.of_bool))])
     |> (fun bunch -> object method data = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method target = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method target_names = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "target_names" |> Wrap_utils.Option.get) method feature_names = (fun py -> Py.List.to_list_map (Py.String.to_string) py) (Py.Object.get_attr_string bunch "feature_names" |> Wrap_utils.Option.get) method descr = Py.String.to_string (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) method filename = Py.String.to_string (Py.Object.get_attr_string bunch "filename" |> Wrap_utils.Option.get) end)
let load_linnerud () =
   Py.Module.get_function_with_keywords __wrap_namespace "load_linnerud"
     [||]
     (Wrap_utils.keyword_args [("return_X_y", Some(false |> Py.Bool.of_bool)); ("as_frame", Some(false |> Py.Bool.of_bool))])
     |> (fun bunch -> object method data = Wrap_utils.id (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method target = Wrap_utils.id (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method feature_names = (fun py -> Py.List.to_list_map (Py.String.to_string) py) (Py.Object.get_attr_string bunch "feature_names" |> Wrap_utils.Option.get) method target_names = (fun py -> Py.List.to_list_map (Py.String.to_string) py) (Py.Object.get_attr_string bunch "target_names" |> Wrap_utils.Option.get) method frame = Wrap_utils.id (Py.Object.get_attr_string bunch "frame" |> Wrap_utils.Option.get) method descr = Py.String.to_string (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) method data_filename = Py.String.to_string (Py.Object.get_attr_string bunch "data_filename" |> Wrap_utils.Option.get) method target_filename = Py.String.to_string (Py.Object.get_attr_string bunch "target_filename" |> Wrap_utils.Option.get) end)
let load_sample_image image_name =
   Py.Module.get_function_with_keywords __wrap_namespace "load_sample_image"
     [||]
     (Wrap_utils.keyword_args [("image_name", Some(image_name ))])
     |> (fun bunch -> object method img = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "img" |> Wrap_utils.Option.get) end)
let load_sample_images () =
   Py.Module.get_function_with_keywords __wrap_namespace "load_sample_images"
     [||]
     []
     |> (fun bunch -> object method data = Wrap_utils.id (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method images = Np.Numpy.Ndarray.List.of_pyobject (Py.Object.get_attr_string bunch "images" |> Wrap_utils.Option.get) method filenames = (Py.List.to_list_map Py.String.to_string) (Py.Object.get_attr_string bunch "filenames" |> Wrap_utils.Option.get) method descr = Py.String.to_string (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) end)
                  let load_svmlight_file ?n_features ?dtype ?multilabel ?zero_based ?query_id ?offset ?length ~f () =
                     Py.Module.get_function_with_keywords __wrap_namespace "load_svmlight_file"
                       [||]
                       (Wrap_utils.keyword_args [("n_features", Wrap_utils.Option.map n_features Py.Int.of_int); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("multilabel", Wrap_utils.Option.map multilabel Py.Bool.of_bool); ("zero_based", Wrap_utils.Option.map zero_based (function
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("query_id", Wrap_utils.Option.map query_id Py.Bool.of_bool); ("offset", Wrap_utils.Option.map offset Py.Int.of_int); ("length", Wrap_utils.Option.map length Py.Int.of_int); ("f", Some(f |> (function
| `S x -> Py.String.of_string x
| `I x -> Py.Int.of_int x
| `File_like x -> Wrap_utils.id x
)))])
                       |> (fun bunch -> object method x = (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t)) (Py.Object.get_attr_string bunch "X" |> Wrap_utils.Option.get) method y = Wrap_utils.id (Py.Object.get_attr_string bunch "y" |> Wrap_utils.Option.get) method query_id = (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Object.get_attr_string bunch "query_id" |> Wrap_utils.Option.get) end)
                  let load_svmlight_files ?n_features ?dtype ?multilabel ?zero_based ?query_id ?offset ?length ~files () =
                     Py.Module.get_function_with_keywords __wrap_namespace "load_svmlight_files"
                       [||]
                       (Wrap_utils.keyword_args [("n_features", Wrap_utils.Option.map n_features Py.Int.of_int); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("multilabel", Wrap_utils.Option.map multilabel Py.Bool.of_bool); ("zero_based", Wrap_utils.Option.map zero_based (function
| `Auto -> Py.String.of_string "auto"
| `Bool x -> Py.Bool.of_bool x
)); ("query_id", Wrap_utils.Option.map query_id Py.Bool.of_bool); ("offset", Wrap_utils.Option.map offset Py.Int.of_int); ("length", Wrap_utils.Option.map length Py.Int.of_int); ("files", Some(files ))])

let load_wine () =
   Py.Module.get_function_with_keywords __wrap_namespace "load_wine"
     [||]
     (Wrap_utils.keyword_args [("return_X_y", Some(false |> Py.Bool.of_bool)); ("as_frame", Some(false |> Py.Bool.of_bool))])
     |> (fun bunch -> object method data = Wrap_utils.id (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method target = Wrap_utils.id (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method feature_names = (fun py -> Py.List.to_list_map (Py.String.to_string) py) (Py.Object.get_attr_string bunch "feature_names" |> Wrap_utils.Option.get) method target_names = (fun py -> Py.List.to_list_map (Py.String.to_string) py) (Py.Object.get_attr_string bunch "target_names" |> Wrap_utils.Option.get) method frame = Wrap_utils.id (Py.Object.get_attr_string bunch "frame" |> Wrap_utils.Option.get) method descr = Py.String.to_string (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) method https = Wrap_utils.id (Py.Object.get_attr_string bunch "https" |> Wrap_utils.Option.get) end)
let make_biclusters ?noise ?minval ?maxval ?shuffle ?random_state ~shape ~n_clusters () =
   Py.Module.get_function_with_keywords __wrap_namespace "make_biclusters"
     [||]
     (Wrap_utils.keyword_args [("noise", Wrap_utils.Option.map noise Py.Float.of_float); ("minval", Wrap_utils.Option.map minval Py.Int.of_int); ("maxval", Wrap_utils.Option.map maxval Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml))); ("n_clusters", Some(n_clusters |> Py.Int.of_int))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 2))))
                  let make_blobs ?n_samples ?n_features ?centers ?cluster_std ?center_box ?shuffle ?random_state ?return_centers () =
                     Py.Module.get_function_with_keywords __wrap_namespace "make_blobs"
                       [||]
                       (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples (function
| `Arr x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("n_features", Wrap_utils.Option.map n_features Py.Int.of_int); ("centers", Wrap_utils.Option.map centers (function
| `Arr x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)); ("cluster_std", Wrap_utils.Option.map cluster_std Np.Obj.to_pyobject); ("center_box", Wrap_utils.Option.map center_box (fun (ml_0, ml_1) -> Py.Tuple.of_list [(Py.Float.of_float ml_0); (Py.Float.of_float ml_1)])); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("return_centers", Wrap_utils.Option.map return_centers Py.Bool.of_bool)])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 2))))
                  let make_checkerboard ?noise ?minval ?maxval ?shuffle ?random_state ~shape ~n_clusters () =
                     Py.Module.get_function_with_keywords __wrap_namespace "make_checkerboard"
                       [||]
                       (Wrap_utils.keyword_args [("noise", Wrap_utils.Option.map noise Py.Float.of_float); ("minval", Wrap_utils.Option.map minval Py.Int.of_int); ("maxval", Wrap_utils.Option.map maxval Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml))); ("n_clusters", Some(n_clusters |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 2))))
                  let make_circles ?n_samples ?shuffle ?noise ?random_state ?factor () =
                     Py.Module.get_function_with_keywords __wrap_namespace "make_circles"
                       [||]
                       (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples (function
| `I x -> Py.Int.of_int x
| `Two_element_tuple x -> Wrap_utils.id x
)); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("noise", Wrap_utils.Option.map noise Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("factor", Wrap_utils.Option.map factor Py.Float.of_float)])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let make_classification ?n_samples ?n_features ?n_informative ?n_redundant ?n_repeated ?n_classes ?n_clusters_per_class ?weights ?flip_y ?class_sep ?hypercube ?shift ?scale ?shuffle ?random_state () =
                     Py.Module.get_function_with_keywords __wrap_namespace "make_classification"
                       [||]
                       (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("n_features", Wrap_utils.Option.map n_features Py.Int.of_int); ("n_informative", Wrap_utils.Option.map n_informative Py.Int.of_int); ("n_redundant", Wrap_utils.Option.map n_redundant Py.Int.of_int); ("n_repeated", Wrap_utils.Option.map n_repeated Py.Int.of_int); ("n_classes", Wrap_utils.Option.map n_classes Py.Int.of_int); ("n_clusters_per_class", Wrap_utils.Option.map n_clusters_per_class Py.Int.of_int); ("weights", Wrap_utils.Option.map weights Np.Obj.to_pyobject); ("flip_y", Wrap_utils.Option.map flip_y Py.Float.of_float); ("class_sep", Wrap_utils.Option.map class_sep Py.Float.of_float); ("hypercube", Wrap_utils.Option.map hypercube Py.Bool.of_bool); ("shift", Wrap_utils.Option.map shift (function
| `Arr x -> Np.Obj.to_pyobject x
| `None -> Py.none
)); ("scale", Wrap_utils.Option.map scale (function
| `Arr x -> Np.Obj.to_pyobject x
| `None -> Py.none
)); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
let make_friedman1 ?n_samples ?n_features ?noise ?random_state () =
   Py.Module.get_function_with_keywords __wrap_namespace "make_friedman1"
     [||]
     (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("n_features", Wrap_utils.Option.map n_features Py.Int.of_int); ("noise", Wrap_utils.Option.map noise Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
let make_friedman2 ?n_samples ?noise ?random_state () =
   Py.Module.get_function_with_keywords __wrap_namespace "make_friedman2"
     [||]
     (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("noise", Wrap_utils.Option.map noise Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
let make_friedman3 ?n_samples ?noise ?random_state () =
   Py.Module.get_function_with_keywords __wrap_namespace "make_friedman3"
     [||]
     (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("noise", Wrap_utils.Option.map noise Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
let make_gaussian_quantiles ?mean ?cov ?n_samples ?n_features ?n_classes ?shuffle ?random_state () =
   Py.Module.get_function_with_keywords __wrap_namespace "make_gaussian_quantiles"
     [||]
     (Wrap_utils.keyword_args [("mean", Wrap_utils.Option.map mean Np.Obj.to_pyobject); ("cov", Wrap_utils.Option.map cov Py.Float.of_float); ("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("n_features", Wrap_utils.Option.map n_features Py.Int.of_int); ("n_classes", Wrap_utils.Option.map n_classes Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
let make_hastie_10_2 ?n_samples ?random_state () =
   Py.Module.get_function_with_keywords __wrap_namespace "make_hastie_10_2"
     [||]
     (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
let make_low_rank_matrix ?n_samples ?n_features ?effective_rank ?tail_strength ?random_state () =
   Py.Module.get_function_with_keywords __wrap_namespace "make_low_rank_matrix"
     [||]
     (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("n_features", Wrap_utils.Option.map n_features Py.Int.of_int); ("effective_rank", Wrap_utils.Option.map effective_rank Py.Int.of_int); ("tail_strength", Wrap_utils.Option.map tail_strength Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let make_moons ?n_samples ?shuffle ?noise ?random_state () =
                     Py.Module.get_function_with_keywords __wrap_namespace "make_moons"
                       [||]
                       (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples (function
| `I x -> Py.Int.of_int x
| `Two_element_tuple x -> Wrap_utils.id x
)); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("noise", Wrap_utils.Option.map noise Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
                  let make_multilabel_classification ?n_samples ?n_features ?n_classes ?n_labels ?length ?allow_unlabeled ?sparse ?return_indicator ?return_distributions ?random_state () =
                     Py.Module.get_function_with_keywords __wrap_namespace "make_multilabel_classification"
                       [||]
                       (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("n_features", Wrap_utils.Option.map n_features Py.Int.of_int); ("n_classes", Wrap_utils.Option.map n_classes Py.Int.of_int); ("n_labels", Wrap_utils.Option.map n_labels Py.Int.of_int); ("length", Wrap_utils.Option.map length Py.Int.of_int); ("allow_unlabeled", Wrap_utils.Option.map allow_unlabeled Py.Bool.of_bool); ("sparse", Wrap_utils.Option.map sparse Py.Bool.of_bool); ("return_indicator", Wrap_utils.Option.map return_indicator (function
| `False -> Py.Bool.f
| `Dense -> Py.String.of_string "dense"
| `Sparse -> Py.String.of_string "sparse"
)); ("return_distributions", Wrap_utils.Option.map return_distributions Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 2)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 3))))
let make_regression ?n_samples ?n_features ?n_informative ?n_targets ?bias ?effective_rank ?tail_strength ?noise ?shuffle ?random_state () =
   Py.Module.get_function_with_keywords __wrap_namespace "make_regression"
     [||]
     (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("n_features", Wrap_utils.Option.map n_features Py.Int.of_int); ("n_informative", Wrap_utils.Option.map n_informative Py.Int.of_int); ("n_targets", Wrap_utils.Option.map n_targets Py.Int.of_int); ("bias", Wrap_utils.Option.map bias Py.Float.of_float); ("effective_rank", Wrap_utils.Option.map effective_rank Py.Int.of_int); ("tail_strength", Wrap_utils.Option.map tail_strength Py.Float.of_float); ("noise", Wrap_utils.Option.map noise Py.Float.of_float); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("coef", Some(true |> Py.Bool.of_bool)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 2))))
let make_s_curve ?n_samples ?noise ?random_state () =
   Py.Module.get_function_with_keywords __wrap_namespace "make_s_curve"
     [||]
     (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("noise", Wrap_utils.Option.map noise Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
let make_sparse_coded_signal ?random_state ~n_samples ~n_components ~n_features ~n_nonzero_coefs () =
   Py.Module.get_function_with_keywords __wrap_namespace "make_sparse_coded_signal"
     [||]
     (Wrap_utils.keyword_args [("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("n_samples", Some(n_samples |> Py.Int.of_int)); ("n_components", Some(n_components |> Py.Int.of_int)); ("n_features", Some(n_features |> Py.Int.of_int)); ("n_nonzero_coefs", Some(n_nonzero_coefs |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 2))))
let make_sparse_spd_matrix ?dim ?alpha ?norm_diag ?smallest_coef ?largest_coef ?random_state () =
   Py.Module.get_function_with_keywords __wrap_namespace "make_sparse_spd_matrix"
     [||]
     (Wrap_utils.keyword_args [("dim", Wrap_utils.Option.map dim Py.Int.of_int); ("alpha", Wrap_utils.Option.map alpha Py.Float.of_float); ("norm_diag", Wrap_utils.Option.map norm_diag Py.Bool.of_bool); ("smallest_coef", Wrap_utils.Option.map smallest_coef Py.Float.of_float); ("largest_coef", Wrap_utils.Option.map largest_coef Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Object|`Spmatrix] Np.Obj.t))
let make_sparse_uncorrelated ?n_samples ?n_features ?random_state () =
   Py.Module.get_function_with_keywords __wrap_namespace "make_sparse_uncorrelated"
     [||]
     (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("n_features", Wrap_utils.Option.map n_features Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
let make_spd_matrix ?random_state ~n_dim () =
   Py.Module.get_function_with_keywords __wrap_namespace "make_spd_matrix"
     [||]
     (Wrap_utils.keyword_args [("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("n_dim", Some(n_dim |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let make_swiss_roll ?n_samples ?noise ?random_state () =
   Py.Module.get_function_with_keywords __wrap_namespace "make_swiss_roll"
     [||]
     (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("noise", Wrap_utils.Option.map noise Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
