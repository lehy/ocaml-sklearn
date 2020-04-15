let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.datasets"

let clear_data_home ?data_home () =
   Py.Module.get_function_with_keywords ns "clear_data_home"
     [||]
     (Wrap_utils.keyword_args [("data_home", Wrap_utils.Option.map data_home Py.String.of_string)])

                  let dump_svmlight_file ?zero_based ?comment ?query_id ?multilabel ~x ~y ~f () =
                     Py.Module.get_function_with_keywords ns "dump_svmlight_file"
                       [||]
                       (Wrap_utils.keyword_args [("zero_based", Wrap_utils.Option.map zero_based Py.Bool.of_bool); ("comment", Wrap_utils.Option.map comment Py.String.of_string); ("query_id", Wrap_utils.Option.map query_id Ndarray.to_pyobject); ("multilabel", Wrap_utils.Option.map multilabel Py.Bool.of_bool); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
))); ("y", Some(y |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("f", Some(f |> (function
| `String x -> Py.String.of_string x
| `PyObject x -> Wrap_utils.id x
)))])

                  let fetch_20newsgroups ?data_home ?subset ?categories ?shuffle ?random_state ?remove ?download_if_missing () =
                     Py.Module.get_function_with_keywords ns "fetch_20newsgroups"
                       [||]
                       (Wrap_utils.keyword_args [("data_home", data_home); ("subset", Wrap_utils.Option.map subset (function
| `Train -> Py.String.of_string "train"
| `Test -> Py.String.of_string "test"
| `All -> Py.String.of_string "all"
)); ("categories", Wrap_utils.Option.map categories (function
| `String x -> Py.String.of_string x
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
)); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("remove", remove); ("download_if_missing", download_if_missing); ("return_X_y", Some(false |> Py.Bool.of_bool))])
                       |> (fun bunch -> object method data = Wrap_utils.id (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method target = Ndarray.of_pyobject (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method filenames = (Py.List.to_list_map Py.String.to_string) (Py.Object.get_attr_string bunch "filenames" |> Wrap_utils.Option.get) method descr = Wrap_utils.id (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) method target_names = Wrap_utils.id (Py.Object.get_attr_string bunch "target_names" |> Wrap_utils.Option.get) end)
                  let fetch_20newsgroups_vectorized ?subset ?remove ?data_home ?download_if_missing ?normalize () =
                     Py.Module.get_function_with_keywords ns "fetch_20newsgroups_vectorized"
                       [||]
                       (Wrap_utils.keyword_args [("subset", Wrap_utils.Option.map subset (function
| `Train -> Py.String.of_string "train"
| `Test -> Py.String.of_string "test"
| `All -> Py.String.of_string "all"
)); ("remove", remove); ("data_home", data_home); ("download_if_missing", download_if_missing); ("return_X_y", Some(false |> Py.Bool.of_bool)); ("normalize", Wrap_utils.Option.map normalize Py.Bool.of_bool)])
                       |> (fun bunch -> object method data = Csr_matrix.of_pyobject (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method target = Ndarray.of_pyobject (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method target_names = Wrap_utils.id (Py.Object.get_attr_string bunch "target_names" |> Wrap_utils.Option.get) method descr = Wrap_utils.id (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) end)
let fetch_california_housing ?data_home ?download_if_missing () =
   Py.Module.get_function_with_keywords ns "fetch_california_housing"
     [||]
     (Wrap_utils.keyword_args [("data_home", data_home); ("download_if_missing", download_if_missing); ("return_X_y", Some(false |> Py.Bool.of_bool))])
     |> (fun bunch -> object method data = Ndarray.of_pyobject (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method target = Ndarray.of_pyobject (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method feature_names = (Py.List.to_list_map Py.String.to_string) (Py.Object.get_attr_string bunch "feature_names" |> Wrap_utils.Option.get) method descr = Py.String.to_string (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) end)
                  let fetch_covtype ?data_home ?download_if_missing ?random_state ?shuffle () =
                     Py.Module.get_function_with_keywords ns "fetch_covtype"
                       [||]
                       (Wrap_utils.keyword_args [("data_home", Wrap_utils.Option.map data_home Py.String.of_string); ("download_if_missing", Wrap_utils.Option.map download_if_missing Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("return_X_y", Some(false |> Py.Bool.of_bool))])
                       |> (fun bunch -> object method data = Ndarray.of_pyobject (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method target = Ndarray.of_pyobject (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method descr = Py.String.to_string (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) end)
                  let fetch_kddcup99 ?subset ?data_home ?shuffle ?random_state ?percent10 ?download_if_missing () =
                     Py.Module.get_function_with_keywords ns "fetch_kddcup99"
                       [||]
                       (Wrap_utils.keyword_args [("subset", Wrap_utils.Option.map subset (function
| `SA -> Py.String.of_string "SA"
| `SF -> Py.String.of_string "SF"
| `Http -> Py.String.of_string "http"
| `Smtp -> Py.String.of_string "smtp"
| `None -> Py.String.of_string "None"
)); ("data_home", Wrap_utils.Option.map data_home Py.String.of_string); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("percent10", Wrap_utils.Option.map percent10 Py.Bool.of_bool); ("download_if_missing", Wrap_utils.Option.map download_if_missing Py.Bool.of_bool); ("return_X_y", Some(false |> Py.Bool.of_bool))])
                       |> (fun bunch -> object method data = Wrap_utils.id (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method target = Wrap_utils.id (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method descr = Wrap_utils.id (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) end)
let fetch_lfw_pairs ?subset ?data_home ?funneled ?resize ?color ?slice_ ?download_if_missing () =
   Py.Module.get_function_with_keywords ns "fetch_lfw_pairs"
     [||]
     (Wrap_utils.keyword_args [("subset", subset); ("data_home", data_home); ("funneled", Wrap_utils.Option.map funneled Py.Bool.of_bool); ("resize", Wrap_utils.Option.map resize Py.Float.of_float); ("color", Wrap_utils.Option.map color Py.Bool.of_bool); ("slice_", slice_); ("download_if_missing", download_if_missing)])
     |> (fun bunch -> object method data = Wrap_utils.id (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method pairs = Wrap_utils.id (Py.Object.get_attr_string bunch "pairs" |> Wrap_utils.Option.get) method target = Wrap_utils.id (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method descr = Py.String.to_string (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) end)
let fetch_lfw_people ?data_home ?funneled ?resize ?min_faces_per_person ?color ?slice_ ?download_if_missing () =
   Py.Module.get_function_with_keywords ns "fetch_lfw_people"
     [||]
     (Wrap_utils.keyword_args [("data_home", data_home); ("funneled", Wrap_utils.Option.map funneled Py.Bool.of_bool); ("resize", Wrap_utils.Option.map resize Py.Float.of_float); ("min_faces_per_person", Wrap_utils.Option.map min_faces_per_person Py.Int.of_int); ("color", Wrap_utils.Option.map color Py.Bool.of_bool); ("slice_", slice_); ("download_if_missing", download_if_missing); ("return_X_y", Some(false |> Py.Bool.of_bool))])
     |> (fun bunch -> object method data = Ndarray.of_pyobject (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method images = Ndarray.of_pyobject (Py.Object.get_attr_string bunch "images" |> Wrap_utils.Option.get) method target = Ndarray.of_pyobject (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method descr = Py.String.to_string (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) end)
                  let fetch_olivetti_faces ?data_home ?shuffle ?random_state ?download_if_missing () =
                     Py.Module.get_function_with_keywords ns "fetch_olivetti_faces"
                       [||]
                       (Wrap_utils.keyword_args [("data_home", data_home); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("download_if_missing", download_if_missing); ("return_X_y", Some(false |> Py.Bool.of_bool))])
                       |> (fun bunch -> object method data = Wrap_utils.id (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method images = Wrap_utils.id (Py.Object.get_attr_string bunch "images" |> Wrap_utils.Option.get) method target = Wrap_utils.id (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method descr = Wrap_utils.id (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) end)
                  let fetch_openml ?name ?version ?data_id ?data_home ?target_column ?cache ?as_frame () =
                     Py.Module.get_function_with_keywords ns "fetch_openml"
                       [||]
                       (Wrap_utils.keyword_args [("name", Wrap_utils.Option.map name Py.String.of_string); ("version", Wrap_utils.Option.map version (function
| `Int x -> Py.Int.of_int x
| `Active -> Py.String.of_string "active"
)); ("data_id", Wrap_utils.Option.map data_id (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("data_home", Wrap_utils.Option.map data_home Py.String.of_string); ("target_column", Wrap_utils.Option.map target_column (function
| `String x -> Py.String.of_string x
| `ArrayLike x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("cache", Wrap_utils.Option.map cache Py.Bool.of_bool); ("return_X_y", Some(false |> Py.Bool.of_bool)); ("as_frame", Wrap_utils.Option.map as_frame Py.Bool.of_bool)])
                       |> (fun bunch -> object method data = Wrap_utils.id (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method target = Wrap_utils.id (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method descr = Py.String.to_string (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) method feature_names = (Py.List.to_list_map Py.String.to_string) (Py.Object.get_attr_string bunch "feature_names" |> Wrap_utils.Option.get) method target_names = (Py.List.to_list_map Py.String.to_string) (Py.Object.get_attr_string bunch "target_names" |> Wrap_utils.Option.get) method categories = Wrap_utils.id (Py.Object.get_attr_string bunch "categories" |> Wrap_utils.Option.get) method details = Wrap_utils.id (Py.Object.get_attr_string bunch "details" |> Wrap_utils.Option.get) method frame = Wrap_utils.id (Py.Object.get_attr_string bunch "frame" |> Wrap_utils.Option.get) end)
                  let fetch_rcv1 ?data_home ?subset ?download_if_missing ?random_state ?shuffle () =
                     Py.Module.get_function_with_keywords ns "fetch_rcv1"
                       [||]
                       (Wrap_utils.keyword_args [("data_home", Wrap_utils.Option.map data_home Py.String.of_string); ("subset", Wrap_utils.Option.map subset (function
| `Train -> Py.String.of_string "train"
| `Test -> Py.String.of_string "test"
| `All -> Py.String.of_string "all"
)); ("download_if_missing", Wrap_utils.Option.map download_if_missing Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("return_X_y", Some(false |> Py.Bool.of_bool))])
                       |> (fun bunch -> object method data = Wrap_utils.id (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method target = Wrap_utils.id (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method sample_id = Wrap_utils.id (Py.Object.get_attr_string bunch "sample_id" |> Wrap_utils.Option.get) method target_names = Wrap_utils.id (Py.Object.get_attr_string bunch "target_names" |> Wrap_utils.Option.get) method descr = Py.String.to_string (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) end)
let fetch_species_distributions ?data_home ?download_if_missing () =
   Py.Module.get_function_with_keywords ns "fetch_species_distributions"
     [||]
     (Wrap_utils.keyword_args [("data_home", data_home); ("download_if_missing", download_if_missing)])
     |> (fun bunch -> object method coverages = Ndarray.of_pyobject (Py.Object.get_attr_string bunch "coverages" |> Wrap_utils.Option.get) method train = Wrap_utils.id (Py.Object.get_attr_string bunch "train" |> Wrap_utils.Option.get) method test = Wrap_utils.id (Py.Object.get_attr_string bunch "test" |> Wrap_utils.Option.get) method nx = Wrap_utils.id (Py.Object.get_attr_string bunch "Nx" |> Wrap_utils.Option.get) method x_left_lower_corner = Wrap_utils.id (Py.Object.get_attr_string bunch "x_left_lower_corner" |> Wrap_utils.Option.get) method grid_size = Py.Float.to_float (Py.Object.get_attr_string bunch "grid_size" |> Wrap_utils.Option.get) end)
let get_data_home ?data_home () =
   Py.Module.get_function_with_keywords ns "get_data_home"
     [||]
     (Wrap_utils.keyword_args [("data_home", Wrap_utils.Option.map data_home Py.String.of_string)])

let load_boston () =
   Py.Module.get_function_with_keywords ns "load_boston"
     [||]
     (Wrap_utils.keyword_args [("return_X_y", Some(false |> Py.Bool.of_bool))])
     |> (fun bunch -> object method data = Wrap_utils.id (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method descr = Wrap_utils.id (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) end)
let load_breast_cancer () =
   Py.Module.get_function_with_keywords ns "load_breast_cancer"
     [||]
     (Wrap_utils.keyword_args [("return_X_y", Some(false |> Py.Bool.of_bool))])
     |> (fun bunch -> object method data = Wrap_utils.id (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method target_names = Wrap_utils.id (Py.Object.get_attr_string bunch "target_names" |> Wrap_utils.Option.get) method https = Wrap_utils.id (Py.Object.get_attr_string bunch "https" |> Wrap_utils.Option.get) end)
let load_diabetes () =
   Py.Module.get_function_with_keywords ns "load_diabetes"
     [||]
     (Wrap_utils.keyword_args [("return_X_y", Some(false |> Py.Bool.of_bool))])
     |> (fun bunch -> object method data = Wrap_utils.id (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method sample = Wrap_utils.id (Py.Object.get_attr_string bunch "sample" |> Wrap_utils.Option.get) end)
                  let load_digits ?n_class () =
                     Py.Module.get_function_with_keywords ns "load_digits"
                       [||]
                       (Wrap_utils.keyword_args [("n_class", Wrap_utils.Option.map n_class (function
| `Int x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)); ("return_X_y", Some(false |> Py.Bool.of_bool))])
                       |> (fun bunch -> object method data = Wrap_utils.id (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method sample = Wrap_utils.id (Py.Object.get_attr_string bunch "sample" |> Wrap_utils.Option.get) method https = Wrap_utils.id (Py.Object.get_attr_string bunch "https" |> Wrap_utils.Option.get) end)
                  let load_files ?description ?categories ?load_content ?shuffle ?encoding ?decode_error ?random_state ~container_path () =
                     Py.Module.get_function_with_keywords ns "load_files"
                       [||]
                       (Wrap_utils.keyword_args [("description", Wrap_utils.Option.map description Py.String.of_string); ("categories", Wrap_utils.Option.map categories (function
| `None -> Py.String.of_string "None"
| `PyObject x -> Wrap_utils.id x
)); ("load_content", Wrap_utils.Option.map load_content Py.Bool.of_bool); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("encoding", Wrap_utils.Option.map encoding (function
| `String x -> Py.String.of_string x
| `PyObject x -> Wrap_utils.id x
)); ("decode_error", Wrap_utils.Option.map decode_error (function
| `Strict -> Py.String.of_string "strict"
| `Ignore -> Py.String.of_string "ignore"
| `Replace -> Py.String.of_string "replace"
)); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("container_path", Some(container_path |> Py.String.of_string))])
                       |> (fun bunch -> object method data = Wrap_utils.id (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method target_names = Wrap_utils.id (Py.Object.get_attr_string bunch "target_names" |> Wrap_utils.Option.get) end)
let load_iris () =
   Py.Module.get_function_with_keywords ns "load_iris"
     [||]
     (Wrap_utils.keyword_args [("return_X_y", Some(false |> Py.Bool.of_bool))])
     |> (fun bunch -> object method data = Ndarray.of_pyobject (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method target = Ndarray.of_pyobject (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method target_names = (fun py -> let len = Py.Sequence.length py in Array.init len (fun i -> Py.String.to_string (Py.Sequence.get_item py i))) (Py.Object.get_attr_string bunch "target_names" |> Wrap_utils.Option.get) method feature_names = (fun py -> let len = Py.Sequence.length py in Array.init len (fun i -> Py.String.to_string (Py.Sequence.get_item py i))) (Py.Object.get_attr_string bunch "feature_names" |> Wrap_utils.Option.get) method descr = Py.String.to_string (Py.Object.get_attr_string bunch "DESCR" |> Wrap_utils.Option.get) method filename = Py.String.to_string (Py.Object.get_attr_string bunch "filename" |> Wrap_utils.Option.get) end)
let load_linnerud () =
   Py.Module.get_function_with_keywords ns "load_linnerud"
     [||]
     (Wrap_utils.keyword_args [("return_X_y", Some(false |> Py.Bool.of_bool))])
     |> (fun bunch -> object method data = Wrap_utils.id (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method target = Wrap_utils.id (Py.Object.get_attr_string bunch "target" |> Wrap_utils.Option.get) method measurements = Wrap_utils.id (Py.Object.get_attr_string bunch "measurements" |> Wrap_utils.Option.get) method target_filename = Wrap_utils.id (Py.Object.get_attr_string bunch "target_filename" |> Wrap_utils.Option.get) end)
let load_sample_image ~image_name () =
   Py.Module.get_function_with_keywords ns "load_sample_image"
     [||]
     (Wrap_utils.keyword_args [("image_name", Some(image_name ))])
     |> (fun bunch -> object method img = Wrap_utils.id (Py.Object.get_attr_string bunch "img" |> Wrap_utils.Option.get) end)
let load_sample_images () =
   Py.Module.get_function_with_keywords ns "load_sample_images"
     [||]
     []
     |> (fun bunch -> object method data = Wrap_utils.id (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) end)
                  let load_svmlight_file ?n_features ?dtype ?multilabel ?zero_based ?query_id ?offset ?length ~f () =
                     Py.Module.get_function_with_keywords ns "load_svmlight_file"
                       [||]
                       (Wrap_utils.keyword_args [("n_features", Wrap_utils.Option.map n_features (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("dtype", dtype); ("multilabel", Wrap_utils.Option.map multilabel Py.Bool.of_bool); ("zero_based", Wrap_utils.Option.map zero_based (function
| `Bool x -> Py.Bool.of_bool x
| `Auto -> Py.String.of_string "auto"
)); ("query_id", Wrap_utils.Option.map query_id Py.Bool.of_bool); ("offset", Wrap_utils.Option.map offset Py.Int.of_int); ("length", Wrap_utils.Option.map length Py.Int.of_int); ("f", Some(f |> (function
| `String x -> Py.String.of_string x
| `Int x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)))])
                       |> (fun bunch -> object method x = Wrap_utils.id (Py.Object.get_attr_string bunch "X" |> Wrap_utils.Option.get) method y = Wrap_utils.id (Py.Object.get_attr_string bunch "y" |> Wrap_utils.Option.get) method query_id = Ndarray.of_pyobject (Py.Object.get_attr_string bunch "query_id" |> Wrap_utils.Option.get) end)
                  let load_svmlight_files ?n_features ?dtype ?multilabel ?zero_based ?query_id ?offset ?length ~files () =
                     Py.Module.get_function_with_keywords ns "load_svmlight_files"
                       [||]
                       (Wrap_utils.keyword_args [("n_features", Wrap_utils.Option.map n_features (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("dtype", dtype); ("multilabel", Wrap_utils.Option.map multilabel Py.Bool.of_bool); ("zero_based", Wrap_utils.Option.map zero_based (function
| `Bool x -> Py.Bool.of_bool x
| `Auto -> Py.String.of_string "auto"
)); ("query_id", Wrap_utils.Option.map query_id Py.Bool.of_bool); ("offset", Wrap_utils.Option.map offset Py.Int.of_int); ("length", Wrap_utils.Option.map length Py.Int.of_int); ("files", Some(files ))])

let load_wine () =
   Py.Module.get_function_with_keywords ns "load_wine"
     [||]
     (Wrap_utils.keyword_args [("return_X_y", Some(false |> Py.Bool.of_bool))])
     |> (fun bunch -> object method data = Wrap_utils.id (Py.Object.get_attr_string bunch "data" |> Wrap_utils.Option.get) method https = Wrap_utils.id (Py.Object.get_attr_string bunch "https" |> Wrap_utils.Option.get) end)
                  let make_biclusters ?noise ?minval ?maxval ?shuffle ?random_state ~shape ~n_clusters () =
                     Py.Module.get_function_with_keywords ns "make_biclusters"
                       [||]
                       (Wrap_utils.keyword_args [("noise", Wrap_utils.Option.map noise Py.Float.of_float); ("minval", Wrap_utils.Option.map minval Py.Int.of_int); ("maxval", Wrap_utils.Option.map maxval Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml))); ("n_clusters", Some(n_clusters |> Py.Int.of_int))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
                  let make_blobs ?n_samples ?n_features ?centers ?cluster_std ?center_box ?shuffle ?random_state () =
                     Py.Module.get_function_with_keywords ns "make_blobs"
                       [||]
                       (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples (function
| `Int x -> Py.Int.of_int x
| `Ndarray x -> Ndarray.to_pyobject x
)); ("n_features", Wrap_utils.Option.map n_features Py.Int.of_int); ("centers", Wrap_utils.Option.map centers (function
| `Int x -> Py.Int.of_int x
| `Ndarray x -> Ndarray.to_pyobject x
)); ("cluster_std", Wrap_utils.Option.map cluster_std (function
| `Float x -> Py.Float.of_float x
| `PyObject x -> Wrap_utils.id x
)); ("center_box", center_box); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])
                       |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
                  let make_checkerboard ?noise ?minval ?maxval ?shuffle ?random_state ~shape ~n_clusters () =
                     Py.Module.get_function_with_keywords ns "make_checkerboard"
                       [||]
                       (Wrap_utils.keyword_args [("noise", Wrap_utils.Option.map noise Py.Float.of_float); ("minval", Wrap_utils.Option.map minval Py.Int.of_int); ("maxval", Wrap_utils.Option.map maxval Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("shape", Some(shape |> (fun ml -> Py.List.of_list_map Py.Int.of_int ml))); ("n_clusters", Some(n_clusters |> (function
| `Int x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2))))
                  let make_circles ?n_samples ?shuffle ?noise ?random_state ?factor () =
                     Py.Module.get_function_with_keywords ns "make_circles"
                       [||]
                       (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("noise", Wrap_utils.Option.map noise (function
| `Float x -> Py.Float.of_float x
| `None -> Py.String.of_string "None"
)); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("factor", factor)])
                       |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
                  let make_classification ?n_samples ?n_features ?n_informative ?n_redundant ?n_repeated ?n_classes ?n_clusters_per_class ?weights ?flip_y ?class_sep ?hypercube ?shift ?scale ?shuffle ?random_state () =
                     Py.Module.get_function_with_keywords ns "make_classification"
                       [||]
                       (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("n_features", Wrap_utils.Option.map n_features Py.Int.of_int); ("n_informative", Wrap_utils.Option.map n_informative Py.Int.of_int); ("n_redundant", Wrap_utils.Option.map n_redundant Py.Int.of_int); ("n_repeated", Wrap_utils.Option.map n_repeated Py.Int.of_int); ("n_classes", Wrap_utils.Option.map n_classes Py.Int.of_int); ("n_clusters_per_class", Wrap_utils.Option.map n_clusters_per_class Py.Int.of_int); ("weights", Wrap_utils.Option.map weights (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("flip_y", Wrap_utils.Option.map flip_y Py.Float.of_float); ("class_sep", Wrap_utils.Option.map class_sep Py.Float.of_float); ("hypercube", Wrap_utils.Option.map hypercube Py.Bool.of_bool); ("shift", Wrap_utils.Option.map shift (function
| `Float x -> Py.Float.of_float x
| `Ndarray x -> Ndarray.to_pyobject x
| `None -> Py.String.of_string "None"
)); ("scale", Wrap_utils.Option.map scale (function
| `Float x -> Py.Float.of_float x
| `Ndarray x -> Ndarray.to_pyobject x
| `None -> Py.String.of_string "None"
)); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])
                       |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
                  let make_friedman1 ?n_samples ?n_features ?noise ?random_state () =
                     Py.Module.get_function_with_keywords ns "make_friedman1"
                       [||]
                       (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("n_features", Wrap_utils.Option.map n_features Py.Int.of_int); ("noise", Wrap_utils.Option.map noise Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])
                       |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
                  let make_friedman2 ?n_samples ?noise ?random_state () =
                     Py.Module.get_function_with_keywords ns "make_friedman2"
                       [||]
                       (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("noise", Wrap_utils.Option.map noise Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])
                       |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
                  let make_friedman3 ?n_samples ?noise ?random_state () =
                     Py.Module.get_function_with_keywords ns "make_friedman3"
                       [||]
                       (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("noise", Wrap_utils.Option.map noise Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])
                       |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
                  let make_gaussian_quantiles ?mean ?cov ?n_samples ?n_features ?n_classes ?shuffle ?random_state () =
                     Py.Module.get_function_with_keywords ns "make_gaussian_quantiles"
                       [||]
                       (Wrap_utils.keyword_args [("mean", Wrap_utils.Option.map mean Ndarray.to_pyobject); ("cov", Wrap_utils.Option.map cov Py.Float.of_float); ("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("n_features", Wrap_utils.Option.map n_features Py.Int.of_int); ("n_classes", Wrap_utils.Option.map n_classes Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])
                       |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
                  let make_hastie_10_2 ?n_samples ?random_state () =
                     Py.Module.get_function_with_keywords ns "make_hastie_10_2"
                       [||]
                       (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])
                       |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
                  let make_low_rank_matrix ?n_samples ?n_features ?effective_rank ?tail_strength ?random_state () =
                     Py.Module.get_function_with_keywords ns "make_low_rank_matrix"
                       [||]
                       (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("n_features", Wrap_utils.Option.map n_features Py.Int.of_int); ("effective_rank", Wrap_utils.Option.map effective_rank Py.Int.of_int); ("tail_strength", tail_strength); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])
                       |> Ndarray.of_pyobject
                  let make_moons ?n_samples ?shuffle ?noise ?random_state () =
                     Py.Module.get_function_with_keywords ns "make_moons"
                       [||]
                       (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("noise", Wrap_utils.Option.map noise (function
| `Float x -> Py.Float.of_float x
| `None -> Py.String.of_string "None"
)); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])
                       |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
                  let make_multilabel_classification ?n_samples ?n_features ?n_classes ?n_labels ?length ?allow_unlabeled ?sparse ?return_indicator ?return_distributions ?random_state () =
                     Py.Module.get_function_with_keywords ns "make_multilabel_classification"
                       [||]
                       (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("n_features", Wrap_utils.Option.map n_features Py.Int.of_int); ("n_classes", Wrap_utils.Option.map n_classes Py.Int.of_int); ("n_labels", Wrap_utils.Option.map n_labels Py.Int.of_int); ("length", Wrap_utils.Option.map length Py.Int.of_int); ("allow_unlabeled", Wrap_utils.Option.map allow_unlabeled Py.Bool.of_bool); ("sparse", Wrap_utils.Option.map sparse Py.Bool.of_bool); ("return_indicator", Wrap_utils.Option.map return_indicator Py.String.of_string); ("return_distributions", Wrap_utils.Option.map return_distributions Py.Bool.of_bool); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])
                       |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Ndarray.of_pyobject (Py.Tuple.get x 2)), (Ndarray.of_pyobject (Py.Tuple.get x 3))))
                  let make_regression ?n_samples ?n_features ?n_informative ?n_targets ?bias ?effective_rank ?tail_strength ?noise ?shuffle ?random_state () =
                     Py.Module.get_function_with_keywords ns "make_regression"
                       [||]
                       (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("n_features", Wrap_utils.Option.map n_features Py.Int.of_int); ("n_informative", Wrap_utils.Option.map n_informative Py.Int.of_int); ("n_targets", Wrap_utils.Option.map n_targets Py.Int.of_int); ("bias", Wrap_utils.Option.map bias Py.Float.of_float); ("effective_rank", Wrap_utils.Option.map effective_rank (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("tail_strength", tail_strength); ("noise", Wrap_utils.Option.map noise Py.Float.of_float); ("shuffle", Wrap_utils.Option.map shuffle Py.Bool.of_bool); ("coef", Some(true |> Py.Bool.of_bool)); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])
                       |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1)), (Ndarray.of_pyobject (Py.Tuple.get x 2))))
                  let make_s_curve ?n_samples ?noise ?random_state () =
                     Py.Module.get_function_with_keywords ns "make_s_curve"
                       [||]
                       (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("noise", Wrap_utils.Option.map noise Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])
                       |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
                  let make_sparse_coded_signal ?random_state ~n_samples ~n_components ~n_features ~n_nonzero_coefs () =
                     Py.Module.get_function_with_keywords ns "make_sparse_coded_signal"
                       [||]
                       (Wrap_utils.keyword_args [("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("n_samples", Some(n_samples |> Py.Int.of_int)); ("n_components", Some(n_components |> Py.Int.of_int)); ("n_features", Some(n_features |> Py.Int.of_int)); ("n_nonzero_coefs", Some(n_nonzero_coefs |> Py.Int.of_int))])
                       |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1)), (Ndarray.of_pyobject (Py.Tuple.get x 2))))
                  let make_sparse_spd_matrix ?dim ?alpha ?norm_diag ?smallest_coef ?largest_coef ?random_state () =
                     Py.Module.get_function_with_keywords ns "make_sparse_spd_matrix"
                       [||]
                       (Wrap_utils.keyword_args [("dim", Wrap_utils.Option.map dim Py.Int.of_int); ("alpha", alpha); ("norm_diag", Wrap_utils.Option.map norm_diag Py.Bool.of_bool); ("smallest_coef", smallest_coef); ("largest_coef", largest_coef); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])
                       |> Csr_matrix.of_pyobject
                  let make_sparse_uncorrelated ?n_samples ?n_features ?random_state () =
                     Py.Module.get_function_with_keywords ns "make_sparse_uncorrelated"
                       [||]
                       (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("n_features", Wrap_utils.Option.map n_features Py.Int.of_int); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])
                       |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
                  let make_spd_matrix ?random_state ~n_dim () =
                     Py.Module.get_function_with_keywords ns "make_spd_matrix"
                       [||]
                       (Wrap_utils.keyword_args [("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("n_dim", Some(n_dim |> Py.Int.of_int))])
                       |> Ndarray.of_pyobject
                  let make_swiss_roll ?n_samples ?noise ?random_state () =
                     Py.Module.get_function_with_keywords ns "make_swiss_roll"
                       [||]
                       (Wrap_utils.keyword_args [("n_samples", Wrap_utils.Option.map n_samples Py.Int.of_int); ("noise", Wrap_utils.Option.map noise Py.Float.of_float); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])
                       |> (fun x -> ((Ndarray.of_pyobject (Py.Tuple.get x 0)), (Ndarray.of_pyobject (Py.Tuple.get x 1))))
