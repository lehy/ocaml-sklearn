let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.feature_extraction"

module DictVectorizer = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?dtype ?separator ?sparse ?sort () =
   Py.Module.get_function_with_keywords ns "DictVectorizer"
     [||]
     (Wrap_utils.keyword_args [("dtype", dtype); ("separator", separator); ("sparse", sparse); ("sort", sort)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x ))])

let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x ))])
     |> Ndarray.of_pyobject
let get_feature_names self =
   Py.Module.get_function_with_keywords self "get_feature_names"
     [||]
     []

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

                  let inverse_transform ?dict_type ~x self =
                     Py.Module.get_function_with_keywords self "inverse_transform"
                       [||]
                       (Wrap_utils.keyword_args [("dict_type", dict_type); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

let restrict ?indices ~support self =
   Py.Module.get_function_with_keywords self "restrict"
     [||]
     (Wrap_utils.keyword_args [("indices", indices); ("support", Some(support |> Ndarray.to_pyobject))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> Ndarray.of_pyobject
let vocabulary_ self =
  match Py.Object.get_attr_string self "vocabulary_" with
| None -> raise (Wrap_utils.Attribute_not_found "vocabulary_")
| Some x -> Wrap_utils.id x
let feature_names_ self =
  match Py.Object.get_attr_string self "feature_names_" with
| None -> raise (Wrap_utils.Attribute_not_found "feature_names_")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module FeatureHasher = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?n_features ?input_type ?dtype ?alternate_sign () =
   Py.Module.get_function_with_keywords ns "FeatureHasher"
     [||]
     (Wrap_utils.keyword_args [("n_features", Wrap_utils.Option.map n_features Py.Int.of_int); ("input_type", input_type); ("dtype", dtype); ("alternate_sign", alternate_sign)])

let fit ?x ?y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Ndarray.to_pyobject); ("y", y)])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~raw_X self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("raw_X", Some(raw_X ))])
     |> Ndarray.of_pyobject
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let grid_to_graph ?n_z ?mask ?return_as ?dtype ~n_x ~n_y () =
   Py.Module.get_function_with_keywords ns "grid_to_graph"
     [||]
     (Wrap_utils.keyword_args [("n_z", n_z); ("mask", mask); ("return_as", return_as); ("dtype", dtype); ("n_x", Some(n_x |> Py.Int.of_int)); ("n_y", Some(n_y ))])

module Image = struct
let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.feature_extraction.image"

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
module PatchExtractor = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?patch_size ?max_patches ?random_state () =
                     Py.Module.get_function_with_keywords ns "PatchExtractor"
                       [||]
                       (Wrap_utils.keyword_args [("patch_size", patch_size); ("max_patches", Wrap_utils.Option.map max_patches (function
| `Int x -> Py.Int.of_int x
| `Float x -> Py.Float.of_float x
)); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
))])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

                  let transform ~x self =
                     Py.Module.get_function_with_keywords self "transform"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])
                       |> Ndarray.of_pyobject
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let as_strided ?shape ?strides ?subok ?writeable ~x () =
   Py.Module.get_function_with_keywords ns "as_strided"
     [||]
     (Wrap_utils.keyword_args [("shape", Wrap_utils.Option.map shape (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("strides", strides); ("subok", subok); ("writeable", Wrap_utils.Option.map writeable Py.Bool.of_bool); ("x", Some(x |> Ndarray.to_pyobject))])
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
                  let extract_patches ?patch_shape ?extraction_step ~arr () =
                     Py.Module.get_function_with_keywords ns "extract_patches"
                       [||]
                       (Wrap_utils.keyword_args [("patch_shape", Wrap_utils.Option.map patch_shape (function
| `Int x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)); ("extraction_step", Wrap_utils.Option.map extraction_step (function
| `Int x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)); ("arr", Some(arr |> Ndarray.to_pyobject))])

                  let extract_patches_2d ?max_patches ?random_state ~image ~patch_size () =
                     Py.Module.get_function_with_keywords ns "extract_patches_2d"
                       [||]
                       (Wrap_utils.keyword_args [("max_patches", Wrap_utils.Option.map max_patches (function
| `Int x -> Py.Int.of_int x
| `Float x -> Py.Float.of_float x
)); ("random_state", Wrap_utils.Option.map random_state (function
| `Int x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("image", Some(image |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("patch_size", Some(patch_size ))])

let grid_to_graph ?n_z ?mask ?return_as ?dtype ~n_x ~n_y () =
   Py.Module.get_function_with_keywords ns "grid_to_graph"
     [||]
     (Wrap_utils.keyword_args [("n_z", n_z); ("mask", mask); ("return_as", return_as); ("dtype", dtype); ("n_x", Some(n_x |> Py.Int.of_int)); ("n_y", Some(n_y ))])

                  let img_to_graph ?mask ?return_as ?dtype ~img () =
                     Py.Module.get_function_with_keywords ns "img_to_graph"
                       [||]
                       (Wrap_utils.keyword_args [("mask", mask); ("return_as", return_as); ("dtype", dtype); ("img", Some(img |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])

                  let reconstruct_from_patches_2d ~patches ~image_size () =
                     Py.Module.get_function_with_keywords ns "reconstruct_from_patches_2d"
                       [||]
                       (Wrap_utils.keyword_args [("patches", Some(patches |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
))); ("image_size", Some(image_size ))])


end
                  let img_to_graph ?mask ?return_as ?dtype ~img () =
                     Py.Module.get_function_with_keywords ns "img_to_graph"
                       [||]
                       (Wrap_utils.keyword_args [("mask", mask); ("return_as", return_as); ("dtype", dtype); ("img", Some(img |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])

module Text = struct
let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.feature_extraction.text"

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
module CountVectorizer = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?input ?encoding ?decode_error ?strip_accents ?lowercase ?preprocessor ?tokenizer ?stop_words ?token_pattern ?ngram_range ?analyzer ?max_df ?min_df ?max_features ?vocabulary ?binary ?dtype () =
                     Py.Module.get_function_with_keywords ns "CountVectorizer"
                       [||]
                       (Wrap_utils.keyword_args [("input", Wrap_utils.Option.map input (function
| `Filename -> Py.String.of_string "filename"
| `File -> Py.String.of_string "file"
| `Content -> Py.String.of_string "content"
)); ("encoding", Wrap_utils.Option.map encoding (function
| `String x -> Py.String.of_string x
| `PyObject x -> Wrap_utils.id x
)); ("decode_error", Wrap_utils.Option.map decode_error (function
| `Strict -> Py.String.of_string "strict"
| `Ignore -> Py.String.of_string "ignore"
| `Replace -> Py.String.of_string "replace"
)); ("strip_accents", Wrap_utils.Option.map strip_accents (function
| `Ascii -> Py.String.of_string "ascii"
| `Unicode -> Py.String.of_string "unicode"
| `None -> Py.String.of_string "None"
)); ("lowercase", Wrap_utils.Option.map lowercase Py.Bool.of_bool); ("preprocessor", Wrap_utils.Option.map preprocessor (function
| `Callable x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("tokenizer", Wrap_utils.Option.map tokenizer (function
| `Callable x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("stop_words", Wrap_utils.Option.map stop_words (function
| `English -> Py.String.of_string "english"
| `ArrayLike x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("token_pattern", Wrap_utils.Option.map token_pattern Py.String.of_string); ("ngram_range", ngram_range); ("analyzer", Wrap_utils.Option.map analyzer (function
| `String x -> Py.String.of_string x
| `Word -> Py.String.of_string "word"
| `Char -> Py.String.of_string "char"
| `Char_wb -> Py.String.of_string "char_wb"
| `Callable x -> Wrap_utils.id x
)); ("max_df", Wrap_utils.Option.map max_df (function
| `Int x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)); ("min_df", Wrap_utils.Option.map min_df (function
| `Int x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)); ("max_features", Wrap_utils.Option.map max_features (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("vocabulary", Wrap_utils.Option.map vocabulary (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("binary", Wrap_utils.Option.map binary Py.Bool.of_bool); ("dtype", dtype)])

let build_analyzer self =
   Py.Module.get_function_with_keywords self "build_analyzer"
     [||]
     []

let build_preprocessor self =
   Py.Module.get_function_with_keywords self "build_preprocessor"
     [||]
     []

let build_tokenizer self =
   Py.Module.get_function_with_keywords self "build_tokenizer"
     [||]
     []

let decode ~doc self =
   Py.Module.get_function_with_keywords self "decode"
     [||]
     (Wrap_utils.keyword_args [("doc", Some(doc |> Py.String.of_string))])
     |> Py.String.to_string
let fit ?y ~raw_documents self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("raw_documents", Some(raw_documents |> Ndarray.to_pyobject))])

let fit_transform ?y ~raw_documents self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("raw_documents", Some(raw_documents |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let get_feature_names self =
   Py.Module.get_function_with_keywords self "get_feature_names"
     [||]
     []
     |> (Py.List.to_list_map Py.String.to_string)
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let get_stop_words self =
   Py.Module.get_function_with_keywords self "get_stop_words"
     [||]
     []

                  let inverse_transform ~x self =
                     Py.Module.get_function_with_keywords self "inverse_transform"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~raw_documents self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("raw_documents", Some(raw_documents |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let vocabulary_ self =
  match Py.Object.get_attr_string self "vocabulary_" with
| None -> raise (Wrap_utils.Attribute_not_found "vocabulary_")
| Some x -> Wrap_utils.id x
let fixed_vocabulary_ self =
  match Py.Object.get_attr_string self "fixed_vocabulary_" with
| None -> raise (Wrap_utils.Attribute_not_found "fixed_vocabulary_")
| Some x -> Py.Bool.to_bool x
let stop_words_ self =
  match Py.Object.get_attr_string self "stop_words_" with
| None -> raise (Wrap_utils.Attribute_not_found "stop_words_")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module FeatureHasher = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create ?n_features ?input_type ?dtype ?alternate_sign () =
   Py.Module.get_function_with_keywords ns "FeatureHasher"
     [||]
     (Wrap_utils.keyword_args [("n_features", Wrap_utils.Option.map n_features Py.Int.of_int); ("input_type", input_type); ("dtype", dtype); ("alternate_sign", alternate_sign)])

let fit ?x ?y self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Ndarray.to_pyobject); ("y", y)])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~raw_X self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("raw_X", Some(raw_X ))])
     |> Ndarray.of_pyobject
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module HashingVectorizer = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?input ?encoding ?decode_error ?strip_accents ?lowercase ?preprocessor ?tokenizer ?stop_words ?token_pattern ?ngram_range ?analyzer ?n_features ?binary ?norm ?alternate_sign ?dtype () =
                     Py.Module.get_function_with_keywords ns "HashingVectorizer"
                       [||]
                       (Wrap_utils.keyword_args [("input", Wrap_utils.Option.map input (function
| `Filename -> Py.String.of_string "filename"
| `File -> Py.String.of_string "file"
| `Content -> Py.String.of_string "content"
)); ("encoding", Wrap_utils.Option.map encoding Py.String.of_string); ("decode_error", Wrap_utils.Option.map decode_error (function
| `Strict -> Py.String.of_string "strict"
| `Ignore -> Py.String.of_string "ignore"
| `Replace -> Py.String.of_string "replace"
)); ("strip_accents", Wrap_utils.Option.map strip_accents (function
| `Ascii -> Py.String.of_string "ascii"
| `Unicode -> Py.String.of_string "unicode"
| `None -> Py.String.of_string "None"
)); ("lowercase", Wrap_utils.Option.map lowercase Py.Bool.of_bool); ("preprocessor", Wrap_utils.Option.map preprocessor (function
| `Callable x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("tokenizer", Wrap_utils.Option.map tokenizer (function
| `Callable x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("stop_words", Wrap_utils.Option.map stop_words (function
| `English -> Py.String.of_string "english"
| `ArrayLike x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("token_pattern", Wrap_utils.Option.map token_pattern Py.String.of_string); ("ngram_range", ngram_range); ("analyzer", Wrap_utils.Option.map analyzer (function
| `String x -> Py.String.of_string x
| `Word -> Py.String.of_string "word"
| `Char -> Py.String.of_string "char"
| `Char_wb -> Py.String.of_string "char_wb"
| `Callable x -> Wrap_utils.id x
)); ("n_features", Wrap_utils.Option.map n_features Py.Int.of_int); ("binary", Wrap_utils.Option.map binary Py.Bool.of_bool); ("norm", Wrap_utils.Option.map norm (function
| `L1 -> Py.String.of_string "l1"
| `L2 -> Py.String.of_string "l2"
| `None -> Py.String.of_string "None"
)); ("alternate_sign", Wrap_utils.Option.map alternate_sign Py.Bool.of_bool); ("dtype", dtype)])

let build_analyzer self =
   Py.Module.get_function_with_keywords self "build_analyzer"
     [||]
     []

let build_preprocessor self =
   Py.Module.get_function_with_keywords self "build_preprocessor"
     [||]
     []

let build_tokenizer self =
   Py.Module.get_function_with_keywords self "build_tokenizer"
     [||]
     []

let decode ~doc self =
   Py.Module.get_function_with_keywords self "decode"
     [||]
     (Wrap_utils.keyword_args [("doc", Some(doc |> Py.String.of_string))])
     |> Py.String.to_string
let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x ))])
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let get_stop_words self =
   Py.Module.get_function_with_keywords self "get_stop_words"
     [||]
     []

let partial_fit ?y ~x self =
   Py.Module.get_function_with_keywords self "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Ndarray.to_pyobject))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x ))])
     |> Ndarray.of_pyobject
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module TfidfTransformer = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?norm ?use_idf ?smooth_idf ?sublinear_tf () =
                     Py.Module.get_function_with_keywords ns "TfidfTransformer"
                       [||]
                       (Wrap_utils.keyword_args [("norm", Wrap_utils.Option.map norm (function
| `L1 -> Py.String.of_string "l1"
| `L2 -> Py.String.of_string "l2"
| `None -> Py.String.of_string "None"
)); ("use_idf", Wrap_utils.Option.map use_idf Py.Bool.of_bool); ("smooth_idf", Wrap_utils.Option.map smooth_idf Py.Bool.of_bool); ("sublinear_tf", Wrap_utils.Option.map sublinear_tf Py.Bool.of_bool)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Csr_matrix.to_pyobject))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Ndarray.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ?copy ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Csr_matrix.to_pyobject))])
     |> Ndarray.of_pyobject
let idf_ self =
  match Py.Object.get_attr_string self "idf_" with
| None -> raise (Wrap_utils.Attribute_not_found "idf_")
| Some x -> Ndarray.of_pyobject x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module TfidfVectorizer = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
                  let create ?input ?encoding ?decode_error ?strip_accents ?lowercase ?preprocessor ?tokenizer ?analyzer ?stop_words ?token_pattern ?ngram_range ?max_df ?min_df ?max_features ?vocabulary ?binary ?dtype ?norm ?use_idf ?smooth_idf ?sublinear_tf () =
                     Py.Module.get_function_with_keywords ns "TfidfVectorizer"
                       [||]
                       (Wrap_utils.keyword_args [("input", Wrap_utils.Option.map input (function
| `Filename -> Py.String.of_string "filename"
| `File -> Py.String.of_string "file"
| `Content -> Py.String.of_string "content"
)); ("encoding", Wrap_utils.Option.map encoding Py.String.of_string); ("decode_error", Wrap_utils.Option.map decode_error (function
| `Strict -> Py.String.of_string "strict"
| `Ignore -> Py.String.of_string "ignore"
| `Replace -> Py.String.of_string "replace"
)); ("strip_accents", Wrap_utils.Option.map strip_accents (function
| `Ascii -> Py.String.of_string "ascii"
| `Unicode -> Py.String.of_string "unicode"
| `None -> Py.String.of_string "None"
)); ("lowercase", Wrap_utils.Option.map lowercase Py.Bool.of_bool); ("preprocessor", Wrap_utils.Option.map preprocessor (function
| `Callable x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("tokenizer", Wrap_utils.Option.map tokenizer (function
| `Callable x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("analyzer", Wrap_utils.Option.map analyzer (function
| `String x -> Py.String.of_string x
| `Word -> Py.String.of_string "word"
| `Char -> Py.String.of_string "char"
| `Char_wb -> Py.String.of_string "char_wb"
| `Callable x -> Wrap_utils.id x
)); ("stop_words", Wrap_utils.Option.map stop_words (function
| `English -> Py.String.of_string "english"
| `ArrayLike x -> Wrap_utils.id x
| `None -> Py.String.of_string "None"
)); ("token_pattern", Wrap_utils.Option.map token_pattern Py.String.of_string); ("ngram_range", ngram_range); ("max_df", Wrap_utils.Option.map max_df (function
| `Int x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)); ("min_df", Wrap_utils.Option.map min_df (function
| `Int x -> Py.Int.of_int x
| `PyObject x -> Wrap_utils.id x
)); ("max_features", Wrap_utils.Option.map max_features (function
| `Int x -> Py.Int.of_int x
| `None -> Py.String.of_string "None"
)); ("vocabulary", Wrap_utils.Option.map vocabulary (function
| `Ndarray x -> Ndarray.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)); ("binary", Wrap_utils.Option.map binary Py.Bool.of_bool); ("dtype", dtype); ("norm", Wrap_utils.Option.map norm (function
| `L1 -> Py.String.of_string "l1"
| `L2 -> Py.String.of_string "l2"
| `None -> Py.String.of_string "None"
)); ("use_idf", Wrap_utils.Option.map use_idf Py.Bool.of_bool); ("smooth_idf", Wrap_utils.Option.map smooth_idf Py.Bool.of_bool); ("sublinear_tf", Wrap_utils.Option.map sublinear_tf Py.Bool.of_bool)])

let build_analyzer self =
   Py.Module.get_function_with_keywords self "build_analyzer"
     [||]
     []

let build_preprocessor self =
   Py.Module.get_function_with_keywords self "build_preprocessor"
     [||]
     []

let build_tokenizer self =
   Py.Module.get_function_with_keywords self "build_tokenizer"
     [||]
     []

let decode ~doc self =
   Py.Module.get_function_with_keywords self "decode"
     [||]
     (Wrap_utils.keyword_args [("doc", Some(doc |> Py.String.of_string))])
     |> Py.String.to_string
let fit ?y ~raw_documents self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("raw_documents", Some(raw_documents |> Ndarray.to_pyobject))])

let fit_transform ?y ~raw_documents self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("raw_documents", Some(raw_documents |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let get_feature_names self =
   Py.Module.get_function_with_keywords self "get_feature_names"
     [||]
     []
     |> (Py.List.to_list_map Py.String.to_string)
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])

let get_stop_words self =
   Py.Module.get_function_with_keywords self "get_stop_words"
     [||]
     []

                  let inverse_transform ~x self =
                     Py.Module.get_function_with_keywords self "inverse_transform"
                       [||]
                       (Wrap_utils.keyword_args [("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ?copy ~raw_documents self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("raw_documents", Some(raw_documents |> Ndarray.to_pyobject))])
     |> Ndarray.of_pyobject
let vocabulary_ self =
  match Py.Object.get_attr_string self "vocabulary_" with
| None -> raise (Wrap_utils.Attribute_not_found "vocabulary_")
| Some x -> Wrap_utils.id x
let fixed_vocabulary_ self =
  match Py.Object.get_attr_string self "fixed_vocabulary_" with
| None -> raise (Wrap_utils.Attribute_not_found "fixed_vocabulary_")
| Some x -> Py.Bool.to_bool x
let idf_ self =
  match Py.Object.get_attr_string self "idf_" with
| None -> raise (Wrap_utils.Attribute_not_found "idf_")
| Some x -> Ndarray.of_pyobject x
let stop_words_ self =
  match Py.Object.get_attr_string self "stop_words_" with
| None -> raise (Wrap_utils.Attribute_not_found "stop_words_")
| Some x -> Wrap_utils.id x
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module TransformerMixin = struct
type t = Py.Object.t
let of_pyobject x = x
let to_pyobject x = x
let create () =
   Py.Module.get_function_with_keywords ns "TransformerMixin"
     [||]
     []

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Ndarray.to_pyobject); ("X", Some(x |> Ndarray.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Ndarray.of_pyobject
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
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
                  let normalize ?norm ?axis ?copy ?return_norm ~x () =
                     Py.Module.get_function_with_keywords ns "normalize"
                       [||]
                       (Wrap_utils.keyword_args [("norm", Wrap_utils.Option.map norm (function
| `L1 -> Py.String.of_string "l1"
| `L2 -> Py.String.of_string "l2"
| `Max -> Py.String.of_string "max"
| `PyObject x -> Wrap_utils.id x
)); ("axis", axis); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("return_norm", Wrap_utils.Option.map return_norm Py.Bool.of_bool); ("X", Some(x |> (function
| `Ndarray x -> Ndarray.to_pyobject x
| `SparseMatrix x -> Csr_matrix.to_pyobject x
)))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let strip_accents_ascii ~s () =
   Py.Module.get_function_with_keywords ns "strip_accents_ascii"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s |> Py.String.of_string))])

let strip_accents_unicode ~s () =
   Py.Module.get_function_with_keywords ns "strip_accents_unicode"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s |> Py.String.of_string))])

let strip_tags ~s () =
   Py.Module.get_function_with_keywords ns "strip_tags"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s |> Py.String.of_string))])


end
