let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.feature_extraction"

let get_py name = Py.Module.get ns name
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
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let get_feature_names self =
   Py.Module.get_function_with_keywords self "get_feature_names"
     [||]
     []

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let inverse_transform ?dict_type ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("dict_type", dict_type); ("X", Some(x |> Arr.to_pyobject))])

let restrict ?indices ~support self =
   Py.Module.get_function_with_keywords self "restrict"
     [||]
     (Wrap_utils.keyword_args [("indices", indices); ("support", Some(support |> Arr.to_pyobject))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let vocabulary_opt self =
  match Py.Object.get_attr_string self "vocabulary_" with
  | None -> failwith "attribute vocabulary_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let vocabulary_ self = match vocabulary_opt self with
  | None -> raise Not_found
  | Some x -> x

let feature_names_opt self =
  match Py.Object.get_attr_string self "feature_names_" with
  | None -> failwith "attribute feature_names_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let feature_names_ self = match feature_names_opt self with
  | None -> raise Not_found
  | Some x -> x
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
     (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Arr.to_pyobject); ("y", y)])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~raw_X self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("raw_X", Some(raw_X ))])
     |> Arr.of_pyobject
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

let get_py name = Py.Module.get ns name
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
     |> Dict.of_pyobject
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
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])

let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
let as_strided ?shape ?strides ?subok ?writeable ~x () =
   Py.Module.get_function_with_keywords ns "as_strided"
     [||]
     (Wrap_utils.keyword_args [("shape", Wrap_utils.Option.map shape (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("strides", strides); ("subok", subok); ("writeable", Wrap_utils.Option.map writeable Py.Bool.of_bool); ("x", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
                  let check_array ?accept_sparse ?accept_large_sparse ?dtype ?order ?copy ?force_all_finite ?ensure_2d ?allow_nd ?ensure_min_samples ?ensure_min_features ?warn_on_dtype ?estimator ~array () =
                     Py.Module.get_function_with_keywords ns "check_array"
                       [||]
                       (Wrap_utils.keyword_args [("accept_sparse", Wrap_utils.Option.map accept_sparse (function
| `S x -> Py.String.of_string x
| `Bool x -> Py.Bool.of_bool x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
)); ("accept_large_sparse", Wrap_utils.Option.map accept_large_sparse Py.Bool.of_bool); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Wrap_utils.id x
| `TypeList x -> Wrap_utils.id x
| `None -> Py.none
)); ("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("force_all_finite", Wrap_utils.Option.map force_all_finite (function
| `Bool x -> Py.Bool.of_bool x
| `Allow_nan -> Py.String.of_string "allow-nan"
)); ("ensure_2d", Wrap_utils.Option.map ensure_2d Py.Bool.of_bool); ("allow_nd", Wrap_utils.Option.map allow_nd Py.Bool.of_bool); ("ensure_min_samples", Wrap_utils.Option.map ensure_min_samples Py.Int.of_int); ("ensure_min_features", Wrap_utils.Option.map ensure_min_features Py.Int.of_int); ("warn_on_dtype", Wrap_utils.Option.map warn_on_dtype Py.Bool.of_bool); ("estimator", Wrap_utils.Option.map estimator (function
| `S x -> Py.String.of_string x
| `Estimator x -> Wrap_utils.id x
)); ("array", Some(array ))])

                  let check_random_state ~seed () =
                     Py.Module.get_function_with_keywords ns "check_random_state"
                       [||]
                       (Wrap_utils.keyword_args [("seed", Some(seed |> (function
| `I x -> Py.Int.of_int x
| `RandomState x -> Wrap_utils.id x
| `None -> Py.none
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
| `I x -> Py.Int.of_int x
| `Tuple x -> Wrap_utils.id x
)); ("extraction_step", Wrap_utils.Option.map extraction_step (function
| `I x -> Py.Int.of_int x
| `Tuple x -> Wrap_utils.id x
)); ("arr", Some(arr |> Arr.to_pyobject))])

                  let extract_patches_2d ?max_patches ?random_state ~image ~patch_size () =
                     Py.Module.get_function_with_keywords ns "extract_patches_2d"
                       [||]
                       (Wrap_utils.keyword_args [("max_patches", Wrap_utils.Option.map max_patches (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("image", Some(image |> (function
| `Arr x -> Arr.to_pyobject x
| `Or x -> Wrap_utils.id x
))); ("patch_size", Some(patch_size ))])

let grid_to_graph ?n_z ?mask ?return_as ?dtype ~n_x ~n_y () =
   Py.Module.get_function_with_keywords ns "grid_to_graph"
     [||]
     (Wrap_utils.keyword_args [("n_z", n_z); ("mask", mask); ("return_as", return_as); ("dtype", dtype); ("n_x", Some(n_x |> Py.Int.of_int)); ("n_y", Some(n_y ))])

                  let img_to_graph ?mask ?return_as ?dtype ~img () =
                     Py.Module.get_function_with_keywords ns "img_to_graph"
                       [||]
                       (Wrap_utils.keyword_args [("mask", mask); ("return_as", return_as); ("dtype", dtype); ("img", Some(img |> (function
| `Arr x -> Arr.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])

                  let reconstruct_from_patches_2d ~patches ~image_size () =
                     Py.Module.get_function_with_keywords ns "reconstruct_from_patches_2d"
                       [||]
                       (Wrap_utils.keyword_args [("patches", Some(patches |> (function
| `Arr x -> Arr.to_pyobject x
| `Or x -> Wrap_utils.id x
))); ("image_size", Some(image_size ))])


end
                  let img_to_graph ?mask ?return_as ?dtype ~img () =
                     Py.Module.get_function_with_keywords ns "img_to_graph"
                       [||]
                       (Wrap_utils.keyword_args [("mask", mask); ("return_as", return_as); ("dtype", dtype); ("img", Some(img |> (function
| `Arr x -> Arr.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])

module Text = struct
let () = Wrap_utils.init ();;
let ns = Py.import "sklearn.feature_extraction.text"

let get_py name = Py.Module.get ns name
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
     |> Dict.of_pyobject
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
| `S x -> Py.String.of_string x
| `T_utf_8_by x -> Wrap_utils.id x
)); ("decode_error", Wrap_utils.Option.map decode_error (function
| `Strict -> Py.String.of_string "strict"
| `Ignore -> Py.String.of_string "ignore"
| `Replace -> Py.String.of_string "replace"
)); ("strip_accents", Wrap_utils.Option.map strip_accents (function
| `Ascii -> Py.String.of_string "ascii"
| `Unicode -> Py.String.of_string "unicode"
)); ("lowercase", Wrap_utils.Option.map lowercase Py.Bool.of_bool); ("preprocessor", preprocessor); ("tokenizer", tokenizer); ("stop_words", Wrap_utils.Option.map stop_words (function
| `English -> Py.String.of_string "english"
| `Arr x -> Arr.to_pyobject x
)); ("token_pattern", Wrap_utils.Option.map token_pattern Py.String.of_string); ("ngram_range", ngram_range); ("analyzer", Wrap_utils.Option.map analyzer (function
| `S x -> Py.String.of_string x
| `Word -> Py.String.of_string "word"
| `Char -> Py.String.of_string "char"
| `Char_wb -> Py.String.of_string "char_wb"
| `Callable x -> Wrap_utils.id x
)); ("max_df", Wrap_utils.Option.map max_df (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("min_df", Wrap_utils.Option.map min_df (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("max_features", Wrap_utils.Option.map max_features Py.Int.of_int); ("vocabulary", Wrap_utils.Option.map vocabulary (function
| `Mapping x -> Wrap_utils.id x
| `Arr x -> Arr.to_pyobject x
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
     (Wrap_utils.keyword_args [("y", y); ("raw_documents", Some(raw_documents |> Arr.to_pyobject))])

let fit_transform ?y ~raw_documents self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("raw_documents", Some(raw_documents |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let get_feature_names self =
   Py.Module.get_function_with_keywords self "get_feature_names"
     [||]
     []
     |> (Py.List.to_list_map Py.String.to_string)
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let get_stop_words self =
   Py.Module.get_function_with_keywords self "get_stop_words"
     [||]
     []

let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~raw_documents self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("raw_documents", Some(raw_documents |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let vocabulary_opt self =
  match Py.Object.get_attr_string self "vocabulary_" with
  | None -> failwith "attribute vocabulary_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let vocabulary_ self = match vocabulary_opt self with
  | None -> raise Not_found
  | Some x -> x

let fixed_vocabulary_opt self =
  match Py.Object.get_attr_string self "fixed_vocabulary_" with
  | None -> failwith "attribute fixed_vocabulary_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Bool.to_bool x)

let fixed_vocabulary_ self = match fixed_vocabulary_opt self with
  | None -> raise Not_found
  | Some x -> x

let stop_words_opt self =
  match Py.Object.get_attr_string self "stop_words_" with
  | None -> failwith "attribute stop_words_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let stop_words_ self = match stop_words_opt self with
  | None -> raise Not_found
  | Some x -> x
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
     (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Arr.to_pyobject); ("y", y)])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~raw_X self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("raw_X", Some(raw_X ))])
     |> Arr.of_pyobject
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
)); ("lowercase", Wrap_utils.Option.map lowercase Py.Bool.of_bool); ("preprocessor", preprocessor); ("tokenizer", tokenizer); ("stop_words", Wrap_utils.Option.map stop_words (function
| `English -> Py.String.of_string "english"
| `Arr x -> Arr.to_pyobject x
)); ("token_pattern", Wrap_utils.Option.map token_pattern Py.String.of_string); ("ngram_range", ngram_range); ("analyzer", Wrap_utils.Option.map analyzer (function
| `S x -> Py.String.of_string x
| `Word -> Py.String.of_string "word"
| `Char -> Py.String.of_string "char"
| `Char_wb -> Py.String.of_string "char_wb"
| `Callable x -> Wrap_utils.id x
)); ("n_features", Wrap_utils.Option.map n_features Py.Int.of_int); ("binary", Wrap_utils.Option.map binary Py.Bool.of_bool); ("norm", Wrap_utils.Option.map norm (function
| `L1 -> Py.String.of_string "l1"
| `L2 -> Py.String.of_string "l2"
| `None -> Py.none
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
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])

let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let get_stop_words self =
   Py.Module.get_function_with_keywords self "get_stop_words"
     [||]
     []

let partial_fit ?y ~x self =
   Py.Module.get_function_with_keywords self "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Arr.to_pyobject))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject
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
| `None -> Py.none
)); ("use_idf", Wrap_utils.Option.map use_idf Py.Bool.of_bool); ("smooth_idf", Wrap_utils.Option.map smooth_idf Py.Bool.of_bool); ("sublinear_tf", Wrap_utils.Option.map sublinear_tf Py.Bool.of_bool)])

let fit ?y ~x self =
   Py.Module.get_function_with_keywords self "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Csr_matrix.to_pyobject))])

let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Arr.of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ?copy ~x self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let idf_opt self =
  match Py.Object.get_attr_string self "idf_" with
  | None -> failwith "attribute idf_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let idf_ self = match idf_opt self with
  | None -> raise Not_found
  | Some x -> x
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
)); ("lowercase", Wrap_utils.Option.map lowercase Py.Bool.of_bool); ("preprocessor", preprocessor); ("tokenizer", tokenizer); ("analyzer", Wrap_utils.Option.map analyzer (function
| `S x -> Py.String.of_string x
| `Word -> Py.String.of_string "word"
| `Char -> Py.String.of_string "char"
| `Char_wb -> Py.String.of_string "char_wb"
| `Callable x -> Wrap_utils.id x
)); ("stop_words", Wrap_utils.Option.map stop_words (function
| `English -> Py.String.of_string "english"
| `Arr x -> Arr.to_pyobject x
)); ("token_pattern", Wrap_utils.Option.map token_pattern Py.String.of_string); ("ngram_range", ngram_range); ("max_df", Wrap_utils.Option.map max_df (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("min_df", Wrap_utils.Option.map min_df (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("max_features", Wrap_utils.Option.map max_features Py.Int.of_int); ("vocabulary", Wrap_utils.Option.map vocabulary (function
| `Mapping x -> Wrap_utils.id x
| `Arr x -> Arr.to_pyobject x
)); ("binary", Wrap_utils.Option.map binary Py.Bool.of_bool); ("dtype", dtype); ("norm", Wrap_utils.Option.map norm (function
| `L1 -> Py.String.of_string "l1"
| `L2 -> Py.String.of_string "l2"
| `None -> Py.none
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
     (Wrap_utils.keyword_args [("y", y); ("raw_documents", Some(raw_documents |> Arr.to_pyobject))])

let fit_transform ?y ~raw_documents self =
   Py.Module.get_function_with_keywords self "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("raw_documents", Some(raw_documents |> Arr.to_pyobject))])
     |> Arr.of_pyobject
let get_feature_names self =
   Py.Module.get_function_with_keywords self "get_feature_names"
     [||]
     []
     |> (Py.List.to_list_map Py.String.to_string)
let get_params ?deep self =
   Py.Module.get_function_with_keywords self "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let get_stop_words self =
   Py.Module.get_function_with_keywords self "get_stop_words"
     [||]
     []

let inverse_transform ~x self =
   Py.Module.get_function_with_keywords self "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Arr.to_pyobject))])

let set_params ?params self =
   Py.Module.get_function_with_keywords self "set_params"
     [||]
     (match params with None -> [] | Some x -> x)

let transform ?copy ~raw_documents self =
   Py.Module.get_function_with_keywords self "transform"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("raw_documents", Some(raw_documents |> Arr.to_pyobject))])
     |> Arr.of_pyobject

let vocabulary_opt self =
  match Py.Object.get_attr_string self "vocabulary_" with
  | None -> failwith "attribute vocabulary_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let vocabulary_ self = match vocabulary_opt self with
  | None -> raise Not_found
  | Some x -> x

let fixed_vocabulary_opt self =
  match Py.Object.get_attr_string self "fixed_vocabulary_" with
  | None -> failwith "attribute fixed_vocabulary_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Bool.to_bool x)

let fixed_vocabulary_ self = match fixed_vocabulary_opt self with
  | None -> raise Not_found
  | Some x -> x

let idf_opt self =
  match Py.Object.get_attr_string self "idf_" with
  | None -> failwith "attribute idf_ not found"
  | Some x -> if Py.is_none x then None else Some (Arr.of_pyobject x)

let idf_ self = match idf_opt self with
  | None -> raise Not_found
  | Some x -> x

let stop_words_opt self =
  match Py.Object.get_attr_string self "stop_words_" with
  | None -> failwith "attribute stop_words_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let stop_words_ self = match stop_words_opt self with
  | None -> raise Not_found
  | Some x -> x
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
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Arr.to_pyobject); ("X", Some(x |> Arr.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> Arr.of_pyobject
let to_string self = Py.Object.to_string self
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let check_array ?accept_sparse ?accept_large_sparse ?dtype ?order ?copy ?force_all_finite ?ensure_2d ?allow_nd ?ensure_min_samples ?ensure_min_features ?warn_on_dtype ?estimator ~array () =
                     Py.Module.get_function_with_keywords ns "check_array"
                       [||]
                       (Wrap_utils.keyword_args [("accept_sparse", Wrap_utils.Option.map accept_sparse (function
| `S x -> Py.String.of_string x
| `Bool x -> Py.Bool.of_bool x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
)); ("accept_large_sparse", Wrap_utils.Option.map accept_large_sparse Py.Bool.of_bool); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Wrap_utils.id x
| `TypeList x -> Wrap_utils.id x
| `None -> Py.none
)); ("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("force_all_finite", Wrap_utils.Option.map force_all_finite (function
| `Bool x -> Py.Bool.of_bool x
| `Allow_nan -> Py.String.of_string "allow-nan"
)); ("ensure_2d", Wrap_utils.Option.map ensure_2d Py.Bool.of_bool); ("allow_nd", Wrap_utils.Option.map allow_nd Py.Bool.of_bool); ("ensure_min_samples", Wrap_utils.Option.map ensure_min_samples Py.Int.of_int); ("ensure_min_features", Wrap_utils.Option.map ensure_min_features Py.Int.of_int); ("warn_on_dtype", Wrap_utils.Option.map warn_on_dtype Py.Bool.of_bool); ("estimator", Wrap_utils.Option.map estimator (function
| `S x -> Py.String.of_string x
| `Estimator x -> Wrap_utils.id x
)); ("array", Some(array ))])

                  let check_is_fitted ?attributes ?msg ?all_or_any ~estimator () =
                     Py.Module.get_function_with_keywords ns "check_is_fitted"
                       [||]
                       (Wrap_utils.keyword_args [("attributes", Wrap_utils.Option.map attributes (function
| `S x -> Py.String.of_string x
| `Arr x -> Arr.to_pyobject x
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
| `T_l2_by x -> Wrap_utils.id x
)); ("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
| `T_1_by x -> Wrap_utils.id x
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("return_norm", Wrap_utils.Option.map return_norm Py.Bool.of_bool); ("X", Some(x |> Arr.to_pyobject))])
                       |> (fun x -> ((Arr.of_pyobject (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
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
