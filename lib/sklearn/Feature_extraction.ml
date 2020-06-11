let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.feature_extraction"

let get_py name = Py.Module.get __wrap_namespace name
module DictVectorizer = struct
type tag = [`DictVectorizer]
type t = [`BaseEstimator | `DictVectorizer | `Object | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let create ?dtype ?separator ?sparse ?sort () =
   Py.Module.get_function_with_keywords __wrap_namespace "DictVectorizer"
     [||]
     (Wrap_utils.keyword_args [("dtype", dtype); ("separator", Wrap_utils.Option.map separator Py.String.of_string); ("sparse", Wrap_utils.Option.map sparse Py.Bool.of_bool); ("sort", Wrap_utils.Option.map sort Py.Bool.of_bool)])
     |> of_pyobject
let fit ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x ))])
     |> of_pyobject
let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_feature_names self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_feature_names"
     [||]
     []

let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let inverse_transform ?dict_type ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("dict_type", dict_type); ("X", Some(x |> Np.Obj.to_pyobject))])

let restrict ?indices ~support self =
   Py.Module.get_function_with_keywords (to_pyobject self) "restrict"
     [||]
     (Wrap_utils.keyword_args [("indices", Wrap_utils.Option.map indices Py.Bool.of_bool); ("support", Some(support |> Np.Obj.to_pyobject))])

let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

let vocabulary_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "vocabulary_" with
  | None -> failwith "attribute vocabulary_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let vocabulary_ self = match vocabulary_opt self with
  | None -> raise Not_found
  | Some x -> x

let feature_names_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "feature_names_" with
  | None -> failwith "attribute feature_names_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let feature_names_ self = match feature_names_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module FeatureHasher = struct
type tag = [`FeatureHasher]
type t = [`BaseEstimator | `FeatureHasher | `Object | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
let create ?n_features ?input_type ?dtype ?alternate_sign () =
   Py.Module.get_function_with_keywords __wrap_namespace "FeatureHasher"
     [||]
     (Wrap_utils.keyword_args [("n_features", Wrap_utils.Option.map n_features Py.Int.of_int); ("input_type", Wrap_utils.Option.map input_type Py.String.of_string); ("dtype", dtype); ("alternate_sign", Wrap_utils.Option.map alternate_sign Py.Bool.of_bool)])
     |> of_pyobject
let fit ?x ?y self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("X", Wrap_utils.Option.map x Np.Obj.to_pyobject); ("y", y)])
     |> of_pyobject
let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let transform ~raw_X self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("raw_X", Some(raw_X ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Image = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.feature_extraction.image"

let get_py name = Py.Module.get __wrap_namespace name
module PatchExtractor = struct
type tag = [`PatchExtractor]
type t = [`BaseEstimator | `Object | `PatchExtractor] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
                  let create ?patch_size ?max_patches ?random_state () =
                     Py.Module.get_function_with_keywords __wrap_namespace "PatchExtractor"
                       [||]
                       (Wrap_utils.keyword_args [("patch_size", patch_size); ("max_patches", Wrap_utils.Option.map max_patches (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int)])
                       |> of_pyobject
let fit ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> of_pyobject
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Product = struct
type tag = [`Product]
type t = [`Object | `Product] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?repeat iterables =
   Py.Module.get_function_with_keywords __wrap_namespace "product"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id iterables)])
     (Wrap_utils.keyword_args [("repeat", repeat)])
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
let as_strided ?shape ?strides ?subok ?writeable ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "as_strided"
     [||]
     (Wrap_utils.keyword_args [("shape", Wrap_utils.Option.map shape (fun ml -> Py.List.of_list_map Py.Int.of_int ml)); ("strides", strides); ("subok", Wrap_utils.Option.map subok Py.Bool.of_bool); ("writeable", Wrap_utils.Option.map writeable Py.Bool.of_bool); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
                  let check_array ?accept_sparse ?accept_large_sparse ?dtype ?order ?copy ?force_all_finite ?ensure_2d ?allow_nd ?ensure_min_samples ?ensure_min_features ?warn_on_dtype ?estimator ~array () =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_array"
                       [||]
                       (Wrap_utils.keyword_args [("accept_sparse", Wrap_utils.Option.map accept_sparse (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `S x -> Py.String.of_string x
| `Bool x -> Py.Bool.of_bool x
)); ("accept_large_sparse", Wrap_utils.Option.map accept_large_sparse Py.Bool.of_bool); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
| `Dtypes x -> (fun ml -> Py.List.of_list_map Np.Dtype.to_pyobject ml) x
| `None -> Py.none
)); ("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("force_all_finite", Wrap_utils.Option.map force_all_finite (function
| `Allow_nan -> Py.String.of_string "allow-nan"
| `Bool x -> Py.Bool.of_bool x
)); ("ensure_2d", Wrap_utils.Option.map ensure_2d Py.Bool.of_bool); ("allow_nd", Wrap_utils.Option.map allow_nd Py.Bool.of_bool); ("ensure_min_samples", Wrap_utils.Option.map ensure_min_samples Py.Int.of_int); ("ensure_min_features", Wrap_utils.Option.map ensure_min_features Py.Int.of_int); ("warn_on_dtype", Wrap_utils.Option.map warn_on_dtype Py.Bool.of_bool); ("estimator", Wrap_utils.Option.map estimator Np.Obj.to_pyobject); ("array", Some(array ))])

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

                  let extract_patches ?patch_shape ?extraction_step ~arr () =
                     Py.Module.get_function_with_keywords __wrap_namespace "extract_patches"
                       [||]
                       (Wrap_utils.keyword_args [("patch_shape", Wrap_utils.Option.map patch_shape (function
| `Tuple x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("extraction_step", Wrap_utils.Option.map extraction_step (function
| `Tuple x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)); ("arr", Some(arr |> Np.Obj.to_pyobject))])

                  let extract_patches_2d ?max_patches ?random_state ~image ~patch_size () =
                     Py.Module.get_function_with_keywords __wrap_namespace "extract_patches_2d"
                       [||]
                       (Wrap_utils.keyword_args [("max_patches", Wrap_utils.Option.map max_patches (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("random_state", Wrap_utils.Option.map random_state Py.Int.of_int); ("image", Some(image )); ("patch_size", Some(patch_size ))])

let grid_to_graph ?n_z ?mask ?return_as ?dtype ~n_x ~n_y () =
   Py.Module.get_function_with_keywords __wrap_namespace "grid_to_graph"
     [||]
     (Wrap_utils.keyword_args [("n_z", Wrap_utils.Option.map n_z Py.Int.of_int); ("mask", mask); ("return_as", return_as); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("n_x", Some(n_x |> Py.Int.of_int)); ("n_y", Some(n_y |> Py.Int.of_int))])

                  let img_to_graph ?mask ?return_as ?dtype ~img () =
                     Py.Module.get_function_with_keywords __wrap_namespace "img_to_graph"
                       [||]
                       (Wrap_utils.keyword_args [("mask", mask); ("return_as", return_as); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("img", Some(img |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])

let reconstruct_from_patches_2d ~patches ~image_size () =
   Py.Module.get_function_with_keywords __wrap_namespace "reconstruct_from_patches_2d"
     [||]
     (Wrap_utils.keyword_args [("patches", Some(patches )); ("image_size", Some(image_size ))])


end
module Text = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "sklearn.feature_extraction.text"

let get_py name = Py.Module.get __wrap_namespace name
module CountVectorizer = struct
type tag = [`CountVectorizer]
type t = [`BaseEstimator | `CountVectorizer | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
                  let create ?input ?encoding ?decode_error ?strip_accents ?lowercase ?preprocessor ?tokenizer ?stop_words ?token_pattern ?ngram_range ?analyzer ?max_df ?min_df ?max_features ?vocabulary ?binary ?dtype () =
                     Py.Module.get_function_with_keywords __wrap_namespace "CountVectorizer"
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
| `Unicode -> Py.String.of_string "unicode"
| `Ascii -> Py.String.of_string "ascii"
)); ("lowercase", Wrap_utils.Option.map lowercase Py.Bool.of_bool); ("preprocessor", preprocessor); ("tokenizer", tokenizer); ("stop_words", Wrap_utils.Option.map stop_words (function
| `Arr x -> Np.Obj.to_pyobject x
| `English -> Py.String.of_string "english"
)); ("token_pattern", Wrap_utils.Option.map token_pattern Py.String.of_string); ("ngram_range", ngram_range); ("analyzer", Wrap_utils.Option.map analyzer (function
| `Callable x -> Wrap_utils.id x
| `Char -> Py.String.of_string "char"
| `S x -> Py.String.of_string x
| `PyObject x -> Wrap_utils.id x
)); ("max_df", Wrap_utils.Option.map max_df (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("min_df", Wrap_utils.Option.map min_df (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("max_features", Wrap_utils.Option.map max_features Py.Int.of_int); ("vocabulary", Wrap_utils.Option.map vocabulary (function
| `Mapping x -> Wrap_utils.id x
| `Arr x -> Np.Obj.to_pyobject x
)); ("binary", Wrap_utils.Option.map binary Py.Bool.of_bool); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject)])
                       |> of_pyobject
let build_analyzer self =
   Py.Module.get_function_with_keywords (to_pyobject self) "build_analyzer"
     [||]
     []

let build_preprocessor self =
   Py.Module.get_function_with_keywords (to_pyobject self) "build_preprocessor"
     [||]
     []

let build_tokenizer self =
   Py.Module.get_function_with_keywords (to_pyobject self) "build_tokenizer"
     [||]
     []

let decode ~doc self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decode"
     [||]
     (Wrap_utils.keyword_args [("doc", Some(doc |> Py.String.of_string))])
     |> Py.String.to_string
let fit ?y ~raw_documents self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("raw_documents", Some(raw_documents |> Np.Obj.to_pyobject))])
     |> of_pyobject
let fit_transform ?y ~raw_documents self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("raw_documents", Some(raw_documents |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_feature_names self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_feature_names"
     [||]
     []
     |> (fun py -> Py.List.to_list_map (Py.String.to_string) py)
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let get_stop_words self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_stop_words"
     [||]
     []
     |> (fun py -> if Py.is_none py then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) py))
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])

let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let transform ~raw_documents self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("raw_documents", Some(raw_documents |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

let vocabulary_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "vocabulary_" with
  | None -> failwith "attribute vocabulary_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let vocabulary_ self = match vocabulary_opt self with
  | None -> raise Not_found
  | Some x -> x

let fixed_vocabulary_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "fixed_vocabulary_" with
  | None -> failwith "attribute fixed_vocabulary_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Bool.to_bool x)

let fixed_vocabulary_ self = match fixed_vocabulary_opt self with
  | None -> raise Not_found
  | Some x -> x

let stop_words_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "stop_words_" with
  | None -> failwith "attribute stop_words_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let stop_words_ self = match stop_words_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module HashingVectorizer = struct
type tag = [`HashingVectorizer]
type t = [`BaseEstimator | `HashingVectorizer | `Object | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
                  let create ?input ?encoding ?decode_error ?strip_accents ?lowercase ?preprocessor ?tokenizer ?stop_words ?token_pattern ?ngram_range ?analyzer ?n_features ?binary ?norm ?alternate_sign ?dtype () =
                     Py.Module.get_function_with_keywords __wrap_namespace "HashingVectorizer"
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
| `Unicode -> Py.String.of_string "unicode"
| `Ascii -> Py.String.of_string "ascii"
)); ("lowercase", Wrap_utils.Option.map lowercase Py.Bool.of_bool); ("preprocessor", preprocessor); ("tokenizer", tokenizer); ("stop_words", Wrap_utils.Option.map stop_words (function
| `Arr x -> Np.Obj.to_pyobject x
| `English -> Py.String.of_string "english"
)); ("token_pattern", Wrap_utils.Option.map token_pattern Py.String.of_string); ("ngram_range", ngram_range); ("analyzer", Wrap_utils.Option.map analyzer (function
| `Callable x -> Wrap_utils.id x
| `Char -> Py.String.of_string "char"
| `S x -> Py.String.of_string x
| `PyObject x -> Wrap_utils.id x
)); ("n_features", Wrap_utils.Option.map n_features Py.Int.of_int); ("binary", Wrap_utils.Option.map binary Py.Bool.of_bool); ("norm", Wrap_utils.Option.map norm (function
| `L2 -> Py.String.of_string "l2"
| `L1 -> Py.String.of_string "l1"
| `None -> Py.none
)); ("alternate_sign", Wrap_utils.Option.map alternate_sign Py.Bool.of_bool); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject)])
                       |> of_pyobject
let build_analyzer self =
   Py.Module.get_function_with_keywords (to_pyobject self) "build_analyzer"
     [||]
     []

let build_preprocessor self =
   Py.Module.get_function_with_keywords (to_pyobject self) "build_preprocessor"
     [||]
     []

let build_tokenizer self =
   Py.Module.get_function_with_keywords (to_pyobject self) "build_tokenizer"
     [||]
     []

let decode ~doc self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decode"
     [||]
     (Wrap_utils.keyword_args [("doc", Some(doc |> Py.String.of_string))])
     |> Py.String.to_string
let fit ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> of_pyobject
let fit_transform ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let get_stop_words self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_stop_words"
     [||]
     []
     |> (fun py -> if Py.is_none py then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) py))
let partial_fit ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "partial_fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Mapping = struct
type tag = [`Mapping]
type t = [`Mapping | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let get_item ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key ))])

let iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []
     |> (fun py -> Py.Iter.to_seq py |> Seq.map Dict.of_pyobject)
let get ?default ~key self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get"
     [||]
     (Wrap_utils.keyword_args [("default", default); ("key", Some(key ))])

let items self =
   Py.Module.get_function_with_keywords (to_pyobject self) "items"
     [||]
     []

let keys self =
   Py.Module.get_function_with_keywords (to_pyobject self) "keys"
     [||]
     []

let values self =
   Py.Module.get_function_with_keywords (to_pyobject self) "values"
     [||]
     []

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module TfidfTransformer = struct
type tag = [`TfidfTransformer]
type t = [`BaseEstimator | `Object | `TfidfTransformer | `TransformerMixin] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_transformer x = (x :> [`TransformerMixin] Obj.t)
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
                  let create ?norm ?use_idf ?smooth_idf ?sublinear_tf () =
                     Py.Module.get_function_with_keywords __wrap_namespace "TfidfTransformer"
                       [||]
                       (Wrap_utils.keyword_args [("norm", Wrap_utils.Option.map norm (function
| `L2 -> Py.String.of_string "l2"
| `L1 -> Py.String.of_string "l1"
| `None -> Py.none
)); ("use_idf", Wrap_utils.Option.map use_idf Py.Bool.of_bool); ("smooth_idf", Wrap_utils.Option.map smooth_idf Py.Bool.of_bool); ("sublinear_tf", Wrap_utils.Option.map sublinear_tf Py.Bool.of_bool)])
                       |> of_pyobject
let fit ?y ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> of_pyobject
let fit_transform ?y ?fit_params ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_transform"
     [||]
     (List.rev_append (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("X", Some(x |> Np.Obj.to_pyobject))]) (match fit_params with None -> [] | Some x -> x))
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let transform ?copy ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

let idf_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "idf_" with
  | None -> failwith "attribute idf_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let idf_ self = match idf_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module TfidfVectorizer = struct
type tag = [`TfidfVectorizer]
type t = [`BaseEstimator | `Object | `TfidfVectorizer] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let as_estimator x = (x :> [`BaseEstimator] Obj.t)
                  let create ?input ?encoding ?decode_error ?strip_accents ?lowercase ?preprocessor ?tokenizer ?analyzer ?stop_words ?token_pattern ?ngram_range ?max_df ?min_df ?max_features ?vocabulary ?binary ?dtype ?norm ?use_idf ?smooth_idf ?sublinear_tf () =
                     Py.Module.get_function_with_keywords __wrap_namespace "TfidfVectorizer"
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
| `Unicode -> Py.String.of_string "unicode"
| `Ascii -> Py.String.of_string "ascii"
)); ("lowercase", Wrap_utils.Option.map lowercase Py.Bool.of_bool); ("preprocessor", preprocessor); ("tokenizer", tokenizer); ("analyzer", Wrap_utils.Option.map analyzer (function
| `Callable x -> Wrap_utils.id x
| `Char -> Py.String.of_string "char"
| `S x -> Py.String.of_string x
| `PyObject x -> Wrap_utils.id x
)); ("stop_words", Wrap_utils.Option.map stop_words (function
| `Arr x -> Np.Obj.to_pyobject x
| `English -> Py.String.of_string "english"
)); ("token_pattern", Wrap_utils.Option.map token_pattern Py.String.of_string); ("ngram_range", ngram_range); ("max_df", Wrap_utils.Option.map max_df (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("min_df", Wrap_utils.Option.map min_df (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)); ("max_features", Wrap_utils.Option.map max_features Py.Int.of_int); ("vocabulary", Wrap_utils.Option.map vocabulary (function
| `Mapping x -> Wrap_utils.id x
| `Arr x -> Np.Obj.to_pyobject x
)); ("binary", Wrap_utils.Option.map binary Py.Bool.of_bool); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("norm", Wrap_utils.Option.map norm (function
| `L2 -> Py.String.of_string "l2"
| `L1 -> Py.String.of_string "l1"
| `None -> Py.none
)); ("use_idf", Wrap_utils.Option.map use_idf Py.Bool.of_bool); ("smooth_idf", Wrap_utils.Option.map smooth_idf Py.Bool.of_bool); ("sublinear_tf", Wrap_utils.Option.map sublinear_tf Py.Bool.of_bool)])
                       |> of_pyobject
let build_analyzer self =
   Py.Module.get_function_with_keywords (to_pyobject self) "build_analyzer"
     [||]
     []

let build_preprocessor self =
   Py.Module.get_function_with_keywords (to_pyobject self) "build_preprocessor"
     [||]
     []

let build_tokenizer self =
   Py.Module.get_function_with_keywords (to_pyobject self) "build_tokenizer"
     [||]
     []

let decode ~doc self =
   Py.Module.get_function_with_keywords (to_pyobject self) "decode"
     [||]
     (Wrap_utils.keyword_args [("doc", Some(doc |> Py.String.of_string))])
     |> Py.String.to_string
let fit ?y ~raw_documents self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("raw_documents", Some(raw_documents |> Np.Obj.to_pyobject))])
     |> of_pyobject
let fit_transform ?y ~raw_documents self =
   Py.Module.get_function_with_keywords (to_pyobject self) "fit_transform"
     [||]
     (Wrap_utils.keyword_args [("y", y); ("raw_documents", Some(raw_documents |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))
let get_feature_names self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_feature_names"
     [||]
     []
     |> (fun py -> Py.List.to_list_map (Py.String.to_string) py)
let get_params ?deep self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_params"
     [||]
     (Wrap_utils.keyword_args [("deep", Wrap_utils.Option.map deep Py.Bool.of_bool)])
     |> Dict.of_pyobject
let get_stop_words self =
   Py.Module.get_function_with_keywords (to_pyobject self) "get_stop_words"
     [||]
     []
     |> (fun py -> if Py.is_none py then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) py))
let inverse_transform ~x self =
   Py.Module.get_function_with_keywords (to_pyobject self) "inverse_transform"
     [||]
     (Wrap_utils.keyword_args [("X", Some(x |> Np.Obj.to_pyobject))])

let set_params ?params self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_params"
     [||]
     (match params with None -> [] | Some x -> x)
     |> of_pyobject
let transform ?copy ~raw_documents self =
   Py.Module.get_function_with_keywords (to_pyobject self) "transform"
     [||]
     (Wrap_utils.keyword_args [("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("raw_documents", Some(raw_documents |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t))

let vocabulary_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "vocabulary_" with
  | None -> failwith "attribute vocabulary_ not found"
  | Some x -> if Py.is_none x then None else Some (Dict.of_pyobject x)

let vocabulary_ self = match vocabulary_opt self with
  | None -> raise Not_found
  | Some x -> x

let fixed_vocabulary_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "fixed_vocabulary_" with
  | None -> failwith "attribute fixed_vocabulary_ not found"
  | Some x -> if Py.is_none x then None else Some (Py.Bool.to_bool x)

let fixed_vocabulary_ self = match fixed_vocabulary_opt self with
  | None -> raise Not_found
  | Some x -> x

let idf_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "idf_" with
  | None -> failwith "attribute idf_ not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) x)

let idf_ self = match idf_opt self with
  | None -> raise Not_found
  | Some x -> x

let stop_words_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "stop_words_" with
  | None -> failwith "attribute stop_words_ not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let stop_words_ self = match stop_words_opt self with
  | None -> raise Not_found
  | Some x -> x
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
module Partial = struct
type tag = [`Partial]
type t = [`Object | `Partial] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?keywords ~func args =
   Py.Module.get_function_with_keywords __wrap_namespace "partial"
     (Array.of_list @@ List.concat [(List.map Wrap_utils.id args)])
     (List.rev_append (Wrap_utils.keyword_args [("func", Some(func ))]) (match keywords with None -> [] | Some x -> x))
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
                  let check_array ?accept_sparse ?accept_large_sparse ?dtype ?order ?copy ?force_all_finite ?ensure_2d ?allow_nd ?ensure_min_samples ?ensure_min_features ?warn_on_dtype ?estimator ~array () =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_array"
                       [||]
                       (Wrap_utils.keyword_args [("accept_sparse", Wrap_utils.Option.map accept_sparse (function
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
| `S x -> Py.String.of_string x
| `Bool x -> Py.Bool.of_bool x
)); ("accept_large_sparse", Wrap_utils.Option.map accept_large_sparse Py.Bool.of_bool); ("dtype", Wrap_utils.Option.map dtype (function
| `S x -> Py.String.of_string x
| `Dtype x -> Np.Dtype.to_pyobject x
| `Dtypes x -> (fun ml -> Py.List.of_list_map Np.Dtype.to_pyobject ml) x
| `None -> Py.none
)); ("order", Wrap_utils.Option.map order (function
| `F -> Py.String.of_string "F"
| `C -> Py.String.of_string "C"
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("force_all_finite", Wrap_utils.Option.map force_all_finite (function
| `Allow_nan -> Py.String.of_string "allow-nan"
| `Bool x -> Py.Bool.of_bool x
)); ("ensure_2d", Wrap_utils.Option.map ensure_2d Py.Bool.of_bool); ("allow_nd", Wrap_utils.Option.map allow_nd Py.Bool.of_bool); ("ensure_min_samples", Wrap_utils.Option.map ensure_min_samples Py.Int.of_int); ("ensure_min_features", Wrap_utils.Option.map ensure_min_features Py.Int.of_int); ("warn_on_dtype", Wrap_utils.Option.map warn_on_dtype Py.Bool.of_bool); ("estimator", Wrap_utils.Option.map estimator Np.Obj.to_pyobject); ("array", Some(array ))])

                  let check_is_fitted ?attributes ?msg ?all_or_any ~estimator () =
                     Py.Module.get_function_with_keywords __wrap_namespace "check_is_fitted"
                       [||]
                       (Wrap_utils.keyword_args [("attributes", Wrap_utils.Option.map attributes (function
| `S x -> Py.String.of_string x
| `Arr x -> Np.Obj.to_pyobject x
| `StringList x -> (Py.List.of_list_map Py.String.of_string) x
)); ("msg", Wrap_utils.Option.map msg Py.String.of_string); ("all_or_any", Wrap_utils.Option.map all_or_any (function
| `Callable x -> Wrap_utils.id x
| `PyObject x -> Wrap_utils.id x
)); ("estimator", Some(estimator |> Np.Obj.to_pyobject))])

                  let normalize ?norm ?axis ?copy ?return_norm ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "normalize"
                       [||]
                       (Wrap_utils.keyword_args [("norm", Wrap_utils.Option.map norm (function
| `L1 -> Py.String.of_string "l1"
| `L2 -> Py.String.of_string "l2"
| `Max -> Py.String.of_string "max"
)); ("axis", Wrap_utils.Option.map axis (function
| `Zero -> Py.Int.of_int 0
| `One -> Py.Int.of_int 1
)); ("copy", Wrap_utils.Option.map copy Py.Bool.of_bool); ("return_norm", Wrap_utils.Option.map return_norm Py.Bool.of_bool); ("X", Some(x |> Np.Obj.to_pyobject))])
                       |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [>`ArrayLike] Np.Obj.t)) (Py.Tuple.get x 1))))
let strip_accents_ascii s =
   Py.Module.get_function_with_keywords __wrap_namespace "strip_accents_ascii"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s |> Py.String.of_string))])

let strip_accents_unicode s =
   Py.Module.get_function_with_keywords __wrap_namespace "strip_accents_unicode"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s |> Py.String.of_string))])

let strip_tags s =
   Py.Module.get_function_with_keywords __wrap_namespace "strip_tags"
     [||]
     (Wrap_utils.keyword_args [("s", Some(s |> Py.String.of_string))])


end
let grid_to_graph ?n_z ?mask ?return_as ?dtype ~n_x ~n_y () =
   Py.Module.get_function_with_keywords __wrap_namespace "grid_to_graph"
     [||]
     (Wrap_utils.keyword_args [("n_z", Wrap_utils.Option.map n_z Py.Int.of_int); ("mask", mask); ("return_as", return_as); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("n_x", Some(n_x |> Py.Int.of_int)); ("n_y", Some(n_y |> Py.Int.of_int))])

                  let img_to_graph ?mask ?return_as ?dtype ~img () =
                     Py.Module.get_function_with_keywords __wrap_namespace "img_to_graph"
                       [||]
                       (Wrap_utils.keyword_args [("mask", mask); ("return_as", return_as); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("img", Some(img |> (function
| `Arr x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)))])

