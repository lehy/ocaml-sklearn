let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.odr"

let get_py name = Py.Module.get __wrap_namespace name
module Data = struct
type tag = [`Data]
type t = [`Data | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?y ?we ?wd ?fix ?meta ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "Data"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("we", Wrap_utils.Option.map we Np.Obj.to_pyobject); ("wd", Wrap_utils.Option.map wd Np.Obj.to_pyobject); ("fix", fix); ("meta", meta); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> of_pyobject
let set_meta ?kwds self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_meta"
     [||]
     (match kwds with None -> [] | Some x -> x)

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Model = struct
type tag = [`Model]
type t = [`Model | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?fjacb ?fjacd ?extra_args ?estimate ?implicit ?meta ~fcn () =
   Py.Module.get_function_with_keywords __wrap_namespace "Model"
     [||]
     (Wrap_utils.keyword_args [("fjacb", fjacb); ("fjacd", fjacd); ("extra_args", extra_args); ("estimate", estimate); ("implicit", Wrap_utils.Option.map implicit Py.Bool.of_bool); ("meta", meta); ("fcn", Some(fcn ))])
     |> of_pyobject
let set_meta ?kwds self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_meta"
     [||]
     (match kwds with None -> [] | Some x -> x)

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module ODR = struct
type tag = [`ODR]
type t = [`ODR | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?beta0 ?delta0 ?ifixb ?ifixx ?job ?iprint ?errfile ?rptfile ?ndigit ?taufac ?sstol ?partol ?maxit ?stpb ?stpd ?sclb ?scld ?work ?iwork ~data ~model () =
   Py.Module.get_function_with_keywords __wrap_namespace "ODR"
     [||]
     (Wrap_utils.keyword_args [("beta0", beta0); ("delta0", delta0); ("ifixb", ifixb); ("ifixx", ifixx); ("job", job); ("iprint", iprint); ("errfile", errfile); ("rptfile", rptfile); ("ndigit", ndigit); ("taufac", taufac); ("sstol", sstol); ("partol", partol); ("maxit", maxit); ("stpb", stpb); ("stpd", stpd); ("sclb", sclb); ("scld", scld); ("work", work); ("iwork", iwork); ("data", Some(data )); ("model", Some(model ))])
     |> of_pyobject
let restart ?iter self =
   Py.Module.get_function_with_keywords (to_pyobject self) "restart"
     [||]
     (Wrap_utils.keyword_args [("iter", Wrap_utils.Option.map iter Py.Int.of_int)])

let run self =
   Py.Module.get_function_with_keywords (to_pyobject self) "run"
     [||]
     []

let set_iprint ?init ?so_init ?iter ?so_iter ?iter_step ?final ?so_final self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_iprint"
     [||]
     (Wrap_utils.keyword_args [("init", init); ("so_init", so_init); ("iter", iter); ("so_iter", so_iter); ("iter_step", iter_step); ("final", final); ("so_final", so_final)])

                  let set_job ?fit_type ?deriv ?var_calc ?del_init ?restart self =
                     Py.Module.get_function_with_keywords (to_pyobject self) "set_job"
                       [||]
                       (Wrap_utils.keyword_args [("fit_type", Wrap_utils.Option.map fit_type (function
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("deriv", Wrap_utils.Option.map deriv (function
| `One -> Py.Int.of_int 1
| `Two -> Py.Int.of_int 2
| `PyObject x -> Wrap_utils.id x
)); ("var_calc", Wrap_utils.Option.map var_calc (function
| `One -> Py.Int.of_int 1
| `PyObject x -> Wrap_utils.id x
)); ("del_init", del_init); ("restart", restart)])


let data_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "data" with
  | None -> failwith "attribute data not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let data self = match data_opt self with
  | None -> raise Not_found
  | Some x -> x

let model_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "model" with
  | None -> failwith "attribute model not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let model self = match model_opt self with
  | None -> raise Not_found
  | Some x -> x

let output_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "output" with
  | None -> failwith "attribute output not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let output self = match output_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module OdrError = struct
type tag = [`OdrError]
type t = [`BaseException | `Object | `OdrError] Obj.t
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
module OdrStop = struct
type tag = [`OdrStop]
type t = [`BaseException | `Object | `OdrStop] Obj.t
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
module OdrWarning = struct
type tag = [`OdrWarning]
type t = [`BaseException | `Object | `OdrWarning] Obj.t
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
module Output = struct
type tag = [`Output]
type t = [`Object | `Output] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create output =
   Py.Module.get_function_with_keywords __wrap_namespace "Output"
     [||]
     (Wrap_utils.keyword_args [("output", Some(output ))])
     |> of_pyobject
let pprint self =
   Py.Module.get_function_with_keywords (to_pyobject self) "pprint"
     [||]
     []


let beta_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "beta" with
  | None -> failwith "attribute beta not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let beta self = match beta_opt self with
  | None -> raise Not_found
  | Some x -> x

let sd_beta_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "sd_beta" with
  | None -> failwith "attribute sd_beta not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let sd_beta self = match sd_beta_opt self with
  | None -> raise Not_found
  | Some x -> x

let cov_beta_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "cov_beta" with
  | None -> failwith "attribute cov_beta not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let cov_beta self = match cov_beta_opt self with
  | None -> raise Not_found
  | Some x -> x

let delta_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "delta" with
  | None -> failwith "attribute delta not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let delta self = match delta_opt self with
  | None -> raise Not_found
  | Some x -> x

let eps_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "eps" with
  | None -> failwith "attribute eps not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let eps self = match eps_opt self with
  | None -> raise Not_found
  | Some x -> x

let xplus_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "xplus" with
  | None -> failwith "attribute xplus not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let xplus self = match xplus_opt self with
  | None -> raise Not_found
  | Some x -> x

let y_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "y" with
  | None -> failwith "attribute y not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let y self = match y_opt self with
  | None -> raise Not_found
  | Some x -> x

let res_var_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "res_var" with
  | None -> failwith "attribute res_var not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let res_var self = match res_var_opt self with
  | None -> raise Not_found
  | Some x -> x

let sum_square_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "sum_square" with
  | None -> failwith "attribute sum_square not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let sum_square self = match sum_square_opt self with
  | None -> raise Not_found
  | Some x -> x

let sum_square_delta_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "sum_square_delta" with
  | None -> failwith "attribute sum_square_delta not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let sum_square_delta self = match sum_square_delta_opt self with
  | None -> raise Not_found
  | Some x -> x

let sum_square_eps_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "sum_square_eps" with
  | None -> failwith "attribute sum_square_eps not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let sum_square_eps self = match sum_square_eps_opt self with
  | None -> raise Not_found
  | Some x -> x

let inv_condnum_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "inv_condnum" with
  | None -> failwith "attribute inv_condnum not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let inv_condnum self = match inv_condnum_opt self with
  | None -> raise Not_found
  | Some x -> x

let rel_error_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "rel_error" with
  | None -> failwith "attribute rel_error not found"
  | Some x -> if Py.is_none x then None else Some (Py.Float.to_float x)

let rel_error self = match rel_error_opt self with
  | None -> raise Not_found
  | Some x -> x

let work_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "work" with
  | None -> failwith "attribute work not found"
  | Some x -> if Py.is_none x then None else Some ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) x)

let work self = match work_opt self with
  | None -> raise Not_found
  | Some x -> x

let work_ind_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "work_ind" with
  | None -> failwith "attribute work_ind not found"
  | Some x -> if Py.is_none x then None else Some (Wrap_utils.id x)

let work_ind self = match work_ind_opt self with
  | None -> raise Not_found
  | Some x -> x

let info_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "info" with
  | None -> failwith "attribute info not found"
  | Some x -> if Py.is_none x then None else Some (Py.Int.to_int x)

let info self = match info_opt self with
  | None -> raise Not_found
  | Some x -> x

let stopreason_opt self =
  match Py.Object.get_attr_string (to_pyobject self) "stopreason" with
  | None -> failwith "attribute stopreason not found"
  | Some x -> if Py.is_none x then None else Some ((Py.List.to_list_map Py.String.to_string) x)

let stopreason self = match stopreason_opt self with
  | None -> raise Not_found
  | Some x -> x
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module RealData = struct
type tag = [`RealData]
type t = [`Object | `RealData] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?y ?sx ?sy ?covx ?covy ?fix ?meta ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "RealData"
     [||]
     (Wrap_utils.keyword_args [("y", Wrap_utils.Option.map y Np.Obj.to_pyobject); ("sx", Wrap_utils.Option.map sx Np.Obj.to_pyobject); ("sy", Wrap_utils.Option.map sy Np.Obj.to_pyobject); ("covx", Wrap_utils.Option.map covx Np.Obj.to_pyobject); ("covy", Wrap_utils.Option.map covy Np.Obj.to_pyobject); ("fix", Wrap_utils.Option.map fix Np.Obj.to_pyobject); ("meta", meta); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> of_pyobject
let set_meta ?kwds self =
   Py.Module.get_function_with_keywords (to_pyobject self) "set_meta"
     [||]
     (match kwds with None -> [] | Some x -> x)

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Add_newdocs = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.odr.add_newdocs"

let get_py name = Py.Module.get __wrap_namespace name
                  let add_newdoc ?warn_on_python ~place ~obj ~doc () =
                     Py.Module.get_function_with_keywords __wrap_namespace "add_newdoc"
                       [||]
                       (Wrap_utils.keyword_args [("warn_on_python", Wrap_utils.Option.map warn_on_python Py.Bool.of_bool); ("place", Some(place |> Py.String.of_string)); ("obj", Some(obj |> Py.String.of_string)); ("doc", Some(doc |> (function
| `S x -> Py.String.of_string x
| `PyObject x -> Wrap_utils.id x
)))])


end
module Models = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.odr.models"

let get_py name = Py.Module.get __wrap_namespace name
                  let polynomial order =
                     Py.Module.get_function_with_keywords __wrap_namespace "polynomial"
                       [||]
                       (Wrap_utils.keyword_args [("order", Some(order |> (function
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)))])


end
module Odrpack = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.odr.odrpack"

let get_py name = Py.Module.get __wrap_namespace name
let odr ?we ?wd ?fjacb ?fjacd ?extra_args ?ifixx ?ifixb ?job ?iprint ?errfile ?rptfile ?ndigit ?taufac ?sstol ?partol ?maxit ?stpb ?stpd ?sclb ?scld ?work ?iwork ?full_output ~fcn ~beta0 ~y ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "odr"
     [||]
     (Wrap_utils.keyword_args [("we", we); ("wd", wd); ("fjacb", fjacb); ("fjacd", fjacd); ("extra_args", extra_args); ("ifixx", ifixx); ("ifixb", ifixb); ("job", job); ("iprint", iprint); ("errfile", errfile); ("rptfile", rptfile); ("ndigit", ndigit); ("taufac", taufac); ("sstol", sstol); ("partol", partol); ("maxit", maxit); ("stpb", stpb); ("stpd", stpd); ("sclb", sclb); ("scld", scld); ("work", work); ("iwork", iwork); ("full_output", full_output); ("fcn", Some(fcn )); ("beta0", Some(beta0 )); ("y", Some(y )); ("x", Some(x ))])

let warn ?category ?stacklevel ?source ~message () =
   Py.Module.get_function_with_keywords __wrap_namespace "warn"
     [||]
     (Wrap_utils.keyword_args [("category", category); ("stacklevel", stacklevel); ("source", source); ("message", Some(message ))])


end
let odr ?we ?wd ?fjacb ?fjacd ?extra_args ?ifixx ?ifixb ?job ?iprint ?errfile ?rptfile ?ndigit ?taufac ?sstol ?partol ?maxit ?stpb ?stpd ?sclb ?scld ?work ?iwork ?full_output ~fcn ~beta0 ~y ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "odr"
     [||]
     (Wrap_utils.keyword_args [("we", we); ("wd", wd); ("fjacb", fjacb); ("fjacd", fjacd); ("extra_args", extra_args); ("ifixx", ifixx); ("ifixb", ifixb); ("job", job); ("iprint", iprint); ("errfile", errfile); ("rptfile", rptfile); ("ndigit", ndigit); ("taufac", taufac); ("sstol", sstol); ("partol", partol); ("maxit", maxit); ("stpb", stpb); ("stpd", stpd); ("sclb", sclb); ("scld", scld); ("work", work); ("iwork", iwork); ("full_output", full_output); ("fcn", Some(fcn )); ("beta0", Some(beta0 )); ("y", Some(y )); ("x", Some(x ))])

                  let polynomial order =
                     Py.Module.get_function_with_keywords __wrap_namespace "polynomial"
                       [||]
                       (Wrap_utils.keyword_args [("order", Some(order |> (function
| `Sequence x -> Wrap_utils.id x
| `I x -> Py.Int.of_int x
)))])

