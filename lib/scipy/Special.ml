let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.special"

let get_py name = Py.Module.get __wrap_namespace name
module SpecialFunctionError = struct
type tag = [`SpecialFunctionError]
type t = [`BaseException | `Object | `SpecialFunctionError] Obj.t
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
module SpecialFunctionWarning = struct
type tag = [`SpecialFunctionWarning]
type t = [`BaseException | `Object | `SpecialFunctionWarning] Obj.t
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
module Errstate = struct
type tag = [`Errstate]
type t = [`Errstate | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?kwargs () =
   Py.Module.get_function_with_keywords __wrap_namespace "errstate"
     [||]
     (match kwargs with None -> [] | Some x -> x)
     |> of_pyobject
let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Orthogonal = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.special.orthogonal"

let get_py name = Py.Module.get __wrap_namespace name
module Int = struct
type tag = [`Int]
type t = [`Int | `Object] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?x () =
   Py.Module.get_function_with_keywords __wrap_namespace "int"
     [||]
     (Wrap_utils.keyword_args [("x", x)])
     |> of_pyobject
let as_integer_ratio self =
   Py.Module.get_function_with_keywords (to_pyobject self) "as_integer_ratio"
     [||]
     []

let bit_length self =
   Py.Module.get_function_with_keywords (to_pyobject self) "bit_length"
     [||]
     []

let from_bytes ?signed ~bytes ~byteorder self =
   Py.Module.get_function_with_keywords (to_pyobject self) "from_bytes"
     [||]
     (Wrap_utils.keyword_args [("signed", signed); ("bytes", Some(bytes )); ("byteorder", Some(byteorder ))])

let to_bytes ?signed ~length ~byteorder self =
   Py.Module.get_function_with_keywords (to_pyobject self) "to_bytes"
     [||]
     (Wrap_utils.keyword_args [("signed", signed); ("length", Some(length )); ("byteorder", Some(byteorder ))])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Orthopoly1d = struct
type tag = [`Orthopoly1d]
type t = [`Object | `Orthopoly1d] Obj.t
let of_pyobject x = ((Obj.of_pyobject x) : t)
let to_pyobject x = Obj.to_pyobject x
let create ?weights ?hn ?kn ?wfunc ?limits ?monic ?eval_func ~roots () =
   Py.Module.get_function_with_keywords __wrap_namespace "orthopoly1d"
     [||]
     (Wrap_utils.keyword_args [("weights", weights); ("hn", hn); ("kn", kn); ("wfunc", wfunc); ("limits", limits); ("monic", monic); ("eval_func", eval_func); ("roots", Some(roots ))])
     |> of_pyobject
let __getitem__ ~val_ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__getitem__"
     [||]
     (Wrap_utils.keyword_args [("val", Some(val_ ))])

let __iter__ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__iter__"
     [||]
     []

let __setitem__ ~key ~val_ self =
   Py.Module.get_function_with_keywords (to_pyobject self) "__setitem__"
     [||]
     (Wrap_utils.keyword_args [("key", Some(key )); ("val", Some(val_ ))])

let deriv ?m self =
   Py.Module.get_function_with_keywords (to_pyobject self) "deriv"
     [||]
     (Wrap_utils.keyword_args [("m", m)])

let integ ?m ?k self =
   Py.Module.get_function_with_keywords (to_pyobject self) "integ"
     [||]
     (Wrap_utils.keyword_args [("m", m); ("k", k)])

let to_string self = Py.Object.to_string (to_pyobject self)
let show self = to_string self
let pp formatter self = Format.fprintf formatter "%s" (show self)

end
module Cephes = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.special._ufuncs"

let get_py name = Py.Module.get __wrap_namespace name
let agm ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "agm"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let airy ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "airy"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let airye ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "airye"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let bdtr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "bdtr"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let bdtrc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "bdtrc"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let bdtri ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "bdtri"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let bdtrik ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "bdtrik"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let bdtrin ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "bdtrin"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let bei ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "bei"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let beip ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "beip"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let ber ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ber"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let berp ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "berp"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let besselpoly ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "besselpoly"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let beta ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "beta"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let betainc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "betainc"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let betaincinv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "betaincinv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let betaln ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "betaln"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let binom ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "binom"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let boxcox ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "boxcox"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let boxcox1p ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "boxcox1p"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let btdtr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "btdtr"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let btdtri ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "btdtri"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let btdtria ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "btdtria"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let btdtrib ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "btdtrib"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let cbrt ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "cbrt"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let chdtr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "chdtr"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let chdtrc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "chdtrc"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let chdtri ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "chdtri"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let chdtriv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "chdtriv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let chndtr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "chndtr"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let chndtridf ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "chndtridf"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let chndtrinc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "chndtrinc"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let chndtrix ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "chndtrix"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let cosdg ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "cosdg"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let cosm1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "cosm1"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let cotdg ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "cotdg"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let dawsn ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "dawsn"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let ellipe ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ellipe"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let ellipeinc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ellipeinc"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let ellipj ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ellipj"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let ellipk ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ellipk"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let ellipkinc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ellipkinc"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let ellipkm1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ellipkm1"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let entr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "entr"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let erf ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "erf"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let erfc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "erfc"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let erfcx ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "erfcx"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let erfi ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "erfi"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let eval_chebyc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_chebyc"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_chebys ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_chebys"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_chebyt ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_chebyt"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_chebyu ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_chebyu"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_gegenbauer ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_gegenbauer"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_genlaguerre ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_genlaguerre"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_hermite ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_hermite"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_hermitenorm ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_hermitenorm"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_jacobi ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_jacobi"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_laguerre ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_laguerre"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_legendre ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_legendre"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_sh_chebyt ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_sh_chebyt"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_sh_chebyu ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_sh_chebyu"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_sh_jacobi ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_sh_jacobi"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_sh_legendre ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_sh_legendre"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let exp1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "exp1"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let exp10 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "exp10"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let exp2 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "exp2"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let expi ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "expi"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let expit ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "expit"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let expm1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "expm1"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let expn ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "expn"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let exprel ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "exprel"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let fdtr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "fdtr"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let fdtrc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "fdtrc"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let fdtri ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "fdtri"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let fdtridfd ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "fdtridfd"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let fresnel ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "fresnel"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let gamma ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "gamma"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let gammainc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "gammainc"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let gammaincc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "gammaincc"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let gammainccinv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "gammainccinv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let gammaincinv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "gammaincinv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let gammaln ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "gammaln"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let gammasgn ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "gammasgn"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let gdtr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "gdtr"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let gdtrc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "gdtrc"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let gdtria ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "gdtria"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let gdtrib ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "gdtrib"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let gdtrix ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "gdtrix"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let hankel1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "hankel1"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let hankel1e ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "hankel1e"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let hankel2 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "hankel2"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let hankel2e ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "hankel2e"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let huber ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "huber"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let hyp0f1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "hyp0f1"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let hyp1f1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "hyp1f1"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let hyp2f1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "hyp2f1"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let hyperu ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "hyperu"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let i0 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "i0"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let i0e ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "i0e"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let i1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "i1"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let i1e ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "i1e"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let inv_boxcox ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "inv_boxcox"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let inv_boxcox1p ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "inv_boxcox1p"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let it2i0k0 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "it2i0k0"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let it2j0y0 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "it2j0y0"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let it2struve0 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "it2struve0"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let itairy ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "itairy"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3))))
let iti0k0 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "iti0k0"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let itj0y0 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "itj0y0"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let itmodstruve0 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "itmodstruve0"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let itstruve0 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "itstruve0"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let iv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "iv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let ive ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ive"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let j0 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "j0"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let j1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "j1"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let jn ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "jn"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let jv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "jv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let jve ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "jve"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let k0 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "k0"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let k0e ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "k0e"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let k1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "k1"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let k1e ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "k1e"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let kei ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "kei"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let keip ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "keip"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let kelvin ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "kelvin"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let ker ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ker"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let kerp ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "kerp"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let kl_div ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "kl_div"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let kn ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "kn"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let kolmogi ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "kolmogi"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let kolmogorov ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "kolmogorov"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let kv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "kv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let kve ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "kve"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let log1p ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "log1p"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

                  let log_ndtr ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "log_ndtr"
                       (Array.of_list @@ List.concat [[x |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)]])
                       (Wrap_utils.keyword_args [("out", out); ("where", where)])

let loggamma ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "loggamma"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let logit ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "logit"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let lpmv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "lpmv"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let mathieu_a ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "mathieu_a"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let mathieu_b ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "mathieu_b"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let mathieu_cem ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "mathieu_cem"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let mathieu_modcem1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "mathieu_modcem1"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let mathieu_modcem2 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "mathieu_modcem2"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let mathieu_modsem1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "mathieu_modsem1"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let mathieu_modsem2 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "mathieu_modsem2"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let mathieu_sem ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "mathieu_sem"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let modfresnelm ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "modfresnelm"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let modfresnelp ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "modfresnelp"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let modstruve ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "modstruve"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let nbdtr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "nbdtr"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let nbdtrc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "nbdtrc"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let nbdtri ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "nbdtri"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let nbdtrik ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "nbdtrik"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let nbdtrin ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "nbdtrin"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let ncfdtr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ncfdtr"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let ncfdtri ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ncfdtri"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> Py.Float.to_float
let ncfdtridfd ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ncfdtridfd"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> Py.Float.to_float
let ncfdtridfn ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ncfdtridfn"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> Py.Float.to_float
let ncfdtrinc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ncfdtrinc"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> Py.Float.to_float
let nctdtr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "nctdtr"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let nctdtridf ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "nctdtridf"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let nctdtrinc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "nctdtrinc"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let nctdtrit ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "nctdtrit"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

                  let ndtr ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ndtr"
                       (Array.of_list @@ List.concat [[x |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)]])
                       (Wrap_utils.keyword_args [("out", out); ("where", where)])

let ndtri ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ndtri"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let nrdtrimn ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "nrdtrimn"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let nrdtrisd ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "nrdtrisd"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let obl_ang1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "obl_ang1"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let obl_ang1_cv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "obl_ang1_cv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let obl_cv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "obl_cv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let obl_rad1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "obl_rad1"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let obl_rad1_cv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "obl_rad1_cv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let obl_rad2 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "obl_rad2"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let obl_rad2_cv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "obl_rad2_cv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let owens_t ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "owens_t"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let pbdv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pbdv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let pbvv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pbvv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let pbwa ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pbwa"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let pdtr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pdtr"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let pdtrc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pdtrc"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let pdtri ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pdtri"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let pdtrik ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pdtrik"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let poch ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "poch"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let pro_ang1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pro_ang1"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let pro_ang1_cv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pro_ang1_cv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let pro_cv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pro_cv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let pro_rad1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pro_rad1"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let pro_rad1_cv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pro_rad1_cv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let pro_rad2 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pro_rad2"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let pro_rad2_cv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pro_rad2_cv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let pseudo_huber ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pseudo_huber"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let psi ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "psi"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let radian ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "radian"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let rel_entr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "rel_entr"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let rgamma ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "rgamma"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let round ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "round"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let shichi ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "shichi"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let sici ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "sici"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let sindg ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "sindg"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let smirnov ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "smirnov"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let smirnovi ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "smirnovi"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let spence ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "spence"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let sph_harm ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "sph_harm"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let stdtr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "stdtr"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let stdtridf ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "stdtridf"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let stdtrit ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "stdtrit"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let struve ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "struve"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let tandg ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "tandg"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let tklmbda ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "tklmbda"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let voigt_profile ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "voigt_profile"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let wofz ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "wofz"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let wrightomega ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "wrightomega"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let xlog1py ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "xlog1py"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let xlogy ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "xlogy"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let y0 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "y0"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let y1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "y1"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let yn ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "yn"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let yv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "yv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let yve ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "yve"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let zetac ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "zetac"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
let airy ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "airy"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

                  let arange ?start ?step ?dtype ~stop () =
                     Py.Module.get_function_with_keywords __wrap_namespace "arange"
                       [||]
                       (Wrap_utils.keyword_args [("start", Wrap_utils.Option.map start (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("step", Wrap_utils.Option.map step (function
| `I x -> Py.Int.of_int x
| `F x -> Py.Float.of_float x
)); ("dtype", Wrap_utils.Option.map dtype Np.Dtype.to_pyobject); ("stop", Some(stop |> (function
| `F x -> Py.Float.of_float x
| `I x -> Py.Int.of_int x
)))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let arccos ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "arccos"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let around ?decimals ?out ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "around"
     [||]
     (Wrap_utils.keyword_args [("decimals", Wrap_utils.Option.map decimals Py.Int.of_int); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let binom ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "binom"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let c_roots ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "c_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let cg_roots ?mu ~n ~alpha () =
   Py.Module.get_function_with_keywords __wrap_namespace "cg_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int)); ("alpha", Some(alpha |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let chebyc ?monic ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "chebyc"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])

let chebys ?monic ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "chebys"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])

let chebyt ?monic ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "chebyt"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])

let chebyu ?monic ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "chebyu"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])

                  let cos ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "cos"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_chebyc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_chebyc"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_chebys ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_chebys"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_chebyt ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_chebyt"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_chebyu ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_chebyu"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_gegenbauer ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_gegenbauer"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_genlaguerre ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_genlaguerre"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_hermite ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_hermite"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_hermitenorm ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_hermitenorm"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_jacobi ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_jacobi"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_laguerre ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_laguerre"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_legendre ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_legendre"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_sh_chebyt ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_sh_chebyt"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_sh_chebyu ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_sh_chebyu"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_sh_jacobi ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_sh_jacobi"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_sh_legendre ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_sh_legendre"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let exp ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "exp"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let floor ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "floor"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let gegenbauer ?monic ~n ~alpha () =
   Py.Module.get_function_with_keywords __wrap_namespace "gegenbauer"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int)); ("alpha", Some(alpha ))])

let genlaguerre ?monic ~n ~alpha () =
   Py.Module.get_function_with_keywords __wrap_namespace "genlaguerre"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int)); ("alpha", Some(alpha |> Py.Float.of_float))])

let h_roots ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "h_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let he_roots ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "he_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let hermite ?monic ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "hermite"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])

let hermitenorm ?monic ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "hermitenorm"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])

let hstack tup =
   Py.Module.get_function_with_keywords __wrap_namespace "hstack"
     [||]
     (Wrap_utils.keyword_args [("tup", Some(tup |> (fun ml -> Py.List.of_list_map Np.Obj.to_pyobject ml)))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let j_roots ?mu ~n ~alpha ~beta () =
   Py.Module.get_function_with_keywords __wrap_namespace "j_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int)); ("alpha", Some(alpha |> Py.Float.of_float)); ("beta", Some(beta |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let jacobi ?monic ~n ~alpha ~beta () =
   Py.Module.get_function_with_keywords __wrap_namespace "jacobi"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int)); ("alpha", Some(alpha |> Py.Float.of_float)); ("beta", Some(beta |> Py.Float.of_float))])

let js_roots ?mu ~n ~p1 ~q1 () =
   Py.Module.get_function_with_keywords __wrap_namespace "js_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int)); ("p1", Some(p1 |> Py.Float.of_float)); ("q1", Some(q1 |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let l_roots ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "l_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let la_roots ?mu ~n ~alpha () =
   Py.Module.get_function_with_keywords __wrap_namespace "la_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int)); ("alpha", Some(alpha |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let laguerre ?monic ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "laguerre"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])

let legendre ?monic ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "legendre"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])

let p_roots ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "p_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let poch ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "poch"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let ps_roots ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "ps_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_chebyc ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_chebyc"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_chebys ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_chebys"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_chebyt ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_chebyt"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_chebyu ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_chebyu"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_gegenbauer ?mu ~n ~alpha () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_gegenbauer"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int)); ("alpha", Some(alpha |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_genlaguerre ?mu ~n ~alpha () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_genlaguerre"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int)); ("alpha", Some(alpha |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_hermite ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_hermite"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_hermitenorm ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_hermitenorm"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_jacobi ?mu ~n ~alpha ~beta () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_jacobi"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int)); ("alpha", Some(alpha |> Py.Float.of_float)); ("beta", Some(beta |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_laguerre ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_laguerre"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_legendre ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_legendre"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_sh_chebyt ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_sh_chebyt"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_sh_chebyu ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_sh_chebyu"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_sh_jacobi ?mu ~n ~p1 ~q1 () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_sh_jacobi"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int)); ("p1", Some(p1 |> Py.Float.of_float)); ("q1", Some(q1 |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_sh_legendre ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_sh_legendre"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let s_roots ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "s_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let sh_chebyt ?monic ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "sh_chebyt"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])

let sh_chebyu ?monic ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "sh_chebyu"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])

let sh_jacobi ?monic ~n ~p ~q () =
   Py.Module.get_function_with_keywords __wrap_namespace "sh_jacobi"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int)); ("p", Some(p |> Py.Float.of_float)); ("q", Some(q |> Py.Float.of_float))])

let sh_legendre ?monic ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "sh_legendre"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])

                  let sin ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "sin"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let sqrt ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "sqrt"
                       (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
                       (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `Tuple_of_ndarray_and_None x -> Wrap_utils.id x
)); ("where", Wrap_utils.Option.map where Np.Obj.to_pyobject)])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let t_roots ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "t_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let ts_roots ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "ts_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let u_roots ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "u_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let us_roots ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "us_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))

end
module Sf_error = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.special.sf_error"

let get_py name = Py.Module.get __wrap_namespace name

end
module Specfun = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.special.specfun"

let get_py name = Py.Module.get __wrap_namespace name
let herzo n =
   Py.Module.get_function_with_keywords __wrap_namespace "herzo"
     [||]
     (Wrap_utils.keyword_args [("n", Some(n ))])

let lagzo n =
   Py.Module.get_function_with_keywords __wrap_namespace "lagzo"
     [||]
     (Wrap_utils.keyword_args [("n", Some(n ))])

let legzo n =
   Py.Module.get_function_with_keywords __wrap_namespace "legzo"
     [||]
     (Wrap_utils.keyword_args [("n", Some(n ))])


end
module Spfun_stats = struct
let () = Wrap_utils.init ();;
let __wrap_namespace = Py.import "scipy.special.spfun_stats"

let get_py name = Py.Module.get __wrap_namespace name
let loggam ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "loggam"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let multigammaln ~a ~d () =
   Py.Module.get_function_with_keywords __wrap_namespace "multigammaln"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Np.Obj.to_pyobject)); ("d", Some(d |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))

end
let agm ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "agm"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let ai_zeros nt =
   Py.Module.get_function_with_keywords __wrap_namespace "ai_zeros"
     [||]
     (Wrap_utils.keyword_args [("nt", Some(nt |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 3))))
let airy ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "airy"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let airye ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "airye"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let assoc_laguerre ?k ~x ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "assoc_laguerre"
     [||]
     (Wrap_utils.keyword_args [("k", k); ("x", Some(x )); ("n", Some(n ))])

let bdtr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "bdtr"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let bdtrc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "bdtrc"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let bdtri ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "bdtri"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let bdtrik ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "bdtrik"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let bdtrin ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "bdtrin"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let bei ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "bei"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let bei_zeros nt =
   Py.Module.get_function_with_keywords __wrap_namespace "bei_zeros"
     [||]
     (Wrap_utils.keyword_args [("nt", Some(nt ))])

let beip ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "beip"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let beip_zeros nt =
   Py.Module.get_function_with_keywords __wrap_namespace "beip_zeros"
     [||]
     (Wrap_utils.keyword_args [("nt", Some(nt ))])

let ber ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ber"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let ber_zeros nt =
   Py.Module.get_function_with_keywords __wrap_namespace "ber_zeros"
     [||]
     (Wrap_utils.keyword_args [("nt", Some(nt ))])

let bernoulli n =
   Py.Module.get_function_with_keywords __wrap_namespace "bernoulli"
     [||]
     (Wrap_utils.keyword_args [("n", Some(n ))])

let berp ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "berp"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let berp_zeros nt =
   Py.Module.get_function_with_keywords __wrap_namespace "berp_zeros"
     [||]
     (Wrap_utils.keyword_args [("nt", Some(nt ))])

let besselpoly ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "besselpoly"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let beta ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "beta"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let betainc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "betainc"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let betaincinv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "betaincinv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let betaln ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "betaln"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let bi_zeros nt =
   Py.Module.get_function_with_keywords __wrap_namespace "bi_zeros"
     [||]
     (Wrap_utils.keyword_args [("nt", Some(nt |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 2)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 3))))
let binom ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "binom"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let boxcox ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "boxcox"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let boxcox1p ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "boxcox1p"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let btdtr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "btdtr"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let btdtri ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "btdtri"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let btdtria ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "btdtria"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let btdtrib ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "btdtrib"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let c_roots ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "c_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let cbrt ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "cbrt"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let cg_roots ?mu ~n ~alpha () =
   Py.Module.get_function_with_keywords __wrap_namespace "cg_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int)); ("alpha", Some(alpha |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let chdtr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "chdtr"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let chdtrc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "chdtrc"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let chdtri ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "chdtri"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let chdtriv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "chdtriv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let chebyc ?monic ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "chebyc"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])

let chebys ?monic ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "chebys"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])

let chebyt ?monic ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "chebyt"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])

let chebyu ?monic ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "chebyu"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])

let chndtr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "chndtr"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let chndtridf ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "chndtridf"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let chndtrinc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "chndtrinc"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let chndtrix ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "chndtrix"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

                  let clpmn ?type_ ~m ~n ~z () =
                     Py.Module.get_function_with_keywords __wrap_namespace "clpmn"
                       [||]
                       (Wrap_utils.keyword_args [("type", Wrap_utils.Option.map type_ Py.Int.of_int); ("m", Some(m |> Py.Int.of_int)); ("n", Some(n |> Py.Int.of_int)); ("z", Some(z |> (function
| `F x -> Py.Float.of_float x
| `Complex x -> Wrap_utils.id x
)))])
                       |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
                  let comb ?exact ?repetition ~n ~k () =
                     Py.Module.get_function_with_keywords __wrap_namespace "comb"
                       [||]
                       (Wrap_utils.keyword_args [("exact", Wrap_utils.Option.map exact Py.Bool.of_bool); ("repetition", Wrap_utils.Option.map repetition Py.Bool.of_bool); ("N", Some(n |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
))); ("k", Some(k |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)))])

let cosdg ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "cosdg"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let cosm1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "cosm1"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let cotdg ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "cotdg"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let dawsn ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "dawsn"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let digamma ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "digamma"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let diric ~x ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "diric"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject)); ("n", Some(n |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let ellip_harm ?signm ?signn ~h2 ~k2 ~n ~p ~s () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ellip_harm"
                       [||]
                       (Wrap_utils.keyword_args [("signm", Wrap_utils.Option.map signm (function
| `One -> Py.Int.of_int 1
| `T_1 x -> Wrap_utils.id x
)); ("signn", Wrap_utils.Option.map signn (function
| `One -> Py.Int.of_int 1
| `T_1 x -> Wrap_utils.id x
)); ("h2", Some(h2 |> Py.Float.of_float)); ("k2", Some(k2 |> Py.Float.of_float)); ("n", Some(n |> Py.Int.of_int)); ("p", Some(p |> Py.Int.of_int)); ("s", Some(s |> Py.Float.of_float))])
                       |> Py.Float.to_float
let ellip_harm_2 ~h2 ~k2 ~n ~p ~s () =
   Py.Module.get_function_with_keywords __wrap_namespace "ellip_harm_2"
     [||]
     (Wrap_utils.keyword_args [("h2", Some(h2 |> Py.Float.of_float)); ("k2", Some(k2 |> Py.Float.of_float)); ("n", Some(n |> Py.Int.of_int)); ("p", Some(p |> Py.Int.of_int)); ("s", Some(s |> Py.Float.of_float))])
     |> Py.Float.to_float
let ellip_normal ~h2 ~k2 ~n ~p () =
   Py.Module.get_function_with_keywords __wrap_namespace "ellip_normal"
     [||]
     (Wrap_utils.keyword_args [("h2", Some(h2 |> Py.Float.of_float)); ("k2", Some(k2 |> Py.Float.of_float)); ("n", Some(n |> Py.Int.of_int)); ("p", Some(p |> Py.Int.of_int))])
     |> Py.Float.to_float
let ellipe ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ellipe"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let ellipeinc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ellipeinc"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let ellipj ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ellipj"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let ellipk ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ellipk"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let ellipkinc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ellipkinc"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let ellipkm1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ellipkm1"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let entr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "entr"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let erf ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "erf"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let erf_zeros nt =
   Py.Module.get_function_with_keywords __wrap_namespace "erf_zeros"
     [||]
     (Wrap_utils.keyword_args [("nt", Some(nt |> Py.Int.of_int))])

let erfc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "erfc"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let erfcinv y =
   Py.Module.get_function_with_keywords __wrap_namespace "erfcinv"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let erfcx ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "erfcx"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let erfi ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "erfi"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let erfinv y =
   Py.Module.get_function_with_keywords __wrap_namespace "erfinv"
     [||]
     (Wrap_utils.keyword_args [("y", Some(y |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let euler n =
   Py.Module.get_function_with_keywords __wrap_namespace "euler"
     [||]
     (Wrap_utils.keyword_args [("n", Some(n |> Py.Int.of_int))])

let eval_chebyc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_chebyc"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_chebys ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_chebys"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_chebyt ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_chebyt"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_chebyu ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_chebyu"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_gegenbauer ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_gegenbauer"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_genlaguerre ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_genlaguerre"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_hermite ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_hermite"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_hermitenorm ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_hermitenorm"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_jacobi ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_jacobi"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_laguerre ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_laguerre"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_legendre ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_legendre"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_sh_chebyt ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_sh_chebyt"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_sh_chebyu ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_sh_chebyu"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_sh_jacobi ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_sh_jacobi"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let eval_sh_legendre ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "eval_sh_legendre"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let exp1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "exp1"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let exp10 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "exp10"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let exp2 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "exp2"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let expi ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "expi"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let expit ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "expit"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let expm1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "expm1"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let expn ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "expn"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let exprel ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "exprel"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

                  let factorial ?exact ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "factorial"
                       [||]
                       (Wrap_utils.keyword_args [("exact", Wrap_utils.Option.map exact Py.Bool.of_bool); ("n", Some(n |> (function
| `I x -> Py.Int.of_int x
| `Array_like_of_ints x -> Wrap_utils.id x
)))])

                  let factorial2 ?exact ~n () =
                     Py.Module.get_function_with_keywords __wrap_namespace "factorial2"
                       [||]
                       (Wrap_utils.keyword_args [("exact", Wrap_utils.Option.map exact Py.Bool.of_bool); ("n", Some(n |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)))])
                       |> (fun x -> if Wrap_utils.check_int x then `I (Py.Int.to_int x) else if Wrap_utils.check_float x then `F (Py.Float.to_float x) else failwith (Printf.sprintf "Sklearn: could not identify type from Python value %s (%s)"
                                                  (Py.Object.to_string x) (Wrap_utils.type_string x)))
let factorialk ?exact ~n ~k () =
   Py.Module.get_function_with_keywords __wrap_namespace "factorialk"
     [||]
     (Wrap_utils.keyword_args [("exact", Wrap_utils.Option.map exact Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int)); ("k", Some(k |> Py.Int.of_int))])
     |> Py.Int.to_int
let fdtr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "fdtr"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let fdtrc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "fdtrc"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let fdtri ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "fdtri"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let fdtridfd ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "fdtridfd"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let fresnel ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "fresnel"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let fresnel_zeros nt =
   Py.Module.get_function_with_keywords __wrap_namespace "fresnel_zeros"
     [||]
     (Wrap_utils.keyword_args [("nt", Some(nt ))])

let fresnelc_zeros nt =
   Py.Module.get_function_with_keywords __wrap_namespace "fresnelc_zeros"
     [||]
     (Wrap_utils.keyword_args [("nt", Some(nt ))])

let fresnels_zeros nt =
   Py.Module.get_function_with_keywords __wrap_namespace "fresnels_zeros"
     [||]
     (Wrap_utils.keyword_args [("nt", Some(nt ))])

let gamma ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "gamma"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let gammainc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "gammainc"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let gammaincc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "gammaincc"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let gammainccinv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "gammainccinv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let gammaincinv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "gammaincinv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let gammaln ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "gammaln"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let gammasgn ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "gammasgn"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let gdtr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "gdtr"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let gdtrc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "gdtrc"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let gdtria ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "gdtria"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let gdtrib ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "gdtrib"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let gdtrix ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "gdtrix"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let gegenbauer ?monic ~n ~alpha () =
   Py.Module.get_function_with_keywords __wrap_namespace "gegenbauer"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int)); ("alpha", Some(alpha ))])

let genlaguerre ?monic ~n ~alpha () =
   Py.Module.get_function_with_keywords __wrap_namespace "genlaguerre"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int)); ("alpha", Some(alpha |> Py.Float.of_float))])

let h1vp ?n ~v ~z () =
   Py.Module.get_function_with_keywords __wrap_namespace "h1vp"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("v", Some(v |> Py.Float.of_float)); ("z", Some(z ))])

let h2vp ?n ~v ~z () =
   Py.Module.get_function_with_keywords __wrap_namespace "h2vp"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("v", Some(v |> Py.Float.of_float)); ("z", Some(z ))])

let h_roots ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "h_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let hankel1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "hankel1"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let hankel1e ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "hankel1e"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let hankel2 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "hankel2"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let hankel2e ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "hankel2e"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let he_roots ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "he_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let hermite ?monic ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "hermite"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])

let hermitenorm ?monic ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "hermitenorm"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])

let huber ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "huber"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let hyp0f1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "hyp0f1"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let hyp1f1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "hyp1f1"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let hyp2f1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "hyp2f1"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let hyperu ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "hyperu"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let i0 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "i0"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let i0e ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "i0e"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let i1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "i1"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let i1e ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "i1e"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let inv_boxcox ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "inv_boxcox"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let inv_boxcox1p ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "inv_boxcox1p"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let it2i0k0 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "it2i0k0"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let it2j0y0 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "it2j0y0"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let it2struve0 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "it2struve0"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let itairy ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "itairy"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1)), (Wrap_utils.id (Py.Tuple.get x 2)), (Wrap_utils.id (Py.Tuple.get x 3))))
let iti0k0 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "iti0k0"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let itj0y0 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "itj0y0"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let itmodstruve0 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "itmodstruve0"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let itstruve0 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "itstruve0"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let iv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "iv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let ive ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ive"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let ivp ?n ~v ~z () =
   Py.Module.get_function_with_keywords __wrap_namespace "ivp"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("v", Some(v |> Np.Obj.to_pyobject)); ("z", Some(z ))])

let j0 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "j0"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let j1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "j1"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let j_roots ?mu ~n ~alpha ~beta () =
   Py.Module.get_function_with_keywords __wrap_namespace "j_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int)); ("alpha", Some(alpha |> Py.Float.of_float)); ("beta", Some(beta |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let jacobi ?monic ~n ~alpha ~beta () =
   Py.Module.get_function_with_keywords __wrap_namespace "jacobi"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int)); ("alpha", Some(alpha |> Py.Float.of_float)); ("beta", Some(beta |> Py.Float.of_float))])

let jn ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "jn"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let jn_zeros ~n ~nt () =
   Py.Module.get_function_with_keywords __wrap_namespace "jn_zeros"
     [||]
     (Wrap_utils.keyword_args [("n", Some(n |> Py.Int.of_int)); ("nt", Some(nt |> Py.Int.of_int))])

let jnjnp_zeros nt =
   Py.Module.get_function_with_keywords __wrap_namespace "jnjnp_zeros"
     [||]
     (Wrap_utils.keyword_args [("nt", Some(nt |> Py.Int.of_int))])

let jnp_zeros ~n ~nt () =
   Py.Module.get_function_with_keywords __wrap_namespace "jnp_zeros"
     [||]
     (Wrap_utils.keyword_args [("n", Some(n |> Py.Int.of_int)); ("nt", Some(nt |> Py.Int.of_int))])

let jnyn_zeros ~n ~nt () =
   Py.Module.get_function_with_keywords __wrap_namespace "jnyn_zeros"
     [||]
     (Wrap_utils.keyword_args [("n", Some(n |> Py.Int.of_int)); ("nt", Some(nt |> Py.Int.of_int))])

let js_roots ?mu ~n ~p1 ~q1 () =
   Py.Module.get_function_with_keywords __wrap_namespace "js_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int)); ("p1", Some(p1 |> Py.Float.of_float)); ("q1", Some(q1 |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let jv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "jv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let jve ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "jve"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let jvp ?n ~v ~z () =
   Py.Module.get_function_with_keywords __wrap_namespace "jvp"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("v", Some(v |> Py.Float.of_float)); ("z", Some(z ))])

let k0 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "k0"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let k0e ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "k0e"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let k1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "k1"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let k1e ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "k1e"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let kei ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "kei"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let kei_zeros nt =
   Py.Module.get_function_with_keywords __wrap_namespace "kei_zeros"
     [||]
     (Wrap_utils.keyword_args [("nt", Some(nt ))])

let keip ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "keip"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let keip_zeros nt =
   Py.Module.get_function_with_keywords __wrap_namespace "keip_zeros"
     [||]
     (Wrap_utils.keyword_args [("nt", Some(nt ))])

let kelvin ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "kelvin"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let kelvin_zeros nt =
   Py.Module.get_function_with_keywords __wrap_namespace "kelvin_zeros"
     [||]
     (Wrap_utils.keyword_args [("nt", Some(nt ))])

let ker ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ker"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let ker_zeros nt =
   Py.Module.get_function_with_keywords __wrap_namespace "ker_zeros"
     [||]
     (Wrap_utils.keyword_args [("nt", Some(nt ))])

let kerp ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "kerp"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let kerp_zeros nt =
   Py.Module.get_function_with_keywords __wrap_namespace "kerp_zeros"
     [||]
     (Wrap_utils.keyword_args [("nt", Some(nt ))])

let kl_div ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "kl_div"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let kn ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "kn"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let kolmogi ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "kolmogi"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let kolmogorov ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "kolmogorov"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let kv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "kv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let kve ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "kve"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let kvp ?n ~v ~z () =
   Py.Module.get_function_with_keywords __wrap_namespace "kvp"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("v", Some(v |> Np.Obj.to_pyobject)); ("z", Some(z ))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let l_roots ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "l_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let la_roots ?mu ~n ~alpha () =
   Py.Module.get_function_with_keywords __wrap_namespace "la_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int)); ("alpha", Some(alpha |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let laguerre ?monic ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "laguerre"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])

let lambertw ?k ?tol ~z () =
   Py.Module.get_function_with_keywords __wrap_namespace "lambertw"
     [||]
     (Wrap_utils.keyword_args [("k", Wrap_utils.Option.map k Py.Int.of_int); ("tol", Wrap_utils.Option.map tol Py.Float.of_float); ("z", Some(z |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let legendre ?monic ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "legendre"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])

let lmbda ~v ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "lmbda"
     [||]
     (Wrap_utils.keyword_args [("v", Some(v |> Py.Float.of_float)); ("x", Some(x |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let log1p ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "log1p"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

                  let log_ndtr ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "log_ndtr"
                       (Array.of_list @@ List.concat [[x |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)]])
                       (Wrap_utils.keyword_args [("out", out); ("where", where)])

let loggamma ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "loggamma"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let logit ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "logit"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let logsumexp ?axis ?b ?keepdims ?return_sign ~a () =
   Py.Module.get_function_with_keywords __wrap_namespace "logsumexp"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("b", Wrap_utils.Option.map b Np.Obj.to_pyobject); ("keepdims", Wrap_utils.Option.map keepdims Py.Bool.of_bool); ("return_sign", Wrap_utils.Option.map return_sign Py.Bool.of_bool); ("a", Some(a |> Np.Obj.to_pyobject))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let lpmn ~m ~n ~z () =
   Py.Module.get_function_with_keywords __wrap_namespace "lpmn"
     [||]
     (Wrap_utils.keyword_args [("m", Some(m |> Py.Int.of_int)); ("n", Some(n |> Py.Int.of_int)); ("z", Some(z |> Py.Float.of_float))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let lpmv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "lpmv"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let lpn ~n ~z () =
   Py.Module.get_function_with_keywords __wrap_namespace "lpn"
     [||]
     (Wrap_utils.keyword_args [("n", Some(n )); ("z", Some(z ))])

let lqmn ~m ~n ~z () =
   Py.Module.get_function_with_keywords __wrap_namespace "lqmn"
     [||]
     (Wrap_utils.keyword_args [("m", Some(m |> Py.Int.of_int)); ("n", Some(n |> Py.Int.of_int)); ("z", Some(z ))])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let lqn ~n ~z () =
   Py.Module.get_function_with_keywords __wrap_namespace "lqn"
     [||]
     (Wrap_utils.keyword_args [("n", Some(n )); ("z", Some(z ))])

let mathieu_a ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "mathieu_a"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let mathieu_b ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "mathieu_b"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let mathieu_cem ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "mathieu_cem"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let mathieu_even_coef ~m ~q () =
   Py.Module.get_function_with_keywords __wrap_namespace "mathieu_even_coef"
     [||]
     (Wrap_utils.keyword_args [("m", Some(m |> Py.Int.of_int)); ("q", Some(q |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let mathieu_modcem1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "mathieu_modcem1"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let mathieu_modcem2 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "mathieu_modcem2"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let mathieu_modsem1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "mathieu_modsem1"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let mathieu_modsem2 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "mathieu_modsem2"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let mathieu_odd_coef ~m ~q () =
   Py.Module.get_function_with_keywords __wrap_namespace "mathieu_odd_coef"
     [||]
     (Wrap_utils.keyword_args [("m", Some(m |> Py.Int.of_int)); ("q", Some(q |> Py.Float.of_float))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let mathieu_sem ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "mathieu_sem"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let modfresnelm ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "modfresnelm"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let modfresnelp ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "modfresnelp"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let modstruve ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "modstruve"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let multigammaln ~a ~d () =
   Py.Module.get_function_with_keywords __wrap_namespace "multigammaln"
     [||]
     (Wrap_utils.keyword_args [("a", Some(a |> Np.Obj.to_pyobject)); ("d", Some(d |> Py.Int.of_int))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let nbdtr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "nbdtr"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let nbdtrc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "nbdtrc"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let nbdtri ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "nbdtri"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let nbdtrik ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "nbdtrik"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let nbdtrin ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "nbdtrin"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let ncfdtr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ncfdtr"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let ncfdtri ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ncfdtri"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> Py.Float.to_float
let ncfdtridfd ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ncfdtridfd"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> Py.Float.to_float
let ncfdtridfn ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ncfdtridfn"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> Py.Float.to_float
let ncfdtrinc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ncfdtrinc"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> Py.Float.to_float
let nctdtr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "nctdtr"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let nctdtridf ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "nctdtridf"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let nctdtrinc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "nctdtrinc"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let nctdtrit ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "nctdtrit"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

                  let ndtr ?out ?where ~x () =
                     Py.Module.get_function_with_keywords __wrap_namespace "ndtr"
                       (Array.of_list @@ List.concat [[x |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `PyObject x -> Wrap_utils.id x
)]])
                       (Wrap_utils.keyword_args [("out", out); ("where", where)])

let ndtri ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "ndtri"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let nrdtrimn ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "nrdtrimn"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let nrdtrisd ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "nrdtrisd"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let obl_ang1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "obl_ang1"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let obl_ang1_cv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "obl_ang1_cv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let obl_cv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "obl_cv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let obl_cv_seq ~m ~n ~c () =
   Py.Module.get_function_with_keywords __wrap_namespace "obl_cv_seq"
     [||]
     (Wrap_utils.keyword_args [("m", Some(m )); ("n", Some(n )); ("c", Some(c ))])

let obl_rad1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "obl_rad1"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let obl_rad1_cv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "obl_rad1_cv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let obl_rad2 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "obl_rad2"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let obl_rad2_cv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "obl_rad2_cv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let owens_t ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "owens_t"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let p_roots ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "p_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let pbdn_seq ~n ~z () =
   Py.Module.get_function_with_keywords __wrap_namespace "pbdn_seq"
     [||]
     (Wrap_utils.keyword_args [("n", Some(n |> Py.Int.of_int)); ("z", Some(z ))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let pbdv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pbdv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let pbdv_seq ~v ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pbdv_seq"
     [||]
     (Wrap_utils.keyword_args [("v", Some(v |> Py.Float.of_float)); ("x", Some(x |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let pbvv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pbvv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let pbvv_seq ~v ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pbvv_seq"
     [||]
     (Wrap_utils.keyword_args [("v", Some(v |> Py.Float.of_float)); ("x", Some(x |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let pbwa ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pbwa"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let pdtr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pdtr"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let pdtrc ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pdtrc"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let pdtri ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pdtri"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let pdtrik ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pdtrik"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

                  let perm ?exact ~n ~k () =
                     Py.Module.get_function_with_keywords __wrap_namespace "perm"
                       [||]
                       (Wrap_utils.keyword_args [("exact", Wrap_utils.Option.map exact Py.Bool.of_bool); ("N", Some(n |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
))); ("k", Some(k |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
)))])

let poch ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "poch"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let polygamma ~n ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "polygamma"
     [||]
     (Wrap_utils.keyword_args [("n", Some(n |> Np.Obj.to_pyobject)); ("x", Some(x |> Np.Obj.to_pyobject))])

let pro_ang1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pro_ang1"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let pro_ang1_cv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pro_ang1_cv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let pro_cv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pro_cv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let pro_cv_seq ~m ~n ~c () =
   Py.Module.get_function_with_keywords __wrap_namespace "pro_cv_seq"
     [||]
     (Wrap_utils.keyword_args [("m", Some(m )); ("n", Some(n )); ("c", Some(c ))])

let pro_rad1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pro_rad1"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let pro_rad1_cv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pro_rad1_cv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let pro_rad2 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pro_rad2"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let pro_rad2_cv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pro_rad2_cv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> ((Wrap_utils.id (Py.Tuple.get x 0)), (Wrap_utils.id (Py.Tuple.get x 1))))
let ps_roots ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "ps_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let pseudo_huber ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "pseudo_huber"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let psi ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "psi"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let radian ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "radian"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let rel_entr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "rel_entr"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let rgamma ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "rgamma"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let riccati_jn ~n ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "riccati_jn"
     [||]
     (Wrap_utils.keyword_args [("n", Some(n |> Py.Int.of_int)); ("x", Some(x |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let riccati_yn ~n ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "riccati_yn"
     [||]
     (Wrap_utils.keyword_args [("n", Some(n |> Py.Int.of_int)); ("x", Some(x |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let roots_chebyc ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_chebyc"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_chebys ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_chebys"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_chebyt ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_chebyt"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_chebyu ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_chebyu"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_gegenbauer ?mu ~n ~alpha () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_gegenbauer"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int)); ("alpha", Some(alpha |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_genlaguerre ?mu ~n ~alpha () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_genlaguerre"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int)); ("alpha", Some(alpha |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_hermite ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_hermite"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_hermitenorm ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_hermitenorm"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_jacobi ?mu ~n ~alpha ~beta () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_jacobi"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int)); ("alpha", Some(alpha |> Py.Float.of_float)); ("beta", Some(beta |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_laguerre ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_laguerre"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_legendre ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_legendre"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_sh_chebyt ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_sh_chebyt"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_sh_chebyu ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_sh_chebyu"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_sh_jacobi ?mu ~n ~p1 ~q1 () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_sh_jacobi"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int)); ("p1", Some(p1 |> Py.Float.of_float)); ("q1", Some(q1 |> Py.Float.of_float))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let roots_sh_legendre ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "roots_sh_legendre"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let round ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "round"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let s_roots ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "s_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let sh_chebyt ?monic ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "sh_chebyt"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])

let sh_chebyu ?monic ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "sh_chebyu"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])

let sh_jacobi ?monic ~n ~p ~q () =
   Py.Module.get_function_with_keywords __wrap_namespace "sh_jacobi"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int)); ("p", Some(p |> Py.Float.of_float)); ("q", Some(q |> Py.Float.of_float))])

let sh_legendre ?monic ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "sh_legendre"
     [||]
     (Wrap_utils.keyword_args [("monic", Wrap_utils.Option.map monic Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])

let shichi ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "shichi"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let sici ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "sici"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let sinc x =
   Py.Module.get_function_with_keywords __wrap_namespace "sinc"
     [||]
     (Wrap_utils.keyword_args [("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let sindg ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "sindg"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let smirnov ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "smirnov"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let smirnovi ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "smirnovi"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let softmax ?axis ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "softmax"
     [||]
     (Wrap_utils.keyword_args [("axis", Wrap_utils.Option.map axis (fun ml -> Py.Tuple.of_list_map Py.Int.of_int ml)); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let spence ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "spence"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let sph_harm ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "sph_harm"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

                  let spherical_in ?derivative ~n ~z () =
                     Py.Module.get_function_with_keywords __wrap_namespace "spherical_in"
                       [||]
                       (Wrap_utils.keyword_args [("derivative", Wrap_utils.Option.map derivative Py.Bool.of_bool); ("n", Some(n |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
))); ("z", Some(z ))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let spherical_jn ?derivative ~n ~z () =
                     Py.Module.get_function_with_keywords __wrap_namespace "spherical_jn"
                       [||]
                       (Wrap_utils.keyword_args [("derivative", Wrap_utils.Option.map derivative Py.Bool.of_bool); ("n", Some(n |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
))); ("z", Some(z ))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let spherical_kn ?derivative ~n ~z () =
                     Py.Module.get_function_with_keywords __wrap_namespace "spherical_kn"
                       [||]
                       (Wrap_utils.keyword_args [("derivative", Wrap_utils.Option.map derivative Py.Bool.of_bool); ("n", Some(n |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
))); ("z", Some(z ))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
                  let spherical_yn ?derivative ~n ~z () =
                     Py.Module.get_function_with_keywords __wrap_namespace "spherical_yn"
                       [||]
                       (Wrap_utils.keyword_args [("derivative", Wrap_utils.Option.map derivative Py.Bool.of_bool); ("n", Some(n |> (function
| `Ndarray x -> Np.Obj.to_pyobject x
| `I x -> Py.Int.of_int x
))); ("z", Some(z ))])
                       |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let stdtr ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "stdtr"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let stdtridf ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "stdtridf"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let stdtrit ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "stdtrit"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let struve ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "struve"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let t_roots ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "t_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let tandg ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "tandg"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let tklmbda ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "tklmbda"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let ts_roots ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "ts_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let u_roots ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "u_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let us_roots ?mu ~n () =
   Py.Module.get_function_with_keywords __wrap_namespace "us_roots"
     [||]
     (Wrap_utils.keyword_args [("mu", Wrap_utils.Option.map mu Py.Bool.of_bool); ("n", Some(n |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1)), (Py.Float.to_float (Py.Tuple.get x 2))))
let voigt_profile ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "voigt_profile"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("where", where)])

let wofz ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "wofz"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])

let wrightomega ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "wrightomega"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let xlog1py ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "xlog1py"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let xlogy ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "xlogy"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let y0 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "y0"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let y0_zeros ?complex ~nt () =
   Py.Module.get_function_with_keywords __wrap_namespace "y0_zeros"
     [||]
     (Wrap_utils.keyword_args [("complex", Wrap_utils.Option.map complex Py.Bool.of_bool); ("nt", Some(nt |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let y1 ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "y1"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let y1_zeros ?complex ~nt () =
   Py.Module.get_function_with_keywords __wrap_namespace "y1_zeros"
     [||]
     (Wrap_utils.keyword_args [("complex", Wrap_utils.Option.map complex Py.Bool.of_bool); ("nt", Some(nt |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let y1p_zeros ?complex ~nt () =
   Py.Module.get_function_with_keywords __wrap_namespace "y1p_zeros"
     [||]
     (Wrap_utils.keyword_args [("complex", Wrap_utils.Option.map complex Py.Bool.of_bool); ("nt", Some(nt |> Py.Int.of_int))])
     |> (fun x -> (((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 0)), ((fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t)) (Py.Tuple.get x 1))))
let yn ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "yn"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let yn_zeros ~n ~nt () =
   Py.Module.get_function_with_keywords __wrap_namespace "yn_zeros"
     [||]
     (Wrap_utils.keyword_args [("n", Some(n |> Py.Int.of_int)); ("nt", Some(nt |> Py.Int.of_int))])

let ynp_zeros ~n ~nt () =
   Py.Module.get_function_with_keywords __wrap_namespace "ynp_zeros"
     [||]
     (Wrap_utils.keyword_args [("n", Some(n |> Py.Int.of_int)); ("nt", Some(nt |> Py.Int.of_int))])

let yv ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "yv"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let yve ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "yve"
     (Array.of_list @@ List.concat [[x ]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let yvp ?n ~v ~z () =
   Py.Module.get_function_with_keywords __wrap_namespace "yvp"
     [||]
     (Wrap_utils.keyword_args [("n", Wrap_utils.Option.map n Py.Int.of_int); ("v", Some(v |> Py.Float.of_float)); ("z", Some(z ))])

let zeta ?q ?out ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "zeta"
     [||]
     (Wrap_utils.keyword_args [("q", Wrap_utils.Option.map q Np.Obj.to_pyobject); ("out", Wrap_utils.Option.map out Np.Obj.to_pyobject); ("x", Some(x |> Np.Obj.to_pyobject))])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
let zetac ?out ?where ~x () =
   Py.Module.get_function_with_keywords __wrap_namespace "zetac"
     (Array.of_list @@ List.concat [[x |> Np.Obj.to_pyobject]])
     (Wrap_utils.keyword_args [("out", out); ("where", where)])
     |> (fun py -> (Np.Obj.of_pyobject py : [`ArrayLike|`Ndarray|`Object] Np.Obj.t))
