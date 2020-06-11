module type Numpy_no_ndarray = module type of struct
  include NumpyRaw
end
with module Ndarray := NumpyRaw.Ndarray

include module type of (NumpyRaw : Numpy_no_ndarray)

module Slice : sig
  type t = Wrap_utils.Slice.t

  val to_pyobject : t -> Py.Object.t
  val of_pyobject : Py.Object.t -> t
end

module Index : sig
  module Element : sig
    type t = [
      | `I of int
      | `Slice of Slice.t
      | `Arr of [`Ndarray] Obj.t
      | `Newaxis
      | `Ellipsis
    ]
    val to_pyobject : t -> Py.Object.t
  end

  type t = Element.t list
  val to_pyobject : t -> Py.Object.t
end

module Ndarray : sig
  include module type of NumpyRaw.Ndarray

  type ndarray = t

  val int : int -> t
  val of_int_list : int list -> t
  val of_int_array : int array -> t
  val vectori : int array -> t
  val matrixi : int array array -> t
  val to_int_array : t -> int array

  val float : float -> t
  val of_float_list : float list -> t
  val of_float_array : float array -> t
  val vectorf : float array -> t
  val matrixf : float array array -> t
  val to_float_array : t -> float array

  val string : string -> t
  val of_string_list : string list -> t
  val of_string_array : string array -> t
  val vectors : string array -> t
  val matrixs : string array array -> t
  val to_string_array : t -> string array

  val of_object_list : [`I of int | `F of float | `S of string | `B of bool] list -> t
  val of_object_array : [`I of int | `F of float | `S of string | `B of bool] array -> t
  val vectoro : [`I of int | `F of float | `S of string | `B of bool] array -> t
  val matrixo : [`I of int | `F of float | `S of string | `B of bool] array array -> t

  val bool : bool -> t
  val iter : t -> t Seq.t

  val slice : ?i:int -> ?j:int -> ?step:int -> unit -> [> `Slice of Slice.t]
  val mask : [> `Ndarray] Obj.t -> Index.Element.t

  module List : sig
    type t
    
    val of_pyobject : Py.Object.t -> t
    val to_pyobject : t -> Py.Object.t

    val create : unit -> t
    val of_list : ndarray list -> t

    val of_list_map : ('a -> ndarray) -> 'a list -> t

    val append : t -> ndarray -> unit

    val show : t -> string
    val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]

    val vectori : int array list -> t
    val vectors : string array list -> t
  end
end

val slice : ?i:int -> ?j:int -> ?step:int -> unit -> [> `Slice of Slice.t]
val mask : [> `Ndarray] Obj.t -> Index.Element.t

val int : int -> Ndarray.t
val vectori : int array -> Ndarray.t
val matrixi : int array array -> Ndarray.t

val float : float -> Ndarray.t
val vectorf : float array -> Ndarray.t
val matrixf : float array array -> Ndarray.t

val string : string -> Ndarray.t
val vectors : string array -> Ndarray.t
val matrixs : string array array -> Ndarray.t

val vectoro : [`I of int | `F of float | `S of string | `B of bool] array -> Ndarray.t
val matrixo : [`I of int | `F of float | `S of string | `B of bool] array array -> Ndarray.t

val bool : bool -> Ndarray.t

val ( - ) : Ndarray.t -> Ndarray.t -> Ndarray.t
val ( + ) : Ndarray.t -> Ndarray.t -> Ndarray.t
val ( * ) : Ndarray.t -> Ndarray.t -> Ndarray.t
val ( / ) : Ndarray.t -> Ndarray.t -> Ndarray.t

val ( < ) : Ndarray.t -> Ndarray.t -> Ndarray.t
val ( <= ) : Ndarray.t -> Ndarray.t -> Ndarray.t
val ( > ) : Ndarray.t -> Ndarray.t -> Ndarray.t
val ( >= ) : Ndarray.t -> Ndarray.t -> Ndarray.t

val ( = ) : Ndarray.t -> Ndarray.t -> Ndarray.t
val ( != ) : Ndarray.t -> Ndarray.t -> Ndarray.t

val pp : Format.formatter -> Ndarray.t -> unit

val arange : ?start:[`I of int | `F of float] -> ?step:[`I of int | `F of float] -> ?dtype:[`Object | `S of string] -> [`I of int | `F of float] -> Ndarray.t

module Obj : sig
  type -'a t = 'a Obj.t

  val to_pyobject : 'a t -> Py.Object.t
  val of_pyobject : Py.Object.t -> 'a t
  val print : 'a t -> unit
  val to_string : 'a t -> string
  val pp : Format.formatter -> 'a t -> unit [@@ocaml.toplevel_printer]
end
