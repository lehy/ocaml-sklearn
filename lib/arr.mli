type t

(** ### show

    Pretty-print an Array into a string. *)
val show : t -> string

(** ### pp

    Pretty-print the Array.  *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]

(** ### get

    Get the underlying dense Ndarray or sparse Csr_matrix. No data is
    copied. *)
val get : t -> [`Ndarray of Ndarray.t | `Csr_matrix of Csr_matrix.t]

(** ### get_ndarray

    Get the underlying dense Ndarray, or raise Invalid_argument if the
    underlying array is sparse.  *)
val get_ndarray : t -> Ndarray.t

(** ### get_csr_matrix

    Get the underlying sparse Csr_matrix, or raise Invalid_argument if the
    underlying array is dense.  *)
val get_csr_matrix : t -> Csr_matrix.t

(** ### of_ndarray

    Build an Array.t from a dense Ndarray.t. Data is shared. *)
val of_ndarray : Ndarray.t -> t

(** ### of_csr_matrix

    Build an Array.t from a Csr_matrix.t. Data is shared. *)
val of_csr_matrix : Csr_matrix.t -> t

(** ### of_bigarray

    Build an Array from a bigarray. *)
val of_bigarray : ('a, 'b, 'c) Bigarray.Genarray.t -> t

(** ## module Array.List

    This is a Python list of Arrays. This is `Sklearn.PyList.Make(Sklearn.Array)`. *)
module List : PyList.S with type elt := t

module Dtype : sig
  type t = [`Object | `S of string]
  val to_pyobject : t -> Py.Object.t
end

(** ### arange

    ~~~python
    arange([start,] stop[, step,], dtype=None)
    ~~~

    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including `start` but excluding `stop`).
    For integer arguments the function is equivalent to the Python built-in
    `range` function, but returns an ndarray rather than a list.

    When using a non-integer step, such as 0.1, the results will often not
    be consistent.  It is better to use `numpy.linspace` for these cases.

    #### Parameters

    ???+ info "start : number, optional"
        Start of interval.  The interval includes this value.  The default
        start value is 0.

    ???+ "stop : number"
        End of interval.  The interval does not include this value, except
        in some cases where `step` is not an integer and floating point
        round-off affects the length of `out`.

    ???+ info "step : number, optional"
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified as a position argument,
        `start` must also be given.

    ???+ info "dtype : dtype"
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.

    #### Returns

    ???+ info "arange : ndarray"
        Array of evenly spaced values.

    For floating point arguments, the length of the result is
    ``ceil((stop - start)/step)``.  Because of floating point overflow,
    this rule may result in the last element of `out` being greater
    than `stop`.

    #### See Also

    numpy.linspace : Evenly spaced numbers with careful handling of endpoints.

    numpy.ogrid: Arrays of evenly spaced numbers in N-dimensions.

    numpy.mgrid: Grid-shaped arrays of evenly spaced numbers in N-dimensions.

    #### Examples

    ~~~python
    >>> np.arange(3)
    array([0, 1, 2])
    >>> np.arange(3.0)
    array([ 0.,  1.,  2.])
    >>> np.arange(3,7)
    array([3, 4, 5, 6])
    >>> np.arange(3,7,2)
    array([3, 5])
    ~~~
*)
val arange : ?start : int -> ?step : int -> int -> t

val ones : ?dtype : Dtype.t -> int list -> t
val zeros : ?dtype : Dtype.t -> int list -> t

val set : [`Colon | `I of int] array -> [`I of int | `F of float | `S of string] -> t -> unit
val get_int : int list -> t -> int
val get_float : int list -> t -> float

(* XXX should we expose this? not sure it's reasonable, because
   get_sub on a Csr_matrix can return something that is not a
   Csr_matrix I think; also should it return `Float of f | `Arr of t ?
   *)
val slice : ?i : int -> ?j : int -> ?step : int -> unit -> [`Slice of Wrap_utils.Slice.t]
val get_sub : [`I of int | `Slice of Wrap_utils.Slice.t] list -> t -> t

module Ops : sig
  val int : int -> t
  val float : float -> t
  val bool : bool -> t
  val string : string -> t
  
  val ( - ) : t -> t -> t
  val ( + ) : t -> t -> t
  val ( * ) : t -> t -> t
  val ( / ) : t -> t -> t
  val ( > ) : t -> t -> t
  val ( >= ) : t -> t -> t
  val ( < ) : t -> t -> t
  val ( <= ) : t -> t -> t
  val ( = ) : t -> t -> t
  val ( != ) : t -> t -> t
end

(** ## module Array.Float

    Build a dense Array containing floats. *)
module Float : sig

  (** ### vector

      Build a vector from an OCaml float array. *)
  val vector : float array -> t

  (** ### matrix

      Build a dense matrix from an OCaml float array array. *)
  val matrix : float array array -> t

  (** ### of_bigarray

      Build a dense tensor from a bigarray. The data is not copied,
      and is shared between the bigarray and the Pyhon Array. You
      may find Owl useful for building the bigarray. *)
  val of_bigarray : (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Genarray.t -> t

  (** ### matrices

      Build a Python list of Arrays, with each Array being a
      matrix initialized from an OCaml float array array. *)
  val matrices : float array array list -> List.t

  (** ### vectors

      Build a Python list of Arrays, with each Array being a
      vector initialized from an OCaml float array. *)
  val vectors : float array list -> List.t
end

(** ## module Array.Int

    Build a dense Array containing integers. The
    integers are actually stored as nativeints inside the Array. *)
module Int : sig

  (** ### vector

      Build a vector from an OCaml int array. *)
  val vector : int array -> t

  (** ### matrix

      Build a dense matrix from an OCaml int array array. *)
  val matrix : int array array -> t

  (** ### of_bigarray

      Build a dense tensor from a bigarray. The data is not copied,
      and is shared between the bigarray and the Pyhon Array. You
      may find Owl useful for building the bigarray. *)
  val of_bigarray : (nativeint, Bigarray.nativeint_elt, Bigarray.c_layout) Bigarray.Genarray.t -> t

  (** ### matrices

      Build a Python list of Arrays, with each Array being a
      matrix initialized from an OCaml int array array. *)
  val matrices : int array array list -> List.t

  (** ### vectors

      Build a Python list of Arrays, with each Array being a
      vector initialized from an OCaml int array. *)
  val vectors : int array list -> List.t
end

(** ## module Array.String

    Build an Array containing strings. *)
module String : sig

  (** ### vector

      Build a vector from an OCaml string array.

      Example :

      ~~~ocaml
      let x = Array.String.vector [|"a"; "answer"|]
      ~~~
  *)
  val vector : string array -> t

  (** ### matrix

      Build a matrix from an OCaml array of arrays.

      Example :

      ~~~ocaml
      let x = Array.String.matrix [| [|"a"; "answer"|]; [|"b"; `"lala"|] |]
      ~~~
  *)
  val matrix : string array array -> t

  (** ### vectors

      Build a Python list of Arrays, with each Array being a
      vector initialized from an OCaml string array.

      Example :

      ~~~ocaml
      let x = Array.String.matrix [| [|"a"|]; [|"b"; `"lala"; "d"|] |]
      ~~~
  *)
  val vectors : string array list -> List.t
end

(** ## module Arr.Object

    Build an Arr containing mixed ints, floats or strings. *)
module Object : sig
  (**
     The type of an element: int (`I), float (`F) or string (`S).
  *)
  type elt = [`I of int | `F of float | `S of string]

  (** ### vector

      Build a vector from an OCaml array.

      Example :

      ~~~ocaml
      let x = Arr.Object.vector [| `I 42; `S "answer"; `F 12.3 |]
      ~~~
  *)
  val vector : elt array -> t

  (** ### matrix

      Build a matrix from an OCaml array of arrays.

      Example :

      ~~~ocaml
      let x = Arr.Object.matrix [| [|`I 42; `S "answer"|]; [|`I 43; `S "lala"|] |]
      ~~~
  *)
  val matrix : elt array array -> t
end

(** ### to_pyobject

    Convert the Array to a Py.Object.t. *)
val to_pyobject : t -> Py.Object.t

(** ### of_pyobject

    Build an Array from a Py.Object.t.  *)
val of_pyobject : Py.Object.t -> t
