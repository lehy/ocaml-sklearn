(**
   This represents a Numpy array. In most cases it is a dense
   ndarray. But it max also be a sparse matrix (as built using
   Csr_matrix, or as returned by scikit-learn).  *)

type t

(** ### show

    Pretty-print an Array into a string. *)
val show : t -> string

(** ### pp

    Pretty-print the Array.  *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]

(** ### of_csr_matrix

    Build an Array.t from a Csr_matrix.t. Data is shared. *)
val of_csr_matrix : Csr_matrix.t -> t

(** ### of_bigarray

    Build an Array from a bigarray. *)
val of_bigarray : ('a, 'b, 'c) Bigarray.Genarray.t -> t

val to_int_array : t -> int array
val to_float_array : t -> float array

(** ## module Array.List

    This is a Python list of Arrays. This is `Sklearn.PyList.Make(Sklearn.Array)`. *)
module List : PyList.S with type elt := t

module Dtype : sig
  type t = [`Object | `S of string]
  val to_pyobject : t -> Py.Object.t
  val of_pyobject : Py.Object.t -> t
end

(** ### shape

    Shape (dimensions) of an Arr. *)
val shape : t -> int array

val reshape : shape:int array -> t -> t

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

val min : t -> float
val max : t -> float

val argsort : t -> t

val ones : ?dtype : Dtype.t -> int list -> t
val zeros : ?dtype : Dtype.t -> int list -> t

val get_int : i:int list -> t -> int
val get_float : i:int list -> t -> float

val slice : ?i : int -> ?j : int -> ?step : int -> unit -> [> `Slice of Wrap_utils.Slice.t]
val get : i:[`I of int | `Slice of Wrap_utils.Slice.t | `Arr of t | `Newaxis | `Ellipsis] list -> t -> t
val set : i:[`I of int | `Slice of Wrap_utils.Slice.t | `Arr of t | `Newaxis | `Ellipsis] list -> v:t -> t -> unit

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

  (** ### to_bigarray

      Build a bigarray that shares the same data as the
      Arr. Raises an exception if the Arr has the wrong dtype
      or layout, or if the Arr is not an ndarray. *)
  val to_bigarray : t -> (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Genarray.t

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

  (** ### to_bigarray

      Build a bigarray that shares the same data as the
      Ndarray. Raises an exception if the Ndarray has the wrong dtype
      or layout, or if the Arr is not an ndarray. *)
  val to_bigarray : t -> (nativeint, Bigarray.nativeint_elt, Bigarray.c_layout) Bigarray.Genarray.t

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
  type elt = [`I of int | `F of float | `S of string | `Arr of t]

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

val argmax : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> t -> [`Arr of t | `I of int]
(**
   Return indices of maximum elements along an axis.

   Implicit zero elements are also taken into account. If there are
   several maximum values, the index of the first occurrence is returned.

   Parameters
   ----------
   axis : {-2, -1, 0, 1, None}, optional
    Axis along which the argmax is computed. If None (default), index
    of the maximum element in the flatten data is returned.
   out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except for
    the default value, as this argument is not used.

   Returns
   -------
   ind : numpy.matrix or int
    Indices of maximum elements. If matrix, its size along `axis` is 1.
*)

val argmin : ?axis:[`Zero | `One | `PyObject of Py.Object.t] -> ?out:Py.Object.t -> t -> [`Arr of t | `I of int]
(**
   Return indices of minimum elements along an axis.

   Implicit zero elements are also taken into account. If there are
   several minimum values, the index of the first occurrence is returned.

   Parameters
   ----------
   axis : {-2, -1, 0, 1, None}, optional
    Axis along which the argmin is computed. If None (default), index
    of the minimum element in the flatten data is returned.
   out : None, optional
    This argument is in the signature *solely* for NumPy
    compatibility reasons. Do not pass in anything except for
    the default value, as this argument is not used.

   Returns
   -------
   ind : numpy.matrix or int
    Indices of minimum elements. If matrix, its size along `axis` is 1.
*)

val mean : ?axis:int list -> ?dtype:Dtype.t -> ?out:t -> ?keepdims:bool -> t -> t

val sum : ?axis:int -> ?dtype:Py.Object.t -> ?out:t -> t -> t
(**
   Sum the matrix elements over a given axis.

   Parameters
   ----------
   axis : {-2, -1, 0, 1, None} optional
    Axis along which the sum is computed. The default is to
    compute the sum of all the matrix elements, returning a scalar
    (i.e. `axis` = `None`).
   dtype : dtype, optional
    The type of the returned matrix and of the accumulator in which
    the elements are summed.  The dtype of `a` is used by default
    unless `a` has an integer dtype of less precision than the default
    platform integer.  In that case, if `a` is signed then the platform
    integer is used while if `a` is unsigned then an unsigned integer
    of the same precision as the platform integer is used.

    .. versionadded:: 0.18.0

   out : np.matrix, optional
    Alternative output matrix in which to place the result. It must
    have the same shape as the expected output, but the type of the
    output values will be cast if necessary.

    .. versionadded:: 0.18.0

   Returns
   -------
   sum_along_axis : np.matrix
    A matrix with the same shape as `self`, with the specified
    axis removed.

   See Also
   --------
   numpy.matrix.sum : NumPy's implementation of 'sum' for matrices
*)

val asarray : ?dtype:Dtype.t -> t -> t
val asanyarray : ?dtype:Dtype.t -> t -> t
val ascontiguousarray : ?dtype:Dtype.t -> t -> t
val asfarray : ?dtype:Dtype.t -> t -> t
val asarray_chkfinite : ?dtype:Dtype.t -> t -> t

val toarray : t -> t
val todense : t -> t

val vstack : t list -> t
val hstack : t list -> t
val dstack : t list -> t

val full : ?dtype:Dtype.t -> shape:int list -> Object.elt -> t

val flatnonzero : t -> t

val iter : t -> t Seq.t

module Random : sig
  val seed : int -> unit
  val random_sample : int list -> t
end
