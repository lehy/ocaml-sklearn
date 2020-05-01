(**

Ndarray.t represents a tensor (a multidimensional array), which can
contain integers, floats and strings.

You can **build an Ndarray from an OCaml array** :

```ocaml
let nda = Sklearn.Ndarray.Float.matrix [| [|1;2;3|]; [|4;5;6|] |] in
Format.printf "%a" Ndarray.pp nda;;
```

You can also **build it from a bigarray**. This way, no data is copied,
the bigarray and the Ndarray share the same memory. You may want to use
the library Owl to build the bigarray (this is useful because the interface
provided here is minimal).

 ```ocaml
let module Matrix = Owl.Dense.Matrix.D in
let x = Sklearn.Ndarray.of_bigarray @@ Matrix.uniform 10 3 in
Format.printf "%a" Ndarray.pp x;;
```

Finally, this module provides functions for **building a Python list of
Ndarrays**, which is useful for functions expecting a collection of
arrays which do not have the same length (and therefore do not fit
into one tensor).

```ocaml
let y = Ndarray.String.vectors [ [|"hello"; "world"|]; [|"how"; "are"; "you"; "gentlemen"|] in
Format.printf "%a" Ndarray.pp y;;
```
*)
type t

(** ### show

    Pretty-print an Ndarray into a string. *)
val show : t -> string

(** ### pp

    Pretty-print the Ndarray.  *)
val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]

(** ### of_bigarray

    Build an Ndarray from a bigarray. *)
val of_bigarray : ('a, 'b, 'c) Bigarray.Genarray.t -> t

(** ### shape

    Shape (dimensions) of an Ndarray. *)
val shape : t -> int array

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

(** ### reshape

    Gives a new shape to an array without changing its data.

    #### Parameters

    ???+ info "a : array_like"
        Array to be reshaped.

    ???+ info : "newshape : int or tuple of ints"
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is
        inferred from the length of the array and remaining dimensions.

    ???+ info "order : {'C', 'F', 'A'}, optional"
        Read the elements of `a` using this index order, and place the
        elements into the reshaped array using this index order.  'C'
        means to read / write the elements using C-like index order,
        with the last axis index changing fastest, back to the first
        axis index changing slowest. 'F' means to read / write the
        elements using Fortran-like index order, with the first index
        changing fastest, and the last index changing slowest. Note that
        the 'C' and 'F' options take no account of the memory layout of
        the underlying array, and only refer to the order of indexing.
        'A' means to read / write the elements in Fortran-like index
        order if `a` is Fortran *contiguous* in memory, C-like order
        otherwise.

    #### Returns

    ???+ info "reshaped_array : ndarray"
        This will be a new view object if possible; otherwise, it will
        be a copy.  Note there is no guarantee of the *memory layout* (C- or
        Fortran- contiguous) of the returned array.

    #### See Also

    ndarray.reshape : Equivalent method.

    #### Notes

    It is not always possible to change the shape of an array without
    copying the data. If you want an error to be raised when the data is copied,
    you should assign the new shape to the shape attribute of the array::

    ~~~python
    >>> a = np.zeros((10, 2))
    ~~~

    ~~~python
    # A transpose makes the array non-contiguous
    >>> b = a.T
    ~~~

    ~~~python
    # Taking a view makes it possible to modify the shape without modifying
    # the initial object.
    >>> c = b.view()
    >>> c.shape = (20)
    Traceback (most recent call last):
    ...
    AttributeError: incompatible shape for a non-contiguous array

    The `order` keyword gives the index ordering both for *fetching* the values
    from `a`, and then *placing* the values into the output array.
    For example, let's say you have an array:
    ~~~

    ~~~python
    >>> a = np.arange(6).reshape((3, 2))
    >>> a
    array([[0, 1],
       [2, 3],
       [4, 5]])
    ~~~

    You can think of reshaping as first raveling the array (using the given
    index order), then inserting the elements from the raveled array into the
    new array using the same kind of index ordering as was used for the
    raveling.

    ~~~python
    >>> np.reshape(a, (2, 3)) # C-like index ordering
    ...skipping...
    ~~~

    reshaped_array : ndarray
    This will be a new view object if possible; otherwise, it will
    be a copy.  Note there is no guarantee of the *memory layout* (C- or
    Fortran- contiguous) of the returned array.

    #### See Also

    ndarray.reshape : Equivalent method.

    #### Notes

    It is not always possible to change the shape of an array without
    copying the data. If you want an error to be raised when the data is copied,
    you should assign the new shape to the shape attribute of the array::

    ~~~python
    >>> a = np.zeros((10, 2))

    # A transpose makes the array non-contiguous
    >>> b = a.T
    ~~~

    ~~~python
    # Taking a view makes it possible to modify the shape without modifying
    # the initial object.
    >>> c = b.view()
    >>> c.shape = (20)
    Traceback (most recent call last):
    ...
    AttributeError: incompatible shape for a non-contiguous array
    ~~~

    The `order` keyword gives the index ordering both for *fetching* the values
    from `a`, and then *placing* the values into the output array.
    For example, let's say you have an array:

    ~~~python
    >>> a = np.arange(6).reshape((3, 2))
    >>> a
    array([[0, 1],
       [2, 3],
       [4, 5]])
    ~~~

    You can think of reshaping as first raveling the array (using the given
    index order), then inserting the elements from the raveled array into the
    new array using the same kind of index ordering as was used for the
    raveling.

    ~~~python
    >>> np.reshape(a, (2, 3)) # C-like index ordering
    array([[0, 1, 2],
       [3, 4, 5]])
    >>> np.reshape(np.ravel(a), (2, 3)) # equivalent to C ravel then C reshape
    array([[0, 1, 2],
       [3, 4, 5]])
    >>> np.reshape(a, (2, 3), order='F') # Fortran-like index ordering
    array([[0, 4, 3],
       [2, 1, 5]])
    >>> np.reshape(np.ravel(a, order='F'), (2, 3), order='F')
    array([[0, 4, 3],
       [2, 1, 5]])
    ~~~

    #### Examples

    ~~~python
    >>> a = np.array([[1,2,3], [4,5,6]])
    >>> np.reshape(a, 6)
    array([1, 2, 3, 4, 5, 6])
    >>> np.reshape(a, 6, order='F')
    array([1, 4, 2, 5, 3, 6])

    >>> np.reshape(a, (3,-1))       # the unspecified value is inferred to be 2
    array([[1, 2],
       [3, 4],
       [5, 6]])
    ~~~
*)
val reshape : shape : int array -> t -> t

val set : [`Colon | `I of int] array -> [`I of int | `F of float | `S of string | `Arr of t] -> t -> unit
val get_int : int list -> t -> int
val get_float : int list -> t -> float

val slice : ?i : int -> ?j : int -> ?step : int -> unit -> [> `Slice of Wrap_utils.Slice.t]
val get_sub : [`I of int | `Slice of Wrap_utils.Slice.t | `Arr of t] list -> t -> t

val to_int_array : t -> int array
val to_float_array : t -> float array

val ravel : t -> t

(* TODO provide axis arg on these  *)
val min : t -> float
val max : t -> float

val argsort : t -> t

(** ## module Ndarray.List

    This is a Python list of Ndarrays. This is `Sklearn.PyList.Make(Sklearn.Ndarray)`. *)
module List : PyList.S with type elt := t

(** ## module Ndarray.Float

    Build an Ndarray containing floats. *)
module Float : sig

  (** ### vector

      Build a vector from an OCaml float array. *)
  val vector : float array -> t

  (** ### matrix

      Build a dense matrix from an OCaml float array array. *)
  val matrix : float array array -> t

  (** ### of_bigarray

      Build a dense tensor from a bigarray. The data is not copied,
      and is shared between the bigarray and the Pyhon Ndarray. You
      may find Owl useful for building the bigarray. *)
  val of_bigarray : (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Genarray.t -> t

  (** ### to_bigarray

      Build a bigarray that shares the same data as the
      Ndarray. Raises an exception if the Ndarray has the wrong dtype
      or layout. *)
  val to_bigarray : t -> (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Genarray.t
  
  (** ### matrices

      Build a Python list of Ndarrays, with each Ndarray being a
      matrix initialized from an OCaml float array array. *)
  val matrices : float array array list -> List.t

  (** ### vectors

      Build a Python list of Ndarrays, with each Ndarray being a
      vector initialized from an OCaml float array. *)
  val vectors : float array list -> List.t
end

(** ## module Ndarray.Int

    Build an Ndarray containing integers. The
    integers are actually stored as nativeints inside the Ndarray. *)
module Int : sig
  
  (** ### vector

      Build a vector from an OCaml int array. *)
  val vector : int array -> t

  (** ### matrix

      Build a dense matrix from an OCaml int array array. *)
  val matrix : int array array -> t

  (** ### of_bigarray

      Build a dense tensor from a bigarray. The data is not copied,
      and is shared between the bigarray and the Pyhon Ndarray. You
      may find Owl useful for building the bigarray. *)
  val of_bigarray : (nativeint, Bigarray.nativeint_elt, Bigarray.c_layout) Bigarray.Genarray.t -> t

  (** ### to_bigarray

      Build a bigarray that shares the same data as the
      Ndarray. Raises an exception if the Ndarray has the wrong dtype
      or layout. *)
  val to_bigarray : t -> (nativeint, Bigarray.nativeint_elt, Bigarray.c_layout) Bigarray.Genarray.t

  (** ### matrices

      Build a Python list of Ndarrays, with each Ndarray being a
      matrix initialized from an OCaml int array array. *)
  val matrices : int array array list -> List.t

  (** ### vectors

      Build a Python list of Ndarrays, with each Ndarray being a
      vector initialized from an OCaml int array. *)
  val vectors : int array list -> List.t
end

(** ## module Ndarray.String

    Build an Ndarray containing strings. *)
module String : sig

  (** ### vector

      Build a vector from an OCaml string array.

      Example :

      ~~~ocaml
      let x = Ndarray.String.vector [|"a"; "answer"|]
      ~~~
  *)
  val vector : string array -> t

  (** ### matrix

      Build a matrix from an OCaml array of arrays.

      Example :

      ~~~ocaml
      let x = Ndarray.String.matrix [| [|"a"; "answer"|]; [|"b"; `"lala"|] |]
      ~~~
  *)
  val matrix : string array array -> t
  
  (** ### vectors

      Build a Python list of Ndarrays, with each Ndarray being a
      vector initialized from an OCaml string array.

      Example :

      ~~~ocaml
      let x = Ndarray.String.matrix [| [|"a"|]; [|"b"; `"lala"; "d"|] |]
      ~~~
  *)
  val vectors : string array list -> List.t
end

(** ## module Ndarray.Object

    Build an Ndarray containing mixed ints, floats or strings. *)
module Object : sig
  (**
     The type of an element: int (`I), float (`F) or string (`S).
  *)
  type elt = [`I of int | `F of float | `S of string | `Arr of t]

  (** ### vector

      Build a vector from an OCaml array.

      Example :
      
      ~~~ocaml
      let x = Ndarray.Object.vector [| `I 42; `S "answer"; `F 12.3 |]
      ~~~
  *)
  val vector : elt array -> t

  (** ### matrix

      Build a matrix from an OCaml array of arrays.

      Example :

      ~~~ocaml
      let x = Ndarray.Object.matrix [| [|`I 42; `S "answer"|]; [|`I 43; `S "lala"|] |]
      ~~~
 *)
  val matrix : elt array array -> t
end

(** ### to_pyobject

    Convert the Ndarray to a Py.Object.t. *)
val to_pyobject : t -> Py.Object.t

(** ### of_pyobject

    Build an Ndarray from a Py.Object.t.  *)
val of_pyobject : Py.Object.t -> t

