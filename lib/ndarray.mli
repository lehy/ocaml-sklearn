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
val pp : Format.formatter -> t -> unit

(** ### to_pyobject

    Convert the Ndarray to a Py.Object.t. *)
val to_pyobject : t -> Py.Object.t

(** ### of_pyobject

    Build an Ndarray from a Py.Object.t.  *)
val of_pyobject : Py.Object.t -> t

(** ### of_bigarray

    Build an Ndarray from a bigarray. *)
val of_bigarray : ('a, 'b, 'c) Bigarray.Genarray.t -> t

(** ### shape

    Shape (dimensions) of an Ndarray. *)
val shape : t -> int array

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

      Build a vector from an OCaml string array. *)
  val vector : string array -> t

  (** ### vectors

      Build a Python list of Ndarrays, with each Ndarray being a
      vector initialized from an OCaml string array. *)
  val vectors : string array list -> List.t
end
