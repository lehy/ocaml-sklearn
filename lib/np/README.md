# Numpy for OCaml

These are bindings to [Numpy](https://numpy.org) for OCaml. They
intend to cover all of Numpy's functionalities.

## Example

~~~ocaml
module Np = Np.Numpy

let print arr = Format.printf "%a\n" Np.pp arr

let%expect_test "arr_shape" =
  Np.(ones [3; 4] |> print);
  [%expect {|
    [[1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]] |}];

  Np.(zeros [] |> print);
  [%expect {| 0.0 |}];

  Np.(zeros [5; 6] |> print);
  [%expect {|
    [[0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0.]] |}];

  let o = Np.ones [2; 3] in
  let z = Np.zeros [5; 6] in
  Np.(Ndarray.set ~key:[slice ~i:1 ~j:3 (); slice ~i:2 ~j:5 ()] ~value:o z);
  print z;
  [%expect {|
      [[0. 0. 0. 0. 0. 0.]
       [0. 0. 1. 1. 1. 0.]
       [0. 0. 1. 1. 1. 0.]
       [0. 0. 0. 0. 0. 0.]
       [0. 0. 0. 0. 0. 0.]] |}];

  Np.(Ndarray.get ~key:[slice ~i:1 ~j:3 (); slice ~i:3 ~j:6 ()] z) |> print;
  [%expect {|
      [[1. 1. 0.]
       [1. 1. 0.]] |}];

  Np.(Ndarray.(get ~key:[`I 1; `I 2] (ones [2; 3]) |> shape |> of_int_list)) |> print;
  [%expect {| [] |}]
~~~

## Note about naming

The package is named `np` to avoid conflicting with an existing module
of the same name from `pyml`. I suggest putting the following at the
top of any file using this:

~~~ocaml
module Np = Np.Numpy
~~~

## How to read the documentation.

Module [`Np.NumpyRaw`](NumpyRaw.md) is included in module `Np.Numpy`; it contains the
automatically generated bindings. Consult it to get the provided OCaml
API (with documentation extracted from the Python library).

Module [`Np.Numpy`](Numpy.md) has some manual additions to make things easier to
use from OCaml. Consult its documentation to see them.

## Installation

~~~sh
opam install np
~~~
