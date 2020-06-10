(* let pp_arr = Sklearn.Arr.pp Format.std_formatter;; *)

let pp_arr = Np.Obj.print
(* module Np = Np.Ops *)

let%expect_test "arr_shape" =
  let open Np.Numpy in
  ones [3; 4] |> pp_arr;
  [%expect {|
    [[1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]] |}];
  zeros [] |> pp_arr;
  [%expect {| 0.0 |}];
  zeros [5; 6] |> pp_arr;
  [%expect {|
    [[0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0.]] |}];
  let o = ones [2; 3] in
  let z = zeros [5; 6] in
  Ndarray.set ~key:[slice ~i:1 ~j:3 (); slice ~i:2 ~j:5 ()] ~value:o z;
  pp_arr z;
  [%expect {|
      [[0. 0. 0. 0. 0. 0.]
       [0. 0. 1. 1. 1. 0.]
       [0. 0. 1. 1. 1. 0.]
       [0. 0. 0. 0. 0. 0.]
       [0. 0. 0. 0. 0. 0.]] |}];
  Ndarray.get ~key:[slice ~i:1 ~j:3 (); slice ~i:3 ~j:6 ()] z |> pp_arr;
  [%expect {|
      [[1. 1. 0.]
       [1. 1. 0.]] |}];
  Ndarray.(get ~key:[`I 1; `I 2] (ones [2; 3]) |> shape |> of_int_list |> pp_arr);
  [%expect {| [] |}]

let%expect_test "of_int_list" =
  let open Np.Numpy in
  pp_arr @@ Ndarray.of_int_list [1;2;3];
  [%expect {| [1 2 3] |}]

let%expect_test "of_float_list" =
  let open Np.Numpy in
  pp_arr @@ Ndarray.of_float_list [1.;2.;3.];
  [%expect {| [1. 2. 3.] |}]

let%expect_test "matrixi" =
  let open Np.Numpy in
  pp_arr @@ Ndarray.matrixi [| [|1;2;3|]; [|4;5;6|]|];
  [%expect {|
    [[1 2 3]
     [4 5 6]] |}]


let%expect_test "matrixf" =
  let open Np.Numpy in
  pp_arr @@ Ndarray.matrixf [| [|1.;2.;3.|]; [|4.;5.;6.|]|];
  [%expect {|
    [[1. 2. 3.]
     [4. 5. 6.]] |}]
