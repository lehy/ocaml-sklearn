let pp_arr = Sklearn.Arr.pp Format.std_formatter;;

let%expect_test "arr_shape" =
  let open Sklearn in
  Arr.(ones [3; 4] |> pp_arr);
  [%expect {|
    [[1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]] |}];
  Arr.(zeros [] |> pp_arr);
  [%expect {| 0.0 |}];
  Arr.(zeros [5; 6] |> pp_arr);
  [%expect {|
    [[0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0.]] |}];
  Arr.(
    let o = ones [2; 3] in
    let z = zeros [5; 6] in
    set ~i:[slice ~i:1 ~j:3 (); slice ~i:2 ~j:5 ()] ~v:o z;
    pp_arr z;
    [%expect {|
      [[0. 0. 0. 0. 0. 0.]
       [0. 0. 1. 1. 1. 0.]
       [0. 0. 1. 1. 1. 0.]
       [0. 0. 0. 0. 0. 0.]
       [0. 0. 0. 0. 0. 0.]] |}];
    get ~i:[slice ~i:1 ~j:3 (); slice ~i:3 ~j:6 ()] z |> pp_arr;
    [%expect {|
      [[1. 1. 0.]
       [1. 1. 0.]] |}]);
  Arr.(get ~i:[`I 1; `I 2] (ones [2; 3]) |> shape |> Int.vector |> pp_arr);
  [%expect {| [] |}]

