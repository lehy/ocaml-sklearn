module Np = Np.Numpy

let print f x = Format.printf "%a" f x

let print_py x = Format.printf "%s" (Py.Object.to_string x)

let print_ndarray = Np.Obj.print

let print_float = Format.printf "%g\n"

let print_string = Format.printf "%s\n"

let print_int = Format.printf "%d\n"

(* BernoulliRBM *)
(*
>>> import numpy as np
>>> from sklearn.neural_network import BernoulliRBM
>>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
>>> model = BernoulliRBM(n_components=2)
>>> model.fit(X)
BernoulliRBM(n_components=2)

*)


let%expect_test "BernoulliRBM" =
  let open Sklearn.Neural_network in
  let x = Np.matrixi [|[|0; 0; 0|]; [|0; 1; 1|]; [|1; 0; 1|]; [|1; 1; 1|]|] in
  let model = BernoulliRBM.create ~n_components:2 () in
  print BernoulliRBM.pp @@ BernoulliRBM.fit ~x model;
  [%expect {|
      BernoulliRBM(n_components=2)
  |}]
