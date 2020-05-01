(* BernoulliRBM *)
(*
>>> import numpy as np
>>> from sklearn.neural_network import BernoulliRBM
>>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
>>> model = BernoulliRBM(n_components=2)
>>> model.fit(X)
BernoulliRBM(n_components=2)

*)

(* TEST TODO
let%expect_test "BernoulliRBM" =
  let open Sklearn.Neural_network in
  let x = .array (matrixi [|[|0; 0; 0|]; [|0; 1; 1|]; [|1; 0; 1|]; [|1; 1; 1|]|]) np in  
  let model = BernoulliRBM.create ~n_components:2 () in  
  print BernoulliRBM.pp @@ BernoulliRBM.fit ~x model;  
  [%expect {|
      BernoulliRBM(n_components=2)      
  |}]

*)



