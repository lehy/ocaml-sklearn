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
let%expect_text "BernoulliRBM" =
    import numpy as np    
    let bernoulliRBM = Sklearn.Neural_network.bernoulliRBM in
    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])    
    model = BernoulliRBM(n_components=2)    
    print @@ fit model x
    [%expect {|
            BernoulliRBM(n_components=2)            
    |}]

*)



