(* CCA *)
(*
>>> from sklearn.cross_decomposition import CCA
>>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]]
>>> Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
>>> cca = CCA(n_components=1)
>>> cca.fit(X, Y)
CCA(n_components=1)
>>> X_c, Y_c = cca.transform(X, Y)

*)

(* TEST TODO
let%expect_test "CCA" =
  let open Sklearn.Cross_decomposition in
  let x = (matrix [|[|0.; 0.; 1.|]; [|1.;0.;0.|]; [|2.;2.;2.|]; [|3.;5.;4.|]|]) in  
  let Y = (matrix [|[|0.1; -0.2|]; [|0.9; 1.1|]; [|6.2; 5.9|]; [|11.9; 12.3|]|]) in  
  let cca = CCA.create ~n_components:1 () in  
  print CCA.pp @@ CCA.fit ~x Y cca;  
  [%expect {|
      CCA(n_components=1)      
  |}]
  let X_c, Y_c = CCA.transform ~x Y cca in  
  [%expect {|
  |}]

*)



(* PLSCanonical *)
(*
>>> from sklearn.cross_decomposition import PLSCanonical
>>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
>>> Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
>>> plsca = PLSCanonical(n_components=2)
>>> plsca.fit(X, Y)
PLSCanonical()
>>> X_c, Y_c = plsca.transform(X, Y)

*)

(* TEST TODO
let%expect_test "PLSCanonical" =
  let open Sklearn.Cross_decomposition in
  let x = (matrix [|[|0.; 0.; 1.|]; [|1.;0.;0.|]; [|2.;2.;2.|]; [|2.;5.;4.|]|]) in  
  let Y = (matrix [|[|0.1; -0.2|]; [|0.9; 1.1|]; [|6.2; 5.9|]; [|11.9; 12.3|]|]) in  
  let plsca = PLSCanonical.create ~n_components:2 () in  
  print PLSCanonical.pp @@ PLSCanonical.fit ~x Y plsca;  
  [%expect {|
      PLSCanonical()      
  |}]
  let X_c, Y_c = PLSCanonical.transform ~x Y plsca in  
  [%expect {|
  |}]

*)



(* PLSRegression *)
(*
>>> from sklearn.cross_decomposition import PLSRegression
>>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
>>> Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
>>> pls2 = PLSRegression(n_components=2)
>>> pls2.fit(X, Y)
PLSRegression()
>>> Y_pred = pls2.predict(X)

*)

(* TEST TODO
let%expect_test "PLSRegression" =
  let open Sklearn.Cross_decomposition in
  let x = (matrix [|[|0.; 0.; 1.|]; [|1.;0.;0.|]; [|2.;2.;2.|]; [|2.;5.;4.|]|]) in  
  let Y = (matrix [|[|0.1; -0.2|]; [|0.9; 1.1|]; [|6.2; 5.9|]; [|11.9; 12.3|]|]) in  
  let pls2 = PLSRegression.create ~n_components:2 () in  
  print_ndarray @@ pls2.fit ~x Y ();  
  [%expect {|
      PLSRegression()      
  |}]
  let Y_pred = pls2.predict ~x () in  
  [%expect {|
  |}]

*)



(* PLSSVD *)
(*
>>> import numpy as np
>>> from sklearn.cross_decomposition import PLSSVD
>>> X = np.array([[0., 0., 1.],
...     [1.,0.,0.],
...     [2.,2.,2.],
...     [2.,5.,4.]])
>>> Y = np.array([[0.1, -0.2],
...     [0.9, 1.1],
...     [6.2, 5.9],
...     [11.9, 12.3]])
>>> plsca = PLSSVD(n_components=2)
>>> plsca.fit(X, Y)
PLSSVD()
>>> X_c, Y_c = plsca.transform(X, Y)
>>> X_c.shape, Y_c.shape
((4, 2), (4, 2))

*)

(* TEST TODO
let%expect_test "PLSSVD" =
  let open Sklearn.Cross_decomposition in
  let x = .array [[0. 0. 1.] [1. 0. 0.] [2. 2. 2.] [2. 5. 4.]] np in  
  let Y = .array [[0.1 -0.2] [0.9 1.1] [6.2 5.9] [11.9 12.3]] np in  
  let plsca = PLSSVD.create ~n_components:2 () in  
  print PLSSVD.pp @@ PLSSVD.fit ~x Y plsca;  
  [%expect {|
      PLSSVD()      
  |}]
  let X_c, Y_c = PLSSVD.transform ~x Y plsca in  
  print_ndarray @@ X_c.shape, Y_c.shape;  
  [%expect {|
      ((4, 2), (4, 2))      
  |}]

*)



