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
let%expect_text "CCA" =
    let cca = Sklearn.Cross_decomposition.cca in
    X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]]    
    Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]    
    cca = CCA(n_components=1)    
    print @@ fit cca x y
    [%expect {|
            CCA(n_components=1)            
    |}]
    X_c, Y_c = cca.transform(X, Y)    
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
let%expect_text "PLSCanonical" =
    let pLSCanonical = Sklearn.Cross_decomposition.pLSCanonical in
    X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]    
    Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]    
    plsca = PLSCanonical(n_components=2)    
    print @@ fit plsca x y
    [%expect {|
            PLSCanonical()            
    |}]
    X_c, Y_c = plsca.transform(X, Y)    
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
let%expect_text "PLSRegression" =
    let pLSRegression = Sklearn.Cross_decomposition.pLSRegression in
    X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]    
    Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]    
    pls2 = PLSRegression(n_components=2)    
    print @@ fit pls2 x y
    [%expect {|
            PLSRegression()            
    |}]
    Y_pred = pls2.predict(X)    
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
let%expect_text "PLSSVD" =
    import numpy as np    
    let plssvd = Sklearn.Cross_decomposition.plssvd in
    X = np.array([[0., 0., 1.],[1.,0.,0.],[2.,2.,2.],[2.,5.,4.]])    
    Y = np.array([[0.1, -0.2],[0.9, 1.1],[6.2, 5.9],[11.9, 12.3]])    
    plsca = PLSSVD(n_components=2)    
    print @@ fit plsca x y
    [%expect {|
            PLSSVD()            
    |}]
    X_c, Y_c = plsca.transform(X, Y)    
    X_c.shape, Y_c.shape    
    [%expect {|
            ((4, 2), (4, 2))            
    |}]

*)



