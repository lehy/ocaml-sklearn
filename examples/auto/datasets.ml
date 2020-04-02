(* make_blobs *)
(*
>>> from sklearn.datasets import make_blobs
>>> X, y = make_blobs(n_samples=10, centers=3, n_features=2,
...                   random_state=0)
>>> print(X.shape)
(10, 2)
>>> y
array([0, 0, 1, 0, 2, 2, 2, 1, 1, 0])
>>> X, y = make_blobs(n_samples=[3, 3, 4], centers=None, n_features=2,
...                   random_state=0)
>>> print(X.shape)
(10, 2)
>>> y
array([0, 1, 2, 0, 2, 2, 2, 1, 1, 0])


*)

(* TEST TODO
let%expect_text "make_blobs" =
    let make_blobs = Sklearn.Datasets.make_blobs in
    let x, y = make_blobs n_samples=10 centers=3 n_features=2 random_state=0 in
    print(X.shape)    
    [%expect {|
            (10, 2)            
    |}]
    y    
    [%expect {|
            array([0, 0, 1, 0, 2, 2, 2, 1, 1, 0])            
    |}]
    let x, y = make_blobs n_samples=[3 3 4] centers=None n_features=2 random_state=0 in
    print(X.shape)    
    [%expect {|
            (10, 2)            
    |}]
    y    
    [%expect {|
            array([0, 1, 2, 0, 2, 2, 2, 1, 1, 0])            
    |}]

*)



