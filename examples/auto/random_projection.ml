(* GaussianRandomProjection *)
(*
>>> import numpy as np
>>> from sklearn.random_projection import GaussianRandomProjection
>>> rng = np.random.RandomState(42)
>>> X = rng.rand(100, 10000)
>>> transformer = GaussianRandomProjection(random_state=rng)
>>> X_new = transformer.fit_transform(X)
>>> X_new.shape
(100, 3947)


*)

(* TEST TODO
let%expect_text "GaussianRandomProjection" =
    import numpy as np    
    let gaussianRandomProjection = Sklearn.Random_projection.gaussianRandomProjection in
    rng = np.random.RandomState(42)    
    X = rng.rand(100, 10000)    
    transformer = GaussianRandomProjection(random_state=rng)    
    X_new = transformer.fit_transform(X)    
    X_new.shape    
    [%expect {|
            (100, 3947)            
    |}]

*)



(* SparseRandomProjection *)
(*
>>> import numpy as np
>>> from sklearn.random_projection import SparseRandomProjection
>>> rng = np.random.RandomState(42)
>>> X = rng.rand(100, 10000)
>>> transformer = SparseRandomProjection(random_state=rng)
>>> X_new = transformer.fit_transform(X)
>>> X_new.shape
(100, 3947)
>>> # very few components are non-zero
>>> np.mean(transformer.components_ != 0)
0.0100...


*)

(* TEST TODO
let%expect_text "SparseRandomProjection" =
    import numpy as np    
    let sparseRandomProjection = Sklearn.Random_projection.sparseRandomProjection in
    rng = np.random.RandomState(42)    
    X = rng.rand(100, 10000)    
    transformer = SparseRandomProjection(random_state=rng)    
    X_new = transformer.fit_transform(X)    
    X_new.shape    
    [%expect {|
            (100, 3947)            
    |}]
    # very few components are non-zero    
    print @@ mean np transformer.components_ != 0
    [%expect {|
            0.0100...            
    |}]

*)



(* deprecated *)
(*
>>> from sklearn.utils import deprecated
>>> deprecated()
<sklearn.utils.deprecation.deprecated object at ...>


*)

(* TEST TODO
let%expect_text "deprecated" =
    let deprecated = Sklearn.Utils.deprecated in
    deprecated()    
    [%expect {|
            <sklearn.utils.deprecation.deprecated object at ...>            
    |}]

*)



(* deprecated *)
(*
>>> @deprecated()
... def some_function(): pass


*)

(* TEST TODO
let%expect_text "deprecated" =
    @deprecated()def some_function(): pass    
    [%expect {|
    |}]

*)



(* johnson_lindenstrauss_min_dim *)
(*
>>> johnson_lindenstrauss_min_dim(1e6, eps=0.5)
663


*)

(* TEST TODO
let%expect_text "johnson_lindenstrauss_min_dim" =
    johnson_lindenstrauss_min_dim(1e6, eps=0.5)    
    [%expect {|
            663            
    |}]

*)



(* johnson_lindenstrauss_min_dim *)
(*
>>> johnson_lindenstrauss_min_dim(1e6, eps=[0.5, 0.1, 0.01])
array([    663,   11841, 1112658])


*)

(* TEST TODO
let%expect_text "johnson_lindenstrauss_min_dim" =
    johnson_lindenstrauss_min_dim(1e6, eps=[0.5, 0.1, 0.01])    
    [%expect {|
            array([    663,   11841, 1112658])            
    |}]

*)



(* johnson_lindenstrauss_min_dim *)
(*
>>> johnson_lindenstrauss_min_dim([1e4, 1e5, 1e6], eps=0.1)
array([ 7894,  9868, 11841])


*)

(* TEST TODO
let%expect_text "johnson_lindenstrauss_min_dim" =
    johnson_lindenstrauss_min_dim([1e4, 1e5, 1e6], eps=0.1)    
    [%expect {|
            array([ 7894,  9868, 11841])            
    |}]

*)



