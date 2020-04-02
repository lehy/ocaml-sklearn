(* EllipticEnvelope *)
(*
>>> import numpy as np
>>> from sklearn.covariance import EllipticEnvelope
>>> true_cov = np.array([[.8, .3],
...                      [.3, .4]])
>>> X = np.random.RandomState(0).multivariate_normal(mean=[0, 0],
...                                                  cov=true_cov,
...                                                  size=500)
>>> cov = EllipticEnvelope(random_state=0).fit(X)
>>> # predict returns 1 for an inlier and -1 for an outlier
>>> cov.predict([[0, 0],
...              [3, 3]])
array([ 1, -1])
>>> cov.covariance_
array([[0.7411..., 0.2535...],
       [0.2535..., 0.3053...]])
>>> cov.location_
array([0.0813... , 0.0427...])


*)

(* TEST TODO
let%expect_text "EllipticEnvelope" =
    import numpy as np    
    let ellipticEnvelope = Sklearn.Covariance.ellipticEnvelope in
    true_cov = np.array([[.8, .3],[.3, .4]])    
    X = np.random.RandomState(0).multivariate_normal(mean=[0, 0],cov=true_cov,size=500)    
    cov = EllipticEnvelope(random_state=0).fit(X)    
    # predict returns 1 for an inlier and -1 for an outlier    
    print @@ predict cov [[0 0] [3 3]]
    [%expect {|
            array([ 1, -1])            
    |}]
    cov.covariance_    
    [%expect {|
            array([[0.7411..., 0.2535...],            
                   [0.2535..., 0.3053...]])            
    |}]
    cov.location_    
    [%expect {|
            array([0.0813... , 0.0427...])            
    |}]

*)



(* GraphicalLasso *)
(*
>>> import numpy as np
>>> from sklearn.covariance import GraphicalLasso
>>> true_cov = np.array([[0.8, 0.0, 0.2, 0.0],
...                      [0.0, 0.4, 0.0, 0.0],
...                      [0.2, 0.0, 0.3, 0.1],
...                      [0.0, 0.0, 0.1, 0.7]])
>>> np.random.seed(0)
>>> X = np.random.multivariate_normal(mean=[0, 0, 0, 0],
...                                   cov=true_cov,
...                                   size=200)
>>> cov = GraphicalLasso().fit(X)
>>> np.around(cov.covariance_, decimals=3)
array([[0.816, 0.049, 0.218, 0.019],
       [0.049, 0.364, 0.017, 0.034],
       [0.218, 0.017, 0.322, 0.093],
       [0.019, 0.034, 0.093, 0.69 ]])
>>> np.around(cov.location_, decimals=3)
array([0.073, 0.04 , 0.038, 0.143])


*)

(* TEST TODO
let%expect_text "GraphicalLasso" =
    import numpy as np    
    let graphicalLasso = Sklearn.Covariance.graphicalLasso in
    true_cov = np.array([[0.8, 0.0, 0.2, 0.0],[0.0, 0.4, 0.0, 0.0],[0.2, 0.0, 0.3, 0.1],[0.0, 0.0, 0.1, 0.7]])    
    np.random.seed(0)    
    X = np.random.multivariate_normal(mean=[0, 0, 0, 0],cov=true_cov,size=200)    
    cov = GraphicalLasso().fit(X)    
    print @@ around np cov.covariance_ decimals=3
    [%expect {|
            array([[0.816, 0.049, 0.218, 0.019],            
                   [0.049, 0.364, 0.017, 0.034],            
                   [0.218, 0.017, 0.322, 0.093],            
                   [0.019, 0.034, 0.093, 0.69 ]])            
    |}]
    print @@ around np cov.location_ decimals=3
    [%expect {|
            array([0.073, 0.04 , 0.038, 0.143])            
    |}]

*)



(* GraphicalLassoCV *)
(*
>>> import numpy as np
>>> from sklearn.covariance import GraphicalLassoCV
>>> true_cov = np.array([[0.8, 0.0, 0.2, 0.0],
...                      [0.0, 0.4, 0.0, 0.0],
...                      [0.2, 0.0, 0.3, 0.1],
...                      [0.0, 0.0, 0.1, 0.7]])
>>> np.random.seed(0)
>>> X = np.random.multivariate_normal(mean=[0, 0, 0, 0],
...                                   cov=true_cov,
...                                   size=200)
>>> cov = GraphicalLassoCV().fit(X)
>>> np.around(cov.covariance_, decimals=3)
array([[0.816, 0.051, 0.22 , 0.017],
       [0.051, 0.364, 0.018, 0.036],
       [0.22 , 0.018, 0.322, 0.094],
       [0.017, 0.036, 0.094, 0.69 ]])
>>> np.around(cov.location_, decimals=3)
array([0.073, 0.04 , 0.038, 0.143])


*)

(* TEST TODO
let%expect_text "GraphicalLassoCV" =
    import numpy as np    
    let graphicalLassoCV = Sklearn.Covariance.graphicalLassoCV in
    true_cov = np.array([[0.8, 0.0, 0.2, 0.0],[0.0, 0.4, 0.0, 0.0],[0.2, 0.0, 0.3, 0.1],[0.0, 0.0, 0.1, 0.7]])    
    np.random.seed(0)    
    X = np.random.multivariate_normal(mean=[0, 0, 0, 0],cov=true_cov,size=200)    
    cov = GraphicalLassoCV().fit(X)    
    print @@ around np cov.covariance_ decimals=3
    [%expect {|
            array([[0.816, 0.051, 0.22 , 0.017],            
                   [0.051, 0.364, 0.018, 0.036],            
                   [0.22 , 0.018, 0.322, 0.094],            
                   [0.017, 0.036, 0.094, 0.69 ]])            
    |}]
    print @@ around np cov.location_ decimals=3
    [%expect {|
            array([0.073, 0.04 , 0.038, 0.143])            
    |}]

*)



(* LedoitWolf *)
(*
>>> import numpy as np
>>> from sklearn.covariance import LedoitWolf
>>> real_cov = np.array([[.4, .2],
...                      [.2, .8]])
>>> np.random.seed(0)
>>> X = np.random.multivariate_normal(mean=[0, 0],
...                                   cov=real_cov,
...                                   size=50)
>>> cov = LedoitWolf().fit(X)
>>> cov.covariance_
array([[0.4406..., 0.1616...],
       [0.1616..., 0.8022...]])
>>> cov.location_
array([ 0.0595... , -0.0075...])


*)

(* TEST TODO
let%expect_text "LedoitWolf" =
    import numpy as np    
    let ledoitWolf = Sklearn.Covariance.ledoitWolf in
    real_cov = np.array([[.4, .2],[.2, .8]])    
    np.random.seed(0)    
    X = np.random.multivariate_normal(mean=[0, 0],cov=real_cov,size=50)    
    cov = LedoitWolf().fit(X)    
    cov.covariance_    
    [%expect {|
            array([[0.4406..., 0.1616...],            
                   [0.1616..., 0.8022...]])            
    |}]
    cov.location_    
    [%expect {|
            array([ 0.0595... , -0.0075...])            
    |}]

*)



(* MinCovDet *)
(*
>>> import numpy as np
>>> from sklearn.covariance import MinCovDet
>>> from sklearn.datasets import make_gaussian_quantiles
>>> real_cov = np.array([[.8, .3],
...                      [.3, .4]])
>>> rng = np.random.RandomState(0)
>>> X = rng.multivariate_normal(mean=[0, 0],
...                                   cov=real_cov,
...                                   size=500)
>>> cov = MinCovDet(random_state=0).fit(X)
>>> cov.covariance_
array([[0.7411..., 0.2535...],
       [0.2535..., 0.3053...]])
>>> cov.location_
array([0.0813... , 0.0427...])


*)

(* TEST TODO
let%expect_text "MinCovDet" =
    import numpy as np    
    let minCovDet = Sklearn.Covariance.minCovDet in
    let make_gaussian_quantiles = Sklearn.Datasets.make_gaussian_quantiles in
    real_cov = np.array([[.8, .3],[.3, .4]])    
    rng = np.random.RandomState(0)    
    X = rng.multivariate_normal(mean=[0, 0],cov=real_cov,size=500)    
    cov = MinCovDet(random_state=0).fit(X)    
    cov.covariance_    
    [%expect {|
            array([[0.7411..., 0.2535...],            
                   [0.2535..., 0.3053...]])            
    |}]
    cov.location_    
    [%expect {|
            array([0.0813... , 0.0427...])            
    |}]

*)



(* ShrunkCovariance *)
(*
>>> import numpy as np
>>> from sklearn.covariance import ShrunkCovariance
>>> from sklearn.datasets import make_gaussian_quantiles
>>> real_cov = np.array([[.8, .3],
...                      [.3, .4]])
>>> rng = np.random.RandomState(0)
>>> X = rng.multivariate_normal(mean=[0, 0],
...                                   cov=real_cov,
...                                   size=500)
>>> cov = ShrunkCovariance().fit(X)
>>> cov.covariance_
array([[0.7387..., 0.2536...],
       [0.2536..., 0.4110...]])
>>> cov.location_
array([0.0622..., 0.0193...])


*)

(* TEST TODO
let%expect_text "ShrunkCovariance" =
    import numpy as np    
    let shrunkCovariance = Sklearn.Covariance.shrunkCovariance in
    let make_gaussian_quantiles = Sklearn.Datasets.make_gaussian_quantiles in
    real_cov = np.array([[.8, .3],[.3, .4]])    
    rng = np.random.RandomState(0)    
    X = rng.multivariate_normal(mean=[0, 0],cov=real_cov,size=500)    
    cov = ShrunkCovariance().fit(X)    
    cov.covariance_    
    [%expect {|
            array([[0.7387..., 0.2536...],            
                   [0.2536..., 0.4110...]])            
    |}]
    cov.location_    
    [%expect {|
            array([0.0622..., 0.0193...])            
    |}]

*)



