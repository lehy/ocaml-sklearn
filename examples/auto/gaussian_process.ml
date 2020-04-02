(* GaussianProcessClassifier *)
(*
>>> from sklearn.datasets import load_iris
>>> from sklearn.gaussian_process import GaussianProcessClassifier
>>> from sklearn.gaussian_process.kernels import RBF
>>> X, y = load_iris(return_X_y=True)
>>> kernel = 1.0 * RBF(1.0)
>>> gpc = GaussianProcessClassifier(kernel=kernel,
...         random_state=0).fit(X, y)
>>> gpc.score(X, y)
0.9866...
>>> gpc.predict_proba(X[:2,:])
array([[0.83548752, 0.03228706, 0.13222543],
       [0.79064206, 0.06525643, 0.14410151]])


*)

(* TEST TODO
let%expect_text "GaussianProcessClassifier" =
    let load_iris = Sklearn.Datasets.load_iris in
    let gaussianProcessClassifier = Sklearn.Gaussian_process.gaussianProcessClassifier in
    let rbf = Sklearn.Gaussian_process.Kernels.rbf in
    let x, y = load_iris return_X_y=True in
    kernel = 1.0 * RBF(1.0)    
    gpc = GaussianProcessClassifier(kernel=kernel,random_state=0).fit(X, y)    
    print @@ score gpc x y
    [%expect {|
            0.9866...            
    |}]
    print @@ predict_proba gpc x[:2 :]
    [%expect {|
            array([[0.83548752, 0.03228706, 0.13222543],            
                   [0.79064206, 0.06525643, 0.14410151]])            
    |}]

*)



(*--------- Examples for module .Gaussian_process.Kernels ----------*)
(* cdist *)
(*
>>> from scipy.spatial import distance
>>> coords = [(35.0456, -85.2672),
...           (35.1174, -89.9711),
...           (35.9728, -83.9422),
...           (36.1667, -86.7833)]
>>> distance.cdist(coords, coords, 'euclidean')
array([[ 0.    ,  4.7044,  1.6172,  1.8856],
       [ 4.7044,  0.    ,  6.0893,  3.3561],
       [ 1.6172,  6.0893,  0.    ,  2.8477],
       [ 1.8856,  3.3561,  2.8477,  0.    ]])


*)

(* TEST TODO
let%expect_text "cdist" =
    let distance = Scipy.Spatial.distance in
    coords = [(35.0456, -85.2672),(35.1174, -89.9711),(35.9728, -83.9422),(36.1667, -86.7833)]    
    print @@ cdist distance coords coords 'euclidean'
    [%expect {|
            array([[ 0.    ,  4.7044,  1.6172,  1.8856],            
                   [ 4.7044,  0.    ,  6.0893,  3.3561],            
                   [ 1.6172,  6.0893,  0.    ,  2.8477],            
                   [ 1.8856,  3.3561,  2.8477,  0.    ]])            
    |}]

*)



