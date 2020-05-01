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
let%expect_test "GaussianProcessClassifier" =
  let open Sklearn.Gaussian_process in
  let x, y = load_iris ~return_X_y:true () in  
  let kernel = 1.0 * RBF(1.0) in  
  let gpc = GaussianProcessClassifier(kernel=kernel,random_state=0).fit ~x y () in  
  print_ndarray @@ GaussianProcessClassifier.score ~x y gpc;  
  [%expect {|
      0.9866...      
  |}]
  print_ndarray @@ GaussianProcessClassifier.predict_proba x[:2 :] gpc;  
  [%expect {|
      array([[0.83548752, 0.03228706, 0.13222543],      
             [0.79064206, 0.06525643, 0.14410151]])      
  |}]

*)



(* GaussianProcessRegressor *)
(*
>>> from sklearn.datasets import make_friedman2
>>> from sklearn.gaussian_process import GaussianProcessRegressor
>>> from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
>>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
>>> kernel = DotProduct() + WhiteKernel()
>>> gpr = GaussianProcessRegressor(kernel=kernel,
...         random_state=0).fit(X, y)
>>> gpr.score(X, y)
0.3680...
>>> gpr.predict(X[:2,:], return_std=True)

*)

(* TEST TODO
let%expect_test "GaussianProcessRegressor" =
  let open Sklearn.Gaussian_process in
  let x, y = make_friedman2(n_samples=500, noise=0, random_state=0) in  
  let kernel = DotProduct() + WhiteKernel() in  
  let gpr = GaussianProcessRegressor(kernel=kernel,random_state=0).fit ~x y () in  
  print_ndarray @@ GaussianProcessRegressor.score ~x y gpr;  
  [%expect {|
      0.3680...      
  |}]
  print_ndarray @@ GaussianProcessRegressor.predict x[:2 :] ~return_std:true gpr;  
  [%expect {|
  |}]

*)



(*--------- Examples for module Sklearn.Gaussian_process.Kernels ----------*)
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
let%expect_test "cdist" =
  let open Sklearn.Gaussian_process in
  let coords = [(35.0456, -85.2672),(35.1174, -89.9711),(35.9728, -83.9422),(36.1667, -86.7833)] in  
  print_ndarray @@ .cdist ~coords coords 'euclidean' distance;  
  [%expect {|
      array([[ 0.    ,  4.7044,  1.6172,  1.8856],      
             [ 4.7044,  0.    ,  6.0893,  3.3561],      
             [ 1.6172,  6.0893,  0.    ,  2.8477],      
             [ 1.8856,  3.3561,  2.8477,  0.    ]])      
  |}]

*)



(* cdist *)
(*
>>> a = np.array([[0, 0, 0],
...               [0, 0, 1],
...               [0, 1, 0],
...               [0, 1, 1],
...               [1, 0, 0],
...               [1, 0, 1],
...               [1, 1, 0],
...               [1, 1, 1]])
>>> b = np.array([[ 0.1,  0.2,  0.4]])
>>> distance.cdist(a, b, 'cityblock')
array([[ 0.7],
       [ 0.9],
       [ 1.3],
       [ 1.5],
       [ 1.5],
       [ 1.7],
       [ 2.1],

*)

(* TEST TODO
let%expect_test "cdist" =
  let open Sklearn.Gaussian_process in
  let a = .array [(vectori [|0; 0; 0|]) (vectori [|0; 0; 1|]) (vectori [|0; 1; 0|]) (vectori [|0; 1; 1|]) (vectori [|1; 0; 0|]) (vectori [|1; 0; 1|]) (vectori [|1; 1; 0|]) (vectori [|1; 1; 1|])] np in  
  let b = .array (matrix [|[| 0.1; 0.2; 0.4|]|]) np in  
  print_ndarray @@ .cdist ~a b 'cityblock' distance;  
  [%expect {|
      array([[ 0.7],      
             [ 0.9],      
             [ 1.3],      
             [ 1.5],      
             [ 1.5],      
             [ 1.7],      
             [ 2.1],      
  |}]

*)



(* namedtuple *)
(*
>>> Point = namedtuple('Point', ['x', 'y'])
>>> Point.__doc__                   # docstring for the new class
'Point(x, y)'
>>> p = Point(11, y=22)             # instantiate with positional args or keywords
>>> p[0] + p[1]                     # indexable like a plain tuple
33
>>> x, y = p                        # unpack like a regular tuple
>>> x, y
(11, 22)
>>> p.x + p.y                       # fields also accessible by name
33
>>> d = p._asdict()                 # convert to a dictionary
>>> d['x']
11
>>> Point( **d)                      # convert from a dictionary
Point(x=11, y=22)
>>> p._replace(x=100)               # _replace() is like str.replace() but targets named fields

*)

(* TEST TODO
let%expect_test "namedtuple" =
  let open Sklearn.Gaussian_process in
  let Point = namedtuple 'Point' ['x' 'y'] () in  
  print_ndarray @@ Point.__doc__ # docstring for the new class;  
  [%expect {|
      'Point(x, y)'      
  |}]
  let p = Point(11, y=22) # instantiate with positional args or keywords in  
  print_ndarray @@ p(vectori [|0|]) + p(vectori [|1|]) # indexable like a plain tuple;  
  [%expect {|
      33      
  |}]
  let x, y = p # unpack like a regular tuple in  
  print_ndarray @@ x, y;  
  [%expect {|
      (11, 22)      
  |}]
  print_ndarray @@ p.x + p.y # fields also accessible by name;  
  [%expect {|
      33      
  |}]
  let d = p._asdict() # convert to a dictionary in  
  print_ndarray @@ d['x'];  
  [%expect {|
      11      
  |}]
  print_ndarray @@ Point( **d) # convert from a dictionary;  
  [%expect {|
      Point(x=11, y=22)      
  |}]
  print_ndarray @@ p._replace(x=100) # _replace() is like .replace str but targets named fields;  
  [%expect {|
  |}]

*)



