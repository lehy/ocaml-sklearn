module Np = Np.Numpy

let print f x = Format.printf "%a" f x

let print_py x = Format.printf "%s" (Py.Object.to_string x)

let print_ndarray = Np.Obj.print

let print_float = Format.printf "%g\n"

let print_string = Format.printf "%s\n"

let print_int = Format.printf "%d\n"

let matrixi = Np.Ndarray.matrixi

let matrixf = Np.Ndarray.matrixf


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

(* let%expect_test "EllipticEnvelope" =
 *   let open Sklearn.Covariance in
 *   let true_cov = [[.8 .3] [.3 .4]] in  
 *   let x = np..randomState 0).multivariate_normal( ~mean:(vectori [|0; 0|]) ~cov:true_cov ~size:500 random in  
 *   let cov = EllipticEnvelope(random_state=0).fit ~x () in  
 *   print_ndarray @@ # predict returns 1 for an inlier and -1 for an outlier;  
 *   print_ndarray @@ EllipticEnvelope.predict [(vectori [|0; 0|]) (vectori [|3; 3|])] cov;  
 *   [%expect {|
 *       array([ 1, -1])      
 *   |}];
 *   print_ndarray @@ EllipticEnvelope.covariance_ cov;  
 *   [%expect {|
 *       array([[0.7411..., 0.2535...],      
 *              [0.2535..., 0.3053...]])      
 *   |}];
 *   print_ndarray @@ EllipticEnvelope.location_ cov;  
 *   [%expect {|
 *       array([0.0813... , 0.0427...])      
 *   |}] *)


(* EmpiricalCovariance *)
(*
>>> import numpy as np
>>> from sklearn.covariance import EmpiricalCovariance
>>> from sklearn.datasets import make_gaussian_quantiles
>>> real_cov = np.array([[.8, .3],
...                      [.3, .4]])
>>> rng = np.random.RandomState(0)
>>> X = rng.multivariate_normal(mean=[0, 0],
...                             cov=real_cov,
...                             size=500)
>>> cov = EmpiricalCovariance().fit(X)
>>> cov.covariance_
array([[0.7569..., 0.2818...],
       [0.2818..., 0.3928...]])
>>> cov.location_

*)

(* TEST TODO
let%expect_test "EmpiricalCovariance" =
  let open Sklearn.Covariance in
  let real_cov = .array [[.8 .3] [.3 .4]] np in  
  let rng = np..randomState ~0 random in  
  let x = .multivariate_normal ~mean:(vectori [|0; 0|]) ~cov:real_cov ~size:500 rng in  
  let cov = EmpiricalCovariance().fit ~x () in  
  print_ndarray @@ EmpiricalCovariance.covariance_ cov;  
  [%expect {|
      array([[0.7569..., 0.2818...],      
             [0.2818..., 0.3928...]])      
  |}]
  print_ndarray @@ EmpiricalCovariance.location_ cov;  
  [%expect {|
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
let%expect_test "GraphicalLasso" =
  let open Sklearn.Covariance in
  let true_cov = .array [[0.8 0.0 0.2 0.0] [0.0 0.4 0.0 0.0] [0.2 0.0 0.3 0.1] [0.0 0.0 0.1 0.7]] np in  
  print_ndarray @@ np..seed ~0 random;  
  let x = np..multivariate_normal ~mean:(vectori [|0; 0; 0; 0|]) ~cov:true_cov ~size:200 random in  
  let cov = GraphicalLasso().fit ~x () in  
  print_ndarray @@ .around cov.covariance_ ~decimals:3 np;  
  [%expect {|
      array([[0.816, 0.049, 0.218, 0.019],      
             [0.049, 0.364, 0.017, 0.034],      
             [0.218, 0.017, 0.322, 0.093],      
             [0.019, 0.034, 0.093, 0.69 ]])      
  |}]
  print_ndarray @@ .around cov.location_ ~decimals:3 np;  
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
let%expect_test "GraphicalLassoCV" =
  let open Sklearn.Covariance in
  let true_cov = .array [[0.8 0.0 0.2 0.0] [0.0 0.4 0.0 0.0] [0.2 0.0 0.3 0.1] [0.0 0.0 0.1 0.7]] np in  
  print_ndarray @@ np..seed ~0 random;  
  let x = np..multivariate_normal ~mean:(vectori [|0; 0; 0; 0|]) ~cov:true_cov ~size:200 random in  
  let cov = GraphicalLassoCV().fit ~x () in  
  print_ndarray @@ .around cov.covariance_ ~decimals:3 np;  
  [%expect {|
      array([[0.816, 0.051, 0.22 , 0.017],      
             [0.051, 0.364, 0.018, 0.036],      
             [0.22 , 0.018, 0.322, 0.094],      
             [0.017, 0.036, 0.094, 0.69 ]])      
  |}]
  print_ndarray @@ .around cov.location_ ~decimals:3 np;  
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
let%expect_test "LedoitWolf" =
  let open Sklearn.Covariance in
  let real_cov = .array [[.4 .2] [.2 .8]] np in  
  print_ndarray @@ np..seed ~0 random;  
  let x = np..multivariate_normal ~mean:(vectori [|0; 0|]) ~cov:real_cov ~size:50 random in  
  let cov = LedoitWolf().fit ~x () in  
  print_ndarray @@ LedoitWolf.covariance_ cov;  
  [%expect {|
      array([[0.4406..., 0.1616...],      
             [0.1616..., 0.8022...]])      
  |}]
  print_ndarray @@ LedoitWolf.location_ cov;  
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
let%expect_test "MinCovDet" =
  let open Sklearn.Covariance in
  let real_cov = .array [[.8 .3] [.3 .4]] np in  
  let rng = np..randomState ~0 random in  
  let x = .multivariate_normal ~mean:(vectori [|0; 0|]) ~cov:real_cov ~size:500 rng in  
  let cov = MinCovDet(random_state=0).fit ~x () in  
  print_ndarray @@ MinCovDet.covariance_ cov;  
  [%expect {|
      array([[0.7411..., 0.2535...],      
             [0.2535..., 0.3053...]])      
  |}]
  print_ndarray @@ MinCovDet.location_ cov;  
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
let%expect_test "ShrunkCovariance" =
  let open Sklearn.Covariance in
  let real_cov = .array [[.8 .3] [.3 .4]] np in  
  let rng = np..randomState ~0 random in  
  let x = .multivariate_normal ~mean:(vectori [|0; 0|]) ~cov:real_cov ~size:500 rng in  
  let cov = ShrunkCovariance().fit ~x () in  
  print_ndarray @@ ShrunkCovariance.covariance_ cov;  
  [%expect {|
      array([[0.7387..., 0.2536...],      
             [0.2536..., 0.4110...]])      
  |}]
  print_ndarray @@ ShrunkCovariance.location_ cov;  
  [%expect {|
      array([0.0622..., 0.0193...])      
  |}]

*)



