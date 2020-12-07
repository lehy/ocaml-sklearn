module Np = Np.Numpy

let print f x = Format.printf "%a" f x

let print_py x = Format.printf "%s" (Py.Object.to_string x)

let print_ndarray = Np.Obj.print

let print_float = Format.printf "%g\n"

let print_string = Format.printf "%s\n"

let print_int = Format.printf "%d\n"

(* LinearSVC *)
(*
>>> from sklearn.svm import LinearSVC
>>> from sklearn.datasets import make_classification
>>> X, y = make_classification(n_features=4, random_state=0)
>>> clf = LinearSVC(random_state=0, tol=1e-5)
>>> clf.fit(X, y)
LinearSVC(random_state=0, tol=1e-05)
>>> print(clf.coef_)
[[0.085... 0.394... 0.498... 0.375...]]
>>> print(clf.intercept_)
[0.284...]
>>> print(clf.predict([[0, 0, 0, 0]]))

*)

(* TEST TODO
let%expect_test "LinearSVC" =
  let open Sklearn.Svm in
  let x, y = make_classification ~n_features:4 ~random_state:0 () in  
  let clf = LinearSVC.create ~random_state:0 ~tol:1e-5 () in  
  print LinearSVC.pp @@ LinearSVC.fit ~x y clf;  
  [%expect {|
      LinearSVC(random_state=0, tol=1e-05)      
  |}]
  print_ndarray @@ print clf.coef_ ();  
  [%expect {|
      [[0.085... 0.394... 0.498... 0.375...]]      
  |}]
  print_ndarray @@ print clf.intercept_ ();  
  [%expect {|
      [0.284...]      
  |}]
  print_ndarray @@ print(LinearSVC.predict (matrixi [|[|0; 0; 0; 0|]|])) clf;  
  [%expect {|
  |}]

*)

(* LinearSVR *)
(*
>>> from sklearn.svm import LinearSVR
>>> from sklearn.datasets import make_regression
>>> X, y = make_regression(n_features=4, random_state=0)
>>> regr = LinearSVR(random_state=0, tol=1e-5)
>>> regr.fit(X, y)
LinearSVR(random_state=0, tol=1e-05)
>>> print(regr.coef_)
[16.35... 26.91... 42.30... 60.47...]
>>> print(regr.intercept_)
[-4.29...]
>>> print(regr.predict([[0, 0, 0, 0]]))
[-4.29...]


*)

let%expect_test "LinearSVR" =
  let x, y, _coef =
    Sklearn.Datasets.make_regression ~n_features:4 ~random_state:0 ()
  in
  let open Sklearn.Svm in
  let regr = LinearSVR.create ~random_state:0 ~tol:1e-5 () in
  print LinearSVR.pp @@ LinearSVR.fit regr ~x ~y;
  [%expect
    {|
          LinearSVR(random_state=0, tol=1e-05)
  |}];
  print_ndarray @@ LinearSVR.coef_ regr;
  [%expect
    {|
            [16.35883704 26.91633994 42.30602221 60.4781483 ]
    |}];
  print_ndarray @@ LinearSVR.intercept_ regr;
  [%expect {|
            [-4.29635256]
    |}];
  print_ndarray
  @@ LinearSVR.predict regr ~x:(Np.matrixf [| [| 0.; 0.; 0.; 0. |] |]);
  [%expect {|
            [-4.29635256]
    |}]

(* NuSVC *)
(*
>>> import numpy as np
>>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
>>> y = np.array([1, 1, 2, 2])
>>> from sklearn.svm import NuSVC
>>> clf = NuSVC()
>>> clf.fit(X, y)
NuSVC()
>>> print(clf.predict([[-0.8, -1]]))
[1]


*)

let%expect_test "NuSVC" =
  let x =
    Np.matrixf [| [| -1.; -1. |]; [| -2.; -1. |]; [| 1.; 1. |]; [| 2.; 1. |] |]
  in
  let y = Np.vectorf [| 1.; 1.; 2.; 2. |] in
  let open Sklearn.Svm in
  let clf = NuSVC.create () in
  print NuSVC.pp @@ NuSVC.fit clf ~x ~y;
  [%expect
    {|
            NuSVC()
    |}];
  print_ndarray @@ NuSVC.predict clf ~x:(Np.matrixf [| [| -0.8; -1. |] |]);
  [%expect {|
            [1.]
    |}]

(* NuSVR *)
(*
>>> from sklearn.svm import NuSVR
>>> import numpy as np
>>> n_samples, n_features = 10, 5
>>> np.random.seed(0)
>>> y = np.random.randn(n_samples)
>>> X = np.random.randn(n_samples, n_features)
>>> clf = NuSVR(C=1.0, nu=0.1)
>>> clf.fit(X, y)
NuSVR(nu=0.1)


*)

let%expect_test "NuSVR" =
  let open Sklearn.Svm in
  let n_samples, n_features = (10, 5) in
  (* Random.init 0; *)
  Np.Random.seed 0;
  let y = Np.Random.randn [ n_samples ] in
  let x = Np.Random.randn [ n_samples; n_features ] in
  let clf = NuSVR.create ~c:1.0 ~nu:0.1 () in
  print NuSVR.pp @@ NuSVR.fit clf ~x ~y;
  [%expect
    {|
            NuSVR(nu=0.1)
    |}]

(* OneClassSVM *)
(*
>>> from sklearn.svm import OneClassSVM
>>> X = [[0], [0.44], [0.45], [0.46], [1]]
>>> clf = OneClassSVM(gamma='auto').fit(X)
>>> clf.predict(X)
array([-1,  1,  1,  1, -1])
>>> clf.score_samples(X)  # doctest: +ELLIPSIS

*)

(* TEST TODO
let%expect_test "OneClassSVM" =
  let open Sklearn.Svm in
  let x = (Np.matrixf [|[|0|]; [|0.44|]; [|0.45|]; [|0.46|]; [|1|]|]) in  
  let clf = OneClassSVM(gamma='auto').fit ~x () in  
  print_ndarray @@ OneClassSVM.predict ~x clf;  
  [%expect {|
      array([-1,  1,  1,  1, -1])      
  |}]
  print_ndarray @@ OneClassSVM.score_samples ~x clf # doctest: +ELLIPSIS;  
  [%expect {|
  |}]

*)

(* SVC *)
(*
>>> import numpy as np
>>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
>>> y = np.array([1, 1, 2, 2])
>>> from sklearn.svm import SVC
>>> clf = SVC(gamma='auto')
>>> clf.fit(X, y)
SVC(gamma='auto')
>>> print(clf.predict([[-0.8, -1]]))
[1]


*)

let%expect_test "SVC" =
  let open Sklearn.Svm in
  let x =
    Np.matrixf [| [| -1.; -1. |]; [| -2.; -1. |]; [| 1.; 1. |]; [| 2.; 1. |] |]
  in
  let y = Np.vectorf [| 1.; 1.; 2.; 2. |] in
  let clf = SVC.create ~gamma:`Auto () in
  print SVC.pp @@ SVC.fit clf ~x ~y;
  [%expect
    {|
            SVC(gamma='auto')
    |}];
  print_ndarray @@ SVC.predict clf ~x:(Np.matrixf [| [| -0.8; -1. |] |]);
  [%expect {|
            [1.]
    |}]

(* SVR *)
(*
>>> from sklearn.svm import SVR
>>> import numpy as np
>>> n_samples, n_features = 10, 5
>>> rng = np.random.RandomState(0)
>>> y = rng.randn(n_samples)
>>> X = rng.randn(n_samples, n_features)
>>> clf = SVR(C=1.0, epsilon=0.2)
>>> clf.fit(X, y)
SVR(epsilon=0.2)


*)

let%expect_test "SVR" =
  let n_samples, n_features = (10, 5) in
  Np.Random.seed 0;
  let y = Np.Random.uniform ~size:[ n_samples ] () in
  let x = Np.Random.uniform ~size:[ n_samples; n_features ] () in
  print_ndarray x;
  [%expect
    {|
    [[0.79172504 0.52889492 0.56804456 0.92559664 0.07103606]
     [0.0871293  0.0202184  0.83261985 0.77815675 0.87001215]
     [0.97861834 0.79915856 0.46147936 0.78052918 0.11827443]
     [0.63992102 0.14335329 0.94466892 0.52184832 0.41466194]
     [0.26455561 0.77423369 0.45615033 0.56843395 0.0187898 ]
     [0.6176355  0.61209572 0.616934   0.94374808 0.6818203 ]
     [0.3595079  0.43703195 0.6976312  0.06022547 0.66676672]
     [0.67063787 0.21038256 0.1289263  0.31542835 0.36371077]
     [0.57019677 0.43860151 0.98837384 0.10204481 0.20887676]
     [0.16130952 0.65310833 0.2532916  0.46631077 0.24442559]] |}];
  print_ndarray y;
  [%expect
    {|
    [0.5488135  0.71518937 0.60276338 0.54488318 0.4236548  0.64589411
     0.43758721 0.891773   0.96366276 0.38344152] |}];
  let open Sklearn.Svm in
  let clf = SVR.create ~c:1.0 ~epsilon:0.2 () in
  print SVR.pp @@ SVR.fit clf ~x ~y;
  [%expect
    {|
            SVR(epsilon=0.2)
    |}]
