let print f x = Format.printf "%a" f x
let print_py x = Format.printf "%s" (Py.Object.to_string x)
let print_ndarray = print Sklearn.Arr.pp

module Matrix = Owl.Dense.Matrix.D
let matrix = Sklearn.Arr.Float.matrix
let vector = Sklearn.Arr.Float.vector
let matrixi = Sklearn.Arr.Float.matrix
let vectori = Sklearn.Arr.Float.vector

let get x = match x with
  | None -> failwith "Option.get"
  | Some x -> x

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
  let x, y = make_classification ~n_features:4 ~random_state:(`Int 0) () in  
  let clf = LinearSVC.create ~random_state:(`Int 0) ~tol:1e-5 () in  
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
  let x, y, _coef = Sklearn.Datasets.make_regression ~n_features:4 ~random_state:(`Int 0) () in
  let open Sklearn.Svm in
  let regr = LinearSVR.create ~random_state:(`Int 0) ~tol:1e-5 () in
  print LinearSVR.pp @@ LinearSVR.fit regr ~x ~y;
  [%expect {|
          LinearSVR(C=1.0, dual=True, epsilon=0.0, fit_intercept=True,
                    intercept_scaling=1.0, loss='epsilon_insensitive', max_iter=1000,
                    random_state=0, tol=1e-05, verbose=0)
  |}];
  print_ndarray @@ LinearSVR.coef_ regr;
  [%expect {|
            [16.35841504 26.91644036 42.30619026 60.47800997]
    |}];
  print_ndarray @@ LinearSVR.intercept_ regr;
  [%expect {|
            [-4.29622263]
    |}];
  print_ndarray @@ LinearSVR.predict regr ~x:(matrix [|[|0.; 0.; 0.; 0.|]|]);
  [%expect {|
            [-4.29622263]
    |}];;



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
  let x = matrix [|[|-1.; -1.|]; [|-2.; -1.|]; [|1.; 1.|]; [|2.; 1.|]|] in
  let y = vector [|1.; 1.; 2.; 2.|] in
  let open Sklearn.Svm in
  let clf = NuSVC.create () in
  print NuSVC.pp @@ NuSVC.fit clf ~x ~y;
  [%expect {|
            NuSVC(break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
                  max_iter=-1, nu=0.5, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)
    |}];
  print_ndarray @@ NuSVC.predict clf ~x:(matrix [|[|-0.8; -1.|]|]);
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
  let n_samples, n_features = 10, 5 in
  Random.init 0;
  let y = Sklearn.Arr.of_bigarray @@ Owl.Arr.uniform [|n_samples|] in
  let x = Sklearn.Arr.of_bigarray @@ Matrix.uniform n_samples n_features in
  let clf = NuSVR.create ~c:1.0 ~nu:0.1 () in
  print NuSVR.pp @@ NuSVR.fit clf ~x ~y;
  [%expect {|
            NuSVR(C=1.0, cache_size=200, coef0=0.0, degree=3, gamma='scale', kernel='rbf',
                  max_iter=-1, nu=0.1, shrinking=True, tol=0.001, verbose=False)
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
  let x = (matrix [|[|0|]; [|0.44|]; [|0.45|]; [|0.46|]; [|1|]|]) in  
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
  let x = matrix [|[|-1.; -1.|]; [|-2.; -1.|]; [|1.; 1.|]; [|2.; 1.|]|] in
  let y = vector [|1.; 1.; 2.; 2.|] in
  let clf = SVC.create ~gamma:`Auto () in    
  print SVC.pp @@ SVC.fit clf ~x ~y;
  [%expect {|
            SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                max_iter=-1, probability=False, random_state=None, shrinking=True,
                tol=0.001, verbose=False)
    |}];
  print_ndarray @@ SVC.predict clf ~x:(matrix [|[|-0.8; -1.|]|]);
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
  let n_samples, n_features = 10, 5 in
  Random.init 0;
  let y = Sklearn.Arr.of_bigarray @@ Owl.Arr.uniform [|n_samples|] in
  let x = Sklearn.Arr.of_bigarray @@ Matrix.uniform n_samples n_features in
  let open Sklearn.Svm in
  let clf = SVR.create ~c:1.0 ~epsilon:0.2 () in
  print SVR.pp @@ SVR.fit clf ~x ~y;
  [%expect {|
            SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='scale',
                kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    |}]



