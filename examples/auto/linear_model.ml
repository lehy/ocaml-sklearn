(* ARDRegression *)
(*
>>> from sklearn import linear_model
>>> clf = linear_model.ARDRegression()
>>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
ARDRegression()
>>> clf.predict([[1, 1]])
array([1.])

*)

(* TEST TODO
let%expect_test "ARDRegression" =
  let open Sklearn.Linear_model in
  let clf = .aRDRegression linear_model in  
  print_ndarray @@ .fit (matrixi [|[|0;0|]; [|1; 1|]; [|2; 2|]|]) (vectori [|0; 1; 2|]) clf;  
  [%expect {|
      ARDRegression()      
  |}]
  print_ndarray @@ .predict (matrixi [|[|1; 1|]|]) clf;  
  [%expect {|
      array([1.])      
  |}]

*)



(* BayesianRidge *)
(*
>>> from sklearn import linear_model
>>> clf = linear_model.BayesianRidge()
>>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
BayesianRidge()
>>> clf.predict([[1, 1]])
array([1.])

*)

(* TEST TODO
let%expect_test "BayesianRidge" =
  let open Sklearn.Linear_model in
  let clf = .bayesianRidge linear_model in  
  print_ndarray @@ .fit (matrixi [|[|0;0|]; [|1; 1|]; [|2; 2|]|]) (vectori [|0; 1; 2|]) clf;  
  [%expect {|
      BayesianRidge()      
  |}]
  print_ndarray @@ .predict (matrixi [|[|1; 1|]|]) clf;  
  [%expect {|
      array([1.])      
  |}]

*)



(* ElasticNet *)
(*
>>> from sklearn.linear_model import ElasticNet
>>> from sklearn.datasets import make_regression

*)

(* TEST TODO
let%expect_test "ElasticNet" =
  let open Sklearn.Linear_model in
  [%expect {|
  |}]

*)



(* ElasticNet *)
(*
>>> X, y = make_regression(n_features=2, random_state=0)
>>> regr = ElasticNet(random_state=0)
>>> regr.fit(X, y)
ElasticNet(random_state=0)
>>> print(regr.coef_)
[18.83816048 64.55968825]
>>> print(regr.intercept_)
1.451...
>>> print(regr.predict([[0, 0]]))
[1.451...]

*)

(* TEST TODO
let%expect_test "ElasticNet" =
  let open Sklearn.Linear_model in
  let x, y = make_regression ~n_features:2 ~random_state:0 () in  
  let regr = ElasticNet.create ~random_state:0 () in  
  print ElasticNet.pp @@ ElasticNet.fit ~x y regr;  
  [%expect {|
      ElasticNet(random_state=0)      
  |}]
  print_ndarray @@ print regr.coef_ ();  
  [%expect {|
      [18.83816048 64.55968825]      
  |}]
  print_ndarray @@ print regr.intercept_ ();  
  [%expect {|
      1.451...      
  |}]
  print_ndarray @@ print(ElasticNet.predict (matrixi [|[|0; 0|]|])) regr;  
  [%expect {|
      [1.451...]      
  |}]

*)



(* ElasticNetCV *)
(*
>>> from sklearn.linear_model import ElasticNetCV
>>> from sklearn.datasets import make_regression

*)

(* TEST TODO
let%expect_test "ElasticNetCV" =
  let open Sklearn.Linear_model in
  [%expect {|
  |}]

*)



(* ElasticNetCV *)
(*
>>> X, y = make_regression(n_features=2, random_state=0)
>>> regr = ElasticNetCV(cv=5, random_state=0)
>>> regr.fit(X, y)
ElasticNetCV(cv=5, random_state=0)
>>> print(regr.alpha_)
0.199...
>>> print(regr.intercept_)
0.398...
>>> print(regr.predict([[0, 0]]))
[0.398...]

*)

(* TEST TODO
let%expect_test "ElasticNetCV" =
  let open Sklearn.Linear_model in
  let x, y = make_regression ~n_features:2 ~random_state:0 () in  
  let regr = ElasticNetCV.create ~cv:5 ~random_state:0 () in  
  print ElasticNetCV.pp @@ ElasticNetCV.fit ~x y regr;  
  [%expect {|
      ElasticNetCV(cv=5, random_state=0)      
  |}]
  print_ndarray @@ print regr.alpha_ ();  
  [%expect {|
      0.199...      
  |}]
  print_ndarray @@ print regr.intercept_ ();  
  [%expect {|
      0.398...      
  |}]
  print_ndarray @@ print(ElasticNetCV.predict (matrixi [|[|0; 0|]|])) regr;  
  [%expect {|
      [0.398...]      
  |}]

*)



(* HuberRegressor *)
(*
>>> import numpy as np
>>> from sklearn.linear_model import HuberRegressor, LinearRegression
>>> from sklearn.datasets import make_regression
>>> rng = np.random.RandomState(0)
>>> X, y, coef = make_regression(
...     n_samples=200, n_features=2, noise=4.0, coef=True, random_state=0)
>>> X[:4] = rng.uniform(10, 20, (4, 2))
>>> y[:4] = rng.uniform(10, 20, 4)
>>> huber = HuberRegressor().fit(X, y)
>>> huber.score(X, y)
-7.284608623514573
>>> huber.predict(X[:1,])
array([806.7200...])
>>> linear = LinearRegression().fit(X, y)
>>> print("True coefficients:", coef)
True coefficients: [20.4923...  34.1698...]
>>> print("Huber coefficients:", huber.coef_)
Huber coefficients: [17.7906... 31.0106...]
>>> print("Linear Regression coefficients:", linear.coef_)
Linear Regression coefficients: [-1.9221...  7.0226...]

*)

(* TEST TODO
let%expect_test "HuberRegressor" =
  let open Sklearn.Linear_model in
  let rng = np..randomState ~0 random in  
  let x, y, coef = make_regression ~n_samples:200 ~n_features:2 ~noise:4.0 ~coef:true ~random_state:0 () in  
  print_ndarray @@ x[:4] = .uniform ~10 20 (4 2) rng;  
  print_ndarray @@ y[:4] = .uniform ~10 20 ~4 rng;  
  let huber = HuberRegressor().fit ~x y () in  
  print_ndarray @@ HuberRegressor.score ~x y huber;  
  [%expect {|
      -7.284608623514573      
  |}]
  print_ndarray @@ HuberRegressor.predict x[:1 ] huber;  
  [%expect {|
      array([806.7200...])      
  |}]
  let linear = LinearRegression().fit ~x y () in  
  print_ndarray @@ print "true coefficients:" ~coef ();  
  [%expect {|
      True coefficients: [20.4923...  34.1698...]      
  |}]
  print_ndarray @@ print "Huber coefficients:" huber.coef_ ();  
  [%expect {|
      Huber coefficients: [17.7906... 31.0106...]      
  |}]
  print_ndarray @@ print "Linear Regression coefficients:" linear.coef_ ();  
  [%expect {|
      Linear Regression coefficients: [-1.9221...  7.0226...]      
  |}]

*)



(* Lars *)
(*
>>> from sklearn import linear_model
>>> reg = linear_model.Lars(n_nonzero_coefs=1)
>>> reg.fit([[-1, 1], [0, 0], [1, 1]], [-1.1111, 0, -1.1111])
Lars(n_nonzero_coefs=1)
>>> print(reg.coef_)
[ 0. -1.11...]

*)

(* TEST TODO
let%expect_test "Lars" =
  let open Sklearn.Linear_model in
  let reg = .lars ~n_nonzero_coefs:1 linear_model in  
  print_ndarray @@ .fit (matrixi [|[|-1; 1|]; [|0; 0|]; [|1; 1|]|]) [-1.1111 ~0 -1.1111] reg;  
  [%expect {|
      Lars(n_nonzero_coefs=1)      
  |}]
  print_ndarray @@ print reg.coef_ ();  
  [%expect {|
      [ 0. -1.11...]      
  |}]

*)



(* LarsCV *)
(*
>>> from sklearn.linear_model import LarsCV
>>> from sklearn.datasets import make_regression
>>> X, y = make_regression(n_samples=200, noise=4.0, random_state=0)
>>> reg = LarsCV(cv=5).fit(X, y)
>>> reg.score(X, y)
0.9996...
>>> reg.alpha_
0.0254...
>>> reg.predict(X[:1,])
array([154.0842...])

*)

(* TEST TODO
let%expect_test "LarsCV" =
  let open Sklearn.Linear_model in
  let x, y = make_regression ~n_samples:200 ~noise:4.0 ~random_state:0 () in  
  let reg = LarsCV(cv=5).fit ~x y () in  
  print_ndarray @@ LarsCV.score ~x y reg;  
  [%expect {|
      0.9996...      
  |}]
  print_ndarray @@ LarsCV.alpha_ reg;  
  [%expect {|
      0.0254...      
  |}]
  print_ndarray @@ LarsCV.predict x[:1 ] reg;  
  [%expect {|
      array([154.0842...])      
  |}]

*)



(* Lasso *)
(*
>>> from sklearn import linear_model
>>> clf = linear_model.Lasso(alpha=0.1)
>>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
Lasso(alpha=0.1)
>>> print(clf.coef_)
[0.85 0.  ]
>>> print(clf.intercept_)
0.15...

*)

(* TEST TODO
let%expect_test "Lasso" =
  let open Sklearn.Linear_model in
  let clf = .lasso ~alpha:0.1 linear_model in  
  print_ndarray @@ .fit (matrixi [|[|0;0|]; [|1; 1|]; [|2; 2|]|]) (vectori [|0; 1; 2|]) clf;  
  [%expect {|
      Lasso(alpha=0.1)      
  |}]
  print_ndarray @@ print clf.coef_ ();  
  [%expect {|
      [0.85 0.  ]      
  |}]
  print_ndarray @@ print clf.intercept_ ();  
  [%expect {|
      0.15...      
  |}]

*)



(* LassoCV *)
(*
>>> from sklearn.linear_model import LassoCV
>>> from sklearn.datasets import make_regression
>>> X, y = make_regression(noise=4, random_state=0)
>>> reg = LassoCV(cv=5, random_state=0).fit(X, y)
>>> reg.score(X, y)
0.9993...
>>> reg.predict(X[:1,])
array([-78.4951...])

*)

(* TEST TODO
let%expect_test "LassoCV" =
  let open Sklearn.Linear_model in
  let x, y = make_regression ~noise:4 ~random_state:0 () in  
  let reg = LassoCV(cv=5, random_state=0).fit ~x y () in  
  print_ndarray @@ LassoCV.score ~x y reg;  
  [%expect {|
      0.9993...      
  |}]
  print_ndarray @@ LassoCV.predict x[:1 ] reg;  
  [%expect {|
      array([-78.4951...])      
  |}]

*)



(* LassoLars *)
(*
>>> from sklearn import linear_model
>>> reg = linear_model.LassoLars(alpha=0.01)
>>> reg.fit([[-1, 1], [0, 0], [1, 1]], [-1, 0, -1])
LassoLars(alpha=0.01)
>>> print(reg.coef_)
[ 0.         -0.963257...]

*)

(* TEST TODO
let%expect_test "LassoLars" =
  let open Sklearn.Linear_model in
  let reg = .lassoLars ~alpha:0.01 linear_model in  
  print_ndarray @@ .fit (matrixi [|[|-1; 1|]; [|0; 0|]; [|1; 1|]|]) [-1 ~0 -1] reg;  
  [%expect {|
      LassoLars(alpha=0.01)      
  |}]
  print_ndarray @@ print reg.coef_ ();  
  [%expect {|
      [ 0.         -0.963257...]      
  |}]

*)



(* LassoLarsCV *)
(*
>>> from sklearn.linear_model import LassoLarsCV
>>> from sklearn.datasets import make_regression
>>> X, y = make_regression(noise=4.0, random_state=0)
>>> reg = LassoLarsCV(cv=5).fit(X, y)
>>> reg.score(X, y)
0.9992...
>>> reg.alpha_
0.0484...
>>> reg.predict(X[:1,])
array([-77.8723...])

*)

(* TEST TODO
let%expect_test "LassoLarsCV" =
  let open Sklearn.Linear_model in
  let x, y = make_regression ~noise:4.0 ~random_state:0 () in  
  let reg = LassoLarsCV(cv=5).fit ~x y () in  
  print_ndarray @@ LassoLarsCV.score ~x y reg;  
  [%expect {|
      0.9992...      
  |}]
  print_ndarray @@ LassoLarsCV.alpha_ reg;  
  [%expect {|
      0.0484...      
  |}]
  print_ndarray @@ LassoLarsCV.predict x[:1 ] reg;  
  [%expect {|
      array([-77.8723...])      
  |}]

*)



(* LassoLarsIC *)
(*
>>> from sklearn import linear_model
>>> reg = linear_model.LassoLarsIC(criterion='bic')
>>> reg.fit([[-1, 1], [0, 0], [1, 1]], [-1.1111, 0, -1.1111])
LassoLarsIC(criterion='bic')
>>> print(reg.coef_)
[ 0.  -1.11...]

*)

(* TEST TODO
let%expect_test "LassoLarsIC" =
  let open Sklearn.Linear_model in
  let reg = .lassoLarsIC ~criterion:'bic' linear_model in  
  print_ndarray @@ .fit (matrixi [|[|-1; 1|]; [|0; 0|]; [|1; 1|]|]) [-1.1111 ~0 -1.1111] reg;  
  [%expect {|
      LassoLarsIC(criterion='bic')      
  |}]
  print_ndarray @@ print reg.coef_ ();  
  [%expect {|
      [ 0.  -1.11...]      
  |}]

*)



(* LinearRegression *)
(*
>>> import numpy as np
>>> from sklearn.linear_model import LinearRegression
>>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
>>> # y = 1 * x_0 + 2 * x_1 + 3
>>> y = np.dot(X, np.array([1, 2])) + 3
>>> reg = LinearRegression().fit(X, y)
>>> reg.score(X, y)
1.0
>>> reg.coef_
array([1., 2.])
>>> reg.intercept_
3.0000...
>>> reg.predict(np.array([[3, 5]]))

*)

(* TEST TODO
let%expect_test "LinearRegression" =
  let open Sklearn.Linear_model in
  let x = .array (matrixi [|[|1; 1|]; [|1; 2|]; [|2; 2|]; [|2; 3|]|]) np in  
  print_ndarray @@ # y = 1 * x_0 + 2 * x_1 + 3;  
  let y = .dot ~x np.array((vectori [|1; 2|])) np + 3 in  
  let reg = LinearRegression().fit ~x y () in  
  print_ndarray @@ LinearRegression.score ~x y reg;  
  [%expect {|
      1.0      
  |}]
  print_ndarray @@ LinearRegression.coef_ reg;  
  [%expect {|
      array([1., 2.])      
  |}]
  print_ndarray @@ LinearRegression.intercept_ reg;  
  [%expect {|
      3.0000...      
  |}]
  print_ndarray @@ LinearRegression.predict np.array((matrixi [|[|3; 5|]|])) reg;  
  [%expect {|
  |}]

*)



(* LogisticRegression *)
(*
>>> from sklearn.datasets import load_iris
>>> from sklearn.linear_model import LogisticRegression
>>> X, y = load_iris(return_X_y=True)
>>> clf = LogisticRegression(random_state=0).fit(X, y)
>>> clf.predict(X[:2, :])
array([0, 0])
>>> clf.predict_proba(X[:2, :])
array([[9.8...e-01, 1.8...e-02, 1.4...e-08],
       [9.7...e-01, 2.8...e-02, ...e-08]])
>>> clf.score(X, y)

*)

(* TEST TODO
let%expect_test "LogisticRegression" =
  let open Sklearn.Linear_model in
  let x, y = load_iris ~return_X_y:true () in  
  let clf = LogisticRegression(random_state=0).fit ~x y () in  
  print_ndarray @@ LogisticRegression.predict x[:2 :] clf;  
  [%expect {|
      array([0, 0])      
  |}]
  print_ndarray @@ LogisticRegression.predict_proba x[:2 :] clf;  
  [%expect {|
      array([[9.8...e-01, 1.8...e-02, 1.4...e-08],      
             [9.7...e-01, 2.8...e-02, ...e-08]])      
  |}]
  print_ndarray @@ LogisticRegression.score ~x y clf;  
  [%expect {|
  |}]

*)



(* LogisticRegressionCV *)
(*
>>> from sklearn.datasets import load_iris
>>> from sklearn.linear_model import LogisticRegressionCV
>>> X, y = load_iris(return_X_y=True)
>>> clf = LogisticRegressionCV(cv=5, random_state=0).fit(X, y)
>>> clf.predict(X[:2, :])
array([0, 0])
>>> clf.predict_proba(X[:2, :]).shape
(2, 3)
>>> clf.score(X, y)
0.98...

*)

(* TEST TODO
let%expect_test "LogisticRegressionCV" =
  let open Sklearn.Linear_model in
  let x, y = load_iris ~return_X_y:true () in  
  let clf = LogisticRegressionCV(cv=5, random_state=0).fit ~x y () in  
  print_ndarray @@ LogisticRegressionCV.predict x[:2 :] clf;  
  [%expect {|
      array([0, 0])      
  |}]
  print_ndarray @@ LogisticRegressionCV.predict_proba x[:2 :] LogisticRegressionCV.shape clf;  
  [%expect {|
      (2, 3)      
  |}]
  print_ndarray @@ LogisticRegressionCV.score ~x y clf;  
  [%expect {|
      0.98...      
  |}]

*)



(* MultiTaskElasticNet *)
(*
>>> from sklearn import linear_model
>>> clf = linear_model.MultiTaskElasticNet(alpha=0.1)
>>> clf.fit([[0,0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]])
MultiTaskElasticNet(alpha=0.1)
>>> print(clf.coef_)
[[0.45663524 0.45612256]
 [0.45663524 0.45612256]]
>>> print(clf.intercept_)
[0.0872422 0.0872422]

*)

(* TEST TODO
let%expect_test "MultiTaskElasticNet" =
  let open Sklearn.Linear_model in
  let clf = .multiTaskElasticNet ~alpha:0.1 linear_model in  
  print_ndarray @@ .fit (matrixi [|[|0;0|]; [|1; 1|]; [|2; 2|]|]) (matrixi [|[|0; 0|]; [|1; 1|]; [|2; 2|]|]) clf;  
  [%expect {|
      MultiTaskElasticNet(alpha=0.1)      
  |}]
  print_ndarray @@ print clf.coef_ ();  
  [%expect {|
      [[0.45663524 0.45612256]      
       [0.45663524 0.45612256]]      
  |}]
  print_ndarray @@ print clf.intercept_ ();  
  [%expect {|
      [0.0872422 0.0872422]      
  |}]

*)



(* MultiTaskElasticNetCV *)
(*
>>> from sklearn import linear_model
>>> clf = linear_model.MultiTaskElasticNetCV(cv=3)
>>> clf.fit([[0,0], [1, 1], [2, 2]],
...         [[0, 0], [1, 1], [2, 2]])
MultiTaskElasticNetCV(cv=3)
>>> print(clf.coef_)
[[0.52875032 0.46958558]
 [0.52875032 0.46958558]]
>>> print(clf.intercept_)
[0.00166409 0.00166409]

*)

(* TEST TODO
let%expect_test "MultiTaskElasticNetCV" =
  let open Sklearn.Linear_model in
  let clf = .multiTaskElasticNetCV ~cv:3 linear_model in  
  print_ndarray @@ .fit (matrixi [|[|0;0|]; [|1; 1|]; [|2; 2|]|]) (matrixi [|[|0; 0|]; [|1; 1|]; [|2; 2|]|]) clf;  
  [%expect {|
      MultiTaskElasticNetCV(cv=3)      
  |}]
  print_ndarray @@ print clf.coef_ ();  
  [%expect {|
      [[0.52875032 0.46958558]      
       [0.52875032 0.46958558]]      
  |}]
  print_ndarray @@ print clf.intercept_ ();  
  [%expect {|
      [0.00166409 0.00166409]      
  |}]

*)



(* MultiTaskLasso *)
(*
>>> from sklearn import linear_model
>>> clf = linear_model.MultiTaskLasso(alpha=0.1)
>>> clf.fit([[0,0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]])
MultiTaskLasso(alpha=0.1)
>>> print(clf.coef_)
[[0.89393398 0.        ]
 [0.89393398 0.        ]]
>>> print(clf.intercept_)
[0.10606602 0.10606602]

*)

(* TEST TODO
let%expect_test "MultiTaskLasso" =
  let open Sklearn.Linear_model in
  let clf = .multiTaskLasso ~alpha:0.1 linear_model in  
  print_ndarray @@ .fit (matrixi [|[|0;0|]; [|1; 1|]; [|2; 2|]|]) (matrixi [|[|0; 0|]; [|1; 1|]; [|2; 2|]|]) clf;  
  [%expect {|
      MultiTaskLasso(alpha=0.1)      
  |}]
  print_ndarray @@ print clf.coef_ ();  
  [%expect {|
      [[0.89393398 0.        ]      
       [0.89393398 0.        ]]      
  |}]
  print_ndarray @@ print clf.intercept_ ();  
  [%expect {|
      [0.10606602 0.10606602]      
  |}]

*)



(* MultiTaskLassoCV *)
(*
>>> from sklearn.linear_model import MultiTaskLassoCV
>>> from sklearn.datasets import make_regression
>>> from sklearn.metrics import r2_score
>>> X, y = make_regression(n_targets=2, noise=4, random_state=0)
>>> reg = MultiTaskLassoCV(cv=5, random_state=0).fit(X, y)
>>> r2_score(y, reg.predict(X))
0.9994...
>>> reg.alpha_
0.5713...
>>> reg.predict(X[:1,])
array([[153.7971...,  94.9015...]])

*)

(* TEST TODO
let%expect_test "MultiTaskLassoCV" =
  let open Sklearn.Linear_model in
  let x, y = make_regression ~n_targets:2 ~noise:4 ~random_state:0 () in  
  let reg = MultiTaskLassoCV(cv=5, random_state=0).fit ~x y () in  
  print_ndarray @@ r2_score(y, MultiTaskLassoCV.predict x) reg;  
  [%expect {|
      0.9994...      
  |}]
  print_ndarray @@ MultiTaskLassoCV.alpha_ reg;  
  [%expect {|
      0.5713...      
  |}]
  print_ndarray @@ MultiTaskLassoCV.predict x[:1 ] reg;  
  [%expect {|
      array([[153.7971...,  94.9015...]])      
  |}]

*)



(* OrthogonalMatchingPursuit *)
(*
>>> from sklearn.linear_model import OrthogonalMatchingPursuit
>>> from sklearn.datasets import make_regression
>>> X, y = make_regression(noise=4, random_state=0)
>>> reg = OrthogonalMatchingPursuit().fit(X, y)
>>> reg.score(X, y)
0.9991...
>>> reg.predict(X[:1,])
array([-78.3854...])

*)

(* TEST TODO
let%expect_test "OrthogonalMatchingPursuit" =
  let open Sklearn.Linear_model in
  let x, y = make_regression ~noise:4 ~random_state:0 () in  
  let reg = OrthogonalMatchingPursuit().fit ~x y () in  
  print_ndarray @@ OrthogonalMatchingPursuit.score ~x y reg;  
  [%expect {|
      0.9991...      
  |}]
  print_ndarray @@ OrthogonalMatchingPursuit.predict x[:1 ] reg;  
  [%expect {|
      array([-78.3854...])      
  |}]

*)



(* OrthogonalMatchingPursuitCV *)
(*
>>> from sklearn.linear_model import OrthogonalMatchingPursuitCV
>>> from sklearn.datasets import make_regression
>>> X, y = make_regression(n_features=100, n_informative=10,
...                        noise=4, random_state=0)
>>> reg = OrthogonalMatchingPursuitCV(cv=5).fit(X, y)
>>> reg.score(X, y)
0.9991...
>>> reg.n_nonzero_coefs_
10
>>> reg.predict(X[:1,])
array([-78.3854...])

*)

(* TEST TODO
let%expect_test "OrthogonalMatchingPursuitCV" =
  let open Sklearn.Linear_model in
  let x, y = make_regression ~n_features:100 ~n_informative:10 ~noise:4 ~random_state:0 () in  
  let reg = OrthogonalMatchingPursuitCV(cv=5).fit ~x y () in  
  print_ndarray @@ OrthogonalMatchingPursuitCV.score ~x y reg;  
  [%expect {|
      0.9991...      
  |}]
  print_ndarray @@ OrthogonalMatchingPursuitCV.n_nonzero_coefs_ reg;  
  [%expect {|
      10      
  |}]
  print_ndarray @@ OrthogonalMatchingPursuitCV.predict x[:1 ] reg;  
  [%expect {|
      array([-78.3854...])      
  |}]

*)



(* PassiveAggressiveClassifier *)
(*
>>> from sklearn.linear_model import PassiveAggressiveClassifier
>>> from sklearn.datasets import make_classification

*)

(* TEST TODO
let%expect_test "PassiveAggressiveClassifier" =
  let open Sklearn.Linear_model in
  [%expect {|
  |}]

*)



(* PassiveAggressiveClassifier *)
(*
>>> X, y = make_classification(n_features=4, random_state=0)
>>> clf = PassiveAggressiveClassifier(max_iter=1000, random_state=0,
... tol=1e-3)
>>> clf.fit(X, y)
PassiveAggressiveClassifier(random_state=0)
>>> print(clf.coef_)
[[0.26642044 0.45070924 0.67251877 0.64185414]]
>>> print(clf.intercept_)
[1.84127814]
>>> print(clf.predict([[0, 0, 0, 0]]))
[1]

*)

(* TEST TODO
let%expect_test "PassiveAggressiveClassifier" =
  let open Sklearn.Linear_model in
  let x, y = make_classification ~n_features:4 ~random_state:0 () in  
  let clf = PassiveAggressiveClassifier.create ~max_iter:1000 ~random_state:0 ~tol:1e-3 () in  
  print PassiveAggressiveClassifier.pp @@ PassiveAggressiveClassifier.fit ~x y clf;  
  [%expect {|
      PassiveAggressiveClassifier(random_state=0)      
  |}]
  print_ndarray @@ print clf.coef_ ();  
  [%expect {|
      [[0.26642044 0.45070924 0.67251877 0.64185414]]      
  |}]
  print_ndarray @@ print clf.intercept_ ();  
  [%expect {|
      [1.84127814]      
  |}]
  print_ndarray @@ print(PassiveAggressiveClassifier.predict (matrixi [|[|0; 0; 0; 0|]|])) clf;  
  [%expect {|
      [1]      
  |}]

*)



(* PassiveAggressiveRegressor *)
(*
>>> from sklearn.linear_model import PassiveAggressiveRegressor
>>> from sklearn.datasets import make_regression

*)

(* TEST TODO
let%expect_test "PassiveAggressiveRegressor" =
  let open Sklearn.Linear_model in
  [%expect {|
  |}]

*)



(* PassiveAggressiveRegressor *)
(*
>>> X, y = make_regression(n_features=4, random_state=0)
>>> regr = PassiveAggressiveRegressor(max_iter=100, random_state=0,
... tol=1e-3)
>>> regr.fit(X, y)
PassiveAggressiveRegressor(max_iter=100, random_state=0)
>>> print(regr.coef_)
[20.48736655 34.18818427 67.59122734 87.94731329]
>>> print(regr.intercept_)
[-0.02306214]
>>> print(regr.predict([[0, 0, 0, 0]]))
[-0.02306214]

*)

(* TEST TODO
let%expect_test "PassiveAggressiveRegressor" =
  let open Sklearn.Linear_model in
  let x, y = make_regression ~n_features:4 ~random_state:0 () in  
  let regr = PassiveAggressiveRegressor.create ~max_iter:100 ~random_state:0 ~tol:1e-3 () in  
  print PassiveAggressiveRegressor.pp @@ PassiveAggressiveRegressor.fit ~x y regr;  
  [%expect {|
      PassiveAggressiveRegressor(max_iter=100, random_state=0)      
  |}]
  print_ndarray @@ print regr.coef_ ();  
  [%expect {|
      [20.48736655 34.18818427 67.59122734 87.94731329]      
  |}]
  print_ndarray @@ print regr.intercept_ ();  
  [%expect {|
      [-0.02306214]      
  |}]
  print_ndarray @@ print(PassiveAggressiveRegressor.predict (matrixi [|[|0; 0; 0; 0|]|])) regr;  
  [%expect {|
      [-0.02306214]      
  |}]

*)



(* Perceptron *)
(*
>>> from sklearn.datasets import load_digits
>>> from sklearn.linear_model import Perceptron
>>> X, y = load_digits(return_X_y=True)
>>> clf = Perceptron(tol=1e-3, random_state=0)
>>> clf.fit(X, y)
Perceptron()
>>> clf.score(X, y)
0.939...

*)

(* TEST TODO
let%expect_test "Perceptron" =
  let open Sklearn.Linear_model in
  let x, y = load_digits ~return_X_y:true () in  
  let clf = Perceptron.create ~tol:1e-3 ~random_state:0 () in  
  print Perceptron.pp @@ Perceptron.fit ~x y clf;  
  [%expect {|
      Perceptron()      
  |}]
  print_ndarray @@ Perceptron.score ~x y clf;  
  [%expect {|
      0.939...      
  |}]

*)



(* RANSACRegressor *)
(*
>>> from sklearn.linear_model import RANSACRegressor
>>> from sklearn.datasets import make_regression
>>> X, y = make_regression(
...     n_samples=200, n_features=2, noise=4.0, random_state=0)
>>> reg = RANSACRegressor(random_state=0).fit(X, y)
>>> reg.score(X, y)
0.9885...
>>> reg.predict(X[:1,])
array([-31.9417...])

*)

(* TEST TODO
let%expect_test "RANSACRegressor" =
  let open Sklearn.Linear_model in
  let x, y = make_regression ~n_samples:200 ~n_features:2 ~noise:4.0 ~random_state:0 () in  
  let reg = RANSACRegressor(random_state=0).fit ~x y () in  
  print_ndarray @@ RANSACRegressor.score ~x y reg;  
  [%expect {|
      0.9885...      
  |}]
  print_ndarray @@ RANSACRegressor.predict x[:1 ] reg;  
  [%expect {|
      array([-31.9417...])      
  |}]

*)



(* Ridge *)
(*
>>> from sklearn.linear_model import Ridge
>>> import numpy as np
>>> n_samples, n_features = 10, 5
>>> rng = np.random.RandomState(0)
>>> y = rng.randn(n_samples)
>>> X = rng.randn(n_samples, n_features)
>>> clf = Ridge(alpha=1.0)
>>> clf.fit(X, y)

*)

(* TEST TODO
let%expect_test "Ridge" =
  let open Sklearn.Linear_model in
  let n_samples, n_features = 10, 5 in  
  let rng = np..randomState ~0 random in  
  let y = .randn ~n_samples rng in  
  let x = .randn ~n_samples n_features rng in  
  let clf = Ridge.create ~alpha:1.0 () in  
  print Ridge.pp @@ Ridge.fit ~x y clf;  
  [%expect {|
  |}]

*)



(* RidgeCV *)
(*
>>> from sklearn.datasets import load_diabetes
>>> from sklearn.linear_model import RidgeCV
>>> X, y = load_diabetes(return_X_y=True)
>>> clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X, y)
>>> clf.score(X, y)
0.5166...

*)

(* TEST TODO
let%expect_test "RidgeCV" =
  let open Sklearn.Linear_model in
  let x, y = load_diabetes ~return_X_y:true () in  
  let clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit ~x y () in  
  print_ndarray @@ RidgeCV.score ~x y clf;  
  [%expect {|
      0.5166...      
  |}]

*)



(* RidgeClassifier *)
(*
>>> from sklearn.datasets import load_breast_cancer
>>> from sklearn.linear_model import RidgeClassifier
>>> X, y = load_breast_cancer(return_X_y=True)
>>> clf = RidgeClassifier().fit(X, y)
>>> clf.score(X, y)

*)

(* TEST TODO
let%expect_test "RidgeClassifier" =
  let open Sklearn.Linear_model in
  let x, y = load_breast_cancer ~return_X_y:true () in  
  let clf = RidgeClassifier().fit ~x y () in  
  print_ndarray @@ RidgeClassifier.score ~x y clf;  
  [%expect {|
  |}]

*)



(* RidgeClassifierCV *)
(*
>>> from sklearn.datasets import load_breast_cancer
>>> from sklearn.linear_model import RidgeClassifierCV
>>> X, y = load_breast_cancer(return_X_y=True)
>>> clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X, y)
>>> clf.score(X, y)
0.9630...

*)

(* TEST TODO
let%expect_test "RidgeClassifierCV" =
  let open Sklearn.Linear_model in
  let x, y = load_breast_cancer ~return_X_y:true () in  
  let clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit ~x y () in  
  print_ndarray @@ RidgeClassifierCV.score ~x y clf;  
  [%expect {|
      0.9630...      
  |}]

*)



(* SGDClassifier *)
(*
>>> import numpy as np
>>> from sklearn import linear_model
>>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
>>> Y = np.array([1, 1, 2, 2])
>>> clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
>>> clf.fit(X, Y)
SGDClassifier()

*)

(* TEST TODO
let%expect_test "SGDClassifier" =
  let open Sklearn.Linear_model in
  let x = .array (matrixi [|[|-1; -1|]; [|-2; -1|]; [|1; 1|]; [|2; 1|]|]) np in  
  let Y = .array (vectori [|1; 1; 2; 2|]) np in  
  let clf = .sGDClassifier ~max_iter:1000 ~tol:1e-3 linear_model in  
  print_ndarray @@ .fit ~x Y clf;  
  [%expect {|
      SGDClassifier()      
  |}]

*)



(* SGDClassifier *)
(*
>>> print(clf.predict([[-0.8, -1]]))

*)

(* TEST TODO
let%expect_test "SGDClassifier" =
  let open Sklearn.Linear_model in
  print_ndarray @@ print(.predict (matrix [|[|-0.8; -1|]|])) clf;  
  [%expect {|
  |}]

*)



(* SGDRegressor *)
(*
>>> import numpy as np
>>> from sklearn import linear_model
>>> n_samples, n_features = 10, 5
>>> rng = np.random.RandomState(0)
>>> y = rng.randn(n_samples)
>>> X = rng.randn(n_samples, n_features)
>>> clf = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
>>> clf.fit(X, y)
SGDRegressor()

*)

(* TEST TODO
let%expect_test "SGDRegressor" =
  let open Sklearn.Linear_model in
  let n_samples, n_features = 10, 5 in  
  let rng = np..randomState ~0 random in  
  let y = .randn ~n_samples rng in  
  let x = .randn ~n_samples n_features rng in  
  let clf = .sGDRegressor ~max_iter:1000 ~tol:1e-3 linear_model in  
  print_ndarray @@ .fit ~x y clf;  
  [%expect {|
      SGDRegressor()      
  |}]

*)



(* TheilSenRegressor *)
(*
>>> from sklearn.linear_model import TheilSenRegressor
>>> from sklearn.datasets import make_regression
>>> X, y = make_regression(
...     n_samples=200, n_features=2, noise=4.0, random_state=0)
>>> reg = TheilSenRegressor(random_state=0).fit(X, y)
>>> reg.score(X, y)
0.9884...
>>> reg.predict(X[:1,])
array([-31.5871...])

*)

(* TEST TODO
let%expect_test "TheilSenRegressor" =
  let open Sklearn.Linear_model in
  let x, y = make_regression ~n_samples:200 ~n_features:2 ~noise:4.0 ~random_state:0 () in  
  let reg = TheilSenRegressor(random_state=0).fit ~x y () in  
  print_ndarray @@ TheilSenRegressor.score ~x y reg;  
  [%expect {|
      0.9884...      
  |}]
  print_ndarray @@ TheilSenRegressor.predict x[:1 ] reg;  
  [%expect {|
      array([-31.5871...])      
  |}]

*)



(* lasso_path *)
(*
>>> X = np.array([[1, 2, 3.1], [2.3, 5.4, 4.3]]).T
>>> y = np.array([1, 2, 3.1])
>>> # Use lasso_path to compute a coefficient path
>>> _, coef_path, _ = lasso_path(X, y, alphas=[5., 1., .5])
>>> print(coef_path)
[[0.         0.         0.46874778]
 [0.2159048  0.4425765  0.23689075]]

*)

(* TEST TODO
let%expect_test "lasso_path" =
  let open Sklearn.Linear_model in
  let x = .array (matrix [|[|1; 2; 3.1|]; [|2.3; 5.4; 4.3|]|]) .t np in  
  let y = .array [1 ~2 3.1] np in  
  print_ndarray @@ # Use lasso_path to compute a coefficient path;  
  let _, coef_path, _ = lasso_path ~x y ~alphas:[5. 1. .5] () in  
  print_ndarray @@ print ~coef_path ();  
  [%expect {|
      [[0.         0.         0.46874778]      
       [0.2159048  0.4425765  0.23689075]]      
  |}]

*)



(* lasso_path *)
(*
>>> # Now use lars_path and 1D linear interpolation to compute the
>>> # same path
>>> from sklearn.linear_model import lars_path
>>> alphas, active, coef_path_lars = lars_path(X, y, method='lasso')
>>> from scipy import interpolate
>>> coef_path_continuous = interpolate.interp1d(alphas[::-1],
...                                             coef_path_lars[:, ::-1])
>>> print(coef_path_continuous([5., 1., .5]))
[[0.         0.         0.46915237]
 [0.2159048  0.4425765  0.23668876]]

*)

(* TEST TODO
let%expect_test "lasso_path" =
  let open Sklearn.Linear_model in
  print_ndarray @@ # Now use lars_path and 1D linear interpolation to compute the;  
  print_ndarray @@ # same path;  
  let alphas, active, coef_path_lars = lars_path ~x y ~method:'lasso' () in  
  let coef_path_continuous = .interp1d alphas[::-1] coef_path_lars[: ::-1] interpolate in  
  print_ndarray @@ print(coef_path_continuous [5. 1. .5] ());  
  [%expect {|
      [[0.         0.         0.46915237]      
       [0.2159048  0.4425765  0.23668876]]      
  |}]

*)



