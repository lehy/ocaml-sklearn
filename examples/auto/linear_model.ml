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
let%expect_text "ARDRegression" =
    let linear_model = Sklearn.linear_model in
    clf = linear_model.ARDRegression()    
    print @@ fit clf [[0 0] [1 1] [2 2]] [0 1 2]
    [%expect {|
            ARDRegression()            
    |}]
    print @@ predict clf [[1 1]]
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
let%expect_text "BayesianRidge" =
    let linear_model = Sklearn.linear_model in
    clf = linear_model.BayesianRidge()    
    print @@ fit clf [[0 0] [1 1] [2 2]] [0 1 2]
    [%expect {|
            BayesianRidge()            
    |}]
    print @@ predict clf [[1 1]]
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
let%expect_text "ElasticNet" =
    let elasticNet = Sklearn.Linear_model.elasticNet in
    let make_regression = Sklearn.Datasets.make_regression in
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
let%expect_text "ElasticNet" =
    let x, y = make_regression n_features=2 random_state=0 in
    regr = ElasticNet(random_state=0)    
    print @@ fit regr x y
    [%expect {|
            ElasticNet(random_state=0)            
    |}]
    print(regr.coef_)    
    [%expect {|
            [18.83816048 64.55968825]            
    |}]
    print(regr.intercept_)    
    [%expect {|
            1.451...            
    |}]
    print(regr.predict([[0, 0]]))    
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
let%expect_text "ElasticNetCV" =
    let elasticNetCV = Sklearn.Linear_model.elasticNetCV in
    let make_regression = Sklearn.Datasets.make_regression in
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
let%expect_text "ElasticNetCV" =
    let x, y = make_regression n_features=2 random_state=0 in
    regr = ElasticNetCV(cv=5, random_state=0)    
    print @@ fit regr x y
    [%expect {|
            ElasticNetCV(cv=5, random_state=0)            
    |}]
    print(regr.alpha_)    
    [%expect {|
            0.199...            
    |}]
    print(regr.intercept_)    
    [%expect {|
            0.398...            
    |}]
    print(regr.predict([[0, 0]]))    
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
let%expect_text "HuberRegressor" =
    import numpy as np    
    from sklearn.linear_model import HuberRegressor, LinearRegression    
    let make_regression = Sklearn.Datasets.make_regression in
    rng = np.random.RandomState(0)    
    X, y, coef = make_regression(n_samples=200, n_features=2, noise=4.0, coef=True, random_state=0)    
    X[:4] = rng.uniform(10, 20, (4, 2))    
    y[:4] = rng.uniform(10, 20, 4)    
    huber = HuberRegressor().fit(X, y)    
    print @@ score huber x y
    [%expect {|
            -7.284608623514573            
    |}]
    print @@ predict huber x[:1 ]
    [%expect {|
            array([806.7200...])            
    |}]
    linear = LinearRegression().fit(X, y)    
    print("True coefficients:", coef)    
    [%expect {|
            True coefficients: [20.4923...  34.1698...]            
    |}]
    print("Huber coefficients:", huber.coef_)    
    [%expect {|
            Huber coefficients: [17.7906... 31.0106...]            
    |}]
    print("Linear Regression coefficients:", linear.coef_)    
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
let%expect_text "Lars" =
    let linear_model = Sklearn.linear_model in
    reg = linear_model.Lars(n_nonzero_coefs=1)    
    print @@ fit reg [[-1 1] [0 0] [1 1]] [-1.1111 0 -1.1111]
    [%expect {|
            Lars(n_nonzero_coefs=1)            
    |}]
    print(reg.coef_)    
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
let%expect_text "LarsCV" =
    let larsCV = Sklearn.Linear_model.larsCV in
    let make_regression = Sklearn.Datasets.make_regression in
    let x, y = make_regression n_samples=200 noise=4.0 random_state=0 in
    reg = LarsCV(cv=5).fit(X, y)    
    print @@ score reg x y
    [%expect {|
            0.9996...            
    |}]
    reg.alpha_    
    [%expect {|
            0.0254...            
    |}]
    print @@ predict reg x[:1 ]
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
let%expect_text "Lasso" =
    let linear_model = Sklearn.linear_model in
    clf = linear_model.Lasso(alpha=0.1)    
    print @@ fit clf [[0 0] [1 1] [2 2]] [0 1 2]
    [%expect {|
            Lasso(alpha=0.1)            
    |}]
    print(clf.coef_)    
    [%expect {|
            [0.85 0.  ]            
    |}]
    print(clf.intercept_)    
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
let%expect_text "LassoCV" =
    let lassoCV = Sklearn.Linear_model.lassoCV in
    let make_regression = Sklearn.Datasets.make_regression in
    let x, y = make_regression noise=4 random_state=0 in
    reg = LassoCV(cv=5, random_state=0).fit(X, y)    
    print @@ score reg x y
    [%expect {|
            0.9993...            
    |}]
    print @@ predict reg x[:1 ]
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
let%expect_text "LassoLars" =
    let linear_model = Sklearn.linear_model in
    reg = linear_model.LassoLars(alpha=0.01)    
    print @@ fit reg [[-1 1] [0 0] [1 1]] [-1 0 -1]
    [%expect {|
            LassoLars(alpha=0.01)            
    |}]
    print(reg.coef_)    
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
let%expect_text "LassoLarsCV" =
    let lassoLarsCV = Sklearn.Linear_model.lassoLarsCV in
    let make_regression = Sklearn.Datasets.make_regression in
    let x, y = make_regression noise=4.0 random_state=0 in
    reg = LassoLarsCV(cv=5).fit(X, y)    
    print @@ score reg x y
    [%expect {|
            0.9992...            
    |}]
    reg.alpha_    
    [%expect {|
            0.0484...            
    |}]
    print @@ predict reg x[:1 ]
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
let%expect_text "LassoLarsIC" =
    let linear_model = Sklearn.linear_model in
    reg = linear_model.LassoLarsIC(criterion='bic')    
    print @@ fit reg [[-1 1] [0 0] [1 1]] [-1.1111 0 -1.1111]
    [%expect {|
            LassoLarsIC(criterion='bic')            
    |}]
    print(reg.coef_)    
    [%expect {|
            [ 0.  -1.11...]            
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
let%expect_text "LogisticRegressionCV" =
    let load_iris = Sklearn.Datasets.load_iris in
    let logisticRegressionCV = Sklearn.Linear_model.logisticRegressionCV in
    let x, y = load_iris return_X_y=True in
    clf = LogisticRegressionCV(cv=5, random_state=0).fit(X, y)    
    print @@ predict clf x[:2 :]
    [%expect {|
            array([0, 0])            
    |}]
    clf.predict_proba(X[:2, :]).shape    
    [%expect {|
            (2, 3)            
    |}]
    print @@ score clf x y
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
let%expect_text "MultiTaskElasticNet" =
    let linear_model = Sklearn.linear_model in
    clf = linear_model.MultiTaskElasticNet(alpha=0.1)    
    print @@ fit clf [[0 0] [1 1] [2 2]] [[0 0] [1 1] [2 2]]
    [%expect {|
            MultiTaskElasticNet(alpha=0.1)            
    |}]
    print(clf.coef_)    
    [%expect {|
            [[0.45663524 0.45612256]            
             [0.45663524 0.45612256]]            
    |}]
    print(clf.intercept_)    
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
let%expect_text "MultiTaskElasticNetCV" =
    let linear_model = Sklearn.linear_model in
    clf = linear_model.MultiTaskElasticNetCV(cv=3)    
    print @@ fit clf [[0 0] [1 1] [2 2]] [[0 0] [1 1] [2 2]]
    [%expect {|
            MultiTaskElasticNetCV(cv=3)            
    |}]
    print(clf.coef_)    
    [%expect {|
            [[0.52875032 0.46958558]            
             [0.52875032 0.46958558]]            
    |}]
    print(clf.intercept_)    
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
let%expect_text "MultiTaskLasso" =
    let linear_model = Sklearn.linear_model in
    clf = linear_model.MultiTaskLasso(alpha=0.1)    
    print @@ fit clf [[0 0] [1 1] [2 2]] [[0 0] [1 1] [2 2]]
    [%expect {|
            MultiTaskLasso(alpha=0.1)            
    |}]
    print(clf.coef_)    
    [%expect {|
            [[0.89393398 0.        ]            
             [0.89393398 0.        ]]            
    |}]
    print(clf.intercept_)    
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
let%expect_text "MultiTaskLassoCV" =
    let multiTaskLassoCV = Sklearn.Linear_model.multiTaskLassoCV in
    let make_regression = Sklearn.Datasets.make_regression in
    let r2_score = Sklearn.Metrics.r2_score in
    let x, y = make_regression n_targets=2 noise=4 random_state=0 in
    reg = MultiTaskLassoCV(cv=5, random_state=0).fit(X, y)    
    r2_score(y, reg.predict(X))    
    [%expect {|
            0.9994...            
    |}]
    reg.alpha_    
    [%expect {|
            0.5713...            
    |}]
    print @@ predict reg x[:1 ]
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
let%expect_text "OrthogonalMatchingPursuit" =
    let orthogonalMatchingPursuit = Sklearn.Linear_model.orthogonalMatchingPursuit in
    let make_regression = Sklearn.Datasets.make_regression in
    let x, y = make_regression noise=4 random_state=0 in
    reg = OrthogonalMatchingPursuit().fit(X, y)    
    print @@ score reg x y
    [%expect {|
            0.9991...            
    |}]
    print @@ predict reg x[:1 ]
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
let%expect_text "OrthogonalMatchingPursuitCV" =
    let orthogonalMatchingPursuitCV = Sklearn.Linear_model.orthogonalMatchingPursuitCV in
    let make_regression = Sklearn.Datasets.make_regression in
    let x, y = make_regression n_features=100 n_informative=10 noise=4 random_state=0 in
    reg = OrthogonalMatchingPursuitCV(cv=5).fit(X, y)    
    print @@ score reg x y
    [%expect {|
            0.9991...            
    |}]
    reg.n_nonzero_coefs_    
    [%expect {|
            10            
    |}]
    print @@ predict reg x[:1 ]
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
let%expect_text "PassiveAggressiveClassifier" =
    let passiveAggressiveClassifier = Sklearn.Linear_model.passiveAggressiveClassifier in
    let make_classification = Sklearn.Datasets.make_classification in
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
let%expect_text "PassiveAggressiveClassifier" =
    let x, y = make_classification n_features=4 random_state=0 in
    clf = PassiveAggressiveClassifier(max_iter=1000, random_state=0,tol=1e-3)    
    print @@ fit clf x y
    [%expect {|
            PassiveAggressiveClassifier(random_state=0)            
    |}]
    print(clf.coef_)    
    [%expect {|
            [[0.26642044 0.45070924 0.67251877 0.64185414]]            
    |}]
    print(clf.intercept_)    
    [%expect {|
            [1.84127814]            
    |}]
    print(clf.predict([[0, 0, 0, 0]]))    
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
let%expect_text "PassiveAggressiveRegressor" =
    let passiveAggressiveRegressor = Sklearn.Linear_model.passiveAggressiveRegressor in
    let make_regression = Sklearn.Datasets.make_regression in
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
let%expect_text "PassiveAggressiveRegressor" =
    let x, y = make_regression n_features=4 random_state=0 in
    regr = PassiveAggressiveRegressor(max_iter=100, random_state=0,tol=1e-3)    
    print @@ fit regr x y
    [%expect {|
            PassiveAggressiveRegressor(max_iter=100, random_state=0)            
    |}]
    print(regr.coef_)    
    [%expect {|
            [20.48736655 34.18818427 67.59122734 87.94731329]            
    |}]
    print(regr.intercept_)    
    [%expect {|
            [-0.02306214]            
    |}]
    print(regr.predict([[0, 0, 0, 0]]))    
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
let%expect_text "Perceptron" =
    let load_digits = Sklearn.Datasets.load_digits in
    let perceptron = Sklearn.Linear_model.perceptron in
    let x, y = load_digits return_X_y=True in
    clf = Perceptron(tol=1e-3, random_state=0)    
    print @@ fit clf x y
    [%expect {|
            Perceptron()            
    |}]
    print @@ score clf x y
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
let%expect_text "RANSACRegressor" =
    let rANSACRegressor = Sklearn.Linear_model.rANSACRegressor in
    let make_regression = Sklearn.Datasets.make_regression in
    let x, y = make_regression n_samples=200 n_features=2 noise=4.0 random_state=0 in
    reg = RANSACRegressor(random_state=0).fit(X, y)    
    print @@ score reg x y
    [%expect {|
            0.9885...            
    |}]
    print @@ predict reg x[:1 ]
    [%expect {|
            array([-31.9417...])            
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
let%expect_text "RidgeCV" =
    let load_diabetes = Sklearn.Datasets.load_diabetes in
    let ridgeCV = Sklearn.Linear_model.ridgeCV in
    let x, y = load_diabetes return_X_y=True in
    clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X, y)    
    print @@ score clf x y
    [%expect {|
            0.5166...            
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
let%expect_text "RidgeClassifierCV" =
    let load_breast_cancer = Sklearn.Datasets.load_breast_cancer in
    let ridgeClassifierCV = Sklearn.Linear_model.ridgeClassifierCV in
    let x, y = load_breast_cancer return_X_y=True in
    clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X, y)    
    print @@ score clf x y
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
let%expect_text "SGDClassifier" =
    import numpy as np    
    let linear_model = Sklearn.linear_model in
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])    
    Y = np.array([1, 1, 2, 2])    
    clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)    
    print @@ fit clf x y
    [%expect {|
            SGDClassifier()            
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
let%expect_text "SGDRegressor" =
    import numpy as np    
    let linear_model = Sklearn.linear_model in
    n_samples, n_features = 10, 5    
    rng = np.random.RandomState(0)    
    y = rng.randn(n_samples)    
    X = rng.randn(n_samples, n_features)    
    clf = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)    
    print @@ fit clf x y
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
let%expect_text "TheilSenRegressor" =
    let theilSenRegressor = Sklearn.Linear_model.theilSenRegressor in
    let make_regression = Sklearn.Datasets.make_regression in
    let x, y = make_regression n_samples=200 n_features=2 noise=4.0 random_state=0 in
    reg = TheilSenRegressor(random_state=0).fit(X, y)    
    print @@ score reg x y
    [%expect {|
            0.9884...            
    |}]
    print @@ predict reg x[:1 ]
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
let%expect_text "lasso_path" =
    X = np.array([[1, 2, 3.1], [2.3, 5.4, 4.3]]).T    
    y = np.array([1, 2, 3.1])    
    # Use lasso_path to compute a coefficient path    
    let h, _ = lasso_path x y alphas=[5. 1. .5] in
    print(coef_path)    
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
let%expect_text "lasso_path" =
    # Now use lars_path and 1D linear interpolation to compute the    
    # same path    
    let lars_path = Sklearn.Linear_model.lars_path in
    let e, coef_path_lars = lars_path x y method='lasso' in
    let interpolate = Scipy.interpolate in
    coef_path_continuous = interpolate.interp1d(alphas[::-1],coef_path_lars[:, ::-1])    
    print(coef_path_continuous([5., 1., .5]))    
    [%expect {|
            [[0.         0.         0.46915237]            
             [0.2159048  0.4425765  0.23668876]]            
    |}]

*)



