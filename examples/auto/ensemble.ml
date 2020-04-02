(* AdaBoostRegressor *)
(*
>>> from sklearn.ensemble import AdaBoostRegressor
>>> from sklearn.datasets import make_regression
>>> X, y = make_regression(n_features=4, n_informative=2,
...                        random_state=0, shuffle=False)
>>> regr = AdaBoostRegressor(random_state=0, n_estimators=100)
>>> regr.fit(X, y)
AdaBoostRegressor(n_estimators=100, random_state=0)
>>> regr.feature_importances_
array([0.2788..., 0.7109..., 0.0065..., 0.0036...])
>>> regr.predict([[0, 0, 0, 0]])
array([4.7972...])
>>> regr.score(X, y)
0.9771...


*)

(* TEST TODO
let%expect_text "AdaBoostRegressor" =
    let adaBoostRegressor = Sklearn.Ensemble.adaBoostRegressor in
    let make_regression = Sklearn.Datasets.make_regression in
    let x, y = make_regression n_features=4 n_informative=2 random_state=0 shuffle=False in
    regr = AdaBoostRegressor(random_state=0, n_estimators=100)    
    print @@ fit regr x y
    [%expect {|
            AdaBoostRegressor(n_estimators=100, random_state=0)            
    |}]
    regr.feature_importances_    
    [%expect {|
            array([0.2788..., 0.7109..., 0.0065..., 0.0036...])            
    |}]
    print @@ predict regr [[0 0 0 0]]
    [%expect {|
            array([4.7972...])            
    |}]
    print @@ score regr x y
    [%expect {|
            0.9771...            
    |}]

*)



(* BaggingClassifier *)
(*
>>> from sklearn.svm import SVC
>>> from sklearn.ensemble import BaggingClassifier
>>> from sklearn.datasets import make_classification
>>> X, y = make_classification(n_samples=100, n_features=4,
...                            n_informative=2, n_redundant=0,
...                            random_state=0, shuffle=False)
>>> clf = BaggingClassifier(base_estimator=SVC(),
...                         n_estimators=10, random_state=0).fit(X, y)
>>> clf.predict([[0, 0, 0, 0]])
array([1])


*)

(* TEST TODO
let%expect_text "BaggingClassifier" =
    let svc = Sklearn.Svm.svc in
    let baggingClassifier = Sklearn.Ensemble.baggingClassifier in
    let make_classification = Sklearn.Datasets.make_classification in
    let x, y = make_classification n_samples=100 n_features=4 n_informative=2 n_redundant=0 random_state=0 shuffle=False in
    clf = BaggingClassifier(base_estimator=SVC(),n_estimators=10, random_state=0).fit(X, y)    
    print @@ predict clf [[0 0 0 0]]
    [%expect {|
            array([1])            
    |}]

*)



(* BaggingRegressor *)
(*
>>> from sklearn.svm import SVR
>>> from sklearn.ensemble import BaggingRegressor
>>> from sklearn.datasets import make_regression
>>> X, y = make_regression(n_samples=100, n_features=4,
...                        n_informative=2, n_targets=1,
...                        random_state=0, shuffle=False)
>>> regr = BaggingRegressor(base_estimator=SVR(),
...                         n_estimators=10, random_state=0).fit(X, y)
>>> regr.predict([[0, 0, 0, 0]])
array([-2.8720...])


*)

(* TEST TODO
let%expect_text "BaggingRegressor" =
    let svr = Sklearn.Svm.svr in
    let baggingRegressor = Sklearn.Ensemble.baggingRegressor in
    let make_regression = Sklearn.Datasets.make_regression in
    let x, y = make_regression n_samples=100 n_features=4 n_informative=2 n_targets=1 random_state=0 shuffle=False in
    regr = BaggingRegressor(base_estimator=SVR(),n_estimators=10, random_state=0).fit(X, y)    
    print @@ predict regr [[0 0 0 0]]
    [%expect {|
            array([-2.8720...])            
    |}]

*)



(* ExtraTreesClassifier *)
(*
>>> from sklearn.ensemble import ExtraTreesClassifier
>>> from sklearn.datasets import make_classification
>>> X, y = make_classification(n_features=4, random_state=0)
>>> clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
>>> clf.fit(X, y)
ExtraTreesClassifier(random_state=0)
>>> clf.predict([[0, 0, 0, 0]])
array([1])


*)

(* TEST TODO
let%expect_text "ExtraTreesClassifier" =
    let extraTreesClassifier = Sklearn.Ensemble.extraTreesClassifier in
    let make_classification = Sklearn.Datasets.make_classification in
    let x, y = make_classification n_features=4 random_state=0 in
    clf = ExtraTreesClassifier(n_estimators=100, random_state=0)    
    print @@ fit clf x y
    [%expect {|
            ExtraTreesClassifier(random_state=0)            
    |}]
    print @@ predict clf [[0 0 0 0]]
    [%expect {|
            array([1])            
    |}]

*)



(* RandomForestClassifier *)
(*
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.datasets import make_classification


*)

(* TEST TODO
let%expect_text "RandomForestClassifier" =
    let randomForestClassifier = Sklearn.Ensemble.randomForestClassifier in
    let make_classification = Sklearn.Datasets.make_classification in
    [%expect {|
    |}]

*)



(* RandomForestClassifier *)
(*
>>> X, y = make_classification(n_samples=1000, n_features=4,
...                            n_informative=2, n_redundant=0,
...                            random_state=0, shuffle=False)
>>> clf = RandomForestClassifier(max_depth=2, random_state=0)
>>> clf.fit(X, y)
RandomForestClassifier(max_depth=2, random_state=0)
>>> print(clf.feature_importances_)
[0.14205973 0.76664038 0.0282433  0.06305659]
>>> print(clf.predict([[0, 0, 0, 0]]))
[1]


*)

(* TEST TODO
let%expect_text "RandomForestClassifier" =
    let x, y = make_classification n_samples=1000 n_features=4 n_informative=2 n_redundant=0 random_state=0 shuffle=False in
    clf = RandomForestClassifier(max_depth=2, random_state=0)    
    print @@ fit clf x y
    [%expect {|
            RandomForestClassifier(max_depth=2, random_state=0)            
    |}]
    print(clf.feature_importances_)    
    [%expect {|
            [0.14205973 0.76664038 0.0282433  0.06305659]            
    |}]
    print(clf.predict([[0, 0, 0, 0]]))    
    [%expect {|
            [1]            
    |}]

*)



(* RandomForestRegressor *)
(*
>>> from sklearn.ensemble import RandomForestRegressor
>>> from sklearn.datasets import make_regression


*)

(* TEST TODO
let%expect_text "RandomForestRegressor" =
    let randomForestRegressor = Sklearn.Ensemble.randomForestRegressor in
    let make_regression = Sklearn.Datasets.make_regression in
    [%expect {|
    |}]

*)



(* RandomForestRegressor *)
(*
>>> X, y = make_regression(n_features=4, n_informative=2,
...                        random_state=0, shuffle=False)
>>> regr = RandomForestRegressor(max_depth=2, random_state=0)
>>> regr.fit(X, y)
RandomForestRegressor(max_depth=2, random_state=0)
>>> print(regr.feature_importances_)
[0.18146984 0.81473937 0.00145312 0.00233767]
>>> print(regr.predict([[0, 0, 0, 0]]))
[-8.32987858]


*)

(* TEST TODO
let%expect_text "RandomForestRegressor" =
    let x, y = make_regression n_features=4 n_informative=2 random_state=0 shuffle=False in
    regr = RandomForestRegressor(max_depth=2, random_state=0)    
    print @@ fit regr x y
    [%expect {|
            RandomForestRegressor(max_depth=2, random_state=0)            
    |}]
    print(regr.feature_importances_)    
    [%expect {|
            [0.18146984 0.81473937 0.00145312 0.00233767]            
    |}]
    print(regr.predict([[0, 0, 0, 0]]))    
    [%expect {|
            [-8.32987858]            
    |}]

*)



(*--------- Examples for module .Ensemble.Partial_dependence ----------*)
(* Parallel *)
(*
>>> from math import sqrt
>>> from joblib import Parallel, delayed
>>> Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10))
[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


*)

(* TEST TODO
let%expect_text "Parallel" =
    let sqrt = Math.sqrt in
    from joblib import Parallel, delayed    
    Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10))    
    [%expect {|
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]            
    |}]

*)



(* Parallel *)
(*
>>> from math import modf
>>> from joblib import Parallel, delayed
>>> r = Parallel(n_jobs=1)(delayed(modf)(i/2.) for i in range(10))
>>> res, i = zip( *r)
>>> res
(0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5)
>>> i
(0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0)


*)

(* TEST TODO
let%expect_text "Parallel" =
    let modf = Math.modf in
    from joblib import Parallel, delayed    
    r = Parallel(n_jobs=1)(delayed(modf)(i/2.) for i in range(10))    
    let res, i = zip *r in
    res    
    [%expect {|
            (0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5)            
    |}]
    i    
    [%expect {|
            (0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0)            
    |}]

*)



(* Parallel *)
(*
>>> from time import sleep
>>> from joblib import Parallel, delayed
>>> r = Parallel(n_jobs=2, verbose=10)(delayed(sleep)(.2) for _ in range(10)) #doctest: +SKIP
[Parallel(n_jobs=2)]: Done   1 tasks      | elapsed:    0.6s
[Parallel(n_jobs=2)]: Done   4 tasks      | elapsed:    0.8s
[Parallel(n_jobs=2)]: Done  10 out of  10 | elapsed:    1.4s finished


*)

(* TEST TODO
let%expect_text "Parallel" =
    let sleep = Time.sleep in
    from joblib import Parallel, delayed    
    r = Parallel(n_jobs=2, verbose=10)(delayed(sleep)(.2) for _ in range(10)) #doctest: +SKIP    
    [%expect {|
            [Parallel(n_jobs=2)]: Done   1 tasks      | elapsed:    0.6s            
            [Parallel(n_jobs=2)]: Done   4 tasks      | elapsed:    0.8s            
            [Parallel(n_jobs=2)]: Done  10 out of  10 | elapsed:    1.4s finished            
    |}]

*)



(* Parallel *)
(*
>>> from heapq import nlargest
>>> from joblib import Parallel, delayed
>>> Parallel(n_jobs=2)(delayed(nlargest)(2, n) for n in (range(4), 'abcde', 3)) #doctest: +SKIP
#...
---------------------------------------------------------------------------
Sub-process traceback:
---------------------------------------------------------------------------
TypeError                                          Mon Nov 12 11:37:46 2012
PID: 12934                                    Python 2.7.3: /usr/bin/python
...........................................................................
/usr/lib/python2.7/heapq.pyc in nlargest(n=2, iterable=3, key=None)
    419         if n >= size:
    420             return sorted(iterable, key=key, reverse=True)[:n]
    421
    422     # When key is none, use simpler decoration
    423     if key is None:
--> 424         it = izip(iterable, count(0,-1))                    # decorate
    425         result = _nlargest(n, it)
    426         return map(itemgetter(0), result)                   # undecorate
    427
    428     # General case, slowest method
 TypeError: izip argument #1 must support iteration
___________________________________________________________________________


*)

(* TEST TODO
let%expect_text "Parallel" =
    let nlargest = Heapq.nlargest in
    from joblib import Parallel, delayed    
    Parallel(n_jobs=2)(delayed(nlargest)(2, n) for n in (range(4), 'abcde', 3)) #doctest: +SKIP    
    [%expect {|
            #...            
            ---------------------------------------------------------------------------            
            Sub-process traceback:            
            ---------------------------------------------------------------------------            
            TypeError                                          Mon Nov 12 11:37:46 2012            
            PID: 12934                                    Python 2.7.3: /usr/bin/python            
            ...........................................................................            
            /usr/lib/python2.7/heapq.pyc in nlargest(n=2, iterable=3, key=None)            
                419         if n >= size:            
                420             return sorted(iterable, key=key, reverse=True)[:n]            
                421            
                422     # When key is none, use simpler decoration            
                423     if key is None:            
            --> 424         it = izip(iterable, count(0,-1))                    # decorate            
                425         result = _nlargest(n, it)            
                426         return map(itemgetter(0), result)                   # undecorate            
                427            
                428     # General case, slowest method            
             TypeError: izip argument #1 must support iteration            
            ___________________________________________________________________________            
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



(* mquantiles *)
(*
>>> from scipy.stats.mstats import mquantiles
>>> a = np.array([6., 47., 49., 15., 42., 41., 7., 39., 43., 40., 36.])
>>> mquantiles(a)
array([ 19.2,  40. ,  42.8])


*)

(* TEST TODO
let%expect_text "mquantiles" =
    let mquantiles = Scipy.Stats.Mstats.mquantiles in
    a = np.array([6., 47., 49., 15., 42., 41., 7., 39., 43., 40., 36.])    
    mquantiles(a)    
    [%expect {|
            array([ 19.2,  40. ,  42.8])            
    |}]

*)



(* mquantiles *)
(*
>>> data = np.array([[   6.,    7.,    1.],
...                  [  47.,   15.,    2.],
...                  [  49.,   36.,    3.],
...                  [  15.,   39.,    4.],
...                  [  42.,   40., -999.],
...                  [  41.,   41., -999.],
...                  [   7., -999., -999.],
...                  [  39., -999., -999.],
...                  [  43., -999., -999.],
...                  [  40., -999., -999.],
...                  [  36., -999., -999.]])
>>> print(mquantiles(data, axis=0, limit=(0, 50)))
[[19.2  14.6   1.45]
 [40.   37.5   2.5 ]
 [42.8  40.05  3.55]]


*)

(* TEST TODO
let%expect_text "mquantiles" =
    data = np.array([[   6.,    7.,    1.],[  47.,   15.,    2.],[  49.,   36.,    3.],[  15.,   39.,    4.],[  42.,   40., -999.],[  41.,   41., -999.],[   7., -999., -999.],[  39., -999., -999.],[  43., -999., -999.],[  40., -999., -999.],[  36., -999., -999.]])    
    print(mquantiles(data, axis=0, limit=(0, 50)))    
    [%expect {|
            [[19.2  14.6   1.45]            
             [40.   37.5   2.5 ]            
             [42.8  40.05  3.55]]            
    |}]

*)



