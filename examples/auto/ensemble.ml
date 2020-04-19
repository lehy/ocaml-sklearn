let print f x = Format.printf "%a" f x
let print_py x = Format.printf "%s" (Py.Object.to_string x)
let print_ndarray = print Sklearn.Ndarray.pp

let matrix = Sklearn.Ndarray.Float.matrix
let vector = Sklearn.Ndarray.Float.vector
let matrixi = Sklearn.Ndarray.Int.matrix
let vectori = Sklearn.Ndarray.Int.vector

let get x = match x with
  | None -> failwith "Option.get"
  | Some x -> x

(* AdaBoostClassifier *)
(*
>>> from sklearn.ensemble import AdaBoostClassifier
>>> from sklearn.datasets import make_classification
>>> X, y = make_classification(n_samples=1000, n_features=4,
...                            n_informative=2, n_redundant=0,
...                            random_state=0, shuffle=False)
>>> clf = AdaBoostClassifier(n_estimators=100, random_state=0)
>>> clf.fit(X, y)
AdaBoostClassifier(n_estimators=100, random_state=0)
>>> clf.feature_importances_
array([0.28..., 0.42..., 0.14..., 0.16...])
>>> clf.predict([[0, 0, 0, 0]])
array([1])
>>> clf.score(X, y)

*)

let%expect_test "AdaBoostClassifier" =
  let open Sklearn.Ensemble in
  let x, y = Sklearn.Datasets.make_classification ~n_samples:1000 ~n_features:4 ~n_informative:2
      ~n_redundant:0 ~random_state:(`Int 0) ~shuffle:false ()
  in
  let clf = AdaBoostClassifier.create ~n_estimators:100 ~random_state:(`Int 0) () in
  print AdaBoostClassifier.pp @@ AdaBoostClassifier.fit ~x: (`Ndarray x) ~y clf;
  [%expect {|
      AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                         n_estimators=100, random_state=0)
  |}];
  print_ndarray @@ get @@ AdaBoostClassifier.feature_importances_ clf;
  [%expect {|
      [0.28 0.42 0.14 0.16]
  |}];
  print_ndarray @@ AdaBoostClassifier.predict ~x:(`Ndarray (matrixi [|[|0; 0; 0; 0|]|])) clf;
  [%expect {|
      [1]
  |}];
  Format.printf "%g" @@ AdaBoostClassifier.score ~x ~y clf;
  [%expect {| 0.983 |}]


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

let%expect_test "AdaBoostRegressor" =
  let open Sklearn.Ensemble in
  let x, y, _coef = Sklearn.Datasets.make_regression ~n_features:4 ~n_informative:2
      ~random_state:(`Int 0) ~shuffle:false ()
  in
  let regr = AdaBoostRegressor.create ~random_state:(`Int 0) ~n_estimators:100 () in
  print AdaBoostRegressor.pp @@ AdaBoostRegressor.fit ~x:(`Ndarray x) ~y regr;
  [%expect {|
        AdaBoostRegressor(base_estimator=None, learning_rate=1.0, loss='linear',
                          n_estimators=100, random_state=0)
     |}];
  print_ndarray @@ get @@ AdaBoostRegressor.feature_importances_ regr;
  [%expect {|
        [0.27885832 0.71092234 0.00654703 0.00367231]
     |}];
  print_ndarray @@ AdaBoostRegressor.predict ~x:(`Ndarray (matrixi [|[|0; 0; 0; 0|]|])) regr;
  [%expect {|
        [4.79722349]
     |}];
  Format.printf "%g" @@ AdaBoostRegressor.score ~x ~y regr;
  [%expect {|
        0.977138
     |}]


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

let%expect_test "BaggingClassifier" =
  let open Sklearn.Ensemble in
  let x, y = Sklearn.Datasets.make_classification ~n_samples:100 ~n_features:4
      ~n_informative:2 ~n_redundant:0 ~random_state:(`Int 0) ~shuffle:false ()
  in
  (* XXX The way SVC() is passed is not satisfying, but I fear that
     most solutions (passsing a first-class module to the function,
     complicating the Py.Object.t type to include some more
     information + subtyping, exposing PYthon classes as OCaml
     classes... would be a lot of work, probably harder to use, for a
     benefit that is relatively mild. Suggestions welcome.  -- Ronan,
     2020-04-19 *)
  let clf = BaggingClassifier.(create ~base_estimator:(`PyObject (Sklearn.Svm.SVC.(create () |> to_pyobject)))
                                 ~n_estimators:10 ~random_state:(`Int 0) ()
                               |> fit ~x:(`Ndarray x) ~y)
  in
  print_ndarray @@ BaggingClassifier.predict ~x:(`Ndarray (matrixi [|[|0; 0; 0; 0|]|])) clf;
  [%expect {|
      [1]
   |}]

(* BaggingRegressor *)
(*
>>> from sklearn.svm import SVR
>>> from sklearn.ensemble import BaggingRegressor
>>> from sklearn.datasets import Sklearn.Datasets.make_regression
>>> X, y = Sklearn.Datasets.make_regression(n_samples=100, n_features=4,
...                        n_informative=2, n_targets=1,
...                        random_state=0, shuffle=False)
>>> regr = BaggingRegressor(base_estimator=SVR(),
...                         n_estimators=10, random_state=0).fit(X, y)
>>> regr.predict([[0, 0, 0, 0]])
array([-2.8720...])

*)

let%expect_test "BaggingRegressor" =
  let open Sklearn.Ensemble in
  let x, y, _coefs = Sklearn.Datasets.make_regression ~n_samples:100 ~n_features:4
      ~n_informative:2 ~n_targets:1 ~random_state:(`Int 0) ~shuffle:false ()
  in
  let regr = BaggingRegressor.(create ~base_estimator:(`PyObject (Sklearn.Svm.SVR.(create () |> to_pyobject)))
                                 ~n_estimators:10 ~random_state:(`Int 0) ()
                               |> fit ~x:(`Ndarray x) ~y)
  in
  print_ndarray @@ BaggingRegressor.predict ~x:(`Ndarray (matrixi [|[|0; 0; 0; 0|]|])) regr;
  [%expect {|
      [-2.87202411]
   |}]


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
   let%expect_test "ExtraTreesClassifier" =
   let open Sklearn.Ensemble in
   let x, y = make_classification ~n_features:4 ~random_state:(`Int 0) () in
   let clf = ExtraTreesClassifier.create ~n_estimators:100 ~random_state:(`Int 0) () in
   print ExtraTreesClassifier.pp @@ ExtraTreesClassifier.fit ~x y clf;
   [%expect {|
      ExtraTreesClassifier(random_state=0)
   |}];
   print_ndarray @@ ExtraTreesClassifier.predict (matrixi [|[|0; 0; 0; 0|]|]) clf;
   [%expect {|
      array([1])
   |}];

*)



(* IsolationForest *)
(*
>>> from sklearn.ensemble import IsolationForest
>>> X = [[-1.1], [0.3], [0.5], [100]]
>>> clf = IsolationForest(random_state=0).fit(X)
>>> clf.predict([[0.1], [0], [90]])

*)

(* TEST TODO
   let%expect_test "IsolationForest" =
   let open Sklearn.Ensemble in
   let x = (matrix [|[|-1.1|]; [|0.3|]; [|0.5|]; [|100|]|]) in
   let clf = IsolationForest(random_state=0).fit ~x () in
   print_ndarray @@ IsolationForest.predict (matrix [|[|0.1|]; [|0|]; [|90|]|]) clf;
   [%expect {|
   |}];

*)



(* RandomForestClassifier *)
(*
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.datasets import make_classification

*)

(* TEST TODO
   let%expect_test "RandomForestClassifier" =
   let open Sklearn.Ensemble in
   [%expect {|
   |}];

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
   let%expect_test "RandomForestClassifier" =
   let open Sklearn.Ensemble in
   let x, y = make_classification ~n_samples:1000 ~n_features:4 ~n_informative:2 ~n_redundant:0 ~random_state:(`Int 0) ~shuffle:false () in
   let clf = RandomForestClassifier.create ~max_depth:2 ~random_state:(`Int 0) () in
   print RandomForestClassifier.pp @@ RandomForestClassifier.fit ~x y clf;
   [%expect {|
      RandomForestClassifier(max_depth=2, random_state=0)
   |}];
   print_ndarray @@ print clf.feature_importances_ ();
   [%expect {|
      [0.14205973 0.76664038 0.0282433  0.06305659]
   |}];
   print_ndarray @@ print(RandomForestClassifier.predict (matrixi [|[|0; 0; 0; 0|]|])) clf;
   [%expect {|
      [1]
   |}];

*)



(* RandomForestRegressor *)
(*
>>> from sklearn.ensemble import RandomForestRegressor
>>> from sklearn.datasets import Sklearn.Datasets.make_regression

*)

(* TEST TODO
   let%expect_test "RandomForestRegressor" =
   let open Sklearn.Ensemble in
   [%expect {|
   |}];

*)



(* RandomForestRegressor *)
(*
>>> X, y = Sklearn.Datasets.make_regression(n_features=4, n_informative=2,
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
   let%expect_test "RandomForestRegressor" =
   let open Sklearn.Ensemble in
   let x, y = Sklearn.Datasets.make_regression ~n_features:4 ~n_informative:2 ~random_state:(`Int 0) ~shuffle:false () in
   let regr = RandomForestRegressor.create ~max_depth:2 ~random_state:(`Int 0) () in
   print RandomForestRegressor.pp @@ RandomForestRegressor.fit ~x y regr;
   [%expect {|
      RandomForestRegressor(max_depth=2, random_state=0)
   |}];
   print_ndarray @@ print regr.feature_importances_ ();
   [%expect {|
      [0.18146984 0.81473937 0.00145312 0.00233767]
   |}];
   print_ndarray @@ print(RandomForestRegressor.predict (matrixi [|[|0; 0; 0; 0|]|])) regr;
   [%expect {|
      [-8.32987858]
   |}];

*)



(* StackingClassifier *)
(*
>>> from sklearn.datasets import load_iris
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.svm import LinearSVC
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.preprocessing import StandardScaler
>>> from sklearn.pipeline import make_pipeline
>>> from sklearn.ensemble import StackingClassifier
>>> X, y = load_iris(return_X_y=True)
>>> estimators = [
...     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
...     ('svr', make_pipeline(StandardScaler(),
...                           LinearSVC(random_state=42)))
... ]
>>> clf = StackingClassifier(
...     estimators=estimators, final_estimator=LogisticRegression()
... )
>>> from sklearn.model_selection import train_test_split
>>> X_train, X_test, y_train, y_test = train_test_split(
...     X, y, stratify=y, random_state=42
... )
>>> clf.fit(X_train, y_train).score(X_test, y_test)

*)

(* TEST TODO
   let%expect_test "StackingClassifier" =
   let open Sklearn.Ensemble in
   let x, y = load_iris ~return_X_y:true () in
   let estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)),('svr', make_pipeline(StandardScaler(),LinearSVC(random_state=42)))] in
   let clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression()) in
   let X_train, X_test, y_train, y_test = train_test_split ~x y ~stratify:y ~random_state:(`Int 42) () in
   print StackingClassifier.pp @@ StackingClassifier.fit ~X_train y_train).score(X_test ~y_test clf;
   [%expect {|
   |}];

*)



(* StackingRegressor *)
(*
>>> from sklearn.datasets import load_diabetes
>>> from sklearn.linear_model import RidgeCV
>>> from sklearn.svm import LinearSVR
>>> from sklearn.ensemble import RandomForestRegressor
>>> from sklearn.ensemble import StackingRegressor
>>> X, y = load_diabetes(return_X_y=True)
>>> estimators = [
...     ('lr', RidgeCV()),
...     ('svr', LinearSVR(random_state=42))
... ]
>>> reg = StackingRegressor(
...     estimators=estimators,
...     final_estimator=RandomForestRegressor(n_estimators=10,
...                                           random_state=42)
... )
>>> from sklearn.model_selection import train_test_split
>>> X_train, X_test, y_train, y_test = train_test_split(
...     X, y, random_state=42
... )
>>> reg.fit(X_train, y_train).score(X_test, y_test)

*)

(* TEST TODO
   let%expect_test "StackingRegressor" =
   let open Sklearn.Ensemble in
   let x, y = load_diabetes ~return_X_y:true () in
   let estimators = [('lr', RidgeCV()),('svr', LinearSVR(random_state=42))] in
   let reg = StackingRegressor(estimators=estimators,final_estimator=RandomForestRegressor(n_estimators=10,random_state=42)) in
   let X_train, X_test, y_train, y_test = train_test_split ~x y ~random_state:(`Int 42) () in
   print StackingRegressor.pp @@ StackingRegressor.fit ~X_train y_train).score(X_test ~y_test reg;
   [%expect {|
   |}];

*)



(* VotingClassifier *)
(*
>>> import numpy as np
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.naive_bayes import GaussianNB
>>> from sklearn.ensemble import RandomForestClassifier, VotingClassifier
>>> clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
>>> clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
>>> clf3 = GaussianNB()
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
>>> y = np.array([1, 1, 1, 2, 2, 2])
>>> eclf1 = VotingClassifier(estimators=[
...         ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
>>> eclf1 = eclf1.fit(X, y)
>>> print(eclf1.predict(X))
[1 1 1 2 2 2]
>>> np.array_equal(eclf1.named_estimators_.lr.predict(X),
...                eclf1.named_estimators_['lr'].predict(X))
True
>>> eclf2 = VotingClassifier(estimators=[
...         ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
...         voting='soft')
>>> eclf2 = eclf2.fit(X, y)
>>> print(eclf2.predict(X))
[1 1 1 2 2 2]
>>> eclf3 = VotingClassifier(estimators=[
...        ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
...        voting='soft', weights=[2,1,1],
...        flatten_transform=True)
>>> eclf3 = eclf3.fit(X, y)
>>> print(eclf3.predict(X))
[1 1 1 2 2 2]
>>> print(eclf3.transform(X).shape)

*)

(* TEST TODO
   let%expect_test "VotingClassifier" =
   let open Sklearn.Ensemble in
   let clf1 = LogisticRegression.create ~multi_class:'multinomial' ~random_state:(`Int 1) () in
   let clf2 = RandomForestClassifier.create ~n_estimators:50 ~random_state:(`Int 1) () in
   let clf3 = GaussianNB.create () in
   let x = .array (matrixi [|[|-1; -1|]; [|-2; -1|]; [|-3; -2|]; [|1; 1|]; [|2; 1|]; [|3; 2|]|]) np in
   let y = .array (vectori [|1; 1; 1; 2; 2; 2|]) np in
   let eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard') in
   let eclf1 = eclf1.fit ~x y () in
   print_ndarray @@ print(eclf1.predict ~x ());
   [%expect {|
      [1 1 1 2 2 2]
   |}];
   print_ndarray @@ .array_equal eclf1.named_estimators_.lr.predict ~x () eclf1.named_estimators_['lr'].predict ~x () np;
   [%expect {|
      True
   |}];
   let eclf2 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='soft') in
   let eclf2 = eclf2.fit ~x y () in
   print_ndarray @@ print(eclf2.predict ~x ());
   [%expect {|
      [1 1 1 2 2 2]
   |}];
   let eclf3 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='soft', weights=(vectori [|2;1;1|]),flatten_transform=true) in
   let eclf3 = eclf3.fit ~x y () in
   print_ndarray @@ print(eclf3.predict ~x ());
   [%expect {|
      [1 1 1 2 2 2]
   |}];
   print_ndarray @@ print(eclf3.transform ~x ().shape);
   [%expect {|
   |}];

*)



(* VotingRegressor *)
(*
>>> import numpy as np
>>> from sklearn.linear_model import LinearRegression
>>> from sklearn.ensemble import RandomForestRegressor
>>> from sklearn.ensemble import VotingRegressor
>>> r1 = LinearRegression()
>>> r2 = RandomForestRegressor(n_estimators=10, random_state=1)
>>> X = np.array([[1, 1], [2, 4], [3, 9], [4, 16], [5, 25], [6, 36]])
>>> y = np.array([2, 6, 12, 20, 30, 42])
>>> er = VotingRegressor([('lr', r1), ('rf', r2)])
>>> print(er.fit(X, y).predict(X))

*)

(* TEST TODO
   let%expect_test "VotingRegressor" =
   let open Sklearn.Ensemble in
   let r1 = LinearRegression.create () in
   let r2 = RandomForestRegressor.create ~n_estimators:10 ~random_state:(`Int 1) () in
   let x = .array (matrixi [|[|1; 1|]; [|2; 4|]; [|3; 9|]; [|4; 16|]; [|5; 25|]; [|6; 36|]|]) np in
   let y = .array [2 ~6 12 ~20 30 42] np in
   let er = VotingRegressor([('lr', r1), ('rf', r2)]) in
   print_ndarray @@ print VotingRegressor.fit ~x y ().predict ~x () er;
   [%expect {|
   |}];

*)



(*--------- Examples for module Sklearn.Ensemble.Partial_dependence ----------*)
(* Parallel *)
(*
>>> from math import sqrt
>>> from joblib import Parallel, delayed
>>> Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10))
[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

*)

(* TEST TODO
   let%expect_test "Parallel" =
   let open Sklearn.Ensemble in
   print_ndarray @@ Parallel(n_jobs=1)(delayed ~sqrt ()(i**2) for i in range ~10 ());
   [%expect {|
      [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
   |}];

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
   let%expect_test "Parallel" =
   let open Sklearn.Ensemble in
   let r = Parallel(n_jobs=1)(delayed ~modf ()(i/2.) for i in range ~10 ()) in
   let res, i = zip *r () in
   print_ndarray @@ res;
   [%expect {|
      (0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5)
   |}];
   print_ndarray @@ i;
   [%expect {|
      (0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0)
   |}];

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
   let%expect_test "Parallel" =
   let open Sklearn.Ensemble in
   let r = Parallel(n_jobs=2, verbose=10)(delayed ~sleep ()(.2) for _ in range ~10 ()) #doctest: +SKIP in
   [%expect {|
      [Parallel(n_jobs=2)]: Done   1 tasks      | elapsed:    0.6s
      [Parallel(n_jobs=2)]: Done   4 tasks      | elapsed:    0.8s
      [Parallel(n_jobs=2)]: Done  10 out of  10 | elapsed:    1.4s finished
   |}];

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
   let%expect_test "Parallel" =
   let open Sklearn.Ensemble in
   print_ndarray @@ Parallel(n_jobs=2)(delayed ~nlargest ()(2, n) for n in (range ~4 (), 'abcde', 3)) #doctest: +SKIP;
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
   |}];

*)



(* Parallel *)
(*
>>> from math import sqrt
>>> from joblib import Parallel, delayed
>>> def producer():
...     for i in range(6):
...         print('Produced %s' % i)
...         yield i
>>> out = Parallel(n_jobs=2, verbose=100, pre_dispatch='1.5*n_jobs')(
...                delayed(sqrt)(i) for i in producer()) #doctest: +SKIP
Produced 0
Produced 1
Produced 2
[Parallel(n_jobs=2)]: Done 1 jobs     | elapsed:  0.0s
Produced 3
[Parallel(n_jobs=2)]: Done 2 jobs     | elapsed:  0.0s
Produced 4
[Parallel(n_jobs=2)]: Done 3 jobs     | elapsed:  0.0s
Produced 5
[Parallel(n_jobs=2)]: Done 4 jobs     | elapsed:  0.0s
[Parallel(n_jobs=2)]: Done 6 out of 6 | elapsed:  0.0s remaining: 0.0s

*)

(* TEST TODO
   let%expect_test "Parallel" =
   let open Sklearn.Ensemble in
   print_ndarray @@ def producer ():for i in range ~6 ():print 'Produced %s' % i ()yield i;
   let out = Parallel(n_jobs=2, verbose=100, pre_dispatch='1.5*n_jobs')(delayed ~sqrt ()(i) for i in producer ()) #doctest: +SKIP in
   [%expect {|
      Produced 0
      Produced 1
      Produced 2
      [Parallel(n_jobs=2)]: Done 1 jobs     | elapsed:  0.0s
      Produced 3
      [Parallel(n_jobs=2)]: Done 2 jobs     | elapsed:  0.0s
      Produced 4
      [Parallel(n_jobs=2)]: Done 3 jobs     | elapsed:  0.0s
      Produced 5
      [Parallel(n_jobs=2)]: Done 4 jobs     | elapsed:  0.0s
      [Parallel(n_jobs=2)]: Done 6 out of 6 | elapsed:  0.0s remaining: 0.0s
   |}];

*)



(* cartesian *)
(*
>>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
array([[1, 4, 6],
       [1, 4, 7],
       [1, 5, 6],
       [1, 5, 7],
       [2, 4, 6],
       [2, 4, 7],
       [2, 5, 6],
       [2, 5, 7],
       [3, 4, 6],
       [3, 4, 7],
       [3, 5, 6],

*)

(* TEST TODO
   let%expect_test "cartesian" =
   let open Sklearn.Ensemble in
   print_ndarray @@ cartesian(((vectori [|1; 2; 3|]), (vectori [|4; 5|]), (vectori [|6; 7|])));
   [%expect {|
      array([[1, 4, 6],
             [1, 4, 7],
             [1, 5, 6],
             [1, 5, 7],
             [2, 4, 6],
             [2, 4, 7],
             [2, 5, 6],
             [2, 5, 7],
             [3, 4, 6],
             [3, 4, 7],
             [3, 5, 6],
   |}];

*)



(* deprecated *)
(*
>>> from sklearn.utils import deprecated
>>> deprecated()
<sklearn.utils.deprecation.deprecated object at ...>

*)

(* TEST TODO
   let%expect_test "deprecated" =
   let open Sklearn.Ensemble in
   print_ndarray @@ deprecated ();
   [%expect {|
      <sklearn.utils.deprecation.deprecated object at ...>
   |}];

*)



(* deprecated *)
(*
>>> @deprecated()
... def some_function(): pass

*)

(* TEST TODO
   let%expect_test "deprecated" =
   let open Sklearn.Ensemble in
   print_ndarray @@ @deprecated ()def some_function (): pass;
   [%expect {|
   |}];

*)



(* mquantiles *)
(*
>>> from scipy.stats.mstats import mquantiles
>>> a = np.array([6., 47., 49., 15., 42., 41., 7., 39., 43., 40., 36.])
>>> mquantiles(a)
array([ 19.2,  40. ,  42.8])

*)

(* TEST TODO
   let%expect_test "mquantiles" =
   let open Sklearn.Ensemble in
   let a = .array [6. 47. 49. 15. 42. 41. 7. 39. 43. 40. 36.] np in
   print_ndarray @@ mquantiles ~a ();
   [%expect {|
      array([ 19.2,  40. ,  42.8])
   |}];

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
   let%expect_test "mquantiles" =
   let open Sklearn.Ensemble in
   let data = .array [[ 6. 7. 1.] [ 47. 15. 2.] [ 49. 36. 3.] [ 15. 39. 4.] [ 42. 40. -999.] [ 41. 41. -999.] [ 7. -999. -999.] [ 39. -999. -999.] [ 43. -999. -999.] [ 40. -999. -999.] [ 36. -999. -999.]] np in
   print_ndarray @@ print(mquantiles(data, axis=0, limit=(0, 50)));
   [%expect {|
      [[19.2  14.6   1.45]
       [40.   37.5   2.5 ]
       [42.8  40.05  3.55]]
   |}];

*)



(* mquantiles *)
(*
>>> data[:, 2] = -999.
>>> print(mquantiles(data, axis=0, limit=(0, 50)))
[[19.200000000000003 14.6 --]
 [40.0 37.5 --]

*)

(* TEST TODO
   let%expect_test "mquantiles" =
   let open Sklearn.Ensemble in
   print_ndarray @@ data vectori [|:; 2|] () = -999.;
   print_ndarray @@ print(mquantiles(data, axis=0, limit=(0, 50)));
   [%expect {|
      [[19.200000000000003 14.6 --]
       [40.0 37.5 --]
   |}];

*)
