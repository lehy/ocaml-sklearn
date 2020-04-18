(* GenericUnivariateSelect *)
(*
>>> from sklearn.datasets import load_breast_cancer
>>> from sklearn.feature_selection import GenericUnivariateSelect, chi2
>>> X, y = load_breast_cancer(return_X_y=True)
>>> X.shape
(569, 30)
>>> transformer = GenericUnivariateSelect(chi2, 'k_best', param=20)
>>> X_new = transformer.fit_transform(X, y)
>>> X_new.shape
(569, 20)

*)

(* TEST TODO
let%expect_test "GenericUnivariateSelect" =
  let open Sklearn.Feature_selection in
  let x, y = load_breast_cancer ~return_X_y:true () in  
  print_ndarray @@ x.shape;  
  [%expect {|
      (569, 30)      
  |}]
  let transformer = GenericUnivariateSelect.create ~chi2 'k_best' ~param:20 () in  
  let X_new = GenericUnivariateSelect.fit_transform ~x y transformer in  
  print_ndarray @@ X_new.shape;  
  [%expect {|
      (569, 20)      
  |}]

*)



(* RFE *)
(*
>>> from sklearn.datasets import make_friedman1
>>> from sklearn.feature_selection import RFE
>>> from sklearn.svm import SVR
>>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
>>> estimator = SVR(kernel="linear")
>>> selector = RFE(estimator, 5, step=1)
>>> selector = selector.fit(X, y)
>>> selector.support_
array([ True,  True,  True,  True,  True, False, False, False, False,
       False])
>>> selector.ranking_
array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])

*)

(* TEST TODO
let%expect_test "RFE" =
  let open Sklearn.Feature_selection in
  let x, y = make_friedman1(n_samples=50, n_features=10, random_state=0) in  
  let estimator = SVR.create ~kernel:"linear" () in  
  let selector = RFE.create ~estimator 5 ~step:1 () in  
  let selector = RFE.fit ~x y selector in  
  print_ndarray @@ RFE.support_ selector;  
  [%expect {|
      array([ True,  True,  True,  True,  True, False, False, False, False,      
             False])      
  |}]
  print_ndarray @@ RFE.ranking_ selector;  
  [%expect {|
      array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])      
  |}]

*)



(* RFECV *)
(*
>>> from sklearn.datasets import make_friedman1
>>> from sklearn.feature_selection import RFECV
>>> from sklearn.svm import SVR
>>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
>>> estimator = SVR(kernel="linear")
>>> selector = RFECV(estimator, step=1, cv=5)
>>> selector = selector.fit(X, y)
>>> selector.support_
array([ True,  True,  True,  True,  True, False, False, False, False,
       False])
>>> selector.ranking_
array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])

*)

(* TEST TODO
let%expect_test "RFECV" =
  let open Sklearn.Feature_selection in
  let x, y = make_friedman1(n_samples=50, n_features=10, random_state=0) in  
  let estimator = SVR.create ~kernel:"linear" () in  
  let selector = RFECV.create estimator ~step:1 ~cv:5 () in  
  let selector = RFECV.fit ~x y selector in  
  print_ndarray @@ RFECV.support_ selector;  
  [%expect {|
      array([ True,  True,  True,  True,  True, False, False, False, False,      
             False])      
  |}]
  print_ndarray @@ RFECV.ranking_ selector;  
  [%expect {|
      array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])      
  |}]

*)



(* SelectFdr *)
(*
>>> from sklearn.datasets import load_breast_cancer
>>> from sklearn.feature_selection import SelectFdr, chi2
>>> X, y = load_breast_cancer(return_X_y=True)
>>> X.shape
(569, 30)
>>> X_new = SelectFdr(chi2, alpha=0.01).fit_transform(X, y)
>>> X_new.shape
(569, 16)

*)

(* TEST TODO
let%expect_test "SelectFdr" =
  let open Sklearn.Feature_selection in
  let x, y = load_breast_cancer ~return_X_y:true () in  
  print_ndarray @@ x.shape;  
  [%expect {|
      (569, 30)      
  |}]
  let X_new = SelectFdr(chi2, alpha=0.01).fit_transform ~x y () in  
  print_ndarray @@ X_new.shape;  
  [%expect {|
      (569, 16)      
  |}]

*)



(* SelectFpr *)
(*
>>> from sklearn.datasets import load_breast_cancer
>>> from sklearn.feature_selection import SelectFpr, chi2
>>> X, y = load_breast_cancer(return_X_y=True)
>>> X.shape
(569, 30)
>>> X_new = SelectFpr(chi2, alpha=0.01).fit_transform(X, y)
>>> X_new.shape
(569, 16)

*)

(* TEST TODO
let%expect_test "SelectFpr" =
  let open Sklearn.Feature_selection in
  let x, y = load_breast_cancer ~return_X_y:true () in  
  print_ndarray @@ x.shape;  
  [%expect {|
      (569, 30)      
  |}]
  let X_new = SelectFpr(chi2, alpha=0.01).fit_transform ~x y () in  
  print_ndarray @@ X_new.shape;  
  [%expect {|
      (569, 16)      
  |}]

*)



(* SelectFromModel *)
(*
>>> from sklearn.feature_selection import SelectFromModel
>>> from sklearn.linear_model import LogisticRegression
>>> X = [[ 0.87, -1.34,  0.31 ],
...      [-2.79, -0.02, -0.85 ],
...      [-1.34, -0.48, -2.55 ],
...      [ 1.92,  1.48,  0.65 ]]
>>> y = [0, 1, 0, 1]
>>> selector = SelectFromModel(estimator=LogisticRegression()).fit(X, y)
>>> selector.estimator_.coef_
array([[-0.3252302 ,  0.83462377,  0.49750423]])
>>> selector.threshold_
0.55245...
>>> selector.get_support()
array([False,  True, False])
>>> selector.transform(X)
array([[-1.34],
       [-0.02],
       [-0.48],

*)

(* TEST TODO
let%expect_test "SelectFromModel" =
  let open Sklearn.Feature_selection in
  let x = [[ 0.87, -1.34, 0.31 ],[-2.79, -0.02, -0.85 ],[-1.34, -0.48, -2.55 ],[ 1.92, 1.48, 0.65 ]] in  
  let y = (vectori [|0; 1; 0; 1|]) in  
  let selector = SelectFromModel(estimator=LogisticRegression()).fit ~x y () in  
  print_ndarray @@ selector..coef_ estimator_;  
  [%expect {|
      array([[-0.3252302 ,  0.83462377,  0.49750423]])      
  |}]
  print_ndarray @@ SelectFromModel.threshold_ selector;  
  [%expect {|
      0.55245...      
  |}]
  print_ndarray @@ SelectFromModel.get_support selector;  
  [%expect {|
      array([False,  True, False])      
  |}]
  print_ndarray @@ SelectFromModel.transform ~x selector;  
  [%expect {|
      array([[-1.34],      
             [-0.02],      
             [-0.48],      
  |}]

*)



(* SelectFwe *)
(*
>>> from sklearn.datasets import load_breast_cancer
>>> from sklearn.feature_selection import SelectFwe, chi2
>>> X, y = load_breast_cancer(return_X_y=True)
>>> X.shape
(569, 30)
>>> X_new = SelectFwe(chi2, alpha=0.01).fit_transform(X, y)
>>> X_new.shape
(569, 15)

*)

(* TEST TODO
let%expect_test "SelectFwe" =
  let open Sklearn.Feature_selection in
  let x, y = load_breast_cancer ~return_X_y:true () in  
  print_ndarray @@ x.shape;  
  [%expect {|
      (569, 30)      
  |}]
  let X_new = SelectFwe(chi2, alpha=0.01).fit_transform ~x y () in  
  print_ndarray @@ X_new.shape;  
  [%expect {|
      (569, 15)      
  |}]

*)



(* SelectKBest *)
(*
>>> from sklearn.datasets import load_digits
>>> from sklearn.feature_selection import SelectKBest, chi2
>>> X, y = load_digits(return_X_y=True)
>>> X.shape
(1797, 64)
>>> X_new = SelectKBest(chi2, k=20).fit_transform(X, y)
>>> X_new.shape
(1797, 20)

*)

(* TEST TODO
let%expect_test "SelectKBest" =
  let open Sklearn.Feature_selection in
  let x, y = load_digits ~return_X_y:true () in  
  print_ndarray @@ x.shape;  
  [%expect {|
      (1797, 64)      
  |}]
  let X_new = SelectKBest(chi2, k=20).fit_transform ~x y () in  
  print_ndarray @@ X_new.shape;  
  [%expect {|
      (1797, 20)      
  |}]

*)



(* SelectPercentile *)
(*
>>> from sklearn.datasets import load_digits
>>> from sklearn.feature_selection import SelectPercentile, chi2
>>> X, y = load_digits(return_X_y=True)
>>> X.shape
(1797, 64)
>>> X_new = SelectPercentile(chi2, percentile=10).fit_transform(X, y)
>>> X_new.shape
(1797, 7)

*)

(* TEST TODO
let%expect_test "SelectPercentile" =
  let open Sklearn.Feature_selection in
  let x, y = load_digits ~return_X_y:true () in  
  print_ndarray @@ x.shape;  
  [%expect {|
      (1797, 64)      
  |}]
  let X_new = SelectPercentile(chi2, percentile=10).fit_transform ~x y () in  
  print_ndarray @@ X_new.shape;  
  [%expect {|
      (1797, 7)      
  |}]

*)



