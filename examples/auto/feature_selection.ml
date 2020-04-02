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
let%expect_text "GenericUnivariateSelect" =
    let load_breast_cancer = Sklearn.Datasets.load_breast_cancer in
    from sklearn.feature_selection import GenericUnivariateSelect, chi2    
    let x, y = load_breast_cancer return_X_y=True in
    X.shape    
    [%expect {|
            (569, 30)            
    |}]
    transformer = GenericUnivariateSelect(chi2, 'k_best', param=20)    
    X_new = transformer.fit_transform(X, y)    
    X_new.shape    
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
let%expect_text "RFE" =
    let make_friedman1 = Sklearn.Datasets.make_friedman1 in
    let rfe = Sklearn.Feature_selection.rfe in
    let svr = Sklearn.Svm.svr in
    let x, y = make_friedman1 n_samples=50 n_features=10 random_state=0 in
    estimator = SVR(kernel="linear")    
    selector = RFE(estimator, 5, step=1)    
    selector = selector.fit(X, y)    
    selector.support_    
    [%expect {|
            array([ True,  True,  True,  True,  True, False, False, False, False,            
                   False])            
    |}]
    selector.ranking_    
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
let%expect_text "RFECV" =
    let make_friedman1 = Sklearn.Datasets.make_friedman1 in
    let rfecv = Sklearn.Feature_selection.rfecv in
    let svr = Sklearn.Svm.svr in
    let x, y = make_friedman1 n_samples=50 n_features=10 random_state=0 in
    estimator = SVR(kernel="linear")    
    selector = RFECV(estimator, step=1, cv=5)    
    selector = selector.fit(X, y)    
    selector.support_    
    [%expect {|
            array([ True,  True,  True,  True,  True, False, False, False, False,            
                   False])            
    |}]
    selector.ranking_    
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
let%expect_text "SelectFdr" =
    let load_breast_cancer = Sklearn.Datasets.load_breast_cancer in
    from sklearn.feature_selection import SelectFdr, chi2    
    let x, y = load_breast_cancer return_X_y=True in
    X.shape    
    [%expect {|
            (569, 30)            
    |}]
    X_new = SelectFdr(chi2, alpha=0.01).fit_transform(X, y)    
    X_new.shape    
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
let%expect_text "SelectFpr" =
    let load_breast_cancer = Sklearn.Datasets.load_breast_cancer in
    from sklearn.feature_selection import SelectFpr, chi2    
    let x, y = load_breast_cancer return_X_y=True in
    X.shape    
    [%expect {|
            (569, 30)            
    |}]
    X_new = SelectFpr(chi2, alpha=0.01).fit_transform(X, y)    
    X_new.shape    
    [%expect {|
            (569, 16)            
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
let%expect_text "SelectFwe" =
    let load_breast_cancer = Sklearn.Datasets.load_breast_cancer in
    from sklearn.feature_selection import SelectFwe, chi2    
    let x, y = load_breast_cancer return_X_y=True in
    X.shape    
    [%expect {|
            (569, 30)            
    |}]
    X_new = SelectFwe(chi2, alpha=0.01).fit_transform(X, y)    
    X_new.shape    
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
let%expect_text "SelectKBest" =
    let load_digits = Sklearn.Datasets.load_digits in
    from sklearn.feature_selection import SelectKBest, chi2    
    let x, y = load_digits return_X_y=True in
    X.shape    
    [%expect {|
            (1797, 64)            
    |}]
    X_new = SelectKBest(chi2, k=20).fit_transform(X, y)    
    X_new.shape    
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
let%expect_text "SelectPercentile" =
    let load_digits = Sklearn.Datasets.load_digits in
    from sklearn.feature_selection import SelectPercentile, chi2    
    let x, y = load_digits return_X_y=True in
    X.shape    
    [%expect {|
            (1797, 64)            
    |}]
    X_new = SelectPercentile(chi2, percentile=10).fit_transform(X, y)    
    X_new.shape    
    [%expect {|
            (1797, 7)            
    |}]

*)



