(* IsotonicRegression *)
(*
>>> from sklearn.datasets import make_regression
>>> from sklearn.isotonic import IsotonicRegression
>>> X, y = make_regression(n_samples=10, n_features=1, random_state=41)
>>> iso_reg = IsotonicRegression().fit(X.flatten(), y)
>>> iso_reg.predict([.1, .2])

*)

(* TEST TODO
let%expect_test "IsotonicRegression" =
  let open Sklearn.Calibration in
  let x, y = make_regression ~n_samples:10 ~n_features:1 ~random_state:41 () in  
  let iso_reg = IsotonicRegression().fit(x.flatten (), y) in  
  print_ndarray @@ IsotonicRegression.predict [.1 .2] iso_reg;  
  [%expect {|
  |}]

*)



(* LabelBinarizer *)
(*
>>> from sklearn import preprocessing
>>> lb = preprocessing.LabelBinarizer()
>>> lb.fit([1, 2, 6, 4, 2])
LabelBinarizer()
>>> lb.classes_
array([1, 2, 4, 6])
>>> lb.transform([1, 6])
array([[1, 0, 0, 0],
       [0, 0, 0, 1]])

*)

(* TEST TODO
let%expect_test "LabelBinarizer" =
  let open Sklearn.Calibration in
  let lb = .labelBinarizer preprocessing in  
  print_ndarray @@ .fit (vectori [|1; 2; 6; 4; 2|]) lb;  
  [%expect {|
      LabelBinarizer()      
  |}]
  print_ndarray @@ .classes_ lb;  
  [%expect {|
      array([1, 2, 4, 6])      
  |}]
  print_ndarray @@ .transform (vectori [|1; 6|]) lb;  
  [%expect {|
      array([[1, 0, 0, 0],      
             [0, 0, 0, 1]])      
  |}]

*)



(* LabelBinarizer *)
(*
>>> lb = preprocessing.LabelBinarizer()
>>> lb.fit_transform(['yes', 'no', 'no', 'yes'])
array([[1],
       [0],
       [0],
       [1]])

*)

(* TEST TODO
let%expect_test "LabelBinarizer" =
  let open Sklearn.Calibration in
  let lb = .labelBinarizer preprocessing in  
  print_ndarray @@ .fit_transform ['yes' 'no' 'no' 'yes'] lb;  
  [%expect {|
      array([[1],      
             [0],      
             [0],      
             [1]])      
  |}]

*)



(* LabelBinarizer *)
(*
>>> import numpy as np
>>> lb.fit(np.array([[0, 1, 1], [1, 0, 0]]))
LabelBinarizer()
>>> lb.classes_
array([0, 1, 2])
>>> lb.transform([0, 1, 2, 1])
array([[1, 0, 0],
       [0, 1, 0],
       [0, 0, 1],
       [0, 1, 0]])

*)

(* TEST TODO
let%expect_test "LabelBinarizer" =
  let open Sklearn.Calibration in
  print_ndarray @@ .fit np.array((matrixi [|[|0; 1; 1|]; [|1; 0; 0|]|])) lb;  
  [%expect {|
      LabelBinarizer()      
  |}]
  print_ndarray @@ .classes_ lb;  
  [%expect {|
      array([0, 1, 2])      
  |}]
  print_ndarray @@ .transform (vectori [|0; 1; 2; 1|]) lb;  
  [%expect {|
      array([[1, 0, 0],      
             [0, 1, 0],      
             [0, 0, 1],      
             [0, 1, 0]])      
  |}]

*)



(* LabelEncoder *)
(*
>>> from sklearn import preprocessing
>>> le = preprocessing.LabelEncoder()
>>> le.fit([1, 2, 2, 6])
LabelEncoder()
>>> le.classes_
array([1, 2, 6])
>>> le.transform([1, 1, 2, 6])
array([0, 0, 1, 2]...)
>>> le.inverse_transform([0, 0, 1, 2])
array([1, 1, 2, 6])

*)

(* TEST TODO
let%expect_test "LabelEncoder" =
  let open Sklearn.Calibration in
  let le = .labelEncoder preprocessing in  
  print_ndarray @@ .fit (vectori [|1; 2; 2; 6|]) le;  
  [%expect {|
      LabelEncoder()      
  |}]
  print_ndarray @@ .classes_ le;  
  [%expect {|
      array([1, 2, 6])      
  |}]
  print_ndarray @@ .transform (vectori [|1; 1; 2; 6|]) le;  
  [%expect {|
      array([0, 0, 1, 2]...)      
  |}]
  print_ndarray @@ .inverse_transform (vectori [|0; 0; 1; 2|]) le;  
  [%expect {|
      array([1, 1, 2, 6])      
  |}]

*)



(* LabelEncoder *)
(*
>>> le = preprocessing.LabelEncoder()
>>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
LabelEncoder()
>>> list(le.classes_)
['amsterdam', 'paris', 'tokyo']
>>> le.transform(["tokyo", "tokyo", "paris"])
array([2, 2, 1]...)
>>> list(le.inverse_transform([2, 2, 1]))
['tokyo', 'tokyo', 'paris']

*)

(* TEST TODO
let%expect_test "LabelEncoder" =
  let open Sklearn.Calibration in
  let le = .labelEncoder preprocessing in  
  print_ndarray @@ .fit ["paris" "paris" "tokyo" "amsterdam"] le;  
  [%expect {|
      LabelEncoder()      
  |}]
  print_ndarray @@ list le.classes_ ();  
  [%expect {|
      ['amsterdam', 'paris', 'tokyo']      
  |}]
  print_ndarray @@ .transform ["tokyo" "tokyo" "paris"] le;  
  [%expect {|
      array([2, 2, 1]...)      
  |}]
  print_ndarray @@ list(.inverse_transform (vectori [|2; 2; 1|])) le;  
  [%expect {|
      ['tokyo', 'tokyo', 'paris']      
  |}]

*)



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
  let open Sklearn.Calibration in
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



(* label_binarize *)
(*
>>> from sklearn.preprocessing import label_binarize
>>> label_binarize([1, 6], classes=[1, 2, 4, 6])
array([[1, 0, 0, 0],
       [0, 0, 0, 1]])

*)

(* TEST TODO
let%expect_test "label_binarize" =
  let open Sklearn.Calibration in
  print_ndarray @@ label_binarize((vectori [|1; 6|]), classes=(vectori [|1; 2; 4; 6|]));  
  [%expect {|
      array([[1, 0, 0, 0],      
             [0, 0, 0, 1]])      
  |}]

*)



(* label_binarize *)
(*
>>> label_binarize([1, 6], classes=[1, 6, 4, 2])
array([[1, 0, 0, 0],
       [0, 1, 0, 0]])

*)

(* TEST TODO
let%expect_test "label_binarize" =
  let open Sklearn.Calibration in
  print_ndarray @@ label_binarize((vectori [|1; 6|]), classes=(vectori [|1; 6; 4; 2|]));  
  [%expect {|
      array([[1, 0, 0, 0],      
             [0, 1, 0, 0]])      
  |}]

*)



(* label_binarize *)
(*
>>> label_binarize(['yes', 'no', 'no', 'yes'], classes=['no', 'yes'])
array([[1],
       [0],
       [0],
       [1]])

*)

(* TEST TODO
let%expect_test "label_binarize" =
  let open Sklearn.Calibration in
  print_ndarray @@ label_binarize ['yes' 'no' 'no' 'yes'] ~classes:['no' 'yes'] ();  
  [%expect {|
      array([[1],      
             [0],      
             [0],      
             [1]])      
  |}]

*)



