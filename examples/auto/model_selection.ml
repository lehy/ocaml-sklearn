(* GridSearchCV *)
(*
>>> from sklearn import svm, datasets
>>> from sklearn.model_selection import GridSearchCV
>>> iris = datasets.load_iris()
>>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
>>> svc = svm.SVC()
>>> clf = GridSearchCV(svc, parameters)
>>> clf.fit(iris.data, iris.target)
GridSearchCV(estimator=SVC(),
             param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')})
>>> sorted(clf.cv_results_.keys())
['mean_fit_time', 'mean_score_time', 'mean_test_score',...
 'param_C', 'param_kernel', 'params',...
 'rank_test_score', 'split0_test_score',...
 'split2_test_score', ...
 'std_fit_time', 'std_score_time', 'std_test_score']

*)

(* TEST TODO
let%expect_test "GridSearchCV" =
  let open Sklearn.Model_selection in
  let iris = .load_iris datasets in  
  let parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]} in  
  let svc = .svc svm in  
  let clf = GridSearchCV.create ~svc parameters () in  
  print GridSearchCV.pp @@ GridSearchCV.fit iris.data iris.target clf;  
  [%expect {|
      GridSearchCV(estimator=SVC(),      
                   param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')})      
  |}]
  print_ndarray @@ sorted clf..keys () cv_results_;  
  [%expect {|
      ['mean_fit_time', 'mean_score_time', 'mean_test_score',...      
       'param_C', 'param_kernel', 'params',...      
       'rank_test_score', 'split0_test_score',...      
       'split2_test_score', ...      
       'std_fit_time', 'std_score_time', 'std_test_score']      
  |}]

*)



(* GroupKFold *)
(*
>>> import numpy as np
>>> from sklearn.model_selection import GroupKFold
>>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
>>> y = np.array([1, 2, 3, 4])
>>> groups = np.array([0, 0, 2, 2])
>>> group_kfold = GroupKFold(n_splits=2)
>>> group_kfold.get_n_splits(X, y, groups)
2
>>> print(group_kfold)
GroupKFold(n_splits=2)
>>> for train_index, test_index in group_kfold.split(X, y, groups):
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]
...     print(X_train, X_test, y_train, y_test)
...
TRAIN: [0 1] TEST: [2 3]
[[1 2]
 [3 4]] [[5 6]
 [7 8]] [1 2] [3 4]
TRAIN: [2 3] TEST: [0 1]
[[5 6]
 [7 8]] [[1 2]
 [3 4]] [3 4] [1 2]

*)

(* TEST TODO
let%expect_test "GroupKFold" =
  let open Sklearn.Model_selection in
  let x = .array (matrixi [|[|1; 2|]; [|3; 4|]; [|5; 6|]; [|7; 8|]|]) np in  
  let y = .array (vectori [|1; 2; 3; 4|]) np in  
  let groups = .array (vectori [|0; 0; 2; 2|]) np in  
  let group_kfold = GroupKFold.create ~n_splits:2 () in  
  print_ndarray @@ GroupKFold.get_n_splits ~x y ~groups group_kfold;  
  [%expect {|
      2      
  |}]
  print_ndarray @@ print ~group_kfold ();  
  [%expect {|
      GroupKFold(n_splits=2)      
  |}]
  print_ndarray @@ for train_index, test_index in GroupKFold.split ~x y groups):print "TRAIN:" ~train_index "TEST:" test_index ()X_train X_test = x[train_index] x[test_index]y_train y_test = y[train_index] y[test_index]print(X_train ~X_test y_train ~y_test group_kfold;  
  [%expect {|
      TRAIN: [0 1] TEST: [2 3]      
      [[1 2]      
       [3 4]] [[5 6]      
       [7 8]] [1 2] [3 4]      
      TRAIN: [2 3] TEST: [0 1]      
      [[5 6]      
       [7 8]] [[1 2]      
       [3 4]] [3 4] [1 2]      
  |}]

*)



(* GroupShuffleSplit *)
(*
>>> import numpy as np
>>> from sklearn.model_selection import GroupShuffleSplit
>>> X = np.ones(shape=(8, 2))
>>> y = np.ones(shape=(8, 1))
>>> groups = np.array([1, 1, 2, 2, 2, 3, 3, 3])
>>> print(groups.shape)
(8,)
>>> gss = GroupShuffleSplit(n_splits=2, train_size=.7, random_state=42)
>>> gss.get_n_splits()
2
>>> for train_idx, test_idx in gss.split(X, y, groups):
...     print("TRAIN:", train_idx, "TEST:", test_idx)
TRAIN: [2 3 4 5 6 7] TEST: [0 1]

*)

(* TEST TODO
let%expect_test "GroupShuffleSplit" =
  let open Sklearn.Model_selection in
  let x = .ones ~shape:(8 2) np in  
  let y = .ones ~shape:(8 1) np in  
  let groups = .array (vectori [|1; 1; 2; 2; 2; 3; 3; 3|]) np in  
  print_ndarray @@ print groups.shape ();  
  [%expect {|
      (8,)      
  |}]
  let gss = GroupShuffleSplit.create ~n_splits:2 ~train_size:.7 ~random_state:42 () in  
  print_ndarray @@ GroupShuffleSplit.get_n_splits gss;  
  [%expect {|
      2      
  |}]
  print_ndarray @@ for train_idx, test_idx in GroupShuffleSplit.split ~x y groups):print("TRAIN:" ~train_idx "TEST:" ~test_idx gss;  
  [%expect {|
      TRAIN: [2 3 4 5 6 7] TEST: [0 1]      
  |}]

*)



(* KFold *)
(*
>>> import numpy as np
>>> from sklearn.model_selection import KFold
>>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
>>> y = np.array([1, 2, 3, 4])
>>> kf = KFold(n_splits=2)
>>> kf.get_n_splits(X)
2
>>> print(kf)
KFold(n_splits=2, random_state=None, shuffle=False)
>>> for train_index, test_index in kf.split(X):
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]
TRAIN: [2 3] TEST: [0 1]
TRAIN: [0 1] TEST: [2 3]

*)

(* TEST TODO
let%expect_test "KFold" =
  let open Sklearn.Model_selection in
  let x = .array (matrixi [|[|1; 2|]; [|3; 4|]; [|1; 2|]; [|3; 4|]|]) np in  
  let y = .array (vectori [|1; 2; 3; 4|]) np in  
  let kf = KFold.create ~n_splits:2 () in  
  print_ndarray @@ KFold.get_n_splits ~x kf;  
  [%expect {|
      2      
  |}]
  print_ndarray @@ print ~kf ();  
  [%expect {|
      KFold(n_splits=2, random_state=None, shuffle=False)      
  |}]
  print_ndarray @@ for train_index, test_index in KFold.split x):print("TRAIN:" ~train_index "TEST:" ~test_index kfX_train, X_test = x[train_index], x[test_index]y_train, y_test = y[train_index], y[test_index];  
  [%expect {|
      TRAIN: [2 3] TEST: [0 1]      
      TRAIN: [0 1] TEST: [2 3]      
  |}]

*)



(* LeaveOneGroupOut *)
(*
>>> import numpy as np
>>> from sklearn.model_selection import LeaveOneGroupOut
>>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
>>> y = np.array([1, 2, 1, 2])
>>> groups = np.array([1, 1, 2, 2])
>>> logo = LeaveOneGroupOut()
>>> logo.get_n_splits(X, y, groups)
2
>>> logo.get_n_splits(groups=groups)  # 'groups' is always required
2
>>> print(logo)
LeaveOneGroupOut()
>>> for train_index, test_index in logo.split(X, y, groups):
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]
...     print(X_train, X_test, y_train, y_test)
TRAIN: [2 3] TEST: [0 1]
[[5 6]
 [7 8]] [[1 2]
 [3 4]] [1 2] [1 2]
TRAIN: [0 1] TEST: [2 3]
[[1 2]
 [3 4]] [[5 6]

*)

(* TEST TODO
let%expect_test "LeaveOneGroupOut" =
  let open Sklearn.Model_selection in
  let x = .array (matrixi [|[|1; 2|]; [|3; 4|]; [|5; 6|]; [|7; 8|]|]) np in  
  let y = .array (vectori [|1; 2; 1; 2|]) np in  
  let groups = .array (vectori [|1; 1; 2; 2|]) np in  
  let logo = LeaveOneGroupOut.create () in  
  print_ndarray @@ LeaveOneGroupOut.get_n_splits ~x y ~groups logo;  
  [%expect {|
      2      
  |}]
  print_ndarray @@ LeaveOneGroupOut.get_n_splits ~groups:groups logo # 'groups' is always required;  
  [%expect {|
      2      
  |}]
  print_ndarray @@ print ~logo ();  
  [%expect {|
      LeaveOneGroupOut()      
  |}]
  print_ndarray @@ for train_index, test_index in LeaveOneGroupOut.split ~x y groups):print "TRAIN:" ~train_index "TEST:" test_index ()X_train X_test = x[train_index] x[test_index]y_train y_test = y[train_index] y[test_index]print(X_train ~X_test y_train ~y_test logo;  
  [%expect {|
      TRAIN: [2 3] TEST: [0 1]      
      [[5 6]      
       [7 8]] [[1 2]      
       [3 4]] [1 2] [1 2]      
      TRAIN: [0 1] TEST: [2 3]      
      [[1 2]      
       [3 4]] [[5 6]      
  |}]

*)



(* LeaveOneOut *)
(*
>>> import numpy as np
>>> from sklearn.model_selection import LeaveOneOut
>>> X = np.array([[1, 2], [3, 4]])
>>> y = np.array([1, 2])
>>> loo = LeaveOneOut()
>>> loo.get_n_splits(X)
2
>>> print(loo)
LeaveOneOut()
>>> for train_index, test_index in loo.split(X):
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]
...     print(X_train, X_test, y_train, y_test)
TRAIN: [1] TEST: [0]
[[3 4]] [[1 2]] [2] [1]
TRAIN: [0] TEST: [1]
[[1 2]] [[3 4]] [1] [2]

*)

(* TEST TODO
let%expect_test "LeaveOneOut" =
  let open Sklearn.Model_selection in
  let x = .array (matrixi [|[|1; 2|]; [|3; 4|]|]) np in  
  let y = .array (vectori [|1; 2|]) np in  
  let loo = LeaveOneOut.create () in  
  print_ndarray @@ LeaveOneOut.get_n_splits ~x loo;  
  [%expect {|
      2      
  |}]
  print_ndarray @@ print ~loo ();  
  [%expect {|
      LeaveOneOut()      
  |}]
  print_ndarray @@ for train_index, test_index in LeaveOneOut.split x):print "TRAIN:" ~train_index "TEST:" test_index ()X_train X_test = x[train_index] x[test_index]y_train y_test = y[train_index] y[test_index]print(X_train ~X_test y_train ~y_test loo;  
  [%expect {|
      TRAIN: [1] TEST: [0]      
      [[3 4]] [[1 2]] [2] [1]      
      TRAIN: [0] TEST: [1]      
      [[1 2]] [[3 4]] [1] [2]      
  |}]

*)



(* LeavePGroupsOut *)
(*
>>> import numpy as np
>>> from sklearn.model_selection import LeavePGroupsOut
>>> X = np.array([[1, 2], [3, 4], [5, 6]])
>>> y = np.array([1, 2, 1])
>>> groups = np.array([1, 2, 3])
>>> lpgo = LeavePGroupsOut(n_groups=2)
>>> lpgo.get_n_splits(X, y, groups)
3
>>> lpgo.get_n_splits(groups=groups)  # 'groups' is always required
3
>>> print(lpgo)
LeavePGroupsOut(n_groups=2)
>>> for train_index, test_index in lpgo.split(X, y, groups):
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]
...     print(X_train, X_test, y_train, y_test)
TRAIN: [2] TEST: [0 1]
[[5 6]] [[1 2]
 [3 4]] [1] [1 2]
TRAIN: [1] TEST: [0 2]
[[3 4]] [[1 2]
 [5 6]] [2] [1 1]
TRAIN: [0] TEST: [1 2]
[[1 2]] [[3 4]
 [5 6]] [1] [2 1]

*)

(* TEST TODO
let%expect_test "LeavePGroupsOut" =
  let open Sklearn.Model_selection in
  let x = .array (matrixi [|[|1; 2|]; [|3; 4|]; [|5; 6|]|]) np in  
  let y = .array (vectori [|1; 2; 1|]) np in  
  let groups = .array (vectori [|1; 2; 3|]) np in  
  let lpgo = LeavePGroupsOut.create ~n_groups:2 () in  
  print_ndarray @@ LeavePGroupsOut.get_n_splits ~x y ~groups lpgo;  
  [%expect {|
      3      
  |}]
  print_ndarray @@ LeavePGroupsOut.get_n_splits ~groups:groups lpgo # 'groups' is always required;  
  [%expect {|
      3      
  |}]
  print_ndarray @@ print ~lpgo ();  
  [%expect {|
      LeavePGroupsOut(n_groups=2)      
  |}]
  print_ndarray @@ for train_index, test_index in LeavePGroupsOut.split ~x y groups):print "TRAIN:" ~train_index "TEST:" test_index ()X_train X_test = x[train_index] x[test_index]y_train y_test = y[train_index] y[test_index]print(X_train ~X_test y_train ~y_test lpgo;  
  [%expect {|
      TRAIN: [2] TEST: [0 1]      
      [[5 6]] [[1 2]      
       [3 4]] [1] [1 2]      
      TRAIN: [1] TEST: [0 2]      
      [[3 4]] [[1 2]      
       [5 6]] [2] [1 1]      
      TRAIN: [0] TEST: [1 2]      
      [[1 2]] [[3 4]      
       [5 6]] [1] [2 1]      
  |}]

*)



(* LeavePOut *)
(*
>>> import numpy as np
>>> from sklearn.model_selection import LeavePOut
>>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
>>> y = np.array([1, 2, 3, 4])
>>> lpo = LeavePOut(2)
>>> lpo.get_n_splits(X)
6
>>> print(lpo)
LeavePOut(p=2)
>>> for train_index, test_index in lpo.split(X):
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]
TRAIN: [2 3] TEST: [0 1]
TRAIN: [1 3] TEST: [0 2]
TRAIN: [1 2] TEST: [0 3]
TRAIN: [0 3] TEST: [1 2]
TRAIN: [0 2] TEST: [1 3]

*)

(* TEST TODO
let%expect_test "LeavePOut" =
  let open Sklearn.Model_selection in
  let x = .array (matrixi [|[|1; 2|]; [|3; 4|]; [|5; 6|]; [|7; 8|]|]) np in  
  let y = .array (vectori [|1; 2; 3; 4|]) np in  
  let lpo = LeavePOut.create ~2 () in  
  print_ndarray @@ LeavePOut.get_n_splits ~x lpo;  
  [%expect {|
      6      
  |}]
  print_ndarray @@ print ~lpo ();  
  [%expect {|
      LeavePOut(p=2)      
  |}]
  print_ndarray @@ for train_index, test_index in LeavePOut.split x):print("TRAIN:" ~train_index "TEST:" ~test_index lpoX_train, X_test = x[train_index], x[test_index]y_train, y_test = y[train_index], y[test_index];  
  [%expect {|
      TRAIN: [2 3] TEST: [0 1]      
      TRAIN: [1 3] TEST: [0 2]      
      TRAIN: [1 2] TEST: [0 3]      
      TRAIN: [0 3] TEST: [1 2]      
      TRAIN: [0 2] TEST: [1 3]      
  |}]

*)



(* ParameterGrid *)
(*
>>> from sklearn.model_selection import ParameterGrid
>>> param_grid = {'a': [1, 2], 'b': [True, False]}
>>> list(ParameterGrid(param_grid)) == (
...    [{'a': 1, 'b': True}, {'a': 1, 'b': False},
...     {'a': 2, 'b': True}, {'a': 2, 'b': False}])
True

*)

(* TEST TODO
let%expect_test "ParameterGrid" =
  let open Sklearn.Model_selection in
  let param_grid = {'a': (vectori [|1; 2|]), 'b': [true, false]} in  
  print_ndarray @@ list(ParameterGrid(param_grid)) == ([{'a': 1, 'b': true}, {'a': 1, 'b': false},{'a': 2, 'b': true}, {'a': 2, 'b': false}]);  
  [%expect {|
      True      
  |}]

*)



(* ParameterGrid *)
(*
>>> grid = [{'kernel': ['linear']}, {'kernel': ['rbf'], 'gamma': [1, 10]}]
>>> list(ParameterGrid(grid)) == [{'kernel': 'linear'},
...                               {'kernel': 'rbf', 'gamma': 1},
...                               {'kernel': 'rbf', 'gamma': 10}]
True
>>> ParameterGrid(grid)[1] == {'kernel': 'rbf', 'gamma': 1}
True

*)

(* TEST TODO
let%expect_test "ParameterGrid" =
  let open Sklearn.Model_selection in
  let grid = [{'kernel': ['linear']}, {'kernel': ['rbf'], 'gamma': [1, 10]}] in  
  print_ndarray @@ list(ParameterGrid(grid)) == [{'kernel': 'linear'},{'kernel': 'rbf', 'gamma': 1},{'kernel': 'rbf', 'gamma': 10}];  
  [%expect {|
      True      
  |}]
  print_ndarray @@ ParameterGrid(grid)(vectori [|1|]) == {'kernel': 'rbf', 'gamma': 1};  
  [%expect {|
      True      
  |}]

*)



(* ParameterSampler *)
(*
>>> from sklearn.model_selection import ParameterSampler
>>> from scipy.stats.distributions import expon
>>> import numpy as np
>>> rng = np.random.RandomState(0)
>>> param_grid = {'a':[1, 2], 'b': expon()}
>>> param_list = list(ParameterSampler(param_grid, n_iter=4,
...                                    random_state=rng))
>>> rounded_list = [dict((k, round(v, 6)) for (k, v) in d.items())
...                 for d in param_list]
>>> rounded_list == [{'b': 0.89856, 'a': 1},
...                  {'b': 0.923223, 'a': 1},
...                  {'b': 1.878964, 'a': 2},
...                  {'b': 1.038159, 'a': 2}]

*)

(* TEST TODO
let%expect_test "ParameterSampler" =
  let open Sklearn.Model_selection in
  let rng = np..randomState ~0 random in  
  let param_grid = {'a':(vectori [|1; 2|]), 'b': expon ()} in  
  let param_list = list(ParameterSampler(param_grid, n_iter=4,random_state=rng)) in  
  let rounded_list = [dict((k, round ~v 6 ()) for (k, v) in d.items ())for d in param_list] in  
  print_ndarray @@ rounded_list == [{'b': 0.89856, 'a': 1},{'b': 0.923223, 'a': 1},{'b': 1.878964, 'a': 2},{'b': 1.038159, 'a': 2}];  
  [%expect {|
  |}]

*)



(* PredefinedSplit *)
(*
>>> import numpy as np
>>> from sklearn.model_selection import PredefinedSplit
>>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
>>> y = np.array([0, 0, 1, 1])
>>> test_fold = [0, 1, -1, 1]
>>> ps = PredefinedSplit(test_fold)
>>> ps.get_n_splits()
2
>>> print(ps)
PredefinedSplit(test_fold=array([ 0,  1, -1,  1]))
>>> for train_index, test_index in ps.split():
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]
TRAIN: [1 2 3] TEST: [0]

*)

(* TEST TODO
let%expect_test "PredefinedSplit" =
  let open Sklearn.Model_selection in
  let x = .array (matrixi [|[|1; 2|]; [|3; 4|]; [|1; 2|]; [|3; 4|]|]) np in  
  let y = .array (vectori [|0; 0; 1; 1|]) np in  
  let test_fold = [0, 1, -1, 1] in  
  let ps = PredefinedSplit.create ~test_fold () in  
  print_ndarray @@ PredefinedSplit.get_n_splits ps;  
  [%expect {|
      2      
  |}]
  print_ndarray @@ print ~ps ();  
  [%expect {|
      PredefinedSplit(test_fold=array([ 0,  1, -1,  1]))      
  |}]
  print_ndarray @@ for train_index, test_index in PredefinedSplit.split ):print("TRAIN:" ~train_index "TEST:" ~test_index psX_train, X_test = x[train_index], x[test_index]y_train, y_test = y[train_index], y[test_index];  
  [%expect {|
      TRAIN: [1 2 3] TEST: [0]      
  |}]

*)



(* RandomizedSearchCV *)
(*
>>> from sklearn.datasets import load_iris
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.model_selection import RandomizedSearchCV
>>> from scipy.stats import uniform
>>> iris = load_iris()
>>> logistic = LogisticRegression(solver='saga', tol=1e-2, max_iter=200,
...                               random_state=0)
>>> distributions = dict(C=uniform(loc=0, scale=4),
...                      penalty=['l2', 'l1'])
>>> clf = RandomizedSearchCV(logistic, distributions, random_state=0)
>>> search = clf.fit(iris.data, iris.target)
>>> search.best_params_

*)

(* TEST TODO
let%expect_test "RandomizedSearchCV" =
  let open Sklearn.Model_selection in
  let iris = load_iris () in  
  let logistic = LogisticRegression.create ~solver:'saga' ~tol:1e-2 ~max_iter:200 ~random_state:0 () in  
  let distributions = dict(C=uniform ~loc:0 ~scale:4 (),penalty=['l2', 'l1']) in  
  let clf = RandomizedSearchCV.create ~logistic distributions ~random_state:0 () in  
  let search = RandomizedSearchCV.fit iris.data iris.target clf in  
  print_ndarray @@ .best_params_ search;  
  [%expect {|
  |}]

*)



(* RepeatedKFold *)
(*
>>> import numpy as np
>>> from sklearn.model_selection import RepeatedKFold
>>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
>>> y = np.array([0, 0, 1, 1])
>>> rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=2652124)
>>> for train_index, test_index in rkf.split(X):
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]
...
TRAIN: [0 1] TEST: [2 3]
TRAIN: [2 3] TEST: [0 1]
TRAIN: [1 2] TEST: [0 3]
TRAIN: [0 3] TEST: [1 2]

*)

(* TEST TODO
let%expect_test "RepeatedKFold" =
  let open Sklearn.Model_selection in
  let x = .array (matrixi [|[|1; 2|]; [|3; 4|]; [|1; 2|]; [|3; 4|]|]) np in  
  let y = .array (vectori [|0; 0; 1; 1|]) np in  
  let rkf = RepeatedKFold.create ~n_splits:2 ~n_repeats:2 ~random_state:2652124 () in  
  print_ndarray @@ for train_index, test_index in RepeatedKFold.split x):print("TRAIN:" ~train_index "TEST:" ~test_index rkfX_train, X_test = x[train_index], x[test_index]y_train, y_test = y[train_index], y[test_index];  
  [%expect {|
      TRAIN: [0 1] TEST: [2 3]      
      TRAIN: [2 3] TEST: [0 1]      
      TRAIN: [1 2] TEST: [0 3]      
      TRAIN: [0 3] TEST: [1 2]      
  |}]

*)



(* RepeatedStratifiedKFold *)
(*
>>> import numpy as np
>>> from sklearn.model_selection import RepeatedStratifiedKFold
>>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
>>> y = np.array([0, 0, 1, 1])
>>> rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2,
...     random_state=36851234)
>>> for train_index, test_index in rskf.split(X, y):
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]
...
TRAIN: [1 2] TEST: [0 3]
TRAIN: [0 3] TEST: [1 2]
TRAIN: [1 3] TEST: [0 2]
TRAIN: [0 2] TEST: [1 3]

*)

(* TEST TODO
let%expect_test "RepeatedStratifiedKFold" =
  let open Sklearn.Model_selection in
  let x = .array (matrixi [|[|1; 2|]; [|3; 4|]; [|1; 2|]; [|3; 4|]|]) np in  
  let y = .array (vectori [|0; 0; 1; 1|]) np in  
  let rskf = RepeatedStratifiedKFold.create ~n_splits:2 ~n_repeats:2 ~random_state:36851234 () in  
  print_ndarray @@ for train_index, test_index in RepeatedStratifiedKFold.split ~x y):print("TRAIN:" ~train_index "TEST:" ~test_index rskfX_train, X_test = x[train_index], x[test_index]y_train, y_test = y[train_index], y[test_index];  
  [%expect {|
      TRAIN: [1 2] TEST: [0 3]      
      TRAIN: [0 3] TEST: [1 2]      
      TRAIN: [1 3] TEST: [0 2]      
      TRAIN: [0 2] TEST: [1 3]      
  |}]

*)



(* ShuffleSplit *)
(*
>>> import numpy as np
>>> from sklearn.model_selection import ShuffleSplit
>>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [3, 4], [5, 6]])
>>> y = np.array([1, 2, 1, 2, 1, 2])
>>> rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
>>> rs.get_n_splits(X)
5
>>> print(rs)
ShuffleSplit(n_splits=5, random_state=0, test_size=0.25, train_size=None)
>>> for train_index, test_index in rs.split(X):
...     print("TRAIN:", train_index, "TEST:", test_index)
TRAIN: [1 3 0 4] TEST: [5 2]
TRAIN: [4 0 2 5] TEST: [1 3]
TRAIN: [1 2 4 0] TEST: [3 5]
TRAIN: [3 4 1 0] TEST: [5 2]
TRAIN: [3 5 1 0] TEST: [2 4]
>>> rs = ShuffleSplit(n_splits=5, train_size=0.5, test_size=.25,
...                   random_state=0)
>>> for train_index, test_index in rs.split(X):
...     print("TRAIN:", train_index, "TEST:", test_index)
TRAIN: [1 3 0] TEST: [5 2]
TRAIN: [4 0 2] TEST: [1 3]
TRAIN: [1 2 4] TEST: [3 5]
TRAIN: [3 4 1] TEST: [5 2]

*)

(* TEST TODO
let%expect_test "ShuffleSplit" =
  let open Sklearn.Model_selection in
  let x = .array (matrixi [|[|1; 2|]; [|3; 4|]; [|5; 6|]; [|7; 8|]; [|3; 4|]; [|5; 6|]|]) np in  
  let y = .array (vectori [|1; 2; 1; 2; 1; 2|]) np in  
  let rs = ShuffleSplit.create ~n_splits:5 ~test_size:.25 ~random_state:0 () in  
  print_ndarray @@ ShuffleSplit.get_n_splits ~x rs;  
  [%expect {|
      5      
  |}]
  print_ndarray @@ print ~rs ();  
  [%expect {|
      ShuffleSplit(n_splits=5, random_state=0, test_size=0.25, train_size=None)      
  |}]
  print_ndarray @@ for train_index, test_index in ShuffleSplit.split x):print("TRAIN:" ~train_index "TEST:" ~test_index rs;  
  [%expect {|
      TRAIN: [1 3 0 4] TEST: [5 2]      
      TRAIN: [4 0 2 5] TEST: [1 3]      
      TRAIN: [1 2 4 0] TEST: [3 5]      
      TRAIN: [3 4 1 0] TEST: [5 2]      
      TRAIN: [3 5 1 0] TEST: [2 4]      
  |}]
  let rs = ShuffleSplit.create ~n_splits:5 ~train_size:0.5 ~test_size:.25 ~random_state:0 () in  
  print_ndarray @@ for train_index, test_index in ShuffleSplit.split x):print("TRAIN:" ~train_index "TEST:" ~test_index rs;  
  [%expect {|
      TRAIN: [1 3 0] TEST: [5 2]      
      TRAIN: [4 0 2] TEST: [1 3]      
      TRAIN: [1 2 4] TEST: [3 5]      
      TRAIN: [3 4 1] TEST: [5 2]      
  |}]

*)



(* StratifiedKFold *)
(*
>>> import numpy as np
>>> from sklearn.model_selection import StratifiedKFold
>>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
>>> y = np.array([0, 0, 1, 1])
>>> skf = StratifiedKFold(n_splits=2)
>>> skf.get_n_splits(X, y)
2
>>> print(skf)
StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
>>> for train_index, test_index in skf.split(X, y):
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]
TRAIN: [1 3] TEST: [0 2]
TRAIN: [0 2] TEST: [1 3]

*)

(* TEST TODO
let%expect_test "StratifiedKFold" =
  let open Sklearn.Model_selection in
  let x = .array (matrixi [|[|1; 2|]; [|3; 4|]; [|1; 2|]; [|3; 4|]|]) np in  
  let y = .array (vectori [|0; 0; 1; 1|]) np in  
  let skf = StratifiedKFold.create ~n_splits:2 () in  
  print_ndarray @@ StratifiedKFold.get_n_splits ~x y skf;  
  [%expect {|
      2      
  |}]
  print_ndarray @@ print ~skf ();  
  [%expect {|
      StratifiedKFold(n_splits=2, random_state=None, shuffle=False)      
  |}]
  print_ndarray @@ for train_index, test_index in StratifiedKFold.split ~x y):print("TRAIN:" ~train_index "TEST:" ~test_index skfX_train, X_test = x[train_index], x[test_index]y_train, y_test = y[train_index], y[test_index];  
  [%expect {|
      TRAIN: [1 3] TEST: [0 2]      
      TRAIN: [0 2] TEST: [1 3]      
  |}]

*)



(* StratifiedShuffleSplit *)
(*
>>> import numpy as np
>>> from sklearn.model_selection import StratifiedShuffleSplit
>>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
>>> y = np.array([0, 0, 0, 1, 1, 1])
>>> sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
>>> sss.get_n_splits(X, y)
5
>>> print(sss)
StratifiedShuffleSplit(n_splits=5, random_state=0, ...)
>>> for train_index, test_index in sss.split(X, y):
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]
TRAIN: [5 2 3] TEST: [4 1 0]
TRAIN: [5 1 4] TEST: [0 2 3]
TRAIN: [5 0 2] TEST: [4 3 1]
TRAIN: [4 1 0] TEST: [2 3 5]

*)

(* TEST TODO
let%expect_test "StratifiedShuffleSplit" =
  let open Sklearn.Model_selection in
  let x = .array (matrixi [|[|1; 2|]; [|3; 4|]; [|1; 2|]; [|3; 4|]; [|1; 2|]; [|3; 4|]|]) np in  
  let y = .array (vectori [|0; 0; 0; 1; 1; 1|]) np in  
  let sss = StratifiedShuffleSplit.create ~n_splits:5 ~test_size:0.5 ~random_state:0 () in  
  print_ndarray @@ StratifiedShuffleSplit.get_n_splits ~x y sss;  
  [%expect {|
      5      
  |}]
  print_ndarray @@ print ~sss ();  
  [%expect {|
      StratifiedShuffleSplit(n_splits=5, random_state=0, ...)      
  |}]
  print_ndarray @@ for train_index, test_index in StratifiedShuffleSplit.split ~x y):print("TRAIN:" ~train_index "TEST:" ~test_index sssX_train, X_test = x[train_index], x[test_index]y_train, y_test = y[train_index], y[test_index];  
  [%expect {|
      TRAIN: [5 2 3] TEST: [4 1 0]      
      TRAIN: [5 1 4] TEST: [0 2 3]      
      TRAIN: [5 0 2] TEST: [4 3 1]      
      TRAIN: [4 1 0] TEST: [2 3 5]      
  |}]

*)



(* TimeSeriesSplit *)
(*
>>> import numpy as np
>>> from sklearn.model_selection import TimeSeriesSplit
>>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
>>> y = np.array([1, 2, 3, 4, 5, 6])
>>> tscv = TimeSeriesSplit()
>>> print(tscv)
TimeSeriesSplit(max_train_size=None, n_splits=5)
>>> for train_index, test_index in tscv.split(X):
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]
TRAIN: [0] TEST: [1]
TRAIN: [0 1] TEST: [2]
TRAIN: [0 1 2] TEST: [3]
TRAIN: [0 1 2 3] TEST: [4]
TRAIN: [0 1 2 3 4] TEST: [5]

*)

(* TEST TODO
let%expect_test "TimeSeriesSplit" =
  let open Sklearn.Model_selection in
  let x = .array (matrixi [|[|1; 2|]; [|3; 4|]; [|1; 2|]; [|3; 4|]; [|1; 2|]; [|3; 4|]|]) np in  
  let y = .array (vectori [|1; 2; 3; 4; 5; 6|]) np in  
  let tscv = TimeSeriesSplit.create () in  
  print_ndarray @@ print ~tscv ();  
  [%expect {|
      TimeSeriesSplit(max_train_size=None, n_splits=5)      
  |}]
  print_ndarray @@ for train_index, test_index in TimeSeriesSplit.split x):print("TRAIN:" ~train_index "TEST:" ~test_index tscvX_train, X_test = x[train_index], x[test_index]y_train, y_test = y[train_index], y[test_index];  
  [%expect {|
      TRAIN: [0] TEST: [1]      
      TRAIN: [0 1] TEST: [2]      
      TRAIN: [0 1 2] TEST: [3]      
      TRAIN: [0 1 2 3] TEST: [4]      
      TRAIN: [0 1 2 3 4] TEST: [5]      
  |}]

*)



(* cross_val_predict *)
(*
>>> from sklearn import datasets, linear_model
>>> from sklearn.model_selection import cross_val_predict
>>> diabetes = datasets.load_diabetes()
>>> X = diabetes.data[:150]
>>> y = diabetes.target[:150]
>>> lasso = linear_model.Lasso()

*)

(* TEST TODO
let%expect_test "cross_val_predict" =
  let open Sklearn.Model_selection in
  let diabetes = .load_diabetes datasets in  
  let x = diabetes.data[:150] in  
  let y = diabetes.target[:150] in  
  let lasso = .lasso linear_model in  
  [%expect {|
  |}]

*)



(* cross_val_score *)
(*
>>> from sklearn import datasets, linear_model
>>> from sklearn.model_selection import cross_val_score
>>> diabetes = datasets.load_diabetes()
>>> X = diabetes.data[:150]
>>> y = diabetes.target[:150]
>>> lasso = linear_model.Lasso()
>>> print(cross_val_score(lasso, X, y, cv=3))
[0.33150734 0.08022311 0.03531764]

*)

(* TEST TODO
let%expect_test "cross_val_score" =
  let open Sklearn.Model_selection in
  let diabetes = .load_diabetes datasets in  
  let x = diabetes.data[:150] in  
  let y = diabetes.target[:150] in  
  let lasso = .lasso linear_model in  
  print_ndarray @@ print(cross_val_score ~lasso x y ~cv:3 ());  
  [%expect {|
      [0.33150734 0.08022311 0.03531764]      
  |}]

*)



(* cross_validate *)
(*
>>> from sklearn import datasets, linear_model
>>> from sklearn.model_selection import cross_validate
>>> from sklearn.metrics import make_scorer
>>> from sklearn.metrics import confusion_matrix
>>> from sklearn.svm import LinearSVC
>>> diabetes = datasets.load_diabetes()
>>> X = diabetes.data[:150]
>>> y = diabetes.target[:150]
>>> lasso = linear_model.Lasso()

*)

(* TEST TODO
let%expect_test "cross_validate" =
  let open Sklearn.Model_selection in
  let diabetes = .load_diabetes datasets in  
  let x = diabetes.data[:150] in  
  let y = diabetes.target[:150] in  
  let lasso = .lasso linear_model in  
  [%expect {|
  |}]

*)



(* cross_validate *)
(*
>>> cv_results = cross_validate(lasso, X, y, cv=3)
>>> sorted(cv_results.keys())
['fit_time', 'score_time', 'test_score']
>>> cv_results['test_score']
array([0.33150734, 0.08022311, 0.03531764])

*)

(* TEST TODO
let%expect_test "cross_validate" =
  let open Sklearn.Model_selection in
  let cv_results = cross_validate ~lasso x y ~cv:3 () in  
  print_ndarray @@ sorted .keys () cv_results;  
  [%expect {|
      ['fit_time', 'score_time', 'test_score']      
  |}]
  print_ndarray @@ cv_results['test_score'];  
  [%expect {|
      array([0.33150734, 0.08022311, 0.03531764])      
  |}]

*)



(* cross_validate *)
(*
>>> scores = cross_validate(lasso, X, y, cv=3,
...                         scoring=('r2', 'neg_mean_squared_error'),
...                         return_train_score=True)
>>> print(scores['test_neg_mean_squared_error'])
[-3635.5... -3573.3... -6114.7...]
>>> print(scores['train_r2'])
[0.28010158 0.39088426 0.22784852]

*)

(* TEST TODO
let%expect_test "cross_validate" =
  let open Sklearn.Model_selection in
  let scores = cross_validate(lasso, x, y, cv=3,scoring=('r2', 'neg_mean_squared_error'),return_train_score=true) in  
  print_ndarray @@ print scores['test_neg_mean_squared_error'] ();  
  [%expect {|
      [-3635.5... -3573.3... -6114.7...]      
  |}]
  print_ndarray @@ print scores['train_r2'] ();  
  [%expect {|
      [0.28010158 0.39088426 0.22784852]      
  |}]

*)



(* train_test_split *)
(*
>>> import numpy as np
>>> from sklearn.model_selection import train_test_split
>>> X, y = np.arange(10).reshape((5, 2)), range(5)
>>> X
array([[0, 1],
       [2, 3],
       [4, 5],
       [6, 7],
       [8, 9]])
>>> list(y)
[0, 1, 2, 3, 4]

*)

(* TEST TODO
let%expect_test "train_test_split" =
  let open Sklearn.Model_selection in
  let x, y = .arange 10).reshape((5 2)) range(5 np in  
  print_ndarray @@ x;  
  [%expect {|
      array([[0, 1],      
             [2, 3],      
             [4, 5],      
             [6, 7],      
             [8, 9]])      
  |}]
  print_ndarray @@ list ~y ();  
  [%expect {|
      [0, 1, 2, 3, 4]      
  |}]

*)



(* train_test_split *)
(*
>>> X_train, X_test, y_train, y_test = train_test_split(
...     X, y, test_size=0.33, random_state=42)
...
>>> X_train
array([[4, 5],
       [0, 1],
       [6, 7]])
>>> y_train
[2, 0, 3]
>>> X_test
array([[2, 3],
       [8, 9]])
>>> y_test
[1, 4]

*)

(* TEST TODO
let%expect_test "train_test_split" =
  let open Sklearn.Model_selection in
  let X_train, X_test, y_train, y_test = train_test_split ~x y ~test_size:0.33 ~random_state:42 () in  
  X_train  
  [%expect {|
      array([[4, 5],      
             [0, 1],      
             [6, 7]])      
  |}]
  y_train  
  [%expect {|
      [2, 0, 3]      
  |}]
  print_ndarray @@ X_test;  
  [%expect {|
      array([[2, 3],      
             [8, 9]])      
  |}]
  print_ndarray @@ y_test;  
  [%expect {|
      [1, 4]      
  |}]

*)



(* train_test_split *)
(*
>>> train_test_split(y, shuffle=False)

*)

(* TEST TODO
let%expect_test "train_test_split" =
  let open Sklearn.Model_selection in
  print_ndarray @@ train_test_split y ~shuffle:false ();  
  [%expect {|
  |}]

*)



