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
let%expect_text "GridSearchCV" =
    from sklearn import svm, datasets    
    let gridSearchCV = Sklearn.Model_selection.gridSearchCV in
    iris = datasets.load_iris()    
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}    
    svc = svm.SVC()    
    clf = GridSearchCV(svc, parameters)    
    print @@ fit clf iris.data iris.target
    [%expect {|
            GridSearchCV(estimator=SVC(),            
                         param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')})            
    |}]
    sorted(clf.cv_results_.keys())    
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
let%expect_text "GroupKFold" =
    import numpy as np    
    let groupKFold = Sklearn.Model_selection.groupKFold in
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])    
    y = np.array([1, 2, 3, 4])    
    groups = np.array([0, 0, 2, 2])    
    group_kfold = GroupKFold(n_splits=2)    
    print @@ get_n_splits group_kfold x y groups
    [%expect {|
            2            
    |}]
    print(group_kfold)    
    [%expect {|
            GroupKFold(n_splits=2)            
    |}]
    for train_index, test_index in group_kfold.split(X, y, groups):print("TRAIN:", train_index, "TEST:", test_index)X_train, X_test = X[train_index], X[test_index]y_train, y_test = y[train_index], y[test_index]print(X_train, X_test, y_train, y_test)    
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
let%expect_text "KFold" =
    import numpy as np    
    let kFold = Sklearn.Model_selection.kFold in
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])    
    y = np.array([1, 2, 3, 4])    
    kf = KFold(n_splits=2)    
    print @@ get_n_splits kf x
    [%expect {|
            2            
    |}]
    print(kf)    
    [%expect {|
            KFold(n_splits=2, random_state=None, shuffle=False)            
    |}]
    for train_index, test_index in kf.split(X):print("TRAIN:", train_index, "TEST:", test_index)X_train, X_test = X[train_index], X[test_index]y_train, y_test = y[train_index], y[test_index]    
    [%expect {|
            TRAIN: [2 3] TEST: [0 1]            
            TRAIN: [0 1] TEST: [2 3]            
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
let%expect_text "LeaveOneOut" =
    import numpy as np    
    let leaveOneOut = Sklearn.Model_selection.leaveOneOut in
    X = np.array([[1, 2], [3, 4]])    
    y = np.array([1, 2])    
    loo = LeaveOneOut()    
    print @@ get_n_splits loo x
    [%expect {|
            2            
    |}]
    print(loo)    
    [%expect {|
            LeaveOneOut()            
    |}]
    for train_index, test_index in loo.split(X):print("TRAIN:", train_index, "TEST:", test_index)X_train, X_test = X[train_index], X[test_index]y_train, y_test = y[train_index], y[test_index]print(X_train, X_test, y_train, y_test)    
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
let%expect_text "LeavePGroupsOut" =
    import numpy as np    
    let leavePGroupsOut = Sklearn.Model_selection.leavePGroupsOut in
    X = np.array([[1, 2], [3, 4], [5, 6]])    
    y = np.array([1, 2, 1])    
    groups = np.array([1, 2, 3])    
    lpgo = LeavePGroupsOut(n_groups=2)    
    print @@ get_n_splits lpgo x y groups
    [%expect {|
            3            
    |}]
    lpgo.get_n_splits(groups=groups)  # 'groups' is always required    
    [%expect {|
            3            
    |}]
    print(lpgo)    
    [%expect {|
            LeavePGroupsOut(n_groups=2)            
    |}]
    for train_index, test_index in lpgo.split(X, y, groups):print("TRAIN:", train_index, "TEST:", test_index)X_train, X_test = X[train_index], X[test_index]y_train, y_test = y[train_index], y[test_index]print(X_train, X_test, y_train, y_test)    
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
let%expect_text "ParameterGrid" =
    let parameterGrid = Sklearn.Model_selection.parameterGrid in
    param_grid = {'a': [1, 2], 'b': [True, False]}    
    list(ParameterGrid(param_grid)) == ([{'a': 1, 'b': True}, {'a': 1, 'b': False},{'a': 2, 'b': True}, {'a': 2, 'b': False}])    
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
let%expect_text "ParameterGrid" =
    grid = [{'kernel': ['linear']}, {'kernel': ['rbf'], 'gamma': [1, 10]}]    
    list(ParameterGrid(grid)) == [{'kernel': 'linear'},{'kernel': 'rbf', 'gamma': 1},{'kernel': 'rbf', 'gamma': 10}]    
    [%expect {|
            True            
    |}]
    ParameterGrid(grid)[1] == {'kernel': 'rbf', 'gamma': 1}    
    [%expect {|
            True            
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
let%expect_text "RepeatedKFold" =
    import numpy as np    
    let repeatedKFold = Sklearn.Model_selection.repeatedKFold in
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])    
    y = np.array([0, 0, 1, 1])    
    rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=2652124)    
    for train_index, test_index in rkf.split(X):print("TRAIN:", train_index, "TEST:", test_index)X_train, X_test = X[train_index], X[test_index]y_train, y_test = y[train_index], y[test_index]    
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
let%expect_text "RepeatedStratifiedKFold" =
    import numpy as np    
    let repeatedStratifiedKFold = Sklearn.Model_selection.repeatedStratifiedKFold in
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])    
    y = np.array([0, 0, 1, 1])    
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2,random_state=36851234)    
    for train_index, test_index in rskf.split(X, y):print("TRAIN:", train_index, "TEST:", test_index)X_train, X_test = X[train_index], X[test_index]y_train, y_test = y[train_index], y[test_index]    
    [%expect {|
            TRAIN: [1 2] TEST: [0 3]            
            TRAIN: [0 3] TEST: [1 2]            
            TRAIN: [1 3] TEST: [0 2]            
            TRAIN: [0 2] TEST: [1 3]            
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
let%expect_text "StratifiedKFold" =
    import numpy as np    
    let stratifiedKFold = Sklearn.Model_selection.stratifiedKFold in
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])    
    y = np.array([0, 0, 1, 1])    
    skf = StratifiedKFold(n_splits=2)    
    print @@ get_n_splits skf x y
    [%expect {|
            2            
    |}]
    print(skf)    
    [%expect {|
            StratifiedKFold(n_splits=2, random_state=None, shuffle=False)            
    |}]
    for train_index, test_index in skf.split(X, y):print("TRAIN:", train_index, "TEST:", test_index)X_train, X_test = X[train_index], X[test_index]y_train, y_test = y[train_index], y[test_index]    
    [%expect {|
            TRAIN: [1 3] TEST: [0 2]            
            TRAIN: [0 2] TEST: [1 3]            
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
let%expect_text "TimeSeriesSplit" =
    import numpy as np    
    let timeSeriesSplit = Sklearn.Model_selection.timeSeriesSplit in
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])    
    y = np.array([1, 2, 3, 4, 5, 6])    
    tscv = TimeSeriesSplit()    
    print(tscv)    
    [%expect {|
            TimeSeriesSplit(max_train_size=None, n_splits=5)            
    |}]
    for train_index, test_index in tscv.split(X):print("TRAIN:", train_index, "TEST:", test_index)X_train, X_test = X[train_index], X[test_index]y_train, y_test = y[train_index], y[test_index]    
    [%expect {|
            TRAIN: [0] TEST: [1]            
            TRAIN: [0 1] TEST: [2]            
            TRAIN: [0 1 2] TEST: [3]            
            TRAIN: [0 1 2 3] TEST: [4]            
            TRAIN: [0 1 2 3 4] TEST: [5]            
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
let%expect_text "cross_val_score" =
    from sklearn import datasets, linear_model    
    let cross_val_score = Sklearn.Model_selection.cross_val_score in
    diabetes = datasets.load_diabetes()    
    X = diabetes.data[:150]    
    y = diabetes.target[:150]    
    lasso = linear_model.Lasso()    
    print(cross_val_score(lasso, X, y, cv=3))    
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
let%expect_text "cross_validate" =
    from sklearn import datasets, linear_model    
    let cross_validate = Sklearn.Model_selection.cross_validate in
    let make_scorer = Sklearn.Metrics.make_scorer in
    let confusion_matrix = Sklearn.Metrics.confusion_matrix in
    let linearSVC = Sklearn.Svm.linearSVC in
    diabetes = datasets.load_diabetes()    
    X = diabetes.data[:150]    
    y = diabetes.target[:150]    
    lasso = linear_model.Lasso()    
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
let%expect_text "cross_validate" =
    cv_results = cross_validate(lasso, X, y, cv=3)    
    sorted(cv_results.keys())    
    [%expect {|
            ['fit_time', 'score_time', 'test_score']            
    |}]
    cv_results['test_score']    
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
let%expect_text "cross_validate" =
    scores = cross_validate(lasso, X, y, cv=3,scoring=('r2', 'neg_mean_squared_error'),return_train_score=True)    
    print(scores['test_neg_mean_squared_error'])    
    [%expect {|
            [-3635.5... -3573.3... -6114.7...]            
    |}]
    print(scores['train_r2'])    
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
let%expect_text "train_test_split" =
    import numpy as np    
    let train_test_split = Sklearn.Model_selection.train_test_split in
    X, y = np.arange(10).reshape((5, 2)), range(5)    
    X    
    [%expect {|
            array([[0, 1],            
                   [2, 3],            
                   [4, 5],            
                   [6, 7],            
                   [8, 9]])            
    |}]
    list(y)    
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
let%expect_text "train_test_split" =
    let n, y_test = train_test_split x y test_size=0.33 random_state=42 in
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
    X_test    
    [%expect {|
            array([[2, 3],            
                   [8, 9]])            
    |}]
    y_test    
    [%expect {|
            [1, 4]            
    |}]

*)



