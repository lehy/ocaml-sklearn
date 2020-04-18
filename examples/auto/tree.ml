(* DecisionTreeClassifier *)
(*
>>> from sklearn.datasets import load_iris
>>> from sklearn.model_selection import cross_val_score
>>> from sklearn.tree import DecisionTreeClassifier
>>> clf = DecisionTreeClassifier(random_state=0)
>>> iris = load_iris()
>>> cross_val_score(clf, iris.data, iris.target, cv=10)
...                             # doctest: +SKIP
...
array([ 1.     ,  0.93...,  0.86...,  0.93...,  0.93...,

*)

(* TEST TODO
let%expect_test "DecisionTreeClassifier" =
  let open Sklearn.Tree in
  let clf = DecisionTreeClassifier.create ~random_state:(`Int 0) () in  
  let iris = load_iris () in  
  print_ndarray @@ cross_val_score ~clf iris.data iris.target ~cv:10 ()# doctest: +SKIP;  
  [%expect {|
      array([ 1.     ,  0.93...,  0.86...,  0.93...,  0.93...,      
  |}]

*)



(* DecisionTreeRegressor *)
(*
>>> from sklearn.datasets import load_boston
>>> from sklearn.model_selection import cross_val_score
>>> from sklearn.tree import DecisionTreeRegressor
>>> X, y = load_boston(return_X_y=True)
>>> regressor = DecisionTreeRegressor(random_state=0)
>>> cross_val_score(regressor, X, y, cv=10)
...                    # doctest: +SKIP
...
array([ 0.61..., 0.57..., -0.34..., 0.41..., 0.75...,

*)

(* TEST TODO
let%expect_test "DecisionTreeRegressor" =
  let open Sklearn.Tree in
  let x, y = load_boston ~return_X_y:true () in  
  let regressor = DecisionTreeRegressor.create ~random_state:(`Int 0) () in  
  print_ndarray @@ cross_val_score ~regressor x y ~cv:10 ()# doctest: +SKIP;  
  [%expect {|
      array([ 0.61..., 0.57..., -0.34..., 0.41..., 0.75...,      
  |}]

*)



(* ExtraTreeRegressor *)
(*
>>> from sklearn.datasets import load_boston
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.ensemble import BaggingRegressor
>>> from sklearn.tree import ExtraTreeRegressor
>>> X, y = load_boston(return_X_y=True)
>>> X_train, X_test, y_train, y_test = train_test_split(
...     X, y, random_state=0)
>>> extra_tree = ExtraTreeRegressor(random_state=0)
>>> reg = BaggingRegressor(extra_tree, random_state=0).fit(
...     X_train, y_train)
>>> reg.score(X_test, y_test)

*)

(* TEST TODO
let%expect_test "ExtraTreeRegressor" =
  let open Sklearn.Tree in
  let x, y = load_boston ~return_X_y:true () in  
  let X_train, X_test, y_train, y_test = train_test_split ~x y ~random_state:(`Int 0) () in  
  let extra_tree = ExtraTreeRegressor.create ~random_state:(`Int 0) () in  
  let reg = BaggingRegressor(extra_tree, random_state=0).fit ~X_train y_train () in  
  print_ndarray @@ BaggingRegressor.score ~X_test y_test reg;  
  [%expect {|
  |}]

*)



(* export_graphviz *)
(*
>>> from sklearn.datasets import load_iris
>>> from sklearn import tree

*)

(* TEST TODO
let%expect_test "export_graphviz" =
  let open Sklearn.Tree in
  [%expect {|
  |}]

*)



(* export_graphviz *)
(*
>>> clf = tree.DecisionTreeClassifier()
>>> iris = load_iris()

*)

(* TEST TODO
let%expect_test "export_graphviz" =
  let open Sklearn.Tree in
  let clf = .decisionTreeClassifier tree in  
  let iris = load_iris () in  
  [%expect {|
  |}]

*)



(* export_graphviz *)
(*
>>> clf = clf.fit(iris.data, iris.target)
>>> tree.export_graphviz(clf)

*)

(* TEST TODO
let%expect_test "export_graphviz" =
  let open Sklearn.Tree in
  let clf = .fit iris.data iris.target clf in  
  print_ndarray @@ .export_graphviz ~clf tree;  
  [%expect {|
  |}]

*)



(* export_text *)
(*
>>> from sklearn.datasets import load_iris
>>> from sklearn.tree import DecisionTreeClassifier
>>> from sklearn.tree import export_text
>>> iris = load_iris()
>>> X = iris['data']
>>> y = iris['target']
>>> decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
>>> decision_tree = decision_tree.fit(X, y)
>>> r = export_text(decision_tree, feature_names=iris['feature_names'])
>>> print(r)
|--- petal width (cm) <= 0.80
|   |--- class: 0
|--- petal width (cm) >  0.80
|   |--- petal width (cm) <= 1.75
|   |   |--- class: 1
|   |--- petal width (cm) >  1.75

*)

(* TEST TODO
let%expect_test "export_text" =
  let open Sklearn.Tree in
  let iris = load_iris () in  
  let x = iris['data'] in  
  let y = iris['target'] in  
  let decision_tree = DecisionTreeClassifier.create ~random_state:(`Int 0) ~max_depth:2 () in  
  let decision_tree = DecisionTreeClassifier.fit ~x y decision_tree in  
  let r = export_text decision_tree ~feature_names:iris['feature_names'] () in  
  print_ndarray @@ print ~r ();  
  [%expect {|
      |--- petal width (cm) <= 0.80      
      |   |--- class: 0      
      |--- petal width (cm) >  0.80      
      |   |--- petal width (cm) <= 1.75      
      |   |   |--- class: 1      
      |   |--- petal width (cm) >  1.75      
  |}]

*)



(* plot_tree *)
(*
>>> from sklearn.datasets import load_iris
>>> from sklearn import tree

*)

(* TEST TODO
let%expect_test "plot_tree" =
  let open Sklearn.Tree in
  [%expect {|
  |}]

*)



(* plot_tree *)
(*
>>> clf = tree.DecisionTreeClassifier(random_state=0)
>>> iris = load_iris()

*)

(* TEST TODO
let%expect_test "plot_tree" =
  let open Sklearn.Tree in
  let clf = .decisionTreeClassifier ~random_state:(`Int 0) tree in  
  let iris = load_iris () in  
  [%expect {|
  |}]

*)



(* plot_tree *)
(*
>>> clf = clf.fit(iris.data, iris.target)
>>> tree.plot_tree(clf)  # doctest: +SKIP

*)

(* TEST TODO
let%expect_test "plot_tree" =
  let open Sklearn.Tree in
  let clf = .fit iris.data iris.target clf in  
  print_ndarray @@ .plot_tree ~clf tree # doctest: +SKIP;  
  [%expect {|
  |}]

*)



