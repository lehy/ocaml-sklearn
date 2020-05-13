let print f x = Format.printf "%a" f x
let print_py x = Format.printf "%s" (Py.Object.to_string x)
let print_ndarray = print Sklearn.Arr.pp
let print_float = Format.printf "%g\n"
let print_string = Format.printf "%s\n"
let print_int = Format.printf "%d\n"

let matrix = Sklearn.Arr.Float.matrix
let vector = Sklearn.Arr.Float.vector
let matrixi = Sklearn.Arr.Int.matrix
let vectori = Sklearn.Arr.Int.vector
let vectors = Sklearn.Arr.String.vector

let option_get = function Some x -> x | None -> invalid_arg "option_get: None"

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

let%expect_test "DecisionTreeClassifier" =
  let open Sklearn.Tree in
  let clf = DecisionTreeClassifier.create ~random_state:0 () in
  let iris = Sklearn.Datasets.load_iris () in
  print_ndarray @@ Sklearn.Model_selection.cross_val_score ~estimator:clf ~x:iris#data ~y:iris#target ~cv:(`I 10) ();
  [%expect {|
      [1.         0.93333333 1.         0.93333333 0.93333333 0.86666667
       0.93333333 1.         1.         1.        ]
  |}]


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

let%expect_test "DecisionTreeRegressor" =
  let open Sklearn.Tree in
  let boston = Sklearn.Datasets.load_boston () in
  let x, y = boston#data, boston#target in
  let regressor = DecisionTreeRegressor.create ~random_state:0 () in
  print_ndarray @@ Sklearn.Model_selection.cross_val_score ~estimator:regressor ~x ~y ~cv:(`I 10) ();
  [%expect {|
      [ 0.52939335  0.60461936 -1.60907519  0.4356399   0.77280671  0.40597035
        0.23656049  0.38709149 -2.06488186 -0.95162992]
  |}]

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

let%expect_test "ExtraTreeRegressor" =
  let open Sklearn.Tree in
  let boston = Sklearn.Datasets.load_boston () in
  let [@ocaml.warning "-8"] [x_train; x_test; y_train; y_test] =
    Sklearn.Model_selection.train_test_split [boston#data; boston#target] ~random_state:0
  in
  let extra_tree = ExtraTreeRegressor.create ~random_state:0 () in
  let reg = Sklearn.Ensemble.BaggingRegressor.(
      create ~base_estimator:extra_tree ~random_state:0 () |> fit ~x:x_train ~y:y_train)
  in
  print_float @@ Sklearn.Ensemble.BaggingRegressor.score ~x:x_test ~y:y_test reg;
  [%expect {| 0.782357 |}]


(* export_graphviz *)
(*
>>> from sklearn.datasets import load_iris
>>> from sklearn import tree
>>> clf = tree.DecisionTreeClassifier()
>>> iris = load_iris()
>>> clf = clf.fit(iris.data, iris.target)
>>> tree.export_graphviz(clf)

*)

(* let take_3 = function
 *   | a::b::c::_ -> [a;b;c]
 *   | [a;b] -> [a;b]
 *   | [a] -> [a]
 *   | [] -> []
 * 
 * let clip s =
 *    (String.split_on_char '\n' s) |> take_3 |> String.concat "\n" *)

let clip s =
  (String.sub s 0 (min (String.length s) 120)) ^ "..."

let print_some_string = function
  | Some x -> print_string (clip x)
  | None -> assert false

let assert_none = function
  | Some _ -> assert false
  | None -> print_string "OK"

let%expect_test "export_graphviz" =
  let open Sklearn.Tree in
  let clf = DecisionTreeClassifier.create ~random_state:0 () in
  let iris = Sklearn.Datasets.load_iris () in
  let clf = DecisionTreeClassifier.fit ~x:iris#data ~y:iris#target clf in
  print_some_string @@ export_graphviz ~decision_tree:clf ();
  [%expect {|
    digraph Tree {
    node [shape=box] ;
    0 [label="X[3] <= 0.8\ngini = 0.667\nsamples = 150\nvalue = [50, 50, 50]"] ;
    1 [label=...
  |}];
  assert_none @@ export_graphviz ~out_file:(`S "/tmp/tree.dot") ~decision_tree:clf ();
  [%expect {| OK |}]

let%expect_test "export_graphviz_extra" =
  let open Sklearn.Tree in
  let clf = ExtraTreeClassifier.create ~random_state:0 () in
  let iris = Sklearn.Datasets.load_iris () in
  let clf = ExtraTreeClassifier.fit ~x:iris#data ~y:iris#target clf in
  print_some_string @@ export_graphviz ~decision_tree:clf ();
  [%expect {|
    digraph Tree {
    node [shape=box] ;
    0 [label="X[0] <= 5.364\ngini = 0.667\nsamples = 150\nvalue = [50, 50, 50]"] ;
    1 [labe...
  |}];
  assert_none @@ export_graphviz ~out_file:(`S "/tmp/tree.dot") ~decision_tree:clf ();
  [%expect {| OK |}]



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

let%expect_test "export_text" =
  let open Sklearn.Tree in
  let iris = Sklearn.Datasets.load_iris () in
  let x = iris#data in
  let y = iris#target in
  let decision_tree = DecisionTreeClassifier.create ~random_state:0 ~max_depth:2 () in
  let decision_tree = DecisionTreeClassifier.fit ~x ~y decision_tree in
  let r = export_text ~decision_tree ~feature_names:iris#feature_names () in
  print_string r;
  [%expect {|
      |--- petal width (cm) <= 0.80
      |   |--- class: 0
      |--- petal width (cm) >  0.80
      |   |--- petal width (cm) <= 1.75
      |   |   |--- class: 1
      |   |--- petal width (cm) >  1.75
      |   |   |--- class: 2
  |}]


(* plot_tree *)
(*
>>> from sklearn.datasets import load_iris
>>> from sklearn import tree
>>> clf = tree.DecisionTreeClassifier(random_state=0)
>>> iris = load_iris()
>>> clf = clf.fit(iris.data, iris.target)
>>> tree.plot_tree(clf)  # doctest: +SKIP

*)


let%expect_test "plot_tree" =
  let open Sklearn.Tree in
  let clf = DecisionTreeClassifier.create ~random_state:0 () in
  let iris = Sklearn.Datasets.load_iris () in
  let clf = DecisionTreeClassifier.fit ~x:iris#data ~y:iris#target clf in
  print_string @@ clip @@ Py.Object.to_string @@ plot_tree ~decision_tree:clf ();
  [%expect {| [Text(248.0, 338.79999999999995, 'X[3] <= 0.8\ngini = 0.667\nsamples = 150\nvalue = [50, 50, 50]'), Text(209.84615384615... |}]


let%expect_test "plot_tree_extra" =
  let open Sklearn.Tree in
  let clf = ExtraTreeClassifier.create ~random_state:0 () in
  let iris = Sklearn.Datasets.load_iris () in
  let clf = ExtraTreeClassifier.fit ~x:iris#data ~y:iris#target clf in
  print_string @@ clip @@ Py.Object.to_string @@ plot_tree ~decision_tree:clf ();
  [%expect {| [Text(186.0, 352.79999999999995, 'X[0] <= 5.364\ngini = 0.667\nsamples = 150\nvalue = [50, 50, 50]'), Text(53.1428571428... |}]
