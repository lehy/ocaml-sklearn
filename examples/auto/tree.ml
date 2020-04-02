(* export_graphviz *)
(*
>>> from sklearn.datasets import load_iris
>>> from sklearn import tree


*)

(* TEST TODO
let%expect_text "export_graphviz" =
    let load_iris = Sklearn.Datasets.load_iris in
    let tree = Sklearn.tree in
    [%expect {|
    |}]

*)



(* export_graphviz *)
(*
>>> clf = tree.DecisionTreeClassifier()
>>> iris = load_iris()


*)

(* TEST TODO
let%expect_text "export_graphviz" =
    clf = tree.DecisionTreeClassifier()    
    iris = load_iris()    
    [%expect {|
    |}]

*)



(* plot_tree *)
(*
>>> from sklearn.datasets import load_iris
>>> from sklearn import tree


*)

(* TEST TODO
let%expect_text "plot_tree" =
    let load_iris = Sklearn.Datasets.load_iris in
    let tree = Sklearn.tree in
    [%expect {|
    |}]

*)



(* plot_tree *)
(*
>>> clf = tree.DecisionTreeClassifier(random_state=0)
>>> iris = load_iris()


*)

(* TEST TODO
let%expect_text "plot_tree" =
    clf = tree.DecisionTreeClassifier(random_state=0)    
    iris = load_iris()    
    [%expect {|
    |}]

*)



