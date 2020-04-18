(* LabelPropagation *)
(*
>>> import numpy as np
>>> from sklearn import datasets
>>> from sklearn.semi_supervised import LabelPropagation
>>> label_prop_model = LabelPropagation()
>>> iris = datasets.load_iris()
>>> rng = np.random.RandomState(42)
>>> random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
>>> labels = np.copy(iris.target)
>>> labels[random_unlabeled_points] = -1
>>> label_prop_model.fit(iris.data, labels)
LabelPropagation(...)

*)

(* TEST TODO
let%expect_test "LabelPropagation" =
  let open Sklearn.Semi_supervised in
  let label_prop_model = LabelPropagation.create () in  
  let iris = .load_iris datasets in  
  let rng = np..randomState ~42 random in  
  let random_unlabeled_points = .rand len iris.target () rng < 0.3 in  
  let labels = .copy iris.target np in  
  print_ndarray @@ labels[random_unlabeled_points] = -1;  
  print LabelPropagation.pp @@ LabelPropagation.fit iris.data ~labels label_prop_model;  
  [%expect {|
      LabelPropagation(...)      
  |}]

*)



(* LabelSpreading *)
(*
>>> import numpy as np
>>> from sklearn import datasets
>>> from sklearn.semi_supervised import LabelSpreading
>>> label_prop_model = LabelSpreading()
>>> iris = datasets.load_iris()
>>> rng = np.random.RandomState(42)
>>> random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
>>> labels = np.copy(iris.target)
>>> labels[random_unlabeled_points] = -1
>>> label_prop_model.fit(iris.data, labels)
LabelSpreading(...)

*)

(* TEST TODO
let%expect_test "LabelSpreading" =
  let open Sklearn.Semi_supervised in
  let label_prop_model = LabelSpreading.create () in  
  let iris = .load_iris datasets in  
  let rng = np..randomState ~42 random in  
  let random_unlabeled_points = .rand len iris.target () rng < 0.3 in  
  let labels = .copy iris.target np in  
  print_ndarray @@ labels[random_unlabeled_points] = -1;  
  print LabelSpreading.pp @@ LabelSpreading.fit iris.data ~labels label_prop_model;  
  [%expect {|
      LabelSpreading(...)      
  |}]

*)



