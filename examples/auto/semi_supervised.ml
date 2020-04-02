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
let%expect_text "LabelPropagation" =
    import numpy as np    
    let datasets = Sklearn.datasets in
    let labelPropagation = Sklearn.Semi_supervised.labelPropagation in
    label_prop_model = LabelPropagation()    
    iris = datasets.load_iris()    
    rng = np.random.RandomState(42)    
    random_unlabeled_points = rng.rand(len(iris.target)) < 0.3    
    labels = np.copy(iris.target)    
    labels[random_unlabeled_points] = -1    
    print @@ fit label_prop_model iris.data labels
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
let%expect_text "LabelSpreading" =
    import numpy as np    
    let datasets = Sklearn.datasets in
    let labelSpreading = Sklearn.Semi_supervised.labelSpreading in
    label_prop_model = LabelSpreading()    
    iris = datasets.load_iris()    
    rng = np.random.RandomState(42)    
    random_unlabeled_points = rng.rand(len(iris.target)) < 0.3    
    labels = np.copy(iris.target)    
    labels[random_unlabeled_points] = -1    
    print @@ fit label_prop_model iris.data labels
    [%expect {|
            LabelSpreading(...)            
    |}]

*)



