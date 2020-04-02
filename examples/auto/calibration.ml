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
let%expect_text "LabelBinarizer" =
    let preprocessing = Sklearn.preprocessing in
    lb = preprocessing.LabelBinarizer()    
    print @@ fit lb [1 2 6 4 2]
    [%expect {|
            LabelBinarizer()            
    |}]
    lb.classes_    
    [%expect {|
            array([1, 2, 4, 6])            
    |}]
    print @@ transform lb [1 6]
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
let%expect_text "LabelBinarizer" =
    lb = preprocessing.LabelBinarizer()    
    print @@ fit_transform lb ['yes' 'no' 'no' 'yes']
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
let%expect_text "LabelBinarizer" =
    import numpy as np    
    lb.fit(np.array([[0, 1, 1], [1, 0, 0]]))    
    [%expect {|
            LabelBinarizer()            
    |}]
    lb.classes_    
    [%expect {|
            array([0, 1, 2])            
    |}]
    print @@ transform lb [0 1 2 1]
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
let%expect_text "LabelEncoder" =
    let preprocessing = Sklearn.preprocessing in
    le = preprocessing.LabelEncoder()    
    print @@ fit le [1 2 2 6]
    [%expect {|
            LabelEncoder()            
    |}]
    le.classes_    
    [%expect {|
            array([1, 2, 6])            
    |}]
    print @@ transform le [1 1 2 6]
    [%expect {|
            array([0, 0, 1, 2]...)            
    |}]
    print @@ inverse_transform le [0 0 1 2]
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
let%expect_text "LabelEncoder" =
    le = preprocessing.LabelEncoder()    
    print @@ fit le ["paris" "paris" "tokyo" "amsterdam"]
    [%expect {|
            LabelEncoder()            
    |}]
    list(le.classes_)    
    [%expect {|
            ['amsterdam', 'paris', 'tokyo']            
    |}]
    print @@ transform le ["tokyo" "tokyo" "paris"]
    [%expect {|
            array([2, 2, 1]...)            
    |}]
    list(le.inverse_transform([2, 2, 1]))    
    [%expect {|
            ['tokyo', 'tokyo', 'paris']            
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
let%expect_text "label_binarize" =
    let label_binarize = Sklearn.Preprocessing.label_binarize in
    label_binarize([1, 6], classes=[1, 2, 4, 6])    
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
let%expect_text "label_binarize" =
    label_binarize([1, 6], classes=[1, 6, 4, 2])    
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
let%expect_text "label_binarize" =
    label_binarize(['yes', 'no', 'no', 'yes'], classes=['no', 'yes'])    
    [%expect {|
            array([[1],            
                   [0],            
                   [0],            
                   [1]])            
    |}]

*)



