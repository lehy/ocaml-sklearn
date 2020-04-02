(* BernoulliNB *)
(*
>>> import numpy as np
>>> rng = np.random.RandomState(1)
>>> X = rng.randint(5, size=(6, 100))
>>> Y = np.array([1, 2, 3, 4, 4, 5])
>>> from sklearn.naive_bayes import BernoulliNB
>>> clf = BernoulliNB()
>>> clf.fit(X, Y)
BernoulliNB()
>>> print(clf.predict(X[2:3]))
[3]


*)

(* TEST TODO
let%expect_text "BernoulliNB" =
    import numpy as np    
    rng = np.random.RandomState(1)    
    X = rng.randint(5, size=(6, 100))    
    Y = np.array([1, 2, 3, 4, 4, 5])    
    let bernoulliNB = Sklearn.Naive_bayes.bernoulliNB in
    clf = BernoulliNB()    
    print @@ fit clf x y
    [%expect {|
            BernoulliNB()            
    |}]
    print(clf.predict(X[2:3]))    
    [%expect {|
            [3]            
    |}]

*)



(* ComplementNB *)
(*
>>> import numpy as np
>>> rng = np.random.RandomState(1)
>>> X = rng.randint(5, size=(6, 100))
>>> y = np.array([1, 2, 3, 4, 5, 6])
>>> from sklearn.naive_bayes import ComplementNB
>>> clf = ComplementNB()
>>> clf.fit(X, y)
ComplementNB()
>>> print(clf.predict(X[2:3]))
[3]


*)

(* TEST TODO
let%expect_text "ComplementNB" =
    import numpy as np    
    rng = np.random.RandomState(1)    
    X = rng.randint(5, size=(6, 100))    
    y = np.array([1, 2, 3, 4, 5, 6])    
    let complementNB = Sklearn.Naive_bayes.complementNB in
    clf = ComplementNB()    
    print @@ fit clf x y
    [%expect {|
            ComplementNB()            
    |}]
    print(clf.predict(X[2:3]))    
    [%expect {|
            [3]            
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



(* MultinomialNB *)
(*
>>> import numpy as np
>>> rng = np.random.RandomState(1)
>>> X = rng.randint(5, size=(6, 100))
>>> y = np.array([1, 2, 3, 4, 5, 6])
>>> from sklearn.naive_bayes import MultinomialNB
>>> clf = MultinomialNB()
>>> clf.fit(X, y)
MultinomialNB()
>>> print(clf.predict(X[2:3]))
[3]


*)

(* TEST TODO
let%expect_text "MultinomialNB" =
    import numpy as np    
    rng = np.random.RandomState(1)    
    X = rng.randint(5, size=(6, 100))    
    y = np.array([1, 2, 3, 4, 5, 6])    
    let multinomialNB = Sklearn.Naive_bayes.multinomialNB in
    clf = MultinomialNB()    
    print @@ fit clf x y
    [%expect {|
            MultinomialNB()            
    |}]
    print(clf.predict(X[2:3]))    
    [%expect {|
            [3]            
    |}]

*)



(* deprecated *)
(*
>>> from sklearn.utils import deprecated
>>> deprecated()
<sklearn.utils.deprecation.deprecated object at ...>


*)

(* TEST TODO
let%expect_text "deprecated" =
    let deprecated = Sklearn.Utils.deprecated in
    deprecated()    
    [%expect {|
            <sklearn.utils.deprecation.deprecated object at ...>            
    |}]

*)



(* deprecated *)
(*
>>> @deprecated()
... def some_function(): pass


*)

(* TEST TODO
let%expect_text "deprecated" =
    @deprecated()def some_function(): pass    
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



(* logsumexp *)
(*
>>> from scipy.special import logsumexp
>>> a = np.arange(10)
>>> np.log(np.sum(np.exp(a)))
9.4586297444267107
>>> logsumexp(a)
9.4586297444267107


*)

(* TEST TODO
let%expect_text "logsumexp" =
    let logsumexp = Scipy.Special.logsumexp in
    a = np.arange(10)    
    np.log(np.sum(np.exp(a)))    
    [%expect {|
            9.4586297444267107            
    |}]
    logsumexp(a)    
    [%expect {|
            9.4586297444267107            
    |}]

*)



(* logsumexp *)
(*
>>> a = np.arange(10)
>>> b = np.arange(10, 0, -1)
>>> logsumexp(a, b=b)
9.9170178533034665
>>> np.log(np.sum(b*np.exp(a)))
9.9170178533034647


*)

(* TEST TODO
let%expect_text "logsumexp" =
    a = np.arange(10)    
    b = np.arange(10, 0, -1)    
    logsumexp(a, b=b)    
    [%expect {|
            9.9170178533034665            
    |}]
    np.log(np.sum(b*np.exp(a)))    
    [%expect {|
            9.9170178533034647            
    |}]

*)



(* logsumexp *)
(*
>>> logsumexp([1,2],b=[1,-1],return_sign=True)
(1.5413248546129181, -1.0)


*)

(* TEST TODO
let%expect_text "logsumexp" =
    logsumexp([1,2],b=[1,-1],return_sign=True)    
    [%expect {|
            (1.5413248546129181, -1.0)            
    |}]

*)



