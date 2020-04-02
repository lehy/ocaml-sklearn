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



(* OutputCodeClassifier *)
(*
>>> from sklearn.multiclass import OutputCodeClassifier
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.datasets import make_classification
>>> X, y = make_classification(n_samples=100, n_features=4,
...                            n_informative=2, n_redundant=0,
...                            random_state=0, shuffle=False)
>>> clf = OutputCodeClassifier(
...     estimator=RandomForestClassifier(random_state=0),
...     random_state=0).fit(X, y)
>>> clf.predict([[0, 0, 0, 0]])
array([1])


*)

(* TEST TODO
let%expect_text "OutputCodeClassifier" =
    let outputCodeClassifier = Sklearn.Multiclass.outputCodeClassifier in
    let randomForestClassifier = Sklearn.Ensemble.randomForestClassifier in
    let make_classification = Sklearn.Datasets.make_classification in
    let x, y = make_classification n_samples=100 n_features=4 n_informative=2 n_redundant=0 random_state=0 shuffle=False in
    clf = OutputCodeClassifier(estimator=RandomForestClassifier(random_state=0),random_state=0).fit(X, y)    
    print @@ predict clf [[0 0 0 0]]
    [%expect {|
            array([1])            
    |}]

*)



(* Parallel *)
(*
>>> from math import sqrt
>>> from joblib import Parallel, delayed
>>> Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10))
[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


*)

(* TEST TODO
let%expect_text "Parallel" =
    let sqrt = Math.sqrt in
    from joblib import Parallel, delayed    
    Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10))    
    [%expect {|
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]            
    |}]

*)



(* Parallel *)
(*
>>> from math import modf
>>> from joblib import Parallel, delayed
>>> r = Parallel(n_jobs=1)(delayed(modf)(i/2.) for i in range(10))
>>> res, i = zip( *r)
>>> res
(0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5)
>>> i
(0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0)


*)

(* TEST TODO
let%expect_text "Parallel" =
    let modf = Math.modf in
    from joblib import Parallel, delayed    
    r = Parallel(n_jobs=1)(delayed(modf)(i/2.) for i in range(10))    
    let res, i = zip *r in
    res    
    [%expect {|
            (0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5)            
    |}]
    i    
    [%expect {|
            (0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0)            
    |}]

*)



(* Parallel *)
(*
>>> from time import sleep
>>> from joblib import Parallel, delayed
>>> r = Parallel(n_jobs=2, verbose=10)(delayed(sleep)(.2) for _ in range(10)) #doctest: +SKIP
[Parallel(n_jobs=2)]: Done   1 tasks      | elapsed:    0.6s
[Parallel(n_jobs=2)]: Done   4 tasks      | elapsed:    0.8s
[Parallel(n_jobs=2)]: Done  10 out of  10 | elapsed:    1.4s finished


*)

(* TEST TODO
let%expect_text "Parallel" =
    let sleep = Time.sleep in
    from joblib import Parallel, delayed    
    r = Parallel(n_jobs=2, verbose=10)(delayed(sleep)(.2) for _ in range(10)) #doctest: +SKIP    
    [%expect {|
            [Parallel(n_jobs=2)]: Done   1 tasks      | elapsed:    0.6s            
            [Parallel(n_jobs=2)]: Done   4 tasks      | elapsed:    0.8s            
            [Parallel(n_jobs=2)]: Done  10 out of  10 | elapsed:    1.4s finished            
    |}]

*)



(* Parallel *)
(*
>>> from heapq import nlargest
>>> from joblib import Parallel, delayed
>>> Parallel(n_jobs=2)(delayed(nlargest)(2, n) for n in (range(4), 'abcde', 3)) #doctest: +SKIP
#...
---------------------------------------------------------------------------
Sub-process traceback:
---------------------------------------------------------------------------
TypeError                                          Mon Nov 12 11:37:46 2012
PID: 12934                                    Python 2.7.3: /usr/bin/python
...........................................................................
/usr/lib/python2.7/heapq.pyc in nlargest(n=2, iterable=3, key=None)
    419         if n >= size:
    420             return sorted(iterable, key=key, reverse=True)[:n]
    421
    422     # When key is none, use simpler decoration
    423     if key is None:
--> 424         it = izip(iterable, count(0,-1))                    # decorate
    425         result = _nlargest(n, it)
    426         return map(itemgetter(0), result)                   # undecorate
    427
    428     # General case, slowest method
 TypeError: izip argument #1 must support iteration
___________________________________________________________________________


*)

(* TEST TODO
let%expect_text "Parallel" =
    let nlargest = Heapq.nlargest in
    from joblib import Parallel, delayed    
    Parallel(n_jobs=2)(delayed(nlargest)(2, n) for n in (range(4), 'abcde', 3)) #doctest: +SKIP    
    [%expect {|
            #...            
            ---------------------------------------------------------------------------            
            Sub-process traceback:            
            ---------------------------------------------------------------------------            
            TypeError                                          Mon Nov 12 11:37:46 2012            
            PID: 12934                                    Python 2.7.3: /usr/bin/python            
            ...........................................................................            
            /usr/lib/python2.7/heapq.pyc in nlargest(n=2, iterable=3, key=None)            
                419         if n >= size:            
                420             return sorted(iterable, key=key, reverse=True)[:n]            
                421            
                422     # When key is none, use simpler decoration            
                423     if key is None:            
            --> 424         it = izip(iterable, count(0,-1))                    # decorate            
                425         result = _nlargest(n, it)            
                426         return map(itemgetter(0), result)                   # undecorate            
                427            
                428     # General case, slowest method            
             TypeError: izip argument #1 must support iteration            
            ___________________________________________________________________________            
    |}]

*)



(* euclidean_distances *)
(*
>>> from sklearn.metrics.pairwise import euclidean_distances
>>> X = [[0, 1], [1, 1]]
>>> # distance between rows of X
>>> euclidean_distances(X, X)
array([[0., 1.],
       [1., 0.]])
>>> # get distance to origin
>>> euclidean_distances(X, [[0, 0]])
array([[1.        ],
       [1.41421356]])


*)

(* TEST TODO
let%expect_text "euclidean_distances" =
    let euclidean_distances = Sklearn.Metrics.Pairwise.euclidean_distances in
    X = [[0, 1], [1, 1]]    
    # distance between rows of X    
    euclidean_distances(X, X)    
    [%expect {|
            array([[0., 1.],            
                   [1., 0.]])            
    |}]
    # get distance to origin    
    euclidean_distances(X, [[0, 0]])    
    [%expect {|
            array([[1.        ],            
                   [1.41421356]])            
    |}]

*)



