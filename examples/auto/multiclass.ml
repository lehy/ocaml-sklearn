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
let%expect_test "LabelBinarizer" =
  let open Sklearn.Multiclass in
  let lb = .labelBinarizer preprocessing in  
  print_ndarray @@ .fit (vectori [|1; 2; 6; 4; 2|]) lb;  
  [%expect {|
      LabelBinarizer()      
  |}]
  print_ndarray @@ .classes_ lb;  
  [%expect {|
      array([1, 2, 4, 6])      
  |}]
  print_ndarray @@ .transform (vectori [|1; 6|]) lb;  
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
let%expect_test "LabelBinarizer" =
  let open Sklearn.Multiclass in
  let lb = .labelBinarizer preprocessing in  
  print_ndarray @@ .fit_transform ['yes' 'no' 'no' 'yes'] lb;  
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
let%expect_test "LabelBinarizer" =
  let open Sklearn.Multiclass in
  print_ndarray @@ .fit np.array((matrixi [|[|0; 1; 1|]; [|1; 0; 0|]|])) lb;  
  [%expect {|
      LabelBinarizer()      
  |}]
  print_ndarray @@ .classes_ lb;  
  [%expect {|
      array([0, 1, 2])      
  |}]
  print_ndarray @@ .transform (vectori [|0; 1; 2; 1|]) lb;  
  [%expect {|
      array([[1, 0, 0],      
             [0, 1, 0],      
             [0, 0, 1],      
             [0, 1, 0]])      
  |}]

*)



(* OneVsRestClassifier *)
(*
>>> import numpy as np
>>> from sklearn.multiclass import OneVsRestClassifier
>>> from sklearn.svm import SVC
>>> X = np.array([
...     [10, 10],
...     [8, 10],
...     [-5, 5.5],
...     [-5.4, 5.5],
...     [-20, -20],
...     [-15, -20]
... ])
>>> y = np.array([0, 0, 1, 1, 2, 2])
>>> clf = OneVsRestClassifier(SVC()).fit(X, y)
>>> clf.predict([[-19, -20], [9, 9], [-5, 5]])

*)

(* TEST TODO
let%expect_test "OneVsRestClassifier" =
  let open Sklearn.Multiclass in
  let x = .array [[10 10] [8 10] [-5 5.5] [-5.4 5.5] [-20 -20] [-15 -20]] np in  
  let y = vectori [|0; 0; 1; 1; 2; 2|] in  
  let clf = OneVsRestClassifier(SVC()).fit ~x y () in  
  print_ndarray @@ OneVsRestClassifier.predict (matrixi [|[|-19; -20|]; [|9; 9|]; [|-5; 5|]|]) clf;  
  [%expect {|
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
let%expect_test "OutputCodeClassifier" =
  let open Sklearn.Multiclass in
  let x, y = make_classification ~n_samples:100 ~n_features:4 ~n_informative:2 ~n_redundant:0 ~random_state:0 ~shuffle:false () in  
  let clf = OutputCodeClassifier(estimator=RandomForestClassifier(random_state=0),random_state=0).fit ~x y () in  
  print_ndarray @@ OutputCodeClassifier.predict (matrixi [|[|0; 0; 0; 0|]|]) clf;  
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
let%expect_test "Parallel" =
  let open Sklearn.Multiclass in
  print_ndarray @@ Parallel(n_jobs=1)(delayed ~sqrt ()(i**2) for i in range ~10 ());  
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
let%expect_test "Parallel" =
  let open Sklearn.Multiclass in
  let r = Parallel(n_jobs=1)(delayed ~modf ()(i/2.) for i in range ~10 ()) in  
  let res, i = zip *r () in  
  print_ndarray @@ res;  
  [%expect {|
      (0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5)      
  |}]
  print_ndarray @@ i;  
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
let%expect_test "Parallel" =
  let open Sklearn.Multiclass in
  let r = Parallel(n_jobs=2, verbose=10)(delayed ~sleep ()(.2) for _ in range ~10 ()) #doctest: +SKIP in  
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
let%expect_test "Parallel" =
  let open Sklearn.Multiclass in
  print_ndarray @@ Parallel(n_jobs=2)(delayed ~nlargest ()(2, n) for n in (range ~4 (), 'abcde', 3)) #doctest: +SKIP;  
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



(* Parallel *)
(*
>>> from math import sqrt
>>> from joblib import Parallel, delayed
>>> def producer():
...     for i in range(6):
...         print('Produced %s' % i)
...         yield i
>>> out = Parallel(n_jobs=2, verbose=100, pre_dispatch='1.5*n_jobs')(
...                delayed(sqrt)(i) for i in producer()) #doctest: +SKIP
Produced 0
Produced 1
Produced 2
[Parallel(n_jobs=2)]: Done 1 jobs     | elapsed:  0.0s
Produced 3
[Parallel(n_jobs=2)]: Done 2 jobs     | elapsed:  0.0s
Produced 4
[Parallel(n_jobs=2)]: Done 3 jobs     | elapsed:  0.0s
Produced 5
[Parallel(n_jobs=2)]: Done 4 jobs     | elapsed:  0.0s
[Parallel(n_jobs=2)]: Done 6 out of 6 | elapsed:  0.0s remaining: 0.0s

*)

(* TEST TODO
let%expect_test "Parallel" =
  let open Sklearn.Multiclass in
  print_ndarray @@ def producer ():for i in range ~6 ():print 'Produced %s' % i ()yield i;  
  let out = Parallel(n_jobs=2, verbose=100, pre_dispatch='1.5*n_jobs')(delayed ~sqrt ()(i) for i in producer ()) #doctest: +SKIP in  
  [%expect {|
      Produced 0      
      Produced 1      
      Produced 2      
      [Parallel(n_jobs=2)]: Done 1 jobs     | elapsed:  0.0s      
      Produced 3      
      [Parallel(n_jobs=2)]: Done 2 jobs     | elapsed:  0.0s      
      Produced 4      
      [Parallel(n_jobs=2)]: Done 3 jobs     | elapsed:  0.0s      
      Produced 5      
      [Parallel(n_jobs=2)]: Done 4 jobs     | elapsed:  0.0s      
      [Parallel(n_jobs=2)]: Done 6 out of 6 | elapsed:  0.0s remaining: 0.0s      
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
let%expect_test "euclidean_distances" =
  let open Sklearn.Multiclass in
  let x = (matrixi [|[|0; 1|]; [|1; 1|]|]) in  
  print_ndarray @@ # distance between rows of x;  
  print_ndarray @@ euclidean_distances ~x x ();  
  [%expect {|
      array([[0., 1.],      
             [1., 0.]])      
  |}]
  # get distance to origin  
  print_ndarray @@ euclidean_distances(x, (matrixi [|[|0; 0|]|]));  
  [%expect {|
      array([[1.        ],      
             [1.41421356]])      
  |}]

*)



