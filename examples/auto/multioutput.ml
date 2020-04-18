(* MultiOutputClassifier *)
(*
>>> import numpy as np
>>> from sklearn.datasets import make_multilabel_classification
>>> from sklearn.multioutput import MultiOutputClassifier
>>> from sklearn.neighbors import KNeighborsClassifier

*)

(* TEST TODO
let%expect_test "MultiOutputClassifier" =
  let open Sklearn.Multioutput in
  [%expect {|
  |}]

*)



(* MultiOutputClassifier *)
(*
>>> X, y = make_multilabel_classification(n_classes=3, random_state=0)
>>> clf = MultiOutputClassifier(KNeighborsClassifier()).fit(X, y)
>>> clf.predict(X[-2:])

*)

(* TEST TODO
let%expect_test "MultiOutputClassifier" =
  let open Sklearn.Multioutput in
  let x, y = make_multilabel_classification ~n_classes:3 ~random_state:(`Int 0) () in  
  let clf = MultiOutputClassifier(KNeighborsClassifier()).fit ~x y () in  
  print_ndarray @@ MultiOutputClassifier.predict x[-2:] clf;  
  [%expect {|
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
  let open Sklearn.Multioutput in
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
  let open Sklearn.Multioutput in
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
  let open Sklearn.Multioutput in
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
  let open Sklearn.Multioutput in
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
  let open Sklearn.Multioutput in
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



(* cross_val_predict *)
(*
>>> from sklearn import datasets, linear_model
>>> from sklearn.model_selection import cross_val_predict
>>> diabetes = datasets.load_diabetes()
>>> X = diabetes.data[:150]
>>> y = diabetes.target[:150]
>>> lasso = linear_model.Lasso()

*)

(* TEST TODO
let%expect_test "cross_val_predict" =
  let open Sklearn.Multioutput in
  let diabetes = .load_diabetes datasets in  
  let x = diabetes.data[:150] in  
  let y = diabetes.target[:150] in  
  let lasso = .lasso linear_model in  
  [%expect {|
  |}]

*)



(* deprecated *)
(*
>>> from sklearn.utils import deprecated
>>> deprecated()
<sklearn.utils.deprecation.deprecated object at ...>

*)

(* TEST TODO
let%expect_test "deprecated" =
  let open Sklearn.Multioutput in
  print_ndarray @@ deprecated ();  
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
let%expect_test "deprecated" =
  let open Sklearn.Multioutput in
  print_ndarray @@ @deprecated ()def some_function (): pass;  
  [%expect {|
  |}]

*)



(* has_fit_parameter *)
(*
>>> from sklearn.svm import SVC
>>> has_fit_parameter(SVC(), "sample_weight")

*)

(* TEST TODO
let%expect_test "has_fit_parameter" =
  let open Sklearn.Multioutput in
  print_ndarray @@ has_fit_parameter(SVC(), "sample_weight");  
  [%expect {|
  |}]

*)



