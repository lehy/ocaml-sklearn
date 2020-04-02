(* MultiOutputClassifier *)
(*
>>> import numpy as np
>>> from sklearn.datasets import make_multilabel_classification
>>> from sklearn.multioutput import MultiOutputClassifier
>>> from sklearn.neighbors import KNeighborsClassifier


*)

(* TEST TODO
let%expect_text "MultiOutputClassifier" =
    import numpy as np    
    let make_multilabel_classification = Sklearn.Datasets.make_multilabel_classification in
    let multiOutputClassifier = Sklearn.Multioutput.multiOutputClassifier in
    let kNeighborsClassifier = Sklearn.Neighbors.kNeighborsClassifier in
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



