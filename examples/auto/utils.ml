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



(*--------- Examples for module .Utils.Arrayfuncs ----------*)
(*--------- Examples for module .Utils.Class_weight ----------*)
(* <no name> *)
(*
>>> from sklearn.utils import deprecated
>>> deprecated()
<sklearn.utils.deprecation.deprecated object at ...>


*)

(* TEST TODO
let%expect_text "<no name>" =
    let deprecated = Sklearn.Utils.deprecated in
    deprecated()    
    [%expect {|
            <sklearn.utils.deprecation.deprecated object at ...>            
    |}]

*)



(* <no name> *)
(*
>>> @deprecated()
... def some_function(): pass


*)

(* TEST TODO
let%expect_text "<no name>" =
    @deprecated()def some_function(): pass    
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



(*--------- Examples for module .Utils.Deprecation ----------*)
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



(*--------- Examples for module .Utils.Extmath ----------*)
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



(* weighted_mode *)
(*
>>> from sklearn.utils.extmath import weighted_mode
>>> x = [4, 1, 4, 2, 4, 2]
>>> weights = [1, 1, 1, 1, 1, 1]
>>> weighted_mode(x, weights)
(array([4.]), array([3.]))


*)

(* TEST TODO
let%expect_text "weighted_mode" =
    let weighted_mode = Sklearn.Utils.Extmath.weighted_mode in
    x = [4, 1, 4, 2, 4, 2]    
    weights = [1, 1, 1, 1, 1, 1]    
    weighted_mode(x, weights)    
    [%expect {|
            (array([4.]), array([3.]))            
    |}]

*)



(* weighted_mode *)
(*
>>> weights = [1, 3, 0.5, 1.5, 1, 2]  # deweight the 4's
>>> weighted_mode(x, weights)
(array([2.]), array([3.5]))


*)

(* TEST TODO
let%expect_text "weighted_mode" =
    weights = [1, 3, 0.5, 1.5, 1, 2]  # deweight the 4's    
    weighted_mode(x, weights)    
    [%expect {|
            (array([2.]), array([3.5]))            
    |}]

*)



(*--------- Examples for module .Utils.Fixes ----------*)
(* compress *)
(*
>>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
>>> x
masked_array(
  data=[[1, --, 3],
        [--, 5, --],
        [7, --, 9]],
  mask=[[False,  True, False],
        [ True, False,  True],
        [False,  True, False]],
  fill_value=999999)
>>> x.compress([1, 0, 1])
masked_array(data=[1, 3],
             mask=[False, False],
       fill_value=999999)


*)

(* TEST TODO
let%expect_text "MaskedArray.compress" =
    x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)    
    x    
    [%expect {|
            masked_array(            
              data=[[1, --, 3],            
                    [--, 5, --],            
                    [7, --, 9]],            
              mask=[[False,  True, False],            
                    [ True, False,  True],            
                    [False,  True, False]],            
              fill_value=999999)            
    |}]
    print @@ compress x [1 0 1]
    [%expect {|
            masked_array(data=[1, 3],            
                         mask=[False, False],            
                   fill_value=999999)            
    |}]

*)



(* copy *)
(*
>>> x = np.array([[1,2,3],[4,5,6]], order='F')


*)

(* TEST TODO
let%expect_text "_arraymethod.<locals>.wrapped_method" =
    x = np.array([[1,2,3],[4,5,6]], order='F')    
    [%expect {|
    |}]

*)



(* copy *)
(*
>>> y = x.copy()


*)

(* TEST TODO
let%expect_text "_arraymethod.<locals>.wrapped_method" =
    y = x.copy()    
    [%expect {|
    |}]

*)



(* copy *)
(*
>>> x.fill(0)


*)

(* TEST TODO
let%expect_text "_arraymethod.<locals>.wrapped_method" =
    print @@ fill x 0
    [%expect {|
    |}]

*)



(* copy *)
(*
>>> x
array([[0, 0, 0],
       [0, 0, 0]])


*)

(* TEST TODO
let%expect_text "_arraymethod.<locals>.wrapped_method" =
    x    
    [%expect {|
            array([[0, 0, 0],            
                   [0, 0, 0]])            
    |}]

*)



(* copy *)
(*
>>> y
array([[1, 2, 3],
       [4, 5, 6]])


*)

(* TEST TODO
let%expect_text "_arraymethod.<locals>.wrapped_method" =
    y    
    [%expect {|
            array([[1, 2, 3],            
                   [4, 5, 6]])            
    |}]

*)



(* count *)
(*
>>> import numpy.ma as ma
>>> a = ma.arange(6).reshape((2, 3))
>>> a[1, :] = ma.masked
>>> a
masked_array(
  data=[[0, 1, 2],
        [--, --, --]],
  mask=[[False, False, False],
        [ True,  True,  True]],
  fill_value=999999)
>>> a.count()
3


*)

(* TEST TODO
let%expect_text "MaskedArray.count" =
    import numpy.ma as ma    
    a = ma.arange(6).reshape((2, 3))    
    a[1, :] = ma.masked    
    a    
    [%expect {|
            masked_array(            
              data=[[0, 1, 2],            
                    [--, --, --]],            
              mask=[[False, False, False],            
                    [ True,  True,  True]],            
              fill_value=999999)            
    |}]
    a.count()    
    [%expect {|
            3            
    |}]

*)



(* filled *)
(*
>>> x = np.ma.array([1,2,3,4,5], mask=[0,0,1,0,1], fill_value=-999)
>>> x.filled()
array([   1,    2, -999,    4, -999])
>>> x.filled(fill_value=1000)
array([   1,    2, 1000,    4, 1000])
>>> type(x.filled())
<class 'numpy.ndarray'>


*)

(* TEST TODO
let%expect_text "MaskedArray.filled" =
    x = np.ma.array([1,2,3,4,5], mask=[0,0,1,0,1], fill_value=-999)    
    x.filled()    
    [%expect {|
            array([   1,    2, -999,    4, -999])            
    |}]
    print @@ filled x fill_value=1000
    [%expect {|
            array([   1,    2, 1000,    4, 1000])            
    |}]
    type(x.filled())    
    [%expect {|
            <class 'numpy.ndarray'>            
    |}]

*)



(* fill_value *)
(*
>>> for dt in [np.int32, np.int64, np.float64, np.complex128]:
...     np.ma.array([0, 1], dtype=dt).get_fill_value()
...
999999
999999
1e+20
(1e+20+0j)


*)

(* TEST TODO
let%expect_text "MaskedArray.fill_value" =
    for dt in [np.int32, np.int64, np.float64, np.complex128]:np.ma.array([0, 1], dtype=dt).get_fill_value()    
    [%expect {|
            999999            
            999999            
            1e+20            
            (1e+20+0j)            
    |}]

*)



(* fill_value *)
(*
>>> x = np.ma.array([0, 1.], fill_value=-np.inf)
>>> x.fill_value
-inf
>>> x.fill_value = np.pi
>>> x.fill_value
3.1415926535897931 # may vary


*)

(* TEST TODO
let%expect_text "MaskedArray.fill_value" =
    x = np.ma.array([0, 1.], fill_value=-np.inf)    
    x.fill_value    
    [%expect {|
            -inf            
    |}]
    x.fill_value = np.pi    
    x.fill_value    
    [%expect {|
            3.1415926535897931 # may vary            
    |}]

*)



(* ids *)
(*
>>> x = np.ma.array([1, 2, 3], mask=[0, 1, 1])
>>> x.ids()
(166670640, 166659832) # may vary


*)

(* TEST TODO
let%expect_text "MaskedArray.ids" =
    x = np.ma.array([1, 2, 3], mask=[0, 1, 1])    
    x.ids()    
    [%expect {|
            (166670640, 166659832) # may vary            
    |}]

*)



(* iscontiguous *)
(*
>>> x = np.ma.array([1, 2, 3])
>>> x.iscontiguous()
True


*)

(* TEST TODO
let%expect_text "MaskedArray.iscontiguous" =
    x = np.ma.array([1, 2, 3])    
    x.iscontiguous()    
    [%expect {|
            True            
    |}]

*)



(* mini *)
(*
>>> x = np.ma.array(np.arange(6), mask=[0 ,1, 0, 0, 0 ,1]).reshape(3, 2)
>>> x
masked_array(
  data=[[0, --],
        [2, 3],
        [4, --]],
  mask=[[False,  True],
        [False, False],
        [False,  True]],
  fill_value=999999)
>>> x.mini()
masked_array(data=0,
             mask=False,
       fill_value=999999)
>>> x.mini(axis=0)
masked_array(data=[0, 3],
             mask=[False, False],
       fill_value=999999)
>>> x.mini(axis=1)
masked_array(data=[0, 2, 4],
             mask=[False, False, False],
       fill_value=999999)


*)

(* TEST TODO
let%expect_text "MaskedArray.mini" =
    x = np.ma.array(np.arange(6), mask=[0 ,1, 0, 0, 0 ,1]).reshape(3, 2)    
    x    
    [%expect {|
            masked_array(            
              data=[[0, --],            
                    [2, 3],            
                    [4, --]],            
              mask=[[False,  True],            
                    [False, False],            
                    [False,  True]],            
              fill_value=999999)            
    |}]
    x.mini()    
    [%expect {|
            masked_array(data=0,            
                         mask=False,            
                   fill_value=999999)            
    |}]
    print @@ mini x axis=0
    [%expect {|
            masked_array(data=[0, 3],            
                         mask=[False, False],            
                   fill_value=999999)            
    |}]
    print @@ mini x axis=1
    [%expect {|
            masked_array(data=[0, 2, 4],            
                         mask=[False, False, False],            
                   fill_value=999999)            
    |}]

*)



(* nonzero *)
(*
>>> import numpy.ma as ma
>>> x = ma.array(np.eye(3))
>>> x
masked_array(
  data=[[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]],
  mask=False,
  fill_value=1e+20)
>>> x.nonzero()
(array([0, 1, 2]), array([0, 1, 2]))


*)

(* TEST TODO
let%expect_text "MaskedArray.nonzero" =
    import numpy.ma as ma    
    x = ma.array(np.eye(3))    
    x    
    [%expect {|
            masked_array(            
              data=[[1., 0., 0.],            
                    [0., 1., 0.],            
                    [0., 0., 1.]],            
              mask=False,            
              fill_value=1e+20)            
    |}]
    x.nonzero()    
    [%expect {|
            (array([0, 1, 2]), array([0, 1, 2]))            
    |}]

*)



(* nonzero *)
(*
>>> x[1, 1] = ma.masked
>>> x
masked_array(
  data=[[1.0, 0.0, 0.0],
        [0.0, --, 0.0],
        [0.0, 0.0, 1.0]],
  mask=[[False, False, False],
        [False,  True, False],
        [False, False, False]],
  fill_value=1e+20)
>>> x.nonzero()
(array([0, 2]), array([0, 2]))


*)

(* TEST TODO
let%expect_text "MaskedArray.nonzero" =
    x[1, 1] = ma.masked    
    x    
    [%expect {|
            masked_array(            
              data=[[1.0, 0.0, 0.0],            
                    [0.0, --, 0.0],            
                    [0.0, 0.0, 1.0]],            
              mask=[[False, False, False],            
                    [False,  True, False],            
                    [False, False, False]],            
              fill_value=1e+20)            
    |}]
    x.nonzero()    
    [%expect {|
            (array([0, 2]), array([0, 2]))            
    |}]

*)



(* nonzero *)
(*
>>> np.transpose(x.nonzero())
array([[0, 0],
       [2, 2]])


*)

(* TEST TODO
let%expect_text "MaskedArray.nonzero" =
    np.transpose(x.nonzero())    
    [%expect {|
            array([[0, 0],            
                   [2, 2]])            
    |}]

*)



(* nonzero *)
(*
>>> a = ma.array([[1,2,3],[4,5,6],[7,8,9]])
>>> a > 3
masked_array(
  data=[[False, False, False],
        [ True,  True,  True],
        [ True,  True,  True]],
  mask=False,
  fill_value=True)
>>> ma.nonzero(a > 3)
(array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))


*)

(* TEST TODO
let%expect_text "MaskedArray.nonzero" =
    a = ma.array([[1,2,3],[4,5,6],[7,8,9]])    
    a > 3    
    [%expect {|
            masked_array(            
              data=[[False, False, False],            
                    [ True,  True,  True],            
                    [ True,  True,  True]],            
              mask=False,            
              fill_value=True)            
    |}]
    print @@ nonzero ma a > 3
    [%expect {|
            (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))            
    |}]

*)



(* partition *)
(*
>>> a = np.array([3, 4, 2, 1])
>>> a.partition(3)
>>> a
array([2, 1, 3, 4])


*)

(* TEST TODO
let%expect_text "MaskedArray.partition" =
    a = np.array([3, 4, 2, 1])    
    print @@ partition a 3
    a    
    [%expect {|
            array([2, 1, 3, 4])            
    |}]

*)



(* put *)
(*
>>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
>>> x
masked_array(
  data=[[1, --, 3],
        [--, 5, --],
        [7, --, 9]],
  mask=[[False,  True, False],
        [ True, False,  True],
        [False,  True, False]],
  fill_value=999999)
>>> x.put([0,4,8],[10,20,30])
>>> x
masked_array(
  data=[[10, --, 3],
        [--, 20, --],
        [7, --, 30]],
  mask=[[False,  True, False],
        [ True, False,  True],
        [False,  True, False]],
  fill_value=999999)


*)

(* TEST TODO
let%expect_text "MaskedArray.put" =
    x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)    
    x    
    [%expect {|
            masked_array(            
              data=[[1, --, 3],            
                    [--, 5, --],            
                    [7, --, 9]],            
              mask=[[False,  True, False],            
                    [ True, False,  True],            
                    [False,  True, False]],            
              fill_value=999999)            
    |}]
    print @@ put x [0 4 8] [10 20 30]
    x    
    [%expect {|
            masked_array(            
              data=[[10, --, 3],            
                    [--, 20, --],            
                    [7, --, 30]],            
              mask=[[False,  True, False],            
                    [ True, False,  True],            
                    [False,  True, False]],            
              fill_value=999999)            
    |}]

*)



(* sort *)
(*
>>> a = np.ma.array([1, 2, 5, 4, 3],mask=[0, 1, 0, 1, 0])
>>> # Default
>>> a.sort()
>>> a
masked_array(data=[1, 3, 5, --, --],
             mask=[False, False, False,  True,  True],
       fill_value=999999)


*)

(* TEST TODO
let%expect_text "MaskedArray.sort" =
    a = np.ma.array([1, 2, 5, 4, 3],mask=[0, 1, 0, 1, 0])    
    # Default    
    a.sort()    
    a    
    [%expect {|
            masked_array(data=[1, 3, 5, --, --],            
                         mask=[False, False, False,  True,  True],            
                   fill_value=999999)            
    |}]

*)



(* sort *)
(*
>>> a = np.ma.array([1, 2, 5, 4, 3],mask=[0, 1, 0, 1, 0])
>>> # Put missing values in the front
>>> a.sort(endwith=False)
>>> a
masked_array(data=[--, --, 1, 3, 5],
             mask=[ True,  True, False, False, False],
       fill_value=999999)


*)

(* TEST TODO
let%expect_text "MaskedArray.sort" =
    a = np.ma.array([1, 2, 5, 4, 3],mask=[0, 1, 0, 1, 0])    
    # Put missing values in the front    
    print @@ sort a endwith=False
    a    
    [%expect {|
            masked_array(data=[--, --, 1, 3, 5],            
                         mask=[ True,  True, False, False, False],            
                   fill_value=999999)            
    |}]

*)



(* var *)
(*
>>> a = np.array([[1, 2], [3, 4]])
>>> np.var(a)
1.25
>>> np.var(a, axis=0)
array([1.,  1.])
>>> np.var(a, axis=1)
array([0.25,  0.25])


*)

(* TEST TODO
let%expect_text "MaskedArray.var" =
    a = np.array([[1, 2], [3, 4]])    
    print @@ var np a
    [%expect {|
            1.25            
    |}]
    print @@ var np a axis=0
    [%expect {|
            array([1.,  1.])            
    |}]
    print @@ var np a axis=1
    [%expect {|
            array([0.25,  0.25])            
    |}]

*)



(* var *)
(*
>>> a = np.zeros((2, 512*512), dtype=np.float32)
>>> a[0, :] = 1.0
>>> a[1, :] = 0.1
>>> np.var(a)
0.20250003


*)

(* TEST TODO
let%expect_text "MaskedArray.var" =
    a = np.zeros((2, 512*512), dtype=np.float32)    
    a[0, :] = 1.0    
    a[1, :] = 0.1    
    print @@ var np a
    [%expect {|
            0.20250003            
    |}]

*)



(* lobpcg *)
(*
>>> import numpy as np
>>> from scipy.sparse import spdiags, issparse
>>> from scipy.sparse.linalg import lobpcg, LinearOperator
>>> n = 100
>>> vals = np.arange(1, n + 1)
>>> A = spdiags(vals, 0, n, n)
>>> A.toarray()
array([[  1.,   0.,   0., ...,   0.,   0.,   0.],
       [  0.,   2.,   0., ...,   0.,   0.,   0.],
       [  0.,   0.,   3., ...,   0.,   0.,   0.],
       ...,
       [  0.,   0.,   0., ...,  98.,   0.,   0.],
       [  0.,   0.,   0., ...,   0.,  99.,   0.],
       [  0.,   0.,   0., ...,   0.,   0., 100.]])


*)

(* TEST TODO
let%expect_text "lobpcg" =
    import numpy as np    
    from scipy.sparse import spdiags, issparse    
    from scipy.sparse.linalg import lobpcg, LinearOperator    
    n = 100    
    vals = np.arange(1, n + 1)    
    A = spdiags(vals, 0, n, n)    
    A.toarray()    
    [%expect {|
            array([[  1.,   0.,   0., ...,   0.,   0.,   0.],            
                   [  0.,   2.,   0., ...,   0.,   0.,   0.],            
                   [  0.,   0.,   3., ...,   0.,   0.,   0.],            
                   ...,            
                   [  0.,   0.,   0., ...,  98.,   0.,   0.],            
                   [  0.,   0.,   0., ...,   0.,  99.,   0.],            
                   [  0.,   0.,   0., ...,   0.,   0., 100.]])            
    |}]

*)



(* lobpcg *)
(*
>>> Y = np.eye(n, 3)


*)

(* TEST TODO
let%expect_text "lobpcg" =
    Y = np.eye(n, 3)    
    [%expect {|
    |}]

*)



(* lobpcg *)
(*
>>> X = np.random.rand(n, 3)


*)

(* TEST TODO
let%expect_text "lobpcg" =
    X = np.random.rand(n, 3)    
    [%expect {|
    |}]

*)



(* lobpcg *)
(*
>>> invA = spdiags([1./vals], 0, n, n)


*)

(* TEST TODO
let%expect_text "lobpcg" =
    invA = spdiags([1./vals], 0, n, n)    
    [%expect {|
    |}]

*)



(* lobpcg *)
(*
>>> def precond( x ):
...     return invA @ x


*)

(* TEST TODO
let%expect_text "lobpcg" =
    def precond( x ):return invA @ x    
    [%expect {|
    |}]

*)



(* lobpcg *)
(*
>>> M = LinearOperator(matvec=precond, matmat=precond,
...                    shape=(n, n), dtype=float)


*)

(* TEST TODO
let%expect_text "lobpcg" =
    M = LinearOperator(matvec=precond, matmat=precond,shape=(n, n), dtype=float)    
    [%expect {|
    |}]

*)



(* lobpcg *)
(*
>>> eigenvalues, _ = lobpcg(A, X, Y=Y, M=M, largest=False)
>>> eigenvalues
array([4., 5., 6.])


*)

(* TEST TODO
let%expect_text "lobpcg" =
    let eigenvalues, _ = lobpcg a x y=y m=m largest=False in
    eigenvalues    
    [%expect {|
            array([4., 5., 6.])            
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



(* <no name> *)
(*
>>> from scipy.stats import Distribution
>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots(1, 1)


*)

(* TEST TODO
let%expect_text "<no name>" =
    let distribution = Scipy.Stats.distribution in
    import matplotlib.pyplot as plt    
    fig, ax = plt.subplots(1, 1)    
    [%expect {|
    |}]

*)



(* <no name> *)
(*
>>> a, b = 
>>> mean, var, skew, kurt = Distribution.stats(a, b, moments='mvsk')


*)

(* TEST TODO
let%expect_text "<no name>" =
    a, b =     
    mean, var, skew, kurt = Distribution.stats(a, b, moments='mvsk')    
    [%expect {|
    |}]

*)



(* <no name> *)
(*
>>> x = np.linspace(Distribution.ppf(0.01, a, b),
...                 Distribution.ppf(0.99, a, b), 100)
>>> ax.plot(x, Distribution.pdf(x, a, b),
...        'r-', lw=5, alpha=0.6, label='Distribution pdf')


*)

(* TEST TODO
let%expect_text "<no name>" =
    x = np.linspace(Distribution.ppf(0.01, a, b),Distribution.ppf(0.99, a, b), 100)    
    ax.plot(x, Distribution.pdf(x, a, b),'r-', lw=5, alpha=0.6, label='Distribution pdf')    
    [%expect {|
    |}]

*)



(* <no name> *)
(*
>>> rv = Distribution(a, b)
>>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')


*)

(* TEST TODO
let%expect_text "<no name>" =
    rv = Distribution(a, b)    
    ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')    
    [%expect {|
    |}]

*)



(* <no name> *)
(*
>>> vals = Distribution.ppf([0.001, 0.5, 0.999], a, b)
>>> np.allclose([0.001, 0.5, 0.999], Distribution.cdf(vals, a, b))
True


*)

(* TEST TODO
let%expect_text "<no name>" =
    vals = Distribution.ppf([0.001, 0.5, 0.999], a, b)    
    np.allclose([0.001, 0.5, 0.999], Distribution.cdf(vals, a, b))    
    [%expect {|
            True            
    |}]

*)



(* <no name> *)
(*
>>> r = Distribution.rvs(a, b, size=1000)


*)

(* TEST TODO
let%expect_text "<no name>" =
    r = Distribution.rvs(a, b, size=1000)    
    [%expect {|
    |}]

*)



(* <no name> *)
(*
>>> ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
>>> ax.legend(loc='best', frameon=False)
>>> plt.show()


*)

(* TEST TODO
let%expect_text "<no name>" =
    print @@ hist ax r density=True histtype='stepfilled' alpha=0.2
    print @@ legend ax loc='best' frameon=False
    plt.show()    
    [%expect {|
    |}]

*)



(* <no name> *)
(*
>>> import numpy as np
>>> fig, ax = plt.subplots(1, 1)
>>> ax.hist(np.log10(r))
>>> ax.set_ylabel("Frequency")
>>> ax.set_xlabel("Value of random variable")
>>> ax.xaxis.set_major_locator(plt.FixedLocator([-2, -1, 0]))
>>> ticks = ["$10^{{ {} }}$".format(i) for i in [-2, -1, 0]]
>>> ax.set_xticklabels(ticks)  # doctest: +SKIP
>>> plt.show()


*)

(* TEST TODO
let%expect_text "<no name>" =
    import numpy as np    
    fig, ax = plt.subplots(1, 1)    
    ax.hist(np.log10(r))    
    print @@ set_ylabel ax "Frequency"
    print @@ set_xlabel ax "Value of random variable"
    ax.xaxis.set_major_locator(plt.FixedLocator([-2, -1, 0]))    
    ticks = ["$10^{{ {} }}$".format(i) for i in [-2, -1, 0]]    
    ax.set_xticklabels(ticks)  # doctest: +SKIP    
    plt.show()    
    [%expect {|
    |}]

*)



(* <no name> *)
(*
>>> rvs = Distribution(2**-2, 2**0).rvs(size=1000)


*)

(* TEST TODO
let%expect_text "<no name>" =
    rvs = Distribution(2**-2, 2**0).rvs(size=1000)    
    [%expect {|
    |}]

*)



(* lsqr *)
(*
>>> from scipy.sparse import csc_matrix
>>> from scipy.sparse.linalg import lsqr
>>> A = csc_matrix([[1., 0.], [1., 1.], [0., 1.]], dtype=float)


*)

(* TEST TODO
let%expect_text "lsqr" =
    let csc_matrix = Scipy.Sparse.csc_matrix in
    let lsqr = Scipy.Sparse.Linalg.lsqr in
    A = csc_matrix([[1., 0.], [1., 1.], [0., 1.]], dtype=float)    
    [%expect {|
    |}]

*)



(* lsqr *)
(*
>>> b = np.array([0., 0., 0.], dtype=float)
>>> x, istop, itn, normr = lsqr(A, b)[:4]
The exact solution is  x = 0
>>> istop
0
>>> x
array([ 0.,  0.])


*)

(* TEST TODO
let%expect_text "lsqr" =
    b = np.array([0., 0., 0.], dtype=float)    
    x, istop, itn, normr = lsqr(A, b)[:4]    
    [%expect {|
            The exact solution is  x = 0            
    |}]
    istop    
    [%expect {|
            0            
    |}]
    x    
    [%expect {|
            array([ 0.,  0.])            
    |}]

*)



(* lsqr *)
(*
>>> b = np.array([1., 0., -1.], dtype=float)
>>> x, istop, itn, r1norm = lsqr(A, b)[:4]
>>> istop
1
>>> x
array([ 1., -1.])
>>> itn
1
>>> r1norm
4.440892098500627e-16


*)

(* TEST TODO
let%expect_text "lsqr" =
    b = np.array([1., 0., -1.], dtype=float)    
    x, istop, itn, r1norm = lsqr(A, b)[:4]    
    istop    
    [%expect {|
            1            
    |}]
    x    
    [%expect {|
            array([ 1., -1.])            
    |}]
    itn    
    [%expect {|
            1            
    |}]
    r1norm    
    [%expect {|
            4.440892098500627e-16            
    |}]

*)



(* lsqr *)
(*
>>> b = np.array([1., 0.01, -1.], dtype=float)
>>> x, istop, itn, r1norm = lsqr(A, b)[:4]
>>> istop
2
>>> x
array([ 1.00333333, -0.99666667])
>>> A.dot(x)-b
array([ 0.00333333, -0.00333333,  0.00333333])
>>> r1norm
0.005773502691896255


*)

(* TEST TODO
let%expect_text "lsqr" =
    b = np.array([1., 0.01, -1.], dtype=float)    
    x, istop, itn, r1norm = lsqr(A, b)[:4]    
    istop    
    [%expect {|
            2            
    |}]
    x    
    [%expect {|
            array([ 1.00333333, -0.99666667])            
    |}]
    A.dot(x)-b    
    [%expect {|
            array([ 0.00333333, -0.00333333,  0.00333333])            
    |}]
    r1norm    
    [%expect {|
            0.005773502691896255            
    |}]

*)



(*--------- Examples for module .Utils.Graph ----------*)
(*--------- Examples for module .Utils.Graph_shortest_path ----------*)
(* csr_matrix *)
(*
>>> import numpy as np
>>> from scipy.sparse import csr_matrix
>>> csr_matrix((3, 4), dtype=np.int8).toarray()
array([[0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]], dtype=int8)


*)

(* TEST TODO
let%expect_text "csr_matrix" =
    import numpy as np    
    let csr_matrix = Scipy.Sparse.csr_matrix in
    csr_matrix((3, 4), dtype=np.int8).toarray()    
    [%expect {|
            array([[0, 0, 0, 0],            
                   [0, 0, 0, 0],            
                   [0, 0, 0, 0]], dtype=int8)            
    |}]

*)



(* csr_matrix *)
(*
>>> row = np.array([0, 0, 1, 2, 2, 2])
>>> col = np.array([0, 2, 2, 0, 1, 2])
>>> data = np.array([1, 2, 3, 4, 5, 6])
>>> csr_matrix((data, (row, col)), shape=(3, 3)).toarray()
array([[1, 0, 2],
       [0, 0, 3],
       [4, 5, 6]])


*)

(* TEST TODO
let%expect_text "csr_matrix" =
    row = np.array([0, 0, 1, 2, 2, 2])    
    col = np.array([0, 2, 2, 0, 1, 2])    
    data = np.array([1, 2, 3, 4, 5, 6])    
    csr_matrix((data, (row, col)), shape=(3, 3)).toarray()    
    [%expect {|
            array([[1, 0, 2],            
                   [0, 0, 3],            
                   [4, 5, 6]])            
    |}]

*)



(* csr_matrix *)
(*
>>> indptr = np.array([0, 2, 3, 6])
>>> indices = np.array([0, 2, 2, 0, 1, 2])
>>> data = np.array([1, 2, 3, 4, 5, 6])
>>> csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()
array([[1, 0, 2],
       [0, 0, 3],
       [4, 5, 6]])


*)

(* TEST TODO
let%expect_text "csr_matrix" =
    indptr = np.array([0, 2, 3, 6])    
    indices = np.array([0, 2, 2, 0, 1, 2])    
    data = np.array([1, 2, 3, 4, 5, 6])    
    csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()    
    [%expect {|
            array([[1, 0, 2],            
                   [0, 0, 3],            
                   [4, 5, 6]])            
    |}]

*)



(* isspmatrix *)
(*
>>> from scipy.sparse import csr_matrix, isspmatrix
>>> isspmatrix(csr_matrix([[5]]))
True


*)

(* TEST TODO
let%expect_text "isspmatrix" =
    from scipy.sparse import csr_matrix, isspmatrix    
    isspmatrix(csr_matrix([[5]]))    
    [%expect {|
            True            
    |}]

*)



(* isspmatrix_csr *)
(*
>>> from scipy.sparse import csr_matrix, isspmatrix_csr
>>> isspmatrix_csr(csr_matrix([[5]]))
True


*)

(* TEST TODO
let%expect_text "isspmatrix_csr" =
    from scipy.sparse import csr_matrix, isspmatrix_csr    
    isspmatrix_csr(csr_matrix([[5]]))    
    [%expect {|
            True            
    |}]

*)



(* isspmatrix *)
(*
>>> from scipy.sparse import csr_matrix, isspmatrix
>>> isspmatrix(csr_matrix([[5]]))
True


*)

(* TEST TODO
let%expect_text "isspmatrix" =
    from scipy.sparse import csr_matrix, isspmatrix    
    isspmatrix(csr_matrix([[5]]))    
    [%expect {|
            True            
    |}]

*)



(*--------- Examples for module .Utils.Metaestimators ----------*)
(*--------- Examples for module .Utils.Multiclass ----------*)
(* isspmatrix *)
(*
>>> from scipy.sparse import csr_matrix, isspmatrix
>>> isspmatrix(csr_matrix([[5]]))
True


*)

(* TEST TODO
let%expect_text "isspmatrix" =
    from scipy.sparse import csr_matrix, isspmatrix    
    isspmatrix(csr_matrix([[5]]))    
    [%expect {|
            True            
    |}]

*)



(*--------- Examples for module .Utils.Murmurhash ----------*)
(*--------- Examples for module .Utils.Optimize ----------*)
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



(* parallel_backend *)
(*
>>> from operator import neg
>>> with parallel_backend('threading'):
...     print(Parallel()(delayed(neg)(i + 1) for i in range(5)))
...
[-1, -2, -3, -4, -5]


*)

(* TEST TODO
let%expect_text "parallel_backend" =
    let neg = Operator.neg in
    with parallel_backend('threading'):print(Parallel()(delayed(neg)(i + 1) for i in range(5)))    
    [%expect {|
            [-1, -2, -3, -4, -5]            
    |}]

*)



(*--------- Examples for module .Utils.Random ----------*)
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



(*--------- Examples for module .Utils.Sparsefuncs ----------*)
(*--------- Examples for module .Utils.Sparsefuncs_fast ----------*)
(*--------- Examples for module .Utils.Stats ----------*)
(*--------- Examples for module .Utils.Validation ----------*)
