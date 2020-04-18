(* Bunch *)
(*
>>> b = Bunch(a=1, b=2)
>>> b['b']
2
>>> b.b
2
>>> b.a = 3
>>> b['a']
3
>>> b.c = 6
>>> b['c']

*)

(* TEST TODO
let%expect_test "Bunch" =
  let open Sklearn.Utils in
  let b = Bunch.create ~a:1 ~b:2 () in  
  print_ndarray @@ b['b'];  
  [%expect {|
      2      
  |}]
  print_ndarray @@ b.b;  
  [%expect {|
      2      
  |}]
  print_ndarray @@ b.a = 3;  
  print_ndarray @@ b['a'];  
  [%expect {|
      3      
  |}]
  print_ndarray @@ b.c = 6;  
  print_ndarray @@ b['c'];  
  [%expect {|
  |}]

*)



(*--------- Examples for module Sklearn.Utils.Arrayfuncs ----------*)
(*--------- Examples for module Sklearn.Utils.Class_weight ----------*)
(* <no name> *)
(*
>>> from sklearn.utils import deprecated
>>> deprecated()
<sklearn.utils.deprecation.deprecated object at ...>

*)

(* TEST TODO
let%expect_test "<no name>" =
  let open Sklearn.Utils in
  print_ndarray @@ deprecated ();  
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
let%expect_test "<no name>" =
  let open Sklearn.Utils in
  print_ndarray @@ @deprecated ()def some_function (): pass;  
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
  let open Sklearn.Utils in
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
  let open Sklearn.Utils in
  print_ndarray @@ @deprecated ()def some_function (): pass;  
  [%expect {|
  |}]

*)



(*--------- Examples for module Sklearn.Utils.Deprecation ----------*)
(* deprecated *)
(*
>>> from sklearn.utils import deprecated
>>> deprecated()
<sklearn.utils.deprecation.deprecated object at ...>

*)

(* TEST TODO
let%expect_test "deprecated" =
  let open Sklearn.Utils in
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
  let open Sklearn.Utils in
  print_ndarray @@ @deprecated ()def some_function (): pass;  
  [%expect {|
  |}]

*)



(*--------- Examples for module Sklearn.Utils.Extmath ----------*)
(* cartesian *)
(*
>>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
array([[1, 4, 6],
       [1, 4, 7],
       [1, 5, 6],
       [1, 5, 7],
       [2, 4, 6],
       [2, 4, 7],
       [2, 5, 6],
       [2, 5, 7],
       [3, 4, 6],
       [3, 4, 7],
       [3, 5, 6],

*)

(* TEST TODO
let%expect_test "cartesian" =
  let open Sklearn.Utils in
  print_ndarray @@ cartesian(((vectori [|1; 2; 3|]), (vectori [|4; 5|]), (vectori [|6; 7|])));  
  [%expect {|
      array([[1, 4, 6],      
             [1, 4, 7],      
             [1, 5, 6],      
             [1, 5, 7],      
             [2, 4, 6],      
             [2, 4, 7],      
             [2, 5, 6],      
             [2, 5, 7],      
             [3, 4, 6],      
             [3, 4, 7],      
             [3, 5, 6],      
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
  let open Sklearn.Utils in
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
  let open Sklearn.Utils in
  print_ndarray @@ @deprecated ()def some_function (): pass;  
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
let%expect_test "weighted_mode" =
  let open Sklearn.Utils in
  let x = (vectori [|4; 1; 4; 2; 4; 2|]) in  
  let weights = (vectori [|1; 1; 1; 1; 1; 1|]) in  
  print_ndarray @@ weighted_mode ~x weights ();  
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
let%expect_test "weighted_mode" =
  let open Sklearn.Utils in
  let weights = [1, 3, 0.5, 1.5, 1, 2] # deweight the 4's in  
  print_ndarray @@ weighted_mode ~x weights ();  
  [%expect {|
      (array([2.]), array([3.5]))      
  |}]

*)



(*--------- Examples for module Sklearn.Utils.Fixes ----------*)
(* all *)
(*
>>> np.ma.array([1,2,3]).all()
True
>>> a = np.ma.array([1,2,3], mask=True)
>>> (a.all() is np.ma.masked)

*)

(* TEST TODO
let%expect_test "MaskedArray.all" =
  let open Sklearn.Utils in
  print_ndarray @@ np..array (vectori [|1;2;3|])).all( ma;  
  [%expect {|
      True      
  |}]
  let a = np..array (vectori [|1;2;3|]) ~mask:true ma in  
  print_ndarray @@ (a.all () is np.ma.masked);  
  [%expect {|
  |}]

*)



(* anom *)
(*
>>> a = np.ma.array([1,2,3])
>>> a.anom()
masked_array(data=[-1.,  0.,  1.],
             mask=False,

*)

(* TEST TODO
let%expect_test "MaskedArray.anom" =
  let open Sklearn.Utils in
  let a = np..array (vectori [|1;2;3|]) ma in  
  print_ndarray @@ a.anom ();  
  [%expect {|
      masked_array(data=[-1.,  0.,  1.],      
                   mask=False,      
  |}]

*)



(* argmax *)
(*
>>> a = np.arange(6).reshape(2,3)
>>> a.argmax()
5
>>> a.argmax(0)
array([1, 1, 1])
>>> a.argmax(1)

*)

(* TEST TODO
let%expect_test "MaskedArray.argmax" =
  let open Sklearn.Utils in
  let a = .arange 6).reshape(2 ~3 np in  
  print_ndarray @@ a.argmax ();  
  [%expect {|
      5      
  |}]
  print_ndarray @@ a.argmax ~0 ();  
  [%expect {|
      array([1, 1, 1])      
  |}]
  print_ndarray @@ a.argmax ~1 ();  
  [%expect {|
  |}]

*)



(* argmin *)
(*
>>> x = np.ma.array(np.arange(4), mask=[1,1,0,0])
>>> x.shape = (2,2)
>>> x
masked_array(
  data=[[--, --],
        [2, 3]],
  mask=[[ True,  True],
        [False, False]],
  fill_value=999999)
>>> x.argmin(axis=0, fill_value=-1)
array([0, 0])
>>> x.argmin(axis=0, fill_value=9)

*)

(* TEST TODO
let%expect_test "MaskedArray.argmin" =
  let open Sklearn.Utils in
  let x = np..array np.arange ~4 () ~mask:(vectori [|1;1;0;0|]) ma in  
  print_ndarray @@ x.shape = (2,2);  
  print_ndarray @@ x;  
  [%expect {|
      masked_array(      
        data=[[--, --],      
              [2, 3]],      
        mask=[[ True,  True],      
              [False, False]],      
        fill_value=999999)      
  |}]
  print_ndarray @@ x.argmin ~axis:0 ~fill_value:-1 ();  
  [%expect {|
      array([0, 0])      
  |}]
  print_ndarray @@ x.argmin ~axis:0 ~fill_value:9 ();  
  [%expect {|
  |}]

*)



(* argsort *)
(*
>>> a = np.ma.array([3,2,1], mask=[False, False, True])
>>> a
masked_array(data=[3, 2, --],
             mask=[False, False,  True],
       fill_value=999999)
>>> a.argsort()

*)

(* TEST TODO
let%expect_test "MaskedArray.argsort" =
  let open Sklearn.Utils in
  let a = np..array (vectori [|3;2;1|]) ~mask:[false ~false true] ma in  
  print_ndarray @@ a;  
  [%expect {|
      masked_array(data=[3, 2, --],      
                   mask=[False, False,  True],      
             fill_value=999999)      
  |}]
  print_ndarray @@ a.argsort ();  
  [%expect {|
  |}]

*)



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
let%expect_test "MaskedArray.compress" =
  let open Sklearn.Utils in
  let x = np..array [(vectori [|1;2;3|]) (vectori [|4;5;6|]) (vectori [|7;8;9|])] ~mask:(vectori [|0|]) + (vectori [|1;0|])*4 ma in  
  print_ndarray @@ x;  
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
  print_ndarray @@ x.compress((vectori [|1; 0; 1|]));  
  [%expect {|
      masked_array(data=[1, 3],      
                   mask=[False, False],      
             fill_value=999999)      
  |}]

*)



(* compress *)
(*
>>> x.compress([1, 0, 1], axis=1)
masked_array(
  data=[[1, 3],
        [--, --],
        [7, 9]],
  mask=[[False, False],
        [ True,  True],
        [False, False]],

*)

(* TEST TODO
let%expect_test "MaskedArray.compress" =
  let open Sklearn.Utils in
  print_ndarray @@ x.compress((vectori [|1; 0; 1|]), axis=1);  
  [%expect {|
      masked_array(      
        data=[[1, 3],      
              [--, --],      
              [7, 9]],      
        mask=[[False, False],      
              [ True,  True],      
              [False, False]],      
  |}]

*)



(* compressed *)
(*
>>> x = np.ma.array(np.arange(5), mask=[0]*2 + [1]*3)
>>> x.compressed()
array([0, 1])
>>> type(x.compressed())

*)

(* TEST TODO
let%expect_test "MaskedArray.compressed" =
  let open Sklearn.Utils in
  let x = np..array np.arange ~5 () ~mask:(vectori [|0|])*2 + (vectori [|1|])*3 ma in  
  print_ndarray @@ x.compressed ();  
  [%expect {|
      array([0, 1])      
  |}]
  print_ndarray @@ type(x.compressed ());  
  [%expect {|
  |}]

*)



(* copy *)
(*
>>> x = np.array([[1,2,3],[4,5,6]], order='F')

*)

(* TEST TODO
let%expect_test "_arraymethod.<locals>.wrapped_method" =
  let open Sklearn.Utils in
  let x = .array [(vectori [|1;2;3|]) (vectori [|4;5;6|])] ~order:'F' np in  
  [%expect {|
  |}]

*)



(* copy *)
(*
>>> y = x.copy()

*)

(* TEST TODO
let%expect_test "_arraymethod.<locals>.wrapped_method" =
  let open Sklearn.Utils in
  let y = x.copy () in  
  [%expect {|
  |}]

*)



(* copy *)
(*
>>> x.fill(0)

*)

(* TEST TODO
let%expect_test "_arraymethod.<locals>.wrapped_method" =
  let open Sklearn.Utils in
  print_ndarray @@ x.fill ~0 ();  
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
let%expect_test "_arraymethod.<locals>.wrapped_method" =
  let open Sklearn.Utils in
  print_ndarray @@ x;  
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
let%expect_test "_arraymethod.<locals>.wrapped_method" =
  let open Sklearn.Utils in
  print_ndarray @@ y;  
  [%expect {|
      array([[1, 2, 3],      
             [4, 5, 6]])      
  |}]

*)



(* copy *)
(*
>>> y.flags['C_CONTIGUOUS']

*)

(* TEST TODO
let%expect_test "_arraymethod.<locals>.wrapped_method" =
  let open Sklearn.Utils in
  print_ndarray @@ y.flags['C_CONTIGUOUS'];  
  [%expect {|
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
let%expect_test "MaskedArray.count" =
  let open Sklearn.Utils in
  let a = .arange 6).reshape((2 3) ma in  
  print_ndarray @@ a(vectori [|1; :|]) = .masked ma;  
  print_ndarray @@ a;  
  [%expect {|
      masked_array(      
        data=[[0, 1, 2],      
              [--, --, --]],      
        mask=[[False, False, False],      
              [ True,  True,  True]],      
        fill_value=999999)      
  |}]
  print_ndarray @@ a.count ();  
  [%expect {|
      3      
  |}]

*)



(* count *)
(*
>>> a.count(axis=0)
array([1, 1, 1])
>>> a.count(axis=1)

*)

(* TEST TODO
let%expect_test "MaskedArray.count" =
  let open Sklearn.Utils in
  print_ndarray @@ a.count ~axis:0 ();  
  [%expect {|
      array([1, 1, 1])      
  |}]
  print_ndarray @@ a.count ~axis:1 ();  
  [%expect {|
  |}]

*)



(* cumsum *)
(*
>>> marr = np.ma.array(np.arange(10), mask=[0,0,0,1,1,1,0,0,0,0])
>>> marr.cumsum()
masked_array(data=[0, 1, 3, --, --, --, 9, 16, 24, 33],
             mask=[False, False, False,  True,  True,  True, False, False,
                   False, False],

*)

(* TEST TODO
let%expect_test "MaskedArray.cumsum" =
  let open Sklearn.Utils in
  let marr = np..array np.arange ~10 () ~mask:(vectori [|0;0;0;1;1;1;0;0;0;0|]) ma in  
  print_ndarray @@ .cumsum marr;  
  [%expect {|
      masked_array(data=[0, 1, 3, --, --, --, 9, 16, 24, 33],      
                   mask=[False, False, False,  True,  True,  True, False, False,      
                         False, False],      
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
let%expect_test "MaskedArray.filled" =
  let open Sklearn.Utils in
  let x = np..array (vectori [|1;2;3;4;5|]) ~mask:(vectori [|0;0;1;0;1|]) ~fill_value:-999 ma in  
  print_ndarray @@ x.filled ();  
  [%expect {|
      array([   1,    2, -999,    4, -999])      
  |}]
  print_ndarray @@ x.filled ~fill_value:1000 ();  
  [%expect {|
      array([   1,    2, 1000,    4, 1000])      
  |}]
  print_ndarray @@ type(x.filled ());  
  [%expect {|
      <class 'numpy.ndarray'>      
  |}]

*)



(* filled *)
(*
>>> x = np.array([(-1, 2), (-3, 4)], dtype='i8,i8').view(np.recarray)
>>> m = np.ma.array(x, mask=[(True, False), (False, True)])
>>> m.filled()
rec.array([(999999,      2), (    -3, 999999)],

*)

(* TEST TODO
let%expect_test "MaskedArray.filled" =
  let open Sklearn.Utils in
  let x = .array [(-1 2) (-3 4)] ~dtype:'i8 i8').view(np.recarray np in  
  let m = np..array x ~mask:[(true false) (false true)] ma in  
  print_ndarray @@ m.filled ();  
  [%expect {|
      rec.array([(999999,      2), (    -3, 999999)],      
  |}]

*)



(* flatten *)
(*
>>> a = np.array([[1,2], [3,4]])
>>> a.flatten()
array([1, 2, 3, 4])
>>> a.flatten('F')

*)

(* TEST TODO
let%expect_test "_arraymethod.<locals>.wrapped_method" =
  let open Sklearn.Utils in
  let a = .array (matrixi [|[|1;2|]; [|3;4|]|]) np in  
  print_ndarray @@ a.flatten ();  
  [%expect {|
      array([1, 2, 3, 4])      
  |}]
  print_ndarray @@ a.flatten 'F' ();  
  [%expect {|
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
let%expect_test "MaskedArray.fill_value" =
  let open Sklearn.Utils in
  print_ndarray @@ for dt in [np.int32, np.int64, np.float64, np.complex128]:np..array (vectori [|0; 1|]) ~dtype:dt).get_fill_value( ma;  
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
let%expect_test "MaskedArray.fill_value" =
  let open Sklearn.Utils in
  let x = np..array [0 1.] ~fill_value:-np.inf ma in  
  print_ndarray @@ x.fill_value;  
  [%expect {|
      -inf      
  |}]
  print_ndarray @@ x.fill_value = .pi np;  
  print_ndarray @@ x.fill_value;  
  [%expect {|
      3.1415926535897931 # may vary      
  |}]

*)



(* fill_value *)
(*
>>> x.fill_value = None
>>> x.fill_value

*)

(* TEST TODO
let%expect_test "MaskedArray.fill_value" =
  let open Sklearn.Utils in
  print_ndarray @@ x.fill_value = None;  
  print_ndarray @@ x.fill_value;  
  [%expect {|
  |}]

*)



(* imag *)
(*
>>> x = np.ma.array([1+1.j, -2j, 3.45+1.6j], mask=[False, True, False])
>>> x.imag
masked_array(data=[1.0, --, 1.6],
             mask=[False,  True, False],

*)

(* TEST TODO
let%expect_test "MaskedArray.imag" =
  let open Sklearn.Utils in
  let x = np..array [1+1.j -2j 3.45+1.6j] ~mask:[false ~true false] ma in  
  print_ndarray @@ x.imag;  
  [%expect {|
      masked_array(data=[1.0, --, 1.6],      
                   mask=[False,  True, False],      
  |}]

*)



(* real *)
(*
>>> x = np.ma.array([1+1.j, -2j, 3.45+1.6j], mask=[False, True, False])
>>> x.real
masked_array(data=[1.0, --, 3.45],
             mask=[False,  True, False],

*)

(* TEST TODO
let%expect_test "MaskedArray.real" =
  let open Sklearn.Utils in
  let x = np..array [1+1.j -2j 3.45+1.6j] ~mask:[false ~true false] ma in  
  print_ndarray @@ x.real;  
  [%expect {|
      masked_array(data=[1.0, --, 3.45],      
                   mask=[False,  True, False],      
  |}]

*)



(* ids *)
(*
>>> x = np.ma.array([1, 2, 3], mask=[0, 1, 1])
>>> x.ids()
(166670640, 166659832) # may vary

*)

(* TEST TODO
let%expect_test "MaskedArray.ids" =
  let open Sklearn.Utils in
  let x = np..array (vectori [|1; 2; 3|]) ~mask:(vectori [|0; 1; 1|]) ma in  
  print_ndarray @@ x.ids ();  
  [%expect {|
      (166670640, 166659832) # may vary      
  |}]

*)



(* ids *)
(*
>>> x = np.ma.array([1, 2, 3])
>>> x.ids()

*)

(* TEST TODO
let%expect_test "MaskedArray.ids" =
  let open Sklearn.Utils in
  let x = np..array (vectori [|1; 2; 3|]) ma in  
  print_ndarray @@ x.ids ();  
  [%expect {|
  |}]

*)



(* iscontiguous *)
(*
>>> x = np.ma.array([1, 2, 3])
>>> x.iscontiguous()
True

*)

(* TEST TODO
let%expect_test "MaskedArray.iscontiguous" =
  let open Sklearn.Utils in
  let x = np..array (vectori [|1; 2; 3|]) ma in  
  print_ndarray @@ x.iscontiguous ();  
  [%expect {|
      True      
  |}]

*)



(* iscontiguous *)
(*
>>> x.flags
  C_CONTIGUOUS : True
  F_CONTIGUOUS : True
  OWNDATA : False
  WRITEABLE : True
  ALIGNED : True
  WRITEBACKIFCOPY : False

*)

(* TEST TODO
let%expect_test "MaskedArray.iscontiguous" =
  let open Sklearn.Utils in
  print_ndarray @@ x.flags;  
  [%expect {|
        C_CONTIGUOUS : True      
        F_CONTIGUOUS : True      
        OWNDATA : False      
        WRITEABLE : True      
        ALIGNED : True      
        WRITEBACKIFCOPY : False      
  |}]

*)



(* mean *)
(*
>>> a = np.ma.array([1,2,3], mask=[False, False, True])
>>> a
masked_array(data=[1, 2, --],
             mask=[False, False,  True],
       fill_value=999999)
>>> a.mean()

*)

(* TEST TODO
let%expect_test "MaskedArray.mean" =
  let open Sklearn.Utils in
  let a = np..array (vectori [|1;2;3|]) ~mask:[false ~false true] ma in  
  print_ndarray @@ a;  
  [%expect {|
      masked_array(data=[1, 2, --],      
                   mask=[False, False,  True],      
             fill_value=999999)      
  |}]
  print_ndarray @@ a.mean ();  
  [%expect {|
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
let%expect_test "MaskedArray.mini" =
  let open Sklearn.Utils in
  let x = np..array np.arange ~6 () ~mask:(vectori [|0 ;1; 0; 0; 0 ;1|])).reshape(3 ~2 ma in  
  print_ndarray @@ x;  
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
  print_ndarray @@ x.mini ();  
  [%expect {|
      masked_array(data=0,      
                   mask=False,      
             fill_value=999999)      
  |}]
  print_ndarray @@ x.mini ~axis:0 ();  
  [%expect {|
      masked_array(data=[0, 3],      
                   mask=[False, False],      
             fill_value=999999)      
  |}]
  print_ndarray @@ x.mini ~axis:1 ();  
  [%expect {|
      masked_array(data=[0, 2, 4],      
                   mask=[False, False, False],      
             fill_value=999999)      
  |}]

*)



(* mini *)
(*
>>> x[:,1].mini(axis=0)
masked_array(data=3,
             mask=False,
       fill_value=999999)
>>> x[:,1].min(axis=0)

*)

(* TEST TODO
let%expect_test "MaskedArray.mini" =
  let open Sklearn.Utils in
  print_ndarray @@ x(vectori [|:;1|]).mini ~axis:0 ();  
  [%expect {|
      masked_array(data=3,      
                   mask=False,      
             fill_value=999999)      
  |}]
  print_ndarray @@ x(vectori [|:;1|]).min ~axis:0 ();  
  [%expect {|
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
let%expect_test "MaskedArray.nonzero" =
  let open Sklearn.Utils in
  let x = .array np.eye ~3 () ma in  
  print_ndarray @@ x;  
  [%expect {|
      masked_array(      
        data=[[1., 0., 0.],      
              [0., 1., 0.],      
              [0., 0., 1.]],      
        mask=False,      
        fill_value=1e+20)      
  |}]
  print_ndarray @@ x.nonzero ();  
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
let%expect_test "MaskedArray.nonzero" =
  let open Sklearn.Utils in
  print_ndarray @@ x(vectori [|1; 1|]) = .masked ma;  
  print_ndarray @@ x;  
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
  print_ndarray @@ x.nonzero ();  
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
let%expect_test "MaskedArray.nonzero" =
  let open Sklearn.Utils in
  print_ndarray @@ .transpose x.nonzero () np;  
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
let%expect_test "MaskedArray.nonzero" =
  let open Sklearn.Utils in
  let a = .array [(vectori [|1;2;3|]) (vectori [|4;5;6|]) (vectori [|7;8;9|])] ma in  
  print_ndarray @@ a > 3;  
  [%expect {|
      masked_array(      
        data=[[False, False, False],      
              [ True,  True,  True],      
              [ True,  True,  True]],      
        mask=False,      
        fill_value=True)      
  |}]
  print_ndarray @@ .nonzero a > 3 ma;  
  [%expect {|
      (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))      
  |}]

*)



(* nonzero *)
(*
>>> (a > 3).nonzero()

*)

(* TEST TODO
let%expect_test "MaskedArray.nonzero" =
  let open Sklearn.Utils in
  print_ndarray @@ (a > 3).nonzero ();  
  [%expect {|
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
let%expect_test "MaskedArray.partition" =
  let open Sklearn.Utils in
  let a = .array (vectori [|3; 4; 2; 1|]) np in  
  print_ndarray @@ a.partition ~3 ();  
  print_ndarray @@ a;  
  [%expect {|
      array([2, 1, 3, 4])      
  |}]

*)



(* partition *)
(*
>>> a.partition((1, 3))
>>> a

*)

(* TEST TODO
let%expect_test "MaskedArray.partition" =
  let open Sklearn.Utils in
  print_ndarray @@ a.partition((1, 3));  
  print_ndarray @@ a;  
  [%expect {|
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
let%expect_test "MaskedArray.put" =
  let open Sklearn.Utils in
  let x = np..array [(vectori [|1;2;3|]) (vectori [|4;5;6|]) (vectori [|7;8;9|])] ~mask:(vectori [|0|]) + (vectori [|1;0|])*4 ma in  
  print_ndarray @@ x;  
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
  print_ndarray @@ x.put((vectori [|0;4;8|]),[10,20,30]);  
  print_ndarray @@ x;  
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



(* put *)
(*
>>> x.put(4,999)
>>> x
masked_array(
  data=[[10, --, 3],
        [--, 999, --],
        [7, --, 30]],
  mask=[[False,  True, False],
        [ True, False,  True],
        [False,  True, False]],

*)

(* TEST TODO
let%expect_test "MaskedArray.put" =
  let open Sklearn.Utils in
  print_ndarray @@ x.put ~4 999 ();  
  print_ndarray @@ x;  
  [%expect {|
      masked_array(      
        data=[[10, --, 3],      
              [--, 999, --],      
              [7, --, 30]],      
        mask=[[False,  True, False],      
              [ True, False,  True],      
              [False,  True, False]],      
  |}]

*)



(* ravel *)
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
>>> x.ravel()
masked_array(data=[1, --, 3, --, 5, --, 7, --, 9],
             mask=[False,  True, False,  True, False,  True, False,  True,
                   False],

*)

(* TEST TODO
let%expect_test "MaskedArray.ravel" =
  let open Sklearn.Utils in
  let x = np..array [(vectori [|1;2;3|]) (vectori [|4;5;6|]) (vectori [|7;8;9|])] ~mask:(vectori [|0|]) + (vectori [|1;0|])*4 ma in  
  print_ndarray @@ x;  
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
  print_ndarray @@ x.ravel ();  
  [%expect {|
      masked_array(data=[1, --, 3, --, 5, --, 7, --, 9],      
                   mask=[False,  True, False,  True, False,  True, False,  True,      
                         False],      
  |}]

*)



(* reshape *)
(*
>>> x = np.ma.array([[1,2],[3,4]], mask=[1,0,0,1])
>>> x
masked_array(
  data=[[--, 2],
        [3, --]],
  mask=[[ True, False],
        [False,  True]],
  fill_value=999999)
>>> x = x.reshape((4,1))
>>> x
masked_array(
  data=[[--],
        [2],
        [3],
        [--]],
  mask=[[ True],
        [False],
        [False],
        [ True]],

*)

(* TEST TODO
let%expect_test "MaskedArray.reshape" =
  let open Sklearn.Utils in
  let x = np..array [(vectori [|1;2|]) (vectori [|3;4|])] ~mask:(vectori [|1;0;0;1|]) ma in  
  print_ndarray @@ x;  
  [%expect {|
      masked_array(      
        data=[[--, 2],      
              [3, --]],      
        mask=[[ True, False],      
              [False,  True]],      
        fill_value=999999)      
  |}]
  let x = x.reshape((4,1)) in  
  print_ndarray @@ x;  
  [%expect {|
      masked_array(      
        data=[[--],      
              [2],      
              [3],      
              [--]],      
        mask=[[ True],      
              [False],      
              [False],      
              [ True]],      
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
let%expect_test "MaskedArray.fill_value" =
  let open Sklearn.Utils in
  print_ndarray @@ for dt in [np.int32, np.int64, np.float64, np.complex128]:np..array (vectori [|0; 1|]) ~dtype:dt).get_fill_value( ma;  
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
let%expect_test "MaskedArray.fill_value" =
  let open Sklearn.Utils in
  let x = np..array [0 1.] ~fill_value:-np.inf ma in  
  print_ndarray @@ x.fill_value;  
  [%expect {|
      -inf      
  |}]
  print_ndarray @@ x.fill_value = .pi np;  
  print_ndarray @@ x.fill_value;  
  [%expect {|
      3.1415926535897931 # may vary      
  |}]

*)



(* fill_value *)
(*
>>> x.fill_value = None
>>> x.fill_value

*)

(* TEST TODO
let%expect_test "MaskedArray.fill_value" =
  let open Sklearn.Utils in
  print_ndarray @@ x.fill_value = None;  
  print_ndarray @@ x.fill_value;  
  [%expect {|
  |}]

*)



(* shrink_mask *)
(*
>>> x = np.ma.array([[1,2 ], [3, 4]], mask=[0]*4)
>>> x.mask
array([[False, False],
       [False, False]])
>>> x.shrink_mask()
masked_array(
  data=[[1, 2],
        [3, 4]],
  mask=False,
  fill_value=999999)
>>> x.mask

*)

(* TEST TODO
let%expect_test "MaskedArray.shrink_mask" =
  let open Sklearn.Utils in
  let x = np..array (matrixi [|[|1;2 |]; [|3; 4|]|]) ~mask:(vectori [|0|])*4 ma in  
  print_ndarray @@ x.mask;  
  [%expect {|
      array([[False, False],      
             [False, False]])      
  |}]
  print_ndarray @@ x.shrink_mask ();  
  [%expect {|
      masked_array(      
        data=[[1, 2],      
              [3, 4]],      
        mask=False,      
        fill_value=999999)      
  |}]
  print_ndarray @@ x.mask;  
  [%expect {|
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
let%expect_test "MaskedArray.sort" =
  let open Sklearn.Utils in
  let a = np..array (vectori [|1; 2; 5; 4; 3|]) ~mask:(vectori [|0; 1; 0; 1; 0|]) ma in  
  print_ndarray @@ # Default;  
  print_ndarray @@ a.sort ();  
  print_ndarray @@ a;  
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
let%expect_test "MaskedArray.sort" =
  let open Sklearn.Utils in
  let a = np..array (vectori [|1; 2; 5; 4; 3|]) ~mask:(vectori [|0; 1; 0; 1; 0|]) ma in  
  print_ndarray @@ # Put missing values in the front;  
  print_ndarray @@ a.sort ~endwith:false ();  
  print_ndarray @@ a;  
  [%expect {|
      masked_array(data=[--, --, 1, 3, 5],      
                   mask=[ True,  True, False, False, False],      
             fill_value=999999)      
  |}]

*)



(* sort *)
(*
>>> a = np.ma.array([1, 2, 5, 4, 3],mask=[0, 1, 0, 1, 0])
>>> # fill_value takes over endwith
>>> a.sort(endwith=False, fill_value=3)
>>> a
masked_array(data=[1, --, --, 3, 5],
             mask=[False,  True,  True, False, False],

*)

(* TEST TODO
let%expect_test "MaskedArray.sort" =
  let open Sklearn.Utils in
  let a = np..array (vectori [|1; 2; 5; 4; 3|]) ~mask:(vectori [|0; 1; 0; 1; 0|]) ma in  
  print_ndarray @@ # fill_value takes over endwith;  
  print_ndarray @@ a.sort ~endwith:false ~fill_value:3 ();  
  print_ndarray @@ a;  
  [%expect {|
      masked_array(data=[1, --, --, 3, 5],      
                   mask=[False,  True,  True, False, False],      
  |}]

*)



(* sum *)
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
>>> x.sum()
25
>>> x.sum(axis=1)
masked_array(data=[4, 5, 16],
             mask=[False, False, False],
       fill_value=999999)
>>> x.sum(axis=0)
masked_array(data=[8, 5, 12],
             mask=[False, False, False],
       fill_value=999999)
>>> print(type(x.sum(axis=0, dtype=np.int64)[0]))

*)

(* TEST TODO
let%expect_test "MaskedArray.sum" =
  let open Sklearn.Utils in
  let x = np..array [(vectori [|1;2;3|]) (vectori [|4;5;6|]) (vectori [|7;8;9|])] ~mask:(vectori [|0|]) + (vectori [|1;0|])*4 ma in  
  print_ndarray @@ x;  
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
  print_ndarray @@ x.sum ();  
  [%expect {|
      25      
  |}]
  print_ndarray @@ x.sum ~axis:1 ();  
  [%expect {|
      masked_array(data=[4, 5, 16],      
                   mask=[False, False, False],      
             fill_value=999999)      
  |}]
  print_ndarray @@ x.sum ~axis:0 ();  
  [%expect {|
      masked_array(data=[8, 5, 12],      
                   mask=[False, False, False],      
             fill_value=999999)      
  |}]
  print_ndarray @@ print(type(x.sum ~axis:0 ~dtype:np.int64 ()(vectori [|0|])));  
  [%expect {|
  |}]

*)



(* tobytes *)
(*
>>> x = np.ma.array(np.array([[1, 2], [3, 4]]), mask=[[0, 1], [1, 0]])
>>> x.tobytes()

*)

(* TEST TODO
let%expect_test "MaskedArray.tobytes" =
  let open Sklearn.Utils in
  let x = np..array np.array((matrixi [|[|1; 2|]; [|3; 4|]|])) ~mask:(matrixi [|[|0; 1|]; [|1; 0|]|]) ma in  
  print_ndarray @@ x.tobytes ();  
  [%expect {|
  |}]

*)



(* toflex *)
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
>>> x.toflex()
array([[(1, False), (2,  True), (3, False)],
       [(4,  True), (5, False), (6,  True)],
       [(7, False), (8,  True), (9, False)]],

*)

(* TEST TODO
let%expect_test "MaskedArray.toflex" =
  let open Sklearn.Utils in
  let x = np..array [(vectori [|1;2;3|]) (vectori [|4;5;6|]) (vectori [|7;8;9|])] ~mask:(vectori [|0|]) + (vectori [|1;0|])*4 ma in  
  print_ndarray @@ x;  
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
  print_ndarray @@ x.toflex ();  
  [%expect {|
      array([[(1, False), (2,  True), (3, False)],      
             [(4,  True), (5, False), (6,  True)],      
             [(7, False), (8,  True), (9, False)]],      
  |}]

*)



(* tolist *)
(*
>>> x = np.ma.array([[1,2,3], [4,5,6], [7,8,9]], mask=[0] + [1,0]*4)
>>> x.tolist()
[[1, None, 3], [None, 5, None], [7, None, 9]]
>>> x.tolist(-999)

*)

(* TEST TODO
let%expect_test "MaskedArray.tolist" =
  let open Sklearn.Utils in
  let x = np..array (matrixi [|[|1;2;3|]; [|4;5;6|]; [|7;8;9|]|]) ~mask:(vectori [|0|]) + (vectori [|1;0|])*4 ma in  
  print_ndarray @@ x.tolist ();  
  [%expect {|
      [[1, None, 3], [None, 5, None], [7, None, 9]]      
  |}]
  print_ndarray @@ x.tolist -999 ();  
  [%expect {|
  |}]

*)



(* toflex *)
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
>>> x.toflex()
array([[(1, False), (2,  True), (3, False)],
       [(4,  True), (5, False), (6,  True)],
       [(7, False), (8,  True), (9, False)]],

*)

(* TEST TODO
let%expect_test "MaskedArray.toflex" =
  let open Sklearn.Utils in
  let x = np..array [(vectori [|1;2;3|]) (vectori [|4;5;6|]) (vectori [|7;8;9|])] ~mask:(vectori [|0|]) + (vectori [|1;0|])*4 ma in  
  print_ndarray @@ x;  
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
  print_ndarray @@ x.toflex ();  
  [%expect {|
      array([[(1, False), (2,  True), (3, False)],      
             [(4,  True), (5, False), (6,  True)],      
             [(7, False), (8,  True), (9, False)]],      
  |}]

*)



(* transpose *)
(*
>>> a = np.array([[1, 2], [3, 4]])
>>> a
array([[1, 2],
       [3, 4]])
>>> a.transpose()
array([[1, 3],
       [2, 4]])
>>> a.transpose((1, 0))
array([[1, 3],
       [2, 4]])
>>> a.transpose(1, 0)
array([[1, 3],

*)

(* TEST TODO
let%expect_test "_arraymethod.<locals>.wrapped_method" =
  let open Sklearn.Utils in
  let a = .array (matrixi [|[|1; 2|]; [|3; 4|]|]) np in  
  print_ndarray @@ a;  
  [%expect {|
      array([[1, 2],      
             [3, 4]])      
  |}]
  print_ndarray @@ a.transpose ();  
  [%expect {|
      array([[1, 3],      
             [2, 4]])      
  |}]
  print_ndarray @@ a.transpose((1, 0));  
  [%expect {|
      array([[1, 3],      
             [2, 4]])      
  |}]
  print_ndarray @@ a.transpose ~1 0 ();  
  [%expect {|
      array([[1, 3],      
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
let%expect_test "MaskedArray.var" =
  let open Sklearn.Utils in
  let a = .array (matrixi [|[|1; 2|]; [|3; 4|]|]) np in  
  print_ndarray @@ .var ~a np;  
  [%expect {|
      1.25      
  |}]
  print_ndarray @@ .var a ~axis:0 np;  
  [%expect {|
      array([1.,  1.])      
  |}]
  print_ndarray @@ .var a ~axis:1 np;  
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
let%expect_test "MaskedArray.var" =
  let open Sklearn.Utils in
  let a = .zeros (2 512*512) ~dtype:np.float32 np in  
  print_ndarray @@ a(vectori [|0; :|]) = 1.0;  
  print_ndarray @@ a(vectori [|1; :|]) = 0.1;  
  print_ndarray @@ .var ~a np;  
  [%expect {|
      0.20250003      
  |}]

*)



(* var *)
(*
>>> np.var(a, dtype=np.float64)
0.20249999932944759 # may vary
>>> ((1-0.55)**2 + (0.1-0.55)**2)/2

*)

(* TEST TODO
let%expect_test "MaskedArray.var" =
  let open Sklearn.Utils in
  print_ndarray @@ .var a ~dtype:np.float64 np;  
  [%expect {|
      0.20249999932944759 # may vary      
  |}]
  print_ndarray @@ ((1-0.55)**2 + (0.1-0.55)**2)/2;  
  [%expect {|
  |}]

*)



(* comb *)
(*
>>> from scipy.special import comb
>>> k = np.array([3, 4])
>>> n = np.array([10, 10])
>>> comb(n, k, exact=False)
array([ 120.,  210.])
>>> comb(10, 3, exact=True)
120L
>>> comb(10, 3, exact=True, repetition=True)

*)

(* TEST TODO
let%expect_test "comb" =
  let open Sklearn.Utils in
  let k = .array (vectori [|3; 4|]) np in  
  let n = .array [10 10] np in  
  print_ndarray @@ comb ~n k ~exact:false ();  
  [%expect {|
      array([ 120.,  210.])      
  |}]
  print_ndarray @@ comb ~10 3 ~exact:true ();  
  [%expect {|
      120L      
  |}]
  print_ndarray @@ comb ~10 3 ~exact:true ~repetition:true ();  
  [%expect {|
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
let%expect_test "lobpcg" =
  let open Sklearn.Utils in
  let n = 100 in  
  let vals = .arange ~1 n + 1 np in  
  let A = spdiags ~vals 0 ~n n () in  
  print_ndarray @@ A.toarray ();  
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
let%expect_test "lobpcg" =
  let open Sklearn.Utils in
  let Y = .eye ~n 3 np in  
  [%expect {|
  |}]

*)



(* lobpcg *)
(*
>>> X = np.random.rand(n, 3)

*)

(* TEST TODO
let%expect_test "lobpcg" =
  let open Sklearn.Utils in
  let x = np..rand ~n 3 random in  
  [%expect {|
  |}]

*)



(* lobpcg *)
(*
>>> invA = spdiags([1./vals], 0, n, n)

*)

(* TEST TODO
let%expect_test "lobpcg" =
  let open Sklearn.Utils in
  let invA = spdiags [1./vals] ~0 n ~n () in  
  [%expect {|
  |}]

*)



(* lobpcg *)
(*
>>> def precond( x ):
...     return invA @ x

*)

(* TEST TODO
let%expect_test "lobpcg" =
  let open Sklearn.Utils in
  print_ndarray @@ def precond x ():return invA @ x;  
  [%expect {|
  |}]

*)



(* lobpcg *)
(*
>>> M = LinearOperator(matvec=precond, matmat=precond,
...                    shape=(n, n), dtype=float)

*)

(* TEST TODO
let%expect_test "lobpcg" =
  let open Sklearn.Utils in
  let M = LinearOperator(matvec=precond, matmat=precond,shape=(n, n), dtype=float) in  
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
let%expect_test "lobpcg" =
  let open Sklearn.Utils in
  let eigenvalues, _ = lobpcg ~A x ~Y:Y ~M:M ~largest:false () in  
  print_ndarray @@ eigenvalues;  
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
let%expect_test "logsumexp" =
  let open Sklearn.Utils in
  let a = .arange ~10 np in  
  print_ndarray @@ .log np.sum(np.exp ~a ()) np;  
  [%expect {|
      9.4586297444267107      
  |}]
  print_ndarray @@ logsumexp ~a ();  
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
let%expect_test "logsumexp" =
  let open Sklearn.Utils in
  let a = .arange ~10 np in  
  let b = .arange ~10 0 -1 np in  
  print_ndarray @@ logsumexp a ~b:b ();  
  [%expect {|
      9.9170178533034665      
  |}]
  print_ndarray @@ .log np.sum(b*np.exp ~a ()) np;  
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
let%expect_test "logsumexp" =
  let open Sklearn.Utils in
  print_ndarray @@ logsumexp((vectori [|1;2|]),b=[1,-1],return_sign=true);  
  [%expect {|
      (1.5413248546129181, -1.0)      
  |}]

*)



(* logsumexp *)
(*
>>> a = np.ma.array([np.log(2), 2, np.log(3)],
...                  mask=[False, True, False])
>>> b = (~a.mask).astype(int)
>>> logsumexp(a.data, b=b), np.log(5)

*)

(* TEST TODO
let%expect_test "logsumexp" =
  let open Sklearn.Utils in
  let a = np..array [np.log ~2 () ~2 np.log ~3 ()] ~mask:[false ~true false] ma in  
  let b = (~a.mask).astype ~int () in  
  print_ndarray @@ logsumexp a.data ~b:b (), .log ~5 np;  
  [%expect {|
  |}]

*)



(* <no name> *)
(*
>>> from scipy.stats import Distribution
>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots(1, 1)

*)

(* TEST TODO
let%expect_test "<no name>" =
  let open Sklearn.Utils in
  let fig, ax = .subplots ~1 1 plt in  
  [%expect {|
  |}]

*)



(* <no name> *)
(*
>>> a, b = 
>>> mean, var, skew, kurt = Distribution.stats(a, b, moments='mvsk')

*)

(* TEST TODO
let%expect_test "<no name>" =
  let open Sklearn.Utils in
  let a, b = in  
  let mean, var, skew, kurt = Distribution.stats ~a b ~moments:'mvsk' () in  
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
let%expect_test "<no name>" =
  let open Sklearn.Utils in
  let x = .linspace Distribution.ppf 0.01 ~a b () Distribution.ppf 0.99 ~a b () ~100 np in  
  print_ndarray @@ .plot ~x Distribution.pdf x ~a b () 'r-' ~lw:5 ~alpha:0.6 ~label:'Distribution pdf' ax;  
  [%expect {|
  |}]

*)



(* <no name> *)
(*
>>> rv = Distribution(a, b)
>>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

*)

(* TEST TODO
let%expect_test "<no name>" =
  let open Sklearn.Utils in
  let rv = Distribution.create ~a b () in  
  print_ndarray @@ .plot ~x rv.pdf ~x () 'k-' ~lw:2 ~label:'frozen pdf' ax;  
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
let%expect_test "<no name>" =
  let open Sklearn.Utils in
  let vals = Distribution.ppf [0.001 0.5 0.999] ~a b () in  
  print_ndarray @@ .allclose [0.001 0.5 0.999] Distribution.cdf vals ~a b () np;  
  [%expect {|
      True      
  |}]

*)



(* <no name> *)
(*
>>> r = Distribution.rvs(a, b, size=1000)

*)

(* TEST TODO
let%expect_test "<no name>" =
  let open Sklearn.Utils in
  let r = Distribution.rvs ~a b ~size:1000 () in  
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
let%expect_test "<no name>" =
  let open Sklearn.Utils in
  print_ndarray @@ .hist r ~density:true ~histtype:'stepfilled' ~alpha:0.2 ax;  
  print_ndarray @@ .legend ~loc:'best' ~frameon:false ax;  
  print_ndarray @@ .show plt;  
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
let%expect_test "<no name>" =
  let open Sklearn.Utils in
  let fig, ax = .subplots ~1 1 plt in  
  print_ndarray @@ .hist np.log10(r) ax;  
  print_ndarray @@ .set_ylabel "Frequency" ax;  
  print_ndarray @@ .set_xlabel "Value of random variable" ax;  
  print_ndarray @@ ax..set_major_locator plt.FixedLocator([-2 -1 0]) xaxis;  
  let ticks = ["$10^{{ {} }}$".format ~i () for i in [-2, -1, 0]] in  
  print_ndarray @@ .set_xticklabels ~ticks ax # doctest: +SKIP;  
  print_ndarray @@ .show plt;  
  [%expect {|
  |}]

*)



(* <no name> *)
(*
>>> rvs = Distribution(2**-2, 2**0).rvs(size=1000)

*)

(* TEST TODO
let%expect_test "<no name>" =
  let open Sklearn.Utils in
  let rvs = Distribution(2**-2, 2**0).rvs ~size:1000 () in  
  [%expect {|
  |}]

*)



(* <no name> *)
(*
>>> fig, ax = plt.subplots(1, 1)
>>> ax.hist(np.log2(rvs))
>>> ax.set_ylabel("Frequency")
>>> ax.set_xlabel("Value of random variable")
>>> ax.xaxis.set_major_locator(plt.FixedLocator([-2, -1, 0]))
>>> ticks = ["$2^{{ {} }}$".format(i) for i in [-2, -1, 0]]
>>> ax.set_xticklabels(ticks)  # doctest: +SKIP

*)

(* TEST TODO
let%expect_test "<no name>" =
  let open Sklearn.Utils in
  let fig, ax = .subplots ~1 1 plt in  
  print_ndarray @@ .hist np.log2(rvs) ax;  
  print_ndarray @@ .set_ylabel "Frequency" ax;  
  print_ndarray @@ .set_xlabel "Value of random variable" ax;  
  print_ndarray @@ ax..set_major_locator plt.FixedLocator([-2 -1 0]) xaxis;  
  let ticks = ["$2^{{ {} }}$".format ~i () for i in [-2, -1, 0]] in  
  print_ndarray @@ .set_xticklabels ~ticks ax # doctest: +SKIP;  
  [%expect {|
  |}]

*)



(* pinvh *)
(*
>>> from scipy.linalg import pinvh
>>> a = np.random.randn(9, 6)
>>> a = np.dot(a, a.T)
>>> B = pinvh(a)
>>> np.allclose(a, np.dot(a, np.dot(B, a)))
True
>>> np.allclose(B, np.dot(B, np.dot(a, B)))

*)

(* TEST TODO
let%expect_test "pinvh" =
  let open Sklearn.Utils in
  let a = np..randn ~9 6 random in  
  let a = .dot ~a a.T np in  
  let B = pinvh ~a () in  
  print_ndarray @@ .allclose ~a np.dot(a np.dot B a ()) np;  
  [%expect {|
      True      
  |}]
  print_ndarray @@ .allclose ~B np.dot(B np.dot a B ()) np;  
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
let%expect_test "lsqr" =
  let open Sklearn.Utils in
  let A = csc_matrix((matrix [|[|1.; 0.|]; [|1.; 1.|]; [|0.; 1.|]|]), dtype=float) in  
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
let%expect_test "lsqr" =
  let open Sklearn.Utils in
  let b = .array [0. 0. 0.] ~dtype:float np in  
  let x, istop, itn, normr = lsqr ~A b ()[:4] in  
  [%expect {|
      The exact solution is  x = 0      
  |}]
  print_ndarray @@ istop;  
  [%expect {|
      0      
  |}]
  print_ndarray @@ x;  
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
let%expect_test "lsqr" =
  let open Sklearn.Utils in
  let b = .array [1. 0. -1.] ~dtype:float np in  
  let x, istop, itn, r1norm = lsqr ~A b ()[:4] in  
  print_ndarray @@ istop;  
  [%expect {|
      1      
  |}]
  print_ndarray @@ x;  
  [%expect {|
      array([ 1., -1.])      
  |}]
  print_ndarray @@ itn;  
  [%expect {|
      1      
  |}]
  print_ndarray @@ r1norm;  
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
let%expect_test "lsqr" =
  let open Sklearn.Utils in
  let b = .array [1. 0.01 -1.] ~dtype:float np in  
  let x, istop, itn, r1norm = lsqr ~A b ()[:4] in  
  print_ndarray @@ istop;  
  [%expect {|
      2      
  |}]
  print_ndarray @@ x;  
  [%expect {|
      array([ 1.00333333, -0.99666667])      
  |}]
  print_ndarray @@ A.dot ~x ()-b;  
  [%expect {|
      array([ 0.00333333, -0.00333333,  0.00333333])      
  |}]
  print_ndarray @@ r1norm;  
  [%expect {|
      0.005773502691896255      
  |}]

*)



(* gen_batches *)
(*
>>> from sklearn.utils import gen_batches
>>> list(gen_batches(7, 3))
[slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
>>> list(gen_batches(6, 3))
[slice(0, 3, None), slice(3, 6, None)]
>>> list(gen_batches(2, 3))
[slice(0, 2, None)]
>>> list(gen_batches(7, 3, min_batch_size=0))
[slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
>>> list(gen_batches(7, 3, min_batch_size=2))

*)

(* TEST TODO
let%expect_test "gen_batches" =
  let open Sklearn.Utils in
  print_ndarray @@ list(gen_batches ~7 3 ());  
  [%expect {|
      [slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]      
  |}]
  print_ndarray @@ list(gen_batches ~6 3 ());  
  [%expect {|
      [slice(0, 3, None), slice(3, 6, None)]      
  |}]
  print_ndarray @@ list(gen_batches ~2 3 ());  
  [%expect {|
      [slice(0, 2, None)]      
  |}]
  print_ndarray @@ list(gen_batches ~7 3 ~min_batch_size:0 ());  
  [%expect {|
      [slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]      
  |}]
  print_ndarray @@ list(gen_batches ~7 3 ~min_batch_size:2 ());  
  [%expect {|
  |}]

*)



(* gen_even_slices *)
(*
>>> from sklearn.utils import gen_even_slices
>>> list(gen_even_slices(10, 1))
[slice(0, 10, None)]
>>> list(gen_even_slices(10, 10))
[slice(0, 1, None), slice(1, 2, None), ..., slice(9, 10, None)]
>>> list(gen_even_slices(10, 5))
[slice(0, 2, None), slice(2, 4, None), ..., slice(8, 10, None)]
>>> list(gen_even_slices(10, 3))

*)

(* TEST TODO
let%expect_test "gen_even_slices" =
  let open Sklearn.Utils in
  print_ndarray @@ list(gen_even_slices ~10 1 ());  
  [%expect {|
      [slice(0, 10, None)]      
  |}]
  print_ndarray @@ list(gen_even_slices ~10 10 ());  
  [%expect {|
      [slice(0, 1, None), slice(1, 2, None), ..., slice(9, 10, None)]      
  |}]
  print_ndarray @@ list(gen_even_slices ~10 5 ());  
  [%expect {|
      [slice(0, 2, None), slice(2, 4, None), ..., slice(8, 10, None)]      
  |}]
  print_ndarray @@ list(gen_even_slices ~10 3 ());  
  [%expect {|
  |}]

*)



(*--------- Examples for module Sklearn.Utils.Graph ----------*)
(* single_source_shortest_path_length *)
(*
>>> from sklearn.utils.graph import single_source_shortest_path_length
>>> import numpy as np
>>> graph = np.array([[ 0, 1, 0, 0],
...                   [ 1, 0, 1, 0],
...                   [ 0, 1, 0, 1],
...                   [ 0, 0, 1, 0]])
>>> list(sorted(single_source_shortest_path_length(graph, 0).items()))
[(0, 0), (1, 1), (2, 2), (3, 3)]
>>> graph = np.ones((6, 6))
>>> list(sorted(single_source_shortest_path_length(graph, 2).items()))

*)

(* TEST TODO
let%expect_test "single_source_shortest_path_length" =
  let open Sklearn.Utils in
  let graph = .array [[ 0 ~1 0 0] [ 1 ~0 1 0] [ 0 ~1 0 1] [ 0 ~0 1 0]] np in  
  print_ndarray @@ list(sorted(single_source_shortest_path_length ~graph 0 ().items ()));  
  [%expect {|
      [(0, 0), (1, 1), (2, 2), (3, 3)]      
  |}]
  let graph = .ones (6 6) np in  
  print_ndarray @@ list(sorted(single_source_shortest_path_length ~graph 2 ().items ()));  
  [%expect {|
  |}]

*)



(*--------- Examples for module Sklearn.Utils.Graph_shortest_path ----------*)
(* fromhex *)
(*
>>> float.fromhex('0x1.ffffp10')
2047.984375
>>> float.fromhex('-0x1p-1074')

*)

(* TEST TODO
let%expect_test "float64.fromhex" =
  let open Sklearn.Utils in
  print_ndarray @@ .fromhex '0x1.ffffp10' float;  
  [%expect {|
      2047.984375      
  |}]
  print_ndarray @@ .fromhex '-0x1p-1074' float;  
  [%expect {|
  |}]

*)



(* hex *)
(*
>>> (-0.1).hex()
'-0x1.999999999999ap-4'
>>> 3.14159.hex()

*)

(* TEST TODO
let%expect_test "float64.hex" =
  let open Sklearn.Utils in
  print_ndarray @@ (-0.1).hex ();  
  [%expect {|
      '-0x1.999999999999ap-4'      
  |}]
  print_ndarray @@ 3.14159.hex ();  
  [%expect {|
  |}]

*)



(* isspmatrix *)
(*
>>> from scipy.sparse import csr_matrix, isspmatrix
>>> isspmatrix(csr_matrix([[5]]))
True

*)

(* TEST TODO
let%expect_test "isspmatrix" =
  let open Sklearn.Utils in
  print_ndarray @@ isspmatrix(csr_matrix((matrixi [|[|5|]|])));  
  [%expect {|
      True      
  |}]

*)



(* isspmatrix *)
(*
>>> from scipy.sparse import isspmatrix
>>> isspmatrix(5)

*)

(* TEST TODO
let%expect_test "isspmatrix" =
  let open Sklearn.Utils in
  print_ndarray @@ isspmatrix ~5 ();  
  [%expect {|
  |}]

*)



(* isspmatrix_csr *)
(*
>>> from scipy.sparse import csr_matrix, isspmatrix_csr
>>> isspmatrix_csr(csr_matrix([[5]]))
True

*)

(* TEST TODO
let%expect_test "isspmatrix_csr" =
  let open Sklearn.Utils in
  print_ndarray @@ isspmatrix_csr(csr_matrix((matrixi [|[|5|]|])));  
  [%expect {|
      True      
  |}]

*)



(* isspmatrix_csr *)
(*
>>> from scipy.sparse import csc_matrix, csr_matrix, isspmatrix_csc
>>> isspmatrix_csr(csc_matrix([[5]]))

*)

(* TEST TODO
let%expect_test "isspmatrix_csr" =
  let open Sklearn.Utils in
  print_ndarray @@ isspmatrix_csr(csc_matrix((matrixi [|[|5|]|])));  
  [%expect {|
  |}]

*)



(* indices_to_mask *)
(*
>>> from sklearn.utils import indices_to_mask
>>> indices = [1, 2 , 3, 4]
>>> indices_to_mask(indices, 5)

*)

(* TEST TODO
let%expect_test "indices_to_mask" =
  let open Sklearn.Utils in
  let indices = (vectori [|1; 2 ; 3; 4|]) in  
  print_ndarray @@ indices_to_mask ~indices 5 ();  
  [%expect {|
  |}]

*)



(* is_scalar_nan *)
(*
>>> is_scalar_nan(np.nan)
True
>>> is_scalar_nan(float("nan"))
True
>>> is_scalar_nan(None)
False
>>> is_scalar_nan("")
False
>>> is_scalar_nan([np.nan])

*)

(* TEST TODO
let%expect_test "is_scalar_nan" =
  let open Sklearn.Utils in
  print_ndarray @@ is_scalar_nan np.nan ();  
  [%expect {|
      True      
  |}]
  print_ndarray @@ is_scalar_nan(float "nan" ());  
  [%expect {|
      True      
  |}]
  print_ndarray @@ is_scalar_nan ~None ();  
  [%expect {|
      False      
  |}]
  print_ndarray @@ is_scalar_nan "" ();  
  [%expect {|
      False      
  |}]
  print_ndarray @@ is_scalar_nan [np.nan] ();  
  [%expect {|
  |}]

*)



(* isspmatrix *)
(*
>>> from scipy.sparse import csr_matrix, isspmatrix
>>> isspmatrix(csr_matrix([[5]]))
True

*)

(* TEST TODO
let%expect_test "isspmatrix" =
  let open Sklearn.Utils in
  print_ndarray @@ isspmatrix(csr_matrix((matrixi [|[|5|]|])));  
  [%expect {|
      True      
  |}]

*)



(* isspmatrix *)
(*
>>> from scipy.sparse import isspmatrix
>>> isspmatrix(5)

*)

(* TEST TODO
let%expect_test "isspmatrix" =
  let open Sklearn.Utils in
  print_ndarray @@ isspmatrix ~5 ();  
  [%expect {|
  |}]

*)



(*--------- Examples for module Sklearn.Utils.Metaestimators ----------*)
(*--------- Examples for module Sklearn.Utils.Multiclass ----------*)
(* dok_matrix *)
(*
>>> import numpy as np
>>> from scipy.sparse import dok_matrix
>>> S = dok_matrix((5, 5), dtype=np.float32)
>>> for i in range(5):
...     for j in range(5):

*)

(* TEST TODO
let%expect_test "dok_matrix" =
  let open Sklearn.Utils in
  let S = dok_matrix((5, 5), dtype=np.float32) in  
  print_ndarray @@ for i in range ~5 ():for j in range ~5 ():;  
  [%expect {|
  |}]

*)



(* diagonal *)
(*
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
>>> A.diagonal()
array([1, 0, 5])
>>> A.diagonal(k=1)

*)

(* TEST TODO
let%expect_test "spmatrix.diagonal" =
  let open Sklearn.Utils in
  let A = csr_matrix((matrixi [|[|1; 2; 0|]; [|0; 0; 3|]; [|4; 0; 5|]|])) in  
  print_ndarray @@ A.diagonal ();  
  [%expect {|
      array([1, 0, 5])      
  |}]
  print_ndarray @@ A.diagonal ~k:1 ();  
  [%expect {|
  |}]

*)



(* dot *)
(*
>>> import numpy as np
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
>>> v = np.array([1, 0, -1])
>>> A.dot(v)

*)

(* TEST TODO
let%expect_test "spmatrix.dot" =
  let open Sklearn.Utils in
  let A = csr_matrix((matrixi [|[|1; 2; 0|]; [|0; 0; 3|]; [|4; 0; 5|]|])) in  
  let v = .array [1 ~0 -1] np in  
  print_ndarray @@ A.dot ~v ();  
  [%expect {|
  |}]

*)



(* nonzero *)
(*
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1,2,0],[0,0,3],[4,0,5]])
>>> A.nonzero()

*)

(* TEST TODO
let%expect_test "spmatrix.nonzero" =
  let open Sklearn.Utils in
  let A = csr_matrix([(vectori [|1;2;0|]),(vectori [|0;0;3|]),(vectori [|4;0;5|])]) in  
  print_ndarray @@ A.nonzero ();  
  [%expect {|
  |}]

*)



(* is_multilabel *)
(*
>>> import numpy as np
>>> from sklearn.utils.multiclass import is_multilabel
>>> is_multilabel([0, 1, 0, 1])
False
>>> is_multilabel([[1], [0, 2], []])
False
>>> is_multilabel(np.array([[1, 0], [0, 0]]))
True
>>> is_multilabel(np.array([[1], [0], [0]]))
False
>>> is_multilabel(np.array([[1, 0, 0]]))

*)

(* TEST TODO
let%expect_test "is_multilabel" =
  let open Sklearn.Utils in
  print_ndarray @@ is_multilabel((vectori [|0; 1; 0; 1|]));  
  [%expect {|
      False      
  |}]
  print_ndarray @@ is_multilabel((matrixi [|[|1|]; [|0; 2|]; [||]|]));  
  [%expect {|
      False      
  |}]
  print_ndarray @@ is_multilabel(.array (matrixi [|[|1; 0|]; [|0; 0|]|])) np;  
  [%expect {|
      True      
  |}]
  print_ndarray @@ is_multilabel(.array (matrixi [|[|1|]; [|0|]; [|0|]|])) np;  
  [%expect {|
      False      
  |}]
  print_ndarray @@ is_multilabel(.array (matrixi [|[|1; 0; 0|]|])) np;  
  [%expect {|
  |}]

*)



(* isspmatrix *)
(*
>>> from scipy.sparse import csr_matrix, isspmatrix
>>> isspmatrix(csr_matrix([[5]]))
True

*)

(* TEST TODO
let%expect_test "isspmatrix" =
  let open Sklearn.Utils in
  print_ndarray @@ isspmatrix(csr_matrix((matrixi [|[|5|]|])));  
  [%expect {|
      True      
  |}]

*)



(* isspmatrix *)
(*
>>> from scipy.sparse import isspmatrix
>>> isspmatrix(5)

*)

(* TEST TODO
let%expect_test "isspmatrix" =
  let open Sklearn.Utils in
  print_ndarray @@ isspmatrix ~5 ();  
  [%expect {|
  |}]

*)



(* diagonal *)
(*
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
>>> A.diagonal()
array([1, 0, 5])
>>> A.diagonal(k=1)

*)

(* TEST TODO
let%expect_test "spmatrix.diagonal" =
  let open Sklearn.Utils in
  let A = csr_matrix((matrixi [|[|1; 2; 0|]; [|0; 0; 3|]; [|4; 0; 5|]|])) in  
  print_ndarray @@ A.diagonal ();  
  [%expect {|
      array([1, 0, 5])      
  |}]
  print_ndarray @@ A.diagonal ~k:1 ();  
  [%expect {|
  |}]

*)



(* dot *)
(*
>>> import numpy as np
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
>>> v = np.array([1, 0, -1])
>>> A.dot(v)

*)

(* TEST TODO
let%expect_test "spmatrix.dot" =
  let open Sklearn.Utils in
  let A = csr_matrix((matrixi [|[|1; 2; 0|]; [|0; 0; 3|]; [|4; 0; 5|]|])) in  
  let v = .array [1 ~0 -1] np in  
  print_ndarray @@ A.dot ~v ();  
  [%expect {|
  |}]

*)



(* nonzero *)
(*
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1,2,0],[0,0,3],[4,0,5]])
>>> A.nonzero()

*)

(* TEST TODO
let%expect_test "spmatrix.nonzero" =
  let open Sklearn.Utils in
  let A = csr_matrix([(vectori [|1;2;0|]),(vectori [|0;0;3|]),(vectori [|4;0;5|])]) in  
  print_ndarray @@ A.nonzero ();  
  [%expect {|
  |}]

*)



(* diagonal *)
(*
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
>>> A.diagonal()
array([1, 0, 5])
>>> A.diagonal(k=1)

*)

(* TEST TODO
let%expect_test "spmatrix.diagonal" =
  let open Sklearn.Utils in
  let A = csr_matrix((matrixi [|[|1; 2; 0|]; [|0; 0; 3|]; [|4; 0; 5|]|])) in  
  print_ndarray @@ A.diagonal ();  
  [%expect {|
      array([1, 0, 5])      
  |}]
  print_ndarray @@ A.diagonal ~k:1 ();  
  [%expect {|
  |}]

*)



(* dot *)
(*
>>> import numpy as np
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
>>> v = np.array([1, 0, -1])
>>> A.dot(v)

*)

(* TEST TODO
let%expect_test "spmatrix.dot" =
  let open Sklearn.Utils in
  let A = csr_matrix((matrixi [|[|1; 2; 0|]; [|0; 0; 3|]; [|4; 0; 5|]|])) in  
  let v = .array [1 ~0 -1] np in  
  print_ndarray @@ A.dot ~v ();  
  [%expect {|
  |}]

*)



(* nonzero *)
(*
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1,2,0],[0,0,3],[4,0,5]])
>>> A.nonzero()

*)

(* TEST TODO
let%expect_test "spmatrix.nonzero" =
  let open Sklearn.Utils in
  let A = csr_matrix([(vectori [|1;2;0|]),(vectori [|0;0;3|]),(vectori [|4;0;5|])]) in  
  print_ndarray @@ A.nonzero ();  
  [%expect {|
  |}]

*)



(* type_of_target *)
(*
>>> import numpy as np
>>> type_of_target([0.1, 0.6])
'continuous'
>>> type_of_target([1, -1, -1, 1])
'binary'
>>> type_of_target(['a', 'b', 'a'])
'binary'
>>> type_of_target([1.0, 2.0])
'binary'
>>> type_of_target([1, 0, 2])
'multiclass'
>>> type_of_target([1.0, 0.0, 3.0])
'multiclass'
>>> type_of_target(['a', 'b', 'c'])
'multiclass'
>>> type_of_target(np.array([[1, 2], [3, 1]]))
'multiclass-multioutput'
>>> type_of_target([[1, 2]])
'multilabel-indicator'
>>> type_of_target(np.array([[1.5, 2.0], [3.0, 1.6]]))
'continuous-multioutput'
>>> type_of_target(np.array([[0, 1], [1, 1]]))

*)

(* TEST TODO
let%expect_test "type_of_target" =
  let open Sklearn.Utils in
  print_ndarray @@ type_of_target [0.1 0.6] ();  
  [%expect {|
      'continuous'      
  |}]
  print_ndarray @@ type_of_target [1 -1 -1 1] ();  
  [%expect {|
      'binary'      
  |}]
  print_ndarray @@ type_of_target ['a' 'b' 'a'] ();  
  [%expect {|
      'binary'      
  |}]
  print_ndarray @@ type_of_target [1.0 2.0] ();  
  [%expect {|
      'binary'      
  |}]
  print_ndarray @@ type_of_target((vectori [|1; 0; 2|]));  
  [%expect {|
      'multiclass'      
  |}]
  print_ndarray @@ type_of_target [1.0 0.0 3.0] ();  
  [%expect {|
      'multiclass'      
  |}]
  print_ndarray @@ type_of_target ['a' 'b' 'c'] ();  
  [%expect {|
      'multiclass'      
  |}]
  print_ndarray @@ type_of_target(.array (matrixi [|[|1; 2|]; [|3; 1|]|])) np;  
  [%expect {|
      'multiclass-multioutput'      
  |}]
  print_ndarray @@ type_of_target((matrixi [|[|1; 2|]|]));  
  [%expect {|
      'multilabel-indicator'      
  |}]
  print_ndarray @@ type_of_target(.array (matrix [|[|1.5; 2.0|]; [|3.0; 1.6|]|])) np;  
  [%expect {|
      'continuous-multioutput'      
  |}]
  print_ndarray @@ type_of_target(.array (matrixi [|[|0; 1|]; [|1; 1|]|])) np;  
  [%expect {|
  |}]

*)



(* unique_labels *)
(*
>>> from sklearn.utils.multiclass import unique_labels
>>> unique_labels([3, 5, 5, 5, 7, 7])
array([3, 5, 7])
>>> unique_labels([1, 2, 3, 4], [2, 2, 3, 4])
array([1, 2, 3, 4])
>>> unique_labels([1, 2, 10], [5, 11])

*)

(* TEST TODO
let%expect_test "unique_labels" =
  let open Sklearn.Utils in
  print_ndarray @@ unique_labels((vectori [|3; 5; 5; 5; 7; 7|]));  
  [%expect {|
      array([3, 5, 7])      
  |}]
  print_ndarray @@ unique_labels((vectori [|1; 2; 3; 4|]), (vectori [|2; 2; 3; 4|]));  
  [%expect {|
      array([1, 2, 3, 4])      
  |}]
  print_ndarray @@ unique_labels [1 ~2 10] [5 11] ();  
  [%expect {|
  |}]

*)



(*--------- Examples for module Sklearn.Utils.Murmurhash ----------*)
(*--------- Examples for module Sklearn.Utils.Optimize ----------*)
(* deprecated *)
(*
>>> from sklearn.utils import deprecated
>>> deprecated()
<sklearn.utils.deprecation.deprecated object at ...>

*)

(* TEST TODO
let%expect_test "deprecated" =
  let open Sklearn.Utils in
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
  let open Sklearn.Utils in
  print_ndarray @@ @deprecated ()def some_function (): pass;  
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
let%expect_test "parallel_backend" =
  let open Sklearn.Utils in
  print_ndarray @@ with parallel_backend 'threading' ():print(Parallel()(delayed ~neg ()(i + 1) for i in range ~5 ()));  
  [%expect {|
      [-1, -2, -3, -4, -5]      
  |}]

*)



(*--------- Examples for module Sklearn.Utils.Random ----------*)
(* deprecated *)
(*
>>> from sklearn.utils import deprecated
>>> deprecated()
<sklearn.utils.deprecation.deprecated object at ...>

*)

(* TEST TODO
let%expect_test "deprecated" =
  let open Sklearn.Utils in
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
  let open Sklearn.Utils in
  print_ndarray @@ @deprecated ()def some_function (): pass;  
  [%expect {|
  |}]

*)



(*--------- Examples for module Sklearn.Utils.Sparsefuncs ----------*)
(*--------- Examples for module Sklearn.Utils.Sparsefuncs_fast ----------*)
(*--------- Examples for module Sklearn.Utils.Stats ----------*)
(*--------- Examples for module Sklearn.Utils.Validation ----------*)
(* has_fit_parameter *)
(*
>>> from sklearn.svm import SVC
>>> has_fit_parameter(SVC(), "sample_weight")

*)

(* TEST TODO
let%expect_test "has_fit_parameter" =
  let open Sklearn.Utils in
  print_ndarray @@ has_fit_parameter(SVC(), "sample_weight");  
  [%expect {|
  |}]

*)



