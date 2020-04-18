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
let%expect_test "BernoulliNB" =
  let open Sklearn.Naive_bayes in
  let rng = np..randomState ~1 random in  
  let x = .randint 5 ~size:(6 100) rng in  
  let Y = .array (vectori [|1; 2; 3; 4; 4; 5|]) np in  
  let clf = BernoulliNB.create () in  
  print BernoulliNB.pp @@ BernoulliNB.fit ~x Y clf;  
  [%expect {|
      BernoulliNB()      
  |}]
  print_ndarray @@ print BernoulliNB.predict x[2:3] () clf;  
  [%expect {|
      [3]      
  |}]

*)



(* CategoricalNB *)
(*
>>> import numpy as np
>>> rng = np.random.RandomState(1)
>>> X = rng.randint(5, size=(6, 100))
>>> y = np.array([1, 2, 3, 4, 5, 6])
>>> from sklearn.naive_bayes import CategoricalNB
>>> clf = CategoricalNB()
>>> clf.fit(X, y)
CategoricalNB()
>>> print(clf.predict(X[2:3]))

*)

(* TEST TODO
let%expect_test "CategoricalNB" =
  let open Sklearn.Naive_bayes in
  let rng = np..randomState ~1 random in  
  let x = .randint 5 ~size:(6 100) rng in  
  let y = .array (vectori [|1; 2; 3; 4; 5; 6|]) np in  
  let clf = CategoricalNB.create () in  
  print CategoricalNB.pp @@ CategoricalNB.fit ~x y clf;  
  [%expect {|
      CategoricalNB()      
  |}]
  print_ndarray @@ print CategoricalNB.predict x[2:3] () clf;  
  [%expect {|
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
let%expect_test "ComplementNB" =
  let open Sklearn.Naive_bayes in
  let rng = np..randomState ~1 random in  
  let x = .randint 5 ~size:(6 100) rng in  
  let y = .array (vectori [|1; 2; 3; 4; 5; 6|]) np in  
  let clf = ComplementNB.create () in  
  print ComplementNB.pp @@ ComplementNB.fit ~x y clf;  
  [%expect {|
      ComplementNB()      
  |}]
  print_ndarray @@ print ComplementNB.predict x[2:3] () clf;  
  [%expect {|
      [3]      
  |}]

*)



(* GaussianNB *)
(*
>>> import numpy as np
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
>>> Y = np.array([1, 1, 1, 2, 2, 2])
>>> from sklearn.naive_bayes import GaussianNB
>>> clf = GaussianNB()
>>> clf.fit(X, Y)
GaussianNB()
>>> print(clf.predict([[-0.8, -1]]))
[1]
>>> clf_pf = GaussianNB()
>>> clf_pf.partial_fit(X, Y, np.unique(Y))
GaussianNB()
>>> print(clf_pf.predict([[-0.8, -1]]))

*)

(* TEST TODO
let%expect_test "GaussianNB" =
  let open Sklearn.Naive_bayes in
  let x = .array (matrixi [|[|-1; -1|]; [|-2; -1|]; [|-3; -2|]; [|1; 1|]; [|2; 1|]; [|3; 2|]|]) np in  
  let Y = .array (vectori [|1; 1; 1; 2; 2; 2|]) np in  
  let clf = GaussianNB.create () in  
  print GaussianNB.pp @@ GaussianNB.fit ~x Y clf;  
  [%expect {|
      GaussianNB()      
  |}]
  print_ndarray @@ print(GaussianNB.predict (matrix [|[|-0.8; -1|]|])) clf;  
  [%expect {|
      [1]      
  |}]
  let clf_pf = GaussianNB.create () in  
  print_ndarray @@ GaussianNB.partial_fit ~x Y np.unique ~Y () clf_pf;  
  [%expect {|
      GaussianNB()      
  |}]
  print_ndarray @@ print(GaussianNB.predict (matrix [|[|-0.8; -1|]|])) clf_pf;  
  [%expect {|
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
let%expect_test "LabelBinarizer" =
  let open Sklearn.Naive_bayes in
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
  let open Sklearn.Naive_bayes in
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
  let open Sklearn.Naive_bayes in
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
let%expect_test "MultinomialNB" =
  let open Sklearn.Naive_bayes in
  let rng = np..randomState ~1 random in  
  let x = .randint 5 ~size:(6 100) rng in  
  let y = .array (vectori [|1; 2; 3; 4; 5; 6|]) np in  
  let clf = MultinomialNB.create () in  
  print MultinomialNB.pp @@ MultinomialNB.fit ~x y clf;  
  [%expect {|
      MultinomialNB()      
  |}]
  print_ndarray @@ print MultinomialNB.predict x[2:3] () clf;  
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
let%expect_test "deprecated" =
  let open Sklearn.Naive_bayes in
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
  let open Sklearn.Naive_bayes in
  print_ndarray @@ @deprecated ()def some_function (): pass;  
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
let%expect_test "label_binarize" =
  let open Sklearn.Naive_bayes in
  print_ndarray @@ label_binarize((vectori [|1; 6|]), classes=(vectori [|1; 2; 4; 6|]));  
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
let%expect_test "label_binarize" =
  let open Sklearn.Naive_bayes in
  print_ndarray @@ label_binarize((vectori [|1; 6|]), classes=(vectori [|1; 6; 4; 2|]));  
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
let%expect_test "label_binarize" =
  let open Sklearn.Naive_bayes in
  print_ndarray @@ label_binarize ['yes' 'no' 'no' 'yes'] ~classes:['no' 'yes'] ();  
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
let%expect_test "logsumexp" =
  let open Sklearn.Naive_bayes in
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
  let open Sklearn.Naive_bayes in
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
  let open Sklearn.Naive_bayes in
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
  let open Sklearn.Naive_bayes in
  let a = np..array [np.log ~2 () ~2 np.log ~3 ()] ~mask:[false ~true false] ma in  
  let b = (~a.mask).astype ~int () in  
  print_ndarray @@ logsumexp a.data ~b:b (), .log ~5 np;  
  [%expect {|
  |}]

*)



