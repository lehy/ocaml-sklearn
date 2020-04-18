(* DictVectorizer *)
(*
>>> from sklearn.feature_extraction import DictVectorizer
>>> v = DictVectorizer(sparse=False)
>>> D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
>>> X = v.fit_transform(D)
>>> X
array([[2., 0., 1.],
       [0., 1., 3.]])
>>> v.inverse_transform(X) ==         [{'bar': 2.0, 'foo': 1.0}, {'baz': 1.0, 'foo': 3.0}]
True
>>> v.transform({'foo': 4, 'unseen_feature': 3})
array([[0., 0., 4.]])

*)

(* TEST TODO
let%expect_test "DictVectorizer" =
  let open Sklearn.Feature_extraction in
  let v = DictVectorizer.create ~sparse:false () in  
  let D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}] in  
  let x = v.fit_transform ~D () in  
  print_ndarray @@ x;  
  [%expect {|
      array([[2., 0., 1.],      
             [0., 1., 3.]])      
  |}]
  print_ndarray @@ v.inverse_transform ~x () == [{'bar': 2.0, 'foo': 1.0}, {'baz': 1.0, 'foo': 3.0}];  
  [%expect {|
      True      
  |}]
  print_ndarray @@ v.transform {'foo': 4 'unseen_feature': 3} ();  
  [%expect {|
      array([[0., 0., 4.]])      
  |}]

*)



(* restrict *)
(*
>>> from sklearn.feature_extraction import DictVectorizer
>>> from sklearn.feature_selection import SelectKBest, chi2
>>> v = DictVectorizer()
>>> D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
>>> X = v.fit_transform(D)
>>> support = SelectKBest(chi2, k=2).fit(X, [0, 1])
>>> v.get_feature_names()
['bar', 'baz', 'foo']
>>> v.restrict(support.get_support())
DictVectorizer()
>>> v.get_feature_names()

*)

(* TEST TODO
let%expect_test "DictVectorizer.restrict" =
  let open Sklearn.Feature_extraction in
  let v = DictVectorizer.create () in  
  let D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}] in  
  let x = v.fit_transform ~D () in  
  let support = SelectKBest(chi2, k=2).fit(x, (vectori [|0; 1|])) in  
  print_ndarray @@ v.get_feature_names ();  
  [%expect {|
      ['bar', 'baz', 'foo']      
  |}]
  print_ndarray @@ v.restrict SelectKBest.get_support () support;  
  [%expect {|
      DictVectorizer()      
  |}]
  print_ndarray @@ v.get_feature_names ();  
  [%expect {|
  |}]

*)



(* FeatureHasher *)
(*
>>> from sklearn.feature_extraction import FeatureHasher
>>> h = FeatureHasher(n_features=10)
>>> D = [{'dog': 1, 'cat':2, 'elephant':4},{'dog': 2, 'run': 5}]
>>> f = h.transform(D)
>>> f.toarray()
array([[ 0.,  0., -4., -1.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 0.,  0.,  0., -2., -5.,  0.,  0.,  0.,  0.,  0.]])

*)

(* TEST TODO
let%expect_test "FeatureHasher" =
  let open Sklearn.Feature_extraction in
  let h = FeatureHasher.create ~n_features:10 () in  
  let D = [{'dog': 1, 'cat':2, 'elephant':4},{'dog': 2, 'run': 5}] in  
  let f = h.transform ~D () in  
  print_ndarray @@ f.toarray ();  
  [%expect {|
      array([[ 0.,  0., -4., -1.,  0.,  0.,  0.,  0.,  0.,  2.],      
             [ 0.,  0.,  0., -2., -5.,  0.,  0.,  0.,  0.,  0.]])      
  |}]

*)



(*--------- Examples for module Sklearn.Feature_extraction.Image ----------*)
(* PatchExtractor *)
(*
>>> from sklearn.datasets import load_sample_images
>>> from sklearn.feature_extraction import image
>>> # Use the array data from the second image in this dataset:
>>> X = load_sample_images().images[1]
>>> print('Image shape: {}'.format(X.shape))
Image shape: (427, 640, 3)
>>> pe = image.PatchExtractor(patch_size=(2, 2))
>>> pe_fit = pe.fit(X)
>>> pe_trans = pe.transform(X)
>>> print('Patches shape: {}'.format(pe_trans.shape))

*)

(* TEST TODO
let%expect_test "PatchExtractor" =
  let open Sklearn.Feature_extraction in
  print_ndarray @@ # Use the array data from the second image in this dataset:;  
  let x = load_sample_images ().images vectori [|1|] () in  
  print_ndarray @@ print('Image shape: {}'.format x.shape ());  
  [%expect {|
      Image shape: (427, 640, 3)      
  |}]
  let pe = .patchExtractor ~patch_size:(2 2) image in  
  let pe_fit = .fit ~x pe in  
  let pe_trans = .transform ~x pe in  
  print_ndarray @@ print('Patches shape: {}'.format pe_trans.shape ());  
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
  let open Sklearn.Feature_extraction in
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
  let open Sklearn.Feature_extraction in
  print_ndarray @@ @deprecated ()def some_function (): pass;  
  [%expect {|
  |}]

*)



(* extract_patches_2d *)
(*
>>> from sklearn.datasets import load_sample_image
>>> from sklearn.feature_extraction import image
>>> # Use the array data from the first image in this dataset:
>>> one_image = load_sample_image("china.jpg")
>>> print('Image shape: {}'.format(one_image.shape))
Image shape: (427, 640, 3)
>>> patches = image.extract_patches_2d(one_image, (2, 2))
>>> print('Patches shape: {}'.format(patches.shape))
Patches shape: (272214, 2, 2, 3)
>>> # Here are just two of these patches:
>>> print(patches[1])
[[[174 201 231]
  [174 201 231]]
 [[173 200 230]
  [173 200 230]]]
>>> print(patches[800])
[[[187 214 243]
  [188 215 244]]
 [[187 214 243]

*)

(* TEST TODO
let%expect_test "extract_patches_2d" =
  let open Sklearn.Feature_extraction in
  print_ndarray @@ # Use the array data from the first image in this dataset:;  
  let one_image = load_sample_image "china.jpg" () in  
  print_ndarray @@ print('Image shape: {}'.format one_image.shape ());  
  [%expect {|
      Image shape: (427, 640, 3)      
  |}]
  let patches = .extract_patches_2d ~one_image (2 2) image in  
  print_ndarray @@ print('Patches shape: {}'.format patches.shape ());  
  [%expect {|
      Patches shape: (272214, 2, 2, 3)      
  |}]
  print_ndarray @@ # Here are just two of these patches:;  
  print_ndarray @@ print(patches vectori [|1|] ());  
  [%expect {|
      [[[174 201 231]      
        [174 201 231]]      
       [[173 200 230]      
        [173 200 230]]]      
  |}]
  print_ndarray @@ print patches[800] ();  
  [%expect {|
      [[[187 214 243]      
        [188 215 244]]      
       [[187 214 243]      
  |}]

*)



(*--------- Examples for module Sklearn.Feature_extraction.Text ----------*)
(* CountVectorizer *)
(*
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> corpus = [
...     'This is the first document.',
...     'This document is the second document.',
...     'And this is the third one.',
...     'Is this the first document?',
... ]
>>> vectorizer = CountVectorizer()
>>> X = vectorizer.fit_transform(corpus)
>>> print(vectorizer.get_feature_names())
['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
>>> print(X.toarray())
[[0 1 1 1 0 0 1 0 1]
 [0 2 0 1 0 1 1 0 1]
 [1 0 0 1 1 0 1 1 1]
 [0 1 1 1 0 0 1 0 1]]
>>> vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
>>> X2 = vectorizer2.fit_transform(corpus)
>>> print(vectorizer2.get_feature_names())
['and this', 'document is', 'first document', 'is the', 'is this',
'second document', 'the first', 'the second', 'the third', 'third one',
 'this document', 'this is', 'this the']
 >>> print(X2.toarray())
 [[0 0 1 1 0 0 1 0 0 0 0 1 0]
 [0 1 0 1 0 1 0 1 0 0 1 0 0]
 [1 0 0 1 0 0 0 0 1 1 0 1 0]
 [0 0 1 0 1 0 1 0 0 0 0 0 1]]

*)

(* TEST TODO
let%expect_test "CountVectorizer" =
  let open Sklearn.Feature_extraction in
  let corpus = ['This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?',] in  
  let vectorizer = CountVectorizer.create () in  
  let x = CountVectorizer.fit_transform ~corpus vectorizer in  
  print_ndarray @@ print CountVectorizer.get_feature_names () vectorizer;  
  [%expect {|
      ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']      
  |}]
  print_ndarray @@ print(x.toarray ());  
  [%expect {|
      [[0 1 1 1 0 0 1 0 1]      
       [0 2 0 1 0 1 1 0 1]      
       [1 0 0 1 1 0 1 1 1]      
       [0 1 1 1 0 0 1 0 1]]      
  |}]
  let vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2)) in  
  let X2 = vectorizer2.fit_transform ~corpus () in  
  print_ndarray @@ print(vectorizer2.get_feature_names ());  
  [%expect {|
      ['and this', 'document is', 'first document', 'is the', 'is this',      
      'second document', 'the first', 'the second', 'the third', 'third one',      
       'this document', 'this is', 'this the']      
       >>> print(X2.toarray())      
       [[0 0 1 1 0 0 1 0 0 0 0 1 0]      
       [0 1 0 1 0 1 0 1 0 0 1 0 0]      
       [1 0 0 1 0 0 0 0 1 1 0 1 0]      
       [0 0 1 0 1 0 1 0 0 0 0 0 1]]      
  |}]

*)



(* FeatureHasher *)
(*
>>> from sklearn.feature_extraction import FeatureHasher
>>> h = FeatureHasher(n_features=10)
>>> D = [{'dog': 1, 'cat':2, 'elephant':4},{'dog': 2, 'run': 5}]
>>> f = h.transform(D)
>>> f.toarray()
array([[ 0.,  0., -4., -1.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 0.,  0.,  0., -2., -5.,  0.,  0.,  0.,  0.,  0.]])

*)

(* TEST TODO
let%expect_test "FeatureHasher" =
  let open Sklearn.Feature_extraction in
  let h = FeatureHasher.create ~n_features:10 () in  
  let D = [{'dog': 1, 'cat':2, 'elephant':4},{'dog': 2, 'run': 5}] in  
  let f = h.transform ~D () in  
  print_ndarray @@ f.toarray ();  
  [%expect {|
      array([[ 0.,  0., -4., -1.,  0.,  0.,  0.,  0.,  0.,  2.],      
             [ 0.,  0.,  0., -2., -5.,  0.,  0.,  0.,  0.,  0.]])      
  |}]

*)



(* HashingVectorizer *)
(*
>>> from sklearn.feature_extraction.text import HashingVectorizer
>>> corpus = [
...     'This is the first document.',
...     'This document is the second document.',
...     'And this is the third one.',
...     'Is this the first document?',
... ]
>>> vectorizer = HashingVectorizer(n_features=2**4)
>>> X = vectorizer.fit_transform(corpus)
>>> print(X.shape)
(4, 16)

*)

(* TEST TODO
let%expect_test "HashingVectorizer" =
  let open Sklearn.Feature_extraction in
  let corpus = ['This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?',] in  
  let vectorizer = HashingVectorizer.create ~n_features:2**4 () in  
  let x = HashingVectorizer.fit_transform ~corpus vectorizer in  
  print_ndarray @@ print x.shape ();  
  [%expect {|
      (4, 16)      
  |}]

*)



(* TfidfTransformer *)
(*
>>> from sklearn.feature_extraction.text import TfidfTransformer
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> from sklearn.pipeline import Pipeline
>>> import numpy as np
>>> corpus = ['this is the first document',
...           'this document is the second document',
...           'and this is the third one',
...           'is this the first document']
>>> vocabulary = ['this', 'document', 'first', 'is', 'second', 'the',
...               'and', 'one']
>>> pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),
...                  ('tfid', TfidfTransformer())]).fit(corpus)
>>> pipe['count'].transform(corpus).toarray()
array([[1, 1, 1, 1, 0, 1, 0, 0],
       [1, 2, 0, 1, 1, 1, 0, 0],
       [1, 0, 0, 1, 0, 1, 1, 1],
       [1, 1, 1, 1, 0, 1, 0, 0]])
>>> pipe['tfid'].idf_
array([1.        , 1.22314355, 1.51082562, 1.        , 1.91629073,
       1.        , 1.91629073, 1.91629073])
>>> pipe.transform(corpus).shape
(4, 8)

*)

(* TEST TODO
let%expect_test "TfidfTransformer" =
  let open Sklearn.Feature_extraction in
  let corpus = ['this is the first document','this document is the second document','and this is the third one','is this the first document'] in  
  let vocabulary = ['this', 'document', 'first', 'is', 'second', 'the','and', 'one'] in  
  let pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),('tfid', TfidfTransformer())]).fit ~corpus () in  
  print_ndarray @@ pipe['count'].transform ~corpus ().toarray ();  
  [%expect {|
      array([[1, 1, 1, 1, 0, 1, 0, 0],      
             [1, 2, 0, 1, 1, 1, 0, 0],      
             [1, 0, 0, 1, 0, 1, 1, 1],      
             [1, 1, 1, 1, 0, 1, 0, 0]])      
  |}]
  print_ndarray @@ pipe['tfid'].idf_;  
  [%expect {|
      array([1.        , 1.22314355, 1.51082562, 1.        , 1.91629073,      
             1.        , 1.91629073, 1.91629073])      
  |}]
  print_ndarray @@ Pipeline.transform ~corpus Pipeline.shape pipe;  
  [%expect {|
      (4, 8)      
  |}]

*)



(* TfidfVectorizer *)
(*
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> corpus = [
...     'This is the first document.',
...     'This document is the second document.',
...     'And this is the third one.',
...     'Is this the first document?',
... ]
>>> vectorizer = TfidfVectorizer()
>>> X = vectorizer.fit_transform(corpus)
>>> print(vectorizer.get_feature_names())
['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
>>> print(X.shape)

*)

(* TEST TODO
let%expect_test "TfidfVectorizer" =
  let open Sklearn.Feature_extraction in
  let corpus = ['This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?',] in  
  let vectorizer = TfidfVectorizer.create () in  
  let x = TfidfVectorizer.fit_transform ~corpus vectorizer in  
  print_ndarray @@ print TfidfVectorizer.get_feature_names () vectorizer;  
  [%expect {|
      ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']      
  |}]
  print_ndarray @@ print x.shape ();  
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
  let open Sklearn.Feature_extraction in
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
  let open Sklearn.Feature_extraction in
  print_ndarray @@ @deprecated ()def some_function (): pass;  
  [%expect {|
  |}]

*)



