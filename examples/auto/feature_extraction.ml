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
let%expect_text "DictVectorizer" =
    let dictVectorizer = Sklearn.Feature_extraction.dictVectorizer in
    v = DictVectorizer(sparse=False)    
    D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]    
    X = v.fit_transform(D)    
    X    
    [%expect {|
            array([[2., 0., 1.],            
                   [0., 1., 3.]])            
    |}]
    v.inverse_transform(X) ==         [{'bar': 2.0, 'foo': 1.0}, {'baz': 1.0, 'foo': 3.0}]    
    [%expect {|
            True            
    |}]
    print @@ transform v {'foo': 4 'unseen_feature': 3}
    [%expect {|
            array([[0., 0., 4.]])            
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
let%expect_text "FeatureHasher" =
    let featureHasher = Sklearn.Feature_extraction.featureHasher in
    h = FeatureHasher(n_features=10)    
    D = [{'dog': 1, 'cat':2, 'elephant':4},{'dog': 2, 'run': 5}]    
    f = h.transform(D)    
    f.toarray()    
    [%expect {|
            array([[ 0.,  0., -4., -1.,  0.,  0.,  0.,  0.,  0.,  2.],            
                   [ 0.,  0.,  0., -2., -5.,  0.,  0.,  0.,  0.,  0.]])            
    |}]

*)



(*--------- Examples for module .Feature_extraction.Image ----------*)
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



(*--------- Examples for module .Feature_extraction.Text ----------*)
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
let%expect_text "CountVectorizer" =
    let countVectorizer = Sklearn.Feature_extraction.Text.countVectorizer in
    corpus = ['This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?',]    
    vectorizer = CountVectorizer()    
    X = vectorizer.fit_transform(corpus)    
    print(vectorizer.get_feature_names())    
    [%expect {|
            ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']            
    |}]
    print(X.toarray())    
    [%expect {|
            [[0 1 1 1 0 0 1 0 1]            
             [0 2 0 1 0 1 1 0 1]            
             [1 0 0 1 1 0 1 1 1]            
             [0 1 1 1 0 0 1 0 1]]            
    |}]
    vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))    
    X2 = vectorizer2.fit_transform(corpus)    
    print(vectorizer2.get_feature_names())    
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
let%expect_text "FeatureHasher" =
    let featureHasher = Sklearn.Feature_extraction.featureHasher in
    h = FeatureHasher(n_features=10)    
    D = [{'dog': 1, 'cat':2, 'elephant':4},{'dog': 2, 'run': 5}]    
    f = h.transform(D)    
    f.toarray()    
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
let%expect_text "HashingVectorizer" =
    let hashingVectorizer = Sklearn.Feature_extraction.Text.hashingVectorizer in
    corpus = ['This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?',]    
    vectorizer = HashingVectorizer(n_features=2**4)    
    X = vectorizer.fit_transform(corpus)    
    print(X.shape)    
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
let%expect_text "TfidfTransformer" =
    let tfidfTransformer = Sklearn.Feature_extraction.Text.tfidfTransformer in
    let countVectorizer = Sklearn.Feature_extraction.Text.countVectorizer in
    let pipeline = Sklearn.Pipeline.pipeline in
    import numpy as np    
    corpus = ['this is the first document','this document is the second document','and this is the third one','is this the first document']    
    vocabulary = ['this', 'document', 'first', 'is', 'second', 'the','and', 'one']    
    pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),('tfid', TfidfTransformer())]).fit(corpus)    
    pipe['count'].transform(corpus).toarray()    
    [%expect {|
            array([[1, 1, 1, 1, 0, 1, 0, 0],            
                   [1, 2, 0, 1, 1, 1, 0, 0],            
                   [1, 0, 0, 1, 0, 1, 1, 1],            
                   [1, 1, 1, 1, 0, 1, 0, 0]])            
    |}]
    pipe['tfid'].idf_    
    [%expect {|
            array([1.        , 1.22314355, 1.51082562, 1.        , 1.91629073,            
                   1.        , 1.91629073, 1.91629073])            
    |}]
    pipe.transform(corpus).shape    
    [%expect {|
            (4, 8)            
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



