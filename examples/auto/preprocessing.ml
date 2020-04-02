(* Binarizer *)
(*
>>> from sklearn.preprocessing import Binarizer
>>> X = [[ 1., -1.,  2.],
...      [ 2.,  0.,  0.],
...      [ 0.,  1., -1.]]
>>> transformer = Binarizer().fit(X)  # fit does nothing.
>>> transformer
Binarizer()
>>> transformer.transform(X)
array([[1., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.]])


*)

(* TEST TODO
let%expect_text "Binarizer" =
    let binarizer = Sklearn.Preprocessing.binarizer in
    X = [[ 1., -1.,  2.],[ 2.,  0.,  0.],[ 0.,  1., -1.]]    
    transformer = Binarizer().fit(X)  # fit does nothing.    
    transformer    
    [%expect {|
            Binarizer()            
    |}]
    print @@ transform transformer x
    [%expect {|
            array([[1., 0., 1.],            
                   [1., 0., 0.],            
                   [0., 1., 0.]])            
    |}]

*)



(* KBinsDiscretizer *)
(*
>>> X = [[-2, 1, -4,   -1],
...      [-1, 2, -3, -0.5],
...      [ 0, 3, -2,  0.5],
...      [ 1, 4, -1,    2]]
>>> est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
>>> est.fit(X)
KBinsDiscretizer(...)
>>> Xt = est.transform(X)
>>> Xt  # doctest: +SKIP
array([[ 0., 0., 0., 0.],
       [ 1., 1., 1., 0.],
       [ 2., 2., 2., 1.],
       [ 2., 2., 2., 2.]])


*)

(* TEST TODO
let%expect_text "KBinsDiscretizer" =
    X = [[-2, 1, -4,   -1],[-1, 2, -3, -0.5],[ 0, 3, -2,  0.5],[ 1, 4, -1,    2]]    
    est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')    
    print @@ fit est x
    [%expect {|
            KBinsDiscretizer(...)            
    |}]
    Xt = est.transform(X)    
    Xt  # doctest: +SKIP    
    [%expect {|
            array([[ 0., 0., 0., 0.],            
                   [ 1., 1., 1., 0.],            
                   [ 2., 2., 2., 1.],            
                   [ 2., 2., 2., 2.]])            
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



(* LabelEncoder *)
(*
>>> from sklearn import preprocessing
>>> le = preprocessing.LabelEncoder()
>>> le.fit([1, 2, 2, 6])
LabelEncoder()
>>> le.classes_
array([1, 2, 6])
>>> le.transform([1, 1, 2, 6])
array([0, 0, 1, 2]...)
>>> le.inverse_transform([0, 0, 1, 2])
array([1, 1, 2, 6])


*)

(* TEST TODO
let%expect_text "LabelEncoder" =
    let preprocessing = Sklearn.preprocessing in
    le = preprocessing.LabelEncoder()    
    print @@ fit le [1 2 2 6]
    [%expect {|
            LabelEncoder()            
    |}]
    le.classes_    
    [%expect {|
            array([1, 2, 6])            
    |}]
    print @@ transform le [1 1 2 6]
    [%expect {|
            array([0, 0, 1, 2]...)            
    |}]
    print @@ inverse_transform le [0 0 1 2]
    [%expect {|
            array([1, 1, 2, 6])            
    |}]

*)



(* LabelEncoder *)
(*
>>> le = preprocessing.LabelEncoder()
>>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
LabelEncoder()
>>> list(le.classes_)
['amsterdam', 'paris', 'tokyo']
>>> le.transform(["tokyo", "tokyo", "paris"])
array([2, 2, 1]...)
>>> list(le.inverse_transform([2, 2, 1]))
['tokyo', 'tokyo', 'paris']


*)

(* TEST TODO
let%expect_text "LabelEncoder" =
    le = preprocessing.LabelEncoder()    
    print @@ fit le ["paris" "paris" "tokyo" "amsterdam"]
    [%expect {|
            LabelEncoder()            
    |}]
    list(le.classes_)    
    [%expect {|
            ['amsterdam', 'paris', 'tokyo']            
    |}]
    print @@ transform le ["tokyo" "tokyo" "paris"]
    [%expect {|
            array([2, 2, 1]...)            
    |}]
    list(le.inverse_transform([2, 2, 1]))    
    [%expect {|
            ['tokyo', 'tokyo', 'paris']            
    |}]

*)



(* MaxAbsScaler *)
(*
>>> from sklearn.preprocessing import MaxAbsScaler
>>> X = [[ 1., -1.,  2.],
...      [ 2.,  0.,  0.],
...      [ 0.,  1., -1.]]
>>> transformer = MaxAbsScaler().fit(X)
>>> transformer
MaxAbsScaler()
>>> transformer.transform(X)
array([[ 0.5, -1. ,  1. ],
       [ 1. ,  0. ,  0. ],
       [ 0. ,  1. , -0.5]])


*)

(* TEST TODO
let%expect_text "MaxAbsScaler" =
    let maxAbsScaler = Sklearn.Preprocessing.maxAbsScaler in
    X = [[ 1., -1.,  2.],[ 2.,  0.,  0.],[ 0.,  1., -1.]]    
    transformer = MaxAbsScaler().fit(X)    
    transformer    
    [%expect {|
            MaxAbsScaler()            
    |}]
    print @@ transform transformer x
    [%expect {|
            array([[ 0.5, -1. ,  1. ],            
                   [ 1. ,  0. ,  0. ],            
                   [ 0. ,  1. , -0.5]])            
    |}]

*)



(* MinMaxScaler *)
(*
>>> from sklearn.preprocessing import MinMaxScaler
>>> data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
>>> scaler = MinMaxScaler()
>>> print(scaler.fit(data))
MinMaxScaler()
>>> print(scaler.data_max_)
[ 1. 18.]
>>> print(scaler.transform(data))
[[0.   0.  ]
 [0.25 0.25]
 [0.5  0.5 ]
 [1.   1.  ]]
>>> print(scaler.transform([[2, 2]]))
[[1.5 0. ]]


*)

(* TEST TODO
let%expect_text "MinMaxScaler" =
    let minMaxScaler = Sklearn.Preprocessing.minMaxScaler in
    data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]    
    scaler = MinMaxScaler()    
    print(scaler.fit(data))    
    [%expect {|
            MinMaxScaler()            
    |}]
    print(scaler.data_max_)    
    [%expect {|
            [ 1. 18.]            
    |}]
    print(scaler.transform(data))    
    [%expect {|
            [[0.   0.  ]            
             [0.25 0.25]            
             [0.5  0.5 ]            
             [1.   1.  ]]            
    |}]
    print(scaler.transform([[2, 2]]))    
    [%expect {|
            [[1.5 0. ]]            
    |}]

*)



(* MultiLabelBinarizer *)
(*
>>> from sklearn.preprocessing import MultiLabelBinarizer
>>> mlb = MultiLabelBinarizer()
>>> mlb.fit_transform([(1, 2), (3,)])
array([[1, 1, 0],
       [0, 0, 1]])
>>> mlb.classes_
array([1, 2, 3])


*)

(* TEST TODO
let%expect_text "MultiLabelBinarizer" =
    let multiLabelBinarizer = Sklearn.Preprocessing.multiLabelBinarizer in
    mlb = MultiLabelBinarizer()    
    mlb.fit_transform([(1, 2), (3,)])    
    [%expect {|
            array([[1, 1, 0],            
                   [0, 0, 1]])            
    |}]
    mlb.classes_    
    [%expect {|
            array([1, 2, 3])            
    |}]

*)



(* MultiLabelBinarizer *)
(*
>>> mlb.fit_transform([{'sci-fi', 'thriller'}, {'comedy'}])
array([[0, 1, 1],
       [1, 0, 0]])
>>> list(mlb.classes_)
['comedy', 'sci-fi', 'thriller']


*)

(* TEST TODO
let%expect_text "MultiLabelBinarizer" =
    print @@ fit_transform mlb [{'sci-fi' 'thriller'} {'comedy'}]
    [%expect {|
            array([[0, 1, 1],            
                   [1, 0, 0]])            
    |}]
    list(mlb.classes_)    
    [%expect {|
            ['comedy', 'sci-fi', 'thriller']            
    |}]

*)



(* MultiLabelBinarizer *)
(*
>>> mlb = MultiLabelBinarizer()
>>> mlb.fit(['sci-fi', 'thriller', 'comedy'])
MultiLabelBinarizer()
>>> mlb.classes_
array(['-', 'c', 'd', 'e', 'f', 'h', 'i', 'l', 'm', 'o', 'r', 's', 't',
    'y'], dtype=object)


*)

(* TEST TODO
let%expect_text "MultiLabelBinarizer" =
    mlb = MultiLabelBinarizer()    
    print @@ fit mlb ['sci-fi' 'thriller' 'comedy']
    [%expect {|
            MultiLabelBinarizer()            
    |}]
    mlb.classes_    
    [%expect {|
            array(['-', 'c', 'd', 'e', 'f', 'h', 'i', 'l', 'm', 'o', 'r', 's', 't',            
                'y'], dtype=object)            
    |}]

*)



(* MultiLabelBinarizer *)
(*
>>> mlb = MultiLabelBinarizer()
>>> mlb.fit([['sci-fi', 'thriller', 'comedy']])
MultiLabelBinarizer()
>>> mlb.classes_
array(['comedy', 'sci-fi', 'thriller'], dtype=object)


*)

(* TEST TODO
let%expect_text "MultiLabelBinarizer" =
    mlb = MultiLabelBinarizer()    
    print @@ fit mlb [['sci-fi' 'thriller' 'comedy']]
    [%expect {|
            MultiLabelBinarizer()            
    |}]
    mlb.classes_    
    [%expect {|
            array(['comedy', 'sci-fi', 'thriller'], dtype=object)            
    |}]

*)



(* Normalizer *)
(*
>>> from sklearn.preprocessing import Normalizer
>>> X = [[4, 1, 2, 2],
...      [1, 3, 9, 3],
...      [5, 7, 5, 1]]
>>> transformer = Normalizer().fit(X)  # fit does nothing.
>>> transformer
Normalizer()
>>> transformer.transform(X)
array([[0.8, 0.2, 0.4, 0.4],
       [0.1, 0.3, 0.9, 0.3],
       [0.5, 0.7, 0.5, 0.1]])


*)

(* TEST TODO
let%expect_text "Normalizer" =
    let normalizer = Sklearn.Preprocessing.normalizer in
    X = [[4, 1, 2, 2],[1, 3, 9, 3],[5, 7, 5, 1]]    
    transformer = Normalizer().fit(X)  # fit does nothing.    
    transformer    
    [%expect {|
            Normalizer()            
    |}]
    print @@ transform transformer x
    [%expect {|
            array([[0.8, 0.2, 0.4, 0.4],            
                   [0.1, 0.3, 0.9, 0.3],            
                   [0.5, 0.7, 0.5, 0.1]])            
    |}]

*)



(* OrdinalEncoder *)
(*
>>> from sklearn.preprocessing import OrdinalEncoder
>>> enc = OrdinalEncoder()
>>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
>>> enc.fit(X)
OrdinalEncoder()
>>> enc.categories_
[array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
>>> enc.transform([['Female', 3], ['Male', 1]])
array([[0., 2.],
       [1., 0.]])


*)

(* TEST TODO
let%expect_text "OrdinalEncoder" =
    let ordinalEncoder = Sklearn.Preprocessing.ordinalEncoder in
    enc = OrdinalEncoder()    
    X = [['Male', 1], ['Female', 3], ['Female', 2]]    
    print @@ fit enc x
    [%expect {|
            OrdinalEncoder()            
    |}]
    enc.categories_    
    [%expect {|
            [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]            
    |}]
    print @@ transform enc [['Female' 3] ['Male' 1]]
    [%expect {|
            array([[0., 2.],            
                   [1., 0.]])            
    |}]

*)



(* PolynomialFeatures *)
(*
>>> import numpy as np
>>> from sklearn.preprocessing import PolynomialFeatures
>>> X = np.arange(6).reshape(3, 2)
>>> X
array([[0, 1],
       [2, 3],
       [4, 5]])
>>> poly = PolynomialFeatures(2)
>>> poly.fit_transform(X)
array([[ 1.,  0.,  1.,  0.,  0.,  1.],
       [ 1.,  2.,  3.,  4.,  6.,  9.],
       [ 1.,  4.,  5., 16., 20., 25.]])
>>> poly = PolynomialFeatures(interaction_only=True)
>>> poly.fit_transform(X)
array([[ 1.,  0.,  1.,  0.],
       [ 1.,  2.,  3.,  6.],
       [ 1.,  4.,  5., 20.]])


*)

(* TEST TODO
let%expect_text "PolynomialFeatures" =
    import numpy as np    
    let polynomialFeatures = Sklearn.Preprocessing.polynomialFeatures in
    X = np.arange(6).reshape(3, 2)    
    X    
    [%expect {|
            array([[0, 1],            
                   [2, 3],            
                   [4, 5]])            
    |}]
    poly = PolynomialFeatures(2)    
    print @@ fit_transform poly x
    [%expect {|
            array([[ 1.,  0.,  1.,  0.,  0.,  1.],            
                   [ 1.,  2.,  3.,  4.,  6.,  9.],            
                   [ 1.,  4.,  5., 16., 20., 25.]])            
    |}]
    poly = PolynomialFeatures(interaction_only=True)    
    print @@ fit_transform poly x
    [%expect {|
            array([[ 1.,  0.,  1.,  0.],            
                   [ 1.,  2.,  3.,  6.],            
                   [ 1.,  4.,  5., 20.]])            
    |}]

*)



(* PowerTransformer *)
(*
>>> import numpy as np
>>> from sklearn.preprocessing import PowerTransformer
>>> pt = PowerTransformer()
>>> data = [[1, 2], [3, 2], [4, 5]]
>>> print(pt.fit(data))
PowerTransformer()
>>> print(pt.lambdas_)
[ 1.386... -3.100...]
>>> print(pt.transform(data))
[[-1.316... -0.707...]
 [ 0.209... -0.707...]
 [ 1.106...  1.414...]]


*)

(* TEST TODO
let%expect_text "PowerTransformer" =
    import numpy as np    
    let powerTransformer = Sklearn.Preprocessing.powerTransformer in
    pt = PowerTransformer()    
    data = [[1, 2], [3, 2], [4, 5]]    
    print(pt.fit(data))    
    [%expect {|
            PowerTransformer()            
    |}]
    print(pt.lambdas_)    
    [%expect {|
            [ 1.386... -3.100...]            
    |}]
    print(pt.transform(data))    
    [%expect {|
            [[-1.316... -0.707...]            
             [ 0.209... -0.707...]            
             [ 1.106...  1.414...]]            
    |}]

*)



(* QuantileTransformer *)
(*
>>> import numpy as np
>>> from sklearn.preprocessing import QuantileTransformer
>>> rng = np.random.RandomState(0)
>>> X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
>>> qt = QuantileTransformer(n_quantiles=10, random_state=0)
>>> qt.fit_transform(X)
array([...])


*)

(* TEST TODO
let%expect_text "QuantileTransformer" =
    import numpy as np    
    let quantileTransformer = Sklearn.Preprocessing.quantileTransformer in
    rng = np.random.RandomState(0)    
    X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)    
    qt = QuantileTransformer(n_quantiles=10, random_state=0)    
    print @@ fit_transform qt x
    [%expect {|
            array([...])            
    |}]

*)



(* RobustScaler *)
(*
>>> from sklearn.preprocessing import RobustScaler
>>> X = [[ 1., -2.,  2.],
...      [ -2.,  1.,  3.],
...      [ 4.,  1., -2.]]
>>> transformer = RobustScaler().fit(X)
>>> transformer
RobustScaler()
>>> transformer.transform(X)
array([[ 0. , -2. ,  0. ],
       [-1. ,  0. ,  0.4],
       [ 1. ,  0. , -1.6]])


*)

(* TEST TODO
let%expect_text "RobustScaler" =
    let robustScaler = Sklearn.Preprocessing.robustScaler in
    X = [[ 1., -2.,  2.],[ -2.,  1.,  3.],[ 4.,  1., -2.]]    
    transformer = RobustScaler().fit(X)    
    transformer    
    [%expect {|
            RobustScaler()            
    |}]
    print @@ transform transformer x
    [%expect {|
            array([[ 0. , -2. ,  0. ],            
                   [-1. ,  0. ,  0.4],            
                   [ 1. ,  0. , -1.6]])            
    |}]

*)



(* StandardScaler *)
(*
>>> from sklearn.preprocessing import StandardScaler
>>> data = [[0, 0], [0, 0], [1, 1], [1, 1]]
>>> scaler = StandardScaler()
>>> print(scaler.fit(data))
StandardScaler()
>>> print(scaler.mean_)
[0.5 0.5]
>>> print(scaler.transform(data))
[[-1. -1.]
 [-1. -1.]
 [ 1.  1.]
 [ 1.  1.]]
>>> print(scaler.transform([[2, 2]]))
[[3. 3.]]


*)

(* TEST TODO
let%expect_text "StandardScaler" =
    let standardScaler = Sklearn.Preprocessing.standardScaler in
    data = [[0, 0], [0, 0], [1, 1], [1, 1]]    
    scaler = StandardScaler()    
    print(scaler.fit(data))    
    [%expect {|
            StandardScaler()            
    |}]
    print(scaler.mean_)    
    [%expect {|
            [0.5 0.5]            
    |}]
    print(scaler.transform(data))    
    [%expect {|
            [[-1. -1.]            
             [-1. -1.]            
             [ 1.  1.]            
             [ 1.  1.]]            
    |}]
    print(scaler.transform([[2, 2]]))    
    [%expect {|
            [[3. 3.]]            
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



(* power_transform *)
(*
>>> import numpy as np
>>> from sklearn.preprocessing import power_transform
>>> data = [[1, 2], [3, 2], [4, 5]]
>>> print(power_transform(data, method='box-cox'))
[[-1.332... -0.707...]
 [ 0.256... -0.707...]
 [ 1.076...  1.414...]]


*)

(* TEST TODO
let%expect_text "power_transform" =
    import numpy as np    
    let power_transform = Sklearn.Preprocessing.power_transform in
    data = [[1, 2], [3, 2], [4, 5]]    
    print(power_transform(data, method='box-cox'))    
    [%expect {|
            [[-1.332... -0.707...]            
             [ 0.256... -0.707...]            
             [ 1.076...  1.414...]]            
    |}]

*)



(* quantile_transform *)
(*
>>> import numpy as np
>>> from sklearn.preprocessing import quantile_transform
>>> rng = np.random.RandomState(0)
>>> X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
>>> quantile_transform(X, n_quantiles=10, random_state=0, copy=True)
array([...])


*)

(* TEST TODO
let%expect_text "quantile_transform" =
    import numpy as np    
    let quantile_transform = Sklearn.Preprocessing.quantile_transform in
    rng = np.random.RandomState(0)    
    X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)    
    quantile_transform(X, n_quantiles=10, random_state=0, copy=True)    
    [%expect {|
            array([...])            
    |}]

*)



