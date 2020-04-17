let print f x = Format.printf "%a" f x
let print_py x = Format.printf "%s" (Py.Object.to_string x)
let print_ndarray = print Sklearn.Ndarray.pp

let matrix = Sklearn.Ndarray.Float.matrix
let vector = Sklearn.Ndarray.Float.vector
let matrixi = Sklearn.Ndarray.Int.matrix
let vectori = Sklearn.Ndarray.Int.vector
let vectors = Sklearn.Ndarray.String.vector

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

let%expect_test "Binarizer" =
  let open Sklearn.Preprocessing in
  let x = matrix [|[| 1.; -1.;  2.|];[| 2.;  0.;  0.|];[| 0.;  1.; -1.|]|] in
  let transformer = Binarizer.(create () |> fit ~x) in  (* fit does nothing.     *)
  print Binarizer.pp transformer;
  [%expect {|
            Binarizer(copy=True, threshold=0.0)
    |}];
  print_ndarray @@ Binarizer.transform transformer ~x:(`Ndarray x);
  [%expect {|
            [[1. 0. 1.]
             [1. 0. 0.]
             [0. 1. 0.]]
    |}]


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

let%expect_test "KBinsDiscretizer" =
  let open Sklearn.Preprocessing in
  let x = matrix [|[|-2.; 1.; -4.; -1.|]; [|-1.; 2.; -3.; -0.5|]; [| 0.; 3.; -2.;  0.5|]; [| 1.; 4.; -1.; 2.|]|] in
  let est = KBinsDiscretizer.create ~n_bins:(`Int 3) ~encode:`Ordinal ~strategy:`Uniform () in
  print KBinsDiscretizer.pp @@ KBinsDiscretizer.fit est ~x;
  [%expect {|
            KBinsDiscretizer(encode='ordinal', n_bins=3, strategy='uniform')
    |}];
  let xt = KBinsDiscretizer.transform est ~x in
  print_ndarray xt;
  [%expect {|
            [[0. 0. 0. 0.]
             [1. 1. 1. 0.]
             [2. 2. 2. 1.]
             [2. 2. 2. 2.]]
    |}]


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

let%expect_test "LabelBinarizer" =
  let open Sklearn.Preprocessing in
  let lb = LabelBinarizer.create () in
  print LabelBinarizer.pp @@ LabelBinarizer.fit lb ~y:(vectori [|1; 2; 6; 4; 2|]);
  [%expect {|
            LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
    |}];
  print_ndarray @@ LabelBinarizer.classes_ lb;
  [%expect {|
            [1 2 4 6]
    |}];
  print_ndarray @@ LabelBinarizer.transform lb ~y:(`Ndarray (vectori [|1; 6|]));
  [%expect {|
            [[1 0 0 0]
             [0 0 0 1]]
    |}]



(* LabelBinarizer *)
(*
>>> lb = preprocessing.LabelBinarizer()
>>> lb.fit_transform(['yes', 'no', 'no', 'yes'])
array([[1],
       [0],
       [0],
       [1]])


*)

let%expect_test "LabelBinarizer" =
  let open Sklearn.Preprocessing in
  let lb = LabelBinarizer.create () in
  print_ndarray @@ LabelBinarizer.fit_transform lb ~y:(`Ndarray (vectors [|"yes"; "no"; "no"; "yes"|]));
  [%expect {|
            [[1]
             [0]
             [0]
             [1]]
    |}]

(* LabelBinarizer *)
(*
>>> import numpy as np
>>> lb.fit(np.array([|[0, 1, 1], [|1, 0, 0]]))
LabelBinarizer()
>>> lb.classes_
array([0, 1, 2])
>>> lb.transform([0, 1, 2, 1])
array([[1, 0, 0],
       [0, 1, 0],
       [0, 0, 1],
       [0, 1, 0]])


*)

let%expect_test "LabelBinarizer" =
  let open Sklearn.Preprocessing in
  let lb = LabelBinarizer.create () in
  print LabelBinarizer.pp @@ LabelBinarizer.fit lb ~y:(matrixi ([|[|0; 1; 1|]; [|1; 0; 0|]|]));
  [%expect {|
            LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
    |}];
  print_ndarray @@ LabelBinarizer.classes_ lb;
  [%expect {|
            [0 1 2]
    |}];
  print_ndarray @@ LabelBinarizer.transform lb ~y:(`Ndarray (vectori [|0; 1; 2; 1|]));
  [%expect {|
            [[1 0 0]
             [0 1 0]
             [0 0 1]
             [0 1 0]]
    |}]


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

let%expect_test "LabelEncoder" =
  let open Sklearn.Preprocessing in
  let le = LabelEncoder.create () in
  print LabelEncoder.pp @@ LabelEncoder.fit le ~y:(vectori [|1; 2; 2; 6|]);
  [%expect {|
            LabelEncoder()
    |}];
  print_ndarray @@ LabelEncoder.classes_ le;
  [%expect {|
            [1 2 6]
    |}];
  print_ndarray @@ LabelEncoder.transform le ~y:(vectori [|1; 1; 2; 6|]);
  [%expect {|
            [0 0 1 2]
    |}];
  print_ndarray @@ LabelEncoder.inverse_transform le ~y:(vectori [|0; 0; 1; 2|]);
  [%expect {|
            [1 1 2 6]
    |}]


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

let%expect_test "LabelEncoder" =
  let open Sklearn.Preprocessing in
  let le = LabelEncoder.create () in
  print LabelEncoder.pp @@ LabelEncoder.fit le ~y:(vectors [|"paris"; "paris"; "tokyo"; "amsterdam"|]);
  [%expect {|
            LabelEncoder()
    |}];
  print_ndarray @@ LabelEncoder.classes_ le;
  [%expect {|
            ['amsterdam' 'paris' 'tokyo']
    |}];
  print_ndarray @@ LabelEncoder.transform le ~y:(vectors [|"tokyo"; "tokyo"; "paris"|]);
  [%expect {|
            [2 2 1]
    |}];
  print_ndarray @@ LabelEncoder.inverse_transform le ~y:(vectori [|2; 2; 1|]);
  [%expect {|
            ['tokyo' 'tokyo' 'paris']
    |}]



(* MaxAbsScaler *)
(*
>>> from sklearn.preprocessing import MaxAbsScaler
>>> X = [|[ 1., -1.,  2.],
...      [| 2.,  0.,  0.],
...      [| 0.,  1., -1.]]
>>> transformer = MaxAbsScaler().fit(X)
>>> transformer
MaxAbsScaler()
>>> transformer.transform(X)
array([[ 0.5, -1. ,  1. ],
       [ 1. ,  0. ,  0. ],
       [ 0. ,  1. , -0.5]])


*)

let%expect_test "MaxAbsScaler" =
  let open Sklearn.Preprocessing in
  let x = matrix [|[| 1.; -1.;  2.|];[| 2.;  0.;  0.|];[| 0.;  1.; -1.|]|] in
  let transformer = MaxAbsScaler.(create () |> fit ~x:(`Ndarray x)) in
  print MaxAbsScaler.pp transformer;
  [%expect {|
            MaxAbsScaler(copy=True)
    |}];
  print_ndarray @@ MaxAbsScaler.transform transformer ~x:(`Ndarray x);
  [%expect {|
            [[ 0.5 -1.   1. ]
             [ 1.   0.   0. ]
             [ 0.   1.  -0.5]]
    |}]


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

let%expect_test "MinMaxScaler" =
  let open Sklearn.Preprocessing in
  let data = matrix [|[|-1.; 2.|]; [|-0.5; 6.|]; [|0.; 10.|]; [|1.; 18.|]|] in
  let scaler = MinMaxScaler.create () in
  print MinMaxScaler.pp @@ MinMaxScaler.fit scaler ~x:data;
  [%expect {|
            MinMaxScaler(copy=True, feature_range=(0, 1))
    |}];
  print_ndarray @@ MinMaxScaler.data_max_ scaler;
  [%expect {|
            [ 1. 18.]
    |}];
  print_ndarray @@ MinMaxScaler.transform scaler ~x:data;
  [%expect {|
            [[0.   0.  ]
             [0.25 0.25]
             [0.5  0.5 ]
             [1.   1.  ]]
    |}];
  print_ndarray @@ MinMaxScaler.transform scaler ~x:(matrixi [|[|2; 2|]|]);
  [%expect {|
            [[1.5 0. ]]
    |}]


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

let%expect_test "MultiLabelBinarizer" =
  let open Sklearn.Preprocessing in
  let mlb = MultiLabelBinarizer.create () in
  (* arg could be List(Ndarray()) ? *)
  print_ndarray @@ MultiLabelBinarizer.fit_transform mlb ~y:(Sklearn.Ndarray.Int.vectors [[|1; 2|]; [|3|]]);
  [%expect {|
            [[1 1 0]
             [0 0 1]]
    |}];
  print_ndarray @@ MultiLabelBinarizer.classes_ mlb;
    [%expect {|
            [1 2 3]
    |}]

(* MultiLabelBinarizer *)
(*
>>> mlb.fit_transform([{'sci-fi', 'thriller'}, {'comedy'}])
array([[0, 1, 1],
       [1, 0, 0]])
>>> list(mlb.classes_)
['comedy', 'sci-fi', 'thriller']


*)

let%expect_test "MultiLabelBinarizer" =
  let open Sklearn.Preprocessing in
  let mlb = MultiLabelBinarizer.create () in
  print_ndarray @@ MultiLabelBinarizer.fit_transform mlb
    ~y:(Sklearn.Ndarray.String.vectors [[|"sci-fi"; "thriller"|]; [|"comedy"|]]);
  [%expect {|
            [[0 1 1]
             [1 0 0]]
    |}];
  print_ndarray @@ MultiLabelBinarizer.classes_ mlb;
  [%expect {|
            ['comedy' 'sci-fi' 'thriller']
    |}]


(* MultiLabelBinarizer *)
(*
>>> mlb = MultiLabelBinarizer()
>>> mlb.fit(['sci-fi', 'thriller', 'comedy'])
MultiLabelBinarizer()
>>> mlb.classes_
array(['-', 'c', 'd', 'e', 'f', 'h', 'i', 'l', 'm', 'o', 'r', 's', 't',
    'y'], dtype=object)


*)

(* This is a example showing what not to do. In OCaml the below fails,
   rightfully.  *)
(* let%expect_test "MultiLabelBinarizer" =
 *   let open Sklearn.Preprocessing in
 *   let mlb = MultiLabelBinarizer.create () in
 *   print MultiLabelBinarizer.pp @@ MultiLabelBinarizer.fit mlb
 *     ~y:(Sklearn.Ndarray.String.vector [|"sci-fi"; "thriller"; "comedy"|]);
 *   [%expect {|
 *             MultiLabelBinarizer()
 *     |}];
 *   print_ndarray @@ MultiLabelBinarizer.classes_ mlb;
 *   [%expect {|
 *             array(['-', 'c', 'd', 'e', 'f', 'h', 'i', 'l', 'm', 'o', 'r', 's', 't',
 *                 'y'], dtype=object)
 *     |}] *)


(* MultiLabelBinarizer *)
(*
>>> mlb = MultiLabelBinarizer()
>>> mlb.fit([['sci-fi', 'thriller', 'comedy']])
MultiLabelBinarizer()
>>> mlb.classes_
array(['comedy', 'sci-fi', 'thriller'], dtype=object)


*)

(* TEST TODO
let%expect_test "MultiLabelBinarizer" =
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
let%expect_test "MultiLabelBinarizer" =
  let open Sklearn.Preprocessing in
  let mlb = MultiLabelBinarizer.create () in
  print MultiLabelBinarizer.pp @@ MultiLabelBinarizer.fit mlb
    ~y:(Sklearn.Ndarray.String.vectors [[|"sci-fi"; "thriller"; "comedy"|]]);
  [%expect {|
            MultiLabelBinarizer(classes=None, sparse_output=False)
    |}];
  print_ndarray @@ MultiLabelBinarizer.classes_ mlb;
  [%expect {|
            ['comedy' 'sci-fi' 'thriller']
    |}]

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

let%expect_test "Normalizer" =
  let open Sklearn.Preprocessing in
  let x = matrixi [|[|4; 1; 2; 2|]; [|1; 3; 9; 3|]; [|5; 7; 5; 1|]|] in
  let transformer = Normalizer.(create () |> fit ~x) in  (* fit does nothing *)
  print Normalizer.pp transformer;
  [%expect {|
            Normalizer(copy=True, norm='l2')
    |}];
  print_ndarray @@ Normalizer.transform transformer ~x:(`Ndarray x);
  [%expect {|
            [[0.8 0.2 0.4 0.4]
             [0.1 0.3 0.9 0.3]
             [0.5 0.7 0.5 0.1]]
    |}]

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

let%expect_test "OrdinalEncoder" =
  let open Sklearn.Preprocessing in
  let enc = OrdinalEncoder.create () in
  let x = Sklearn.Ndarray.Object.matrix [|[|`S "Male";   `I 1|];
                                          [|`S "Female"; `I 3|];
                                          [|`S "Female"; `I 2|]|] in
  print_ndarray x;
  [%expect {|
    [['Male' 1]
     ['Female' 3]
     ['Female' 2]] |}];
  print OrdinalEncoder.pp @@ OrdinalEncoder.fit enc ~x;
  [%expect {|
            OrdinalEncoder(categories='auto', dtype=<class 'numpy.float64'>)
    |}];
  print Sklearn.Ndarray.List.pp @@ OrdinalEncoder.categories_ enc;
  [%expect {|
            [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    |}];
  print_ndarray @@ OrdinalEncoder.transform enc
    ~x:(Sklearn.Ndarray.Object.matrix [|[|`S "Female"; `I 3|];
                                        [|`S "Male"; `I 1|]|]);
  [%expect {|
            [[0. 2.]
             [1. 0.]]
    |}]


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

let%expect_test "PolynomialFeatures" =
  let open Sklearn.Preprocessing in
  let x =  Sklearn.Ndarray.(arange 6 |> reshape ~shape:[|3; 2|]) in
  print_ndarray x;
  [%expect {|
            [[0 1]
             [2 3]
             [4 5]]
    |}];
  let poly = PolynomialFeatures.create ~degree:2 () in
  print_ndarray @@ PolynomialFeatures.fit_transform poly ~x;
  [%expect {|
            [[ 1.  0.  1.  0.  0.  1.]
             [ 1.  2.  3.  4.  6.  9.]
             [ 1.  4.  5. 16. 20. 25.]]
    |}];
  let poly = PolynomialFeatures.create ~interaction_only:true () in
  print_ndarray @@ PolynomialFeatures.fit_transform poly ~x;
  [%expect {|
            [[ 1.  0.  1.  0.]
             [ 1.  2.  3.  6.]
             [ 1.  4.  5. 20.]]
    |}]

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
let%expect_test "PowerTransformer" =
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
let%expect_test "QuantileTransformer" =
    import numpy as np
    let quantileTransformer = Sklearn.Preprocessing.quantileTransformer in
    rng = np.random.RandomState(0)
    let x = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
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
let%expect_test "RobustScaler" =
    let robustScaler = Sklearn.Preprocessing.robustScaler in
    let x = [[ 1., -2.,  2.],[ -2.,  1.,  3.],[ 4.,  1., -2.]]
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
let%expect_test "StandardScaler" =
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
let%expect_test "label_binarize" =
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
let%expect_test "label_binarize" =
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
let%expect_test "label_binarize" =
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
let%expect_test "power_transform" =
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
let%expect_test "quantile_transform" =
    import numpy as np
    let quantile_transform = Sklearn.Preprocessing.quantile_transform in
    rng = np.random.RandomState(0)
    let x = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
    quantile_transform(X, n_quantiles=10, random_state=0, copy=True)
    [%expect {|
            array([...])
    |}]

*)
