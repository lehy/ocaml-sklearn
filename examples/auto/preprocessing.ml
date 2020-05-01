let print f x = Format.printf "%a" f x
let print_py x = Format.printf "%s" (Py.Object.to_string x)
let print_ndarray = print Sklearn.Arr.pp

let matrix = Sklearn.Arr.Float.matrix
let vector = Sklearn.Arr.Float.vector
let matrixi = Sklearn.Arr.Int.matrix
let vectori = Sklearn.Arr.Int.vector
let vectors = Sklearn.Arr.String.vector

let get x = match x with
  | None -> failwith "Option.get"
  | Some x -> x

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
  print_ndarray @@ Binarizer.transform transformer ~x;
  [%expect {|
            [[1. 0. 1.]
             [1. 0. 0.]
             [0. 1. 0.]]
    |}]


(* FunctionTransformer *)
(*
>>> import numpy as np
>>> from sklearn.preprocessing import FunctionTransformer
>>> transformer = FunctionTransformer(np.log1p)
>>> X = np.array([[0, 1], [2, 3]])
>>> transformer.transform(X)
array([[0.       , 0.6931...],

*)

(* TEST TODO
let%expect_test "FunctionTransformer" =
  let open Sklearn.Preprocessing in
  let transformer = FunctionTransformer.create np.log1p () in  
  let x = .array (matrixi [|[|0; 1|]; [|2; 3|]|]) np in  
  print_ndarray @@ FunctionTransformer.transform ~x transformer;  
  [%expect {|
      array([[0.       , 0.6931...],      
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
let%expect_test "KBinsDiscretizer" =
  let open Sklearn.Preprocessing in
  let x = [[-2, 1, -4, -1],[-1, 2, -3, -0.5],[ 0, 3, -2, 0.5],[ 1, 4, -1, 2]] in  
  let est = KBinsDiscretizer.create ~n_bins:3 ~encode:'ordinal' ~strategy:'uniform' () in  
  print KBinsDiscretizer.pp @@ KBinsDiscretizer.fit ~x est;  
  [%expect {|
      KBinsDiscretizer(...)      
  |}]
  let Xt = KBinsDiscretizer.transform ~x est in  
  print_ndarray @@ Xt # doctest: +SKIP;  
  [%expect {|
      array([[ 0., 0., 0., 0.],      
             [ 1., 1., 1., 0.],      
             [ 2., 2., 2., 1.],      
             [ 2., 2., 2., 2.]])      
  |}]

*)


(* KBinsDiscretizer *)
(*
>>> est.bin_edges_[0]
array([-2., -1.,  0.,  1.])
>>> est.inverse_transform(Xt)
array([[-1.5,  1.5, -3.5, -0.5],
       [-0.5,  2.5, -2.5, -0.5],
       [ 0.5,  3.5, -1.5,  0.5],

*)

(* TEST TODO
let%expect_test "KBinsDiscretizer" =
  let open Sklearn.Preprocessing in
  print_ndarray @@ .bin_edges_ vectori [|0|] est;  
  [%expect {|
      array([-2., -1.,  0.,  1.])      
  |}]
  print_ndarray @@ .inverse_transform ~Xt est;  
  [%expect {|
      array([[-1.5,  1.5, -3.5, -0.5],      
             [-0.5,  2.5, -2.5, -0.5],      
             [ 0.5,  3.5, -1.5,  0.5],      
  |}]

*)

(* KernelCenterer *)
(*
>>> from sklearn.preprocessing import KernelCenterer
>>> from sklearn.metrics.pairwise import pairwise_kernels
>>> X = [[ 1., -2.,  2.],
...      [ -2.,  1.,  3.],
...      [ 4.,  1., -2.]]
>>> K = pairwise_kernels(X, metric='linear')
>>> K
array([[  9.,   2.,  -2.],
       [  2.,  14., -13.],
       [ -2., -13.,  21.]])
>>> transformer = KernelCenterer().fit(K)
>>> transformer
KernelCenterer()
>>> transformer.transform(K)
array([[  5.,   0.,  -5.],
       [  0.,  14., -14.],

*)

(* TEST TODO
let%expect_test "KernelCenterer" =
  let open Sklearn.Preprocessing in
  let x = [[ 1., -2., 2.],[ -2., 1., 3.],[ 4., 1., -2.]] in  
  let K = pairwise_kernels x ~metric:'linear' () in  
  print_ndarray @@ K;  
  [%expect {|
      array([[  9.,   2.,  -2.],      
             [  2.,  14., -13.],      
             [ -2., -13.,  21.]])      
  |}]
  let transformer = KernelCenterer().fit ~K () in  
  print_ndarray @@ transformer;  
  [%expect {|
      KernelCenterer()      
  |}]
  print_ndarray @@ KernelCenterer.transform ~K transformer;  
  [%expect {|
      array([[  5.,   0.,  -5.],      
             [  0.,  14., -14.],      
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

let%expect_test "KBinsDiscretizer" =
  let open Sklearn.Preprocessing in
  let x = matrix [|[|-2.; 1.; -4.; -1.|]; [|-1.; 2.; -3.; -0.5|]; [| 0.; 3.; -2.;  0.5|]; [| 1.; 4.; -1.; 2.|]|] in
  let est = KBinsDiscretizer.create ~n_bins:(`I 3) ~encode:`Ordinal ~strategy:`Uniform () in
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
  print_ndarray @@ LabelBinarizer.transform lb ~y:(vectori [|1; 6|]);
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
  print_ndarray @@ LabelBinarizer.fit_transform lb ~y:(vectors [|"yes"; "no"; "no"; "yes"|]);
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
  print_ndarray @@ LabelBinarizer.transform lb ~y:(vectori [|0; 1; 2; 1|]);
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
  let transformer = MaxAbsScaler.(create () |> fit ~x) in
  print MaxAbsScaler.pp transformer;
  [%expect {|
            MaxAbsScaler(copy=True)
    |}];
  print_ndarray @@ MaxAbsScaler.transform transformer ~x;
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
  print_ndarray @@ MultiLabelBinarizer.fit_transform mlb ~y:(Sklearn.Arr.Int.vectors [[|1; 2|]; [|3|]]);
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
    ~y:(Sklearn.Arr.String.vectors [[|"sci-fi"; "thriller"|]; [|"comedy"|]]);
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
    ~y:(Sklearn.Arr.String.vectors [[|"sci-fi"; "thriller"; "comedy"|]]);
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
  print_ndarray @@ Normalizer.transform transformer ~x;
  [%expect {|
            [[0.8 0.2 0.4 0.4]
             [0.1 0.3 0.9 0.3]
             [0.5 0.7 0.5 0.1]]
    |}]


(* OneHotEncoder *)
(*
>>> from sklearn.preprocessing import OneHotEncoder
>>> enc = OneHotEncoder(handle_unknown='ignore')
>>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
>>> enc.fit(X)
OneHotEncoder(handle_unknown='ignore')
>>> enc.categories_
[array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
>>> enc.transform([['Female', 1], ['Male', 4]]).toarray()
array([[1., 0., 1., 0., 0.],
       [0., 1., 0., 0., 0.]])
>>> enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])
array([['Male', 1],
       [None, 2]], dtype=object)
>>> enc.get_feature_names(['gender', 'group'])
array(['gender_Female', 'gender_Male', 'group_1', 'group_2', 'group_3'],
  dtype=object)
>>> drop_enc = OneHotEncoder(drop='first').fit(X)
>>> drop_enc.categories_
[array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
>>> drop_enc.transform([['Female', 1], ['Male', 2]]).toarray()
array([[0., 0., 0.],

*)

(* TEST TODO
let%expect_test "OneHotEncoder" =
  let open Sklearn.Preprocessing in
  let enc = OneHotEncoder.create ~handle_unknown:'ignore' () in  
  let x = (matrixi [|[|'Male'; 1|]; [|'Female'; 3|]; [|'Female'; 2|]|]) in  
  print OneHotEncoder.pp @@ OneHotEncoder.fit ~x enc;  
  [%expect {|
      OneHotEncoder(handle_unknown='ignore')      
  |}]
  print_ndarray @@ OneHotEncoder.categories_ enc;  
  [%expect {|
      [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]      
  |}]
  print_ndarray @@ OneHotEncoder.transform (matrixi [|[|'Female'; 1|]; [|'Male'; 4|]|])).toarray( enc;  
  [%expect {|
      array([[1., 0., 1., 0., 0.],      
             [0., 1., 0., 0., 0.]])      
  |}]
  print_ndarray @@ OneHotEncoder.inverse_transform (matrixi [|[|0; 1; 1; 0; 0|]; [|0; 0; 0; 1; 0|]|]) enc;  
  [%expect {|
      array([['Male', 1],      
             [None, 2]], dtype=object)      
  |}]
  print_ndarray @@ OneHotEncoder.get_feature_names ['gender' 'group'] enc;  
  [%expect {|
      array(['gender_Female', 'gender_Male', 'group_1', 'group_2', 'group_3'],      
        dtype=object)      
  |}]
  let drop_enc = OneHotEncoder(drop='first').fit ~x () in  
  print_ndarray @@ OneHotEncoder.categories_ drop_enc;  
  [%expect {|
      [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]      
  |}]
  print_ndarray @@ OneHotEncoder.transform (matrixi [|[|'Female'; 1|]; [|'Male'; 2|]|])).toarray( drop_enc;  
  [%expect {|
      array([[0., 0., 0.],      
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

let%expect_test "OrdinalEncoder" =
  let open Sklearn.Preprocessing in
  let enc = OrdinalEncoder.create () in
  let x = Sklearn.Arr.Object.matrix [|[|`S "Male";   `I 1|];
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
  print Sklearn.Arr.List.pp @@ OrdinalEncoder.categories_ enc;
  [%expect {|
            [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    |}];
  print_ndarray @@ OrdinalEncoder.transform enc
    ~x:(Sklearn.Arr.Object.matrix [|[|`S "Female"; `I 3|];
                                    [|`S "Male"; `I 1|]|]);
  [%expect {|
            [[0. 2.]
             [1. 0.]]
    |}]





(* OrdinalEncoder *)
(*
>>> enc.inverse_transform([[1, 0], [0, 1]])
array([['Male', 1],

*)

(* TEST TODO
let%expect_test "OrdinalEncoder" =
  let open Sklearn.Preprocessing in
  print_ndarray @@ .inverse_transform (matrixi [|[|1; 0|]; [|0; 1|]|]) enc;  
  [%expect {|
      array([['Male', 1],      
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

let%expect_test "PolynomialFeatures" =
  let open Sklearn.Preprocessing in
  let x =  Sklearn.Arr.(arange 6 |> reshape ~shape:[|3; 2|]) in
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

let%expect_test "PowerTransformer" =
  let open Sklearn.Preprocessing in
  let pt = PowerTransformer.create () in
  let data = matrixi [|[|1; 2|]; [|3; 2|]; [|4; 5|]|] in
  print PowerTransformer.pp @@ PowerTransformer.fit pt ~x:data;
  [%expect {|
            PowerTransformer(copy=True, method='yeo-johnson', standardize=True)
    |}];
  print_ndarray @@ PowerTransformer.lambdas_ pt;
  [%expect {|
            [ 1.38668178 -3.10053309]
    |}];
  print_ndarray @@ PowerTransformer.transform pt ~x:data;
  [%expect {|
            [[-1.31616039 -0.70710678]
             [ 0.20998268 -0.70710678]
             [ 1.1061777   1.41421356]]
    |}]


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

let%expect_test "QuantileTransformer" =
  let open Sklearn.Preprocessing in
  let module Matrix = Owl.Dense.Matrix.D in
  Owl_stats_prng.init 0;
  let x = Matrix.(gaussian ~mu:0.5 ~sigma:0.25 25 1 |> sort) |> Sklearn.Arr.of_bigarray in
  let qt = QuantileTransformer.create ~n_quantiles:10 ~random_state:0 () in
  print_ndarray @@ QuantileTransformer.fit_transform qt ~x;
  [%expect {|
            [[0.        ]
             [0.0465588 ]
             [0.07236635]
             [0.13435383]
             [0.19038255]
             [0.20744256]
             [0.27077313]
             [0.3072948 ]
             [0.33333333]
             [0.40459044]
             [0.4263402 ]
             [0.46393428]
             [0.52235714]
             [0.54881137]
             [0.59860807]
             [0.60721673]
             [0.66666667]
             [0.69638456]
             [0.74844166]
             [0.78237337]
             [0.80823856]
             [0.86774379]
             [0.90414956]
             [0.95992288]
             [1.        ]]
    |}]


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

let%expect_test "RobustScaler" =
  let open Sklearn.Preprocessing in
  let x = matrix [|[| 1.; -2.;  2.|];
                   [| -2.;  1.;  3.|];
                   [| 4.;  1.; -2.|]|]
  in
  let transformer = RobustScaler.(create () |> fit ~x) in
  print RobustScaler.pp transformer;
  [%expect {|
            RobustScaler(copy=True, quantile_range=(25.0, 75.0), with_centering=True,
                         with_scaling=True)
    |}];
  print_ndarray @@ RobustScaler.transform transformer ~x;
  [%expect {|
            [[ 0.  -2.   0. ]
             [-1.   0.   0.4]
             [ 1.   0.  -1.6]]
    |}]


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

let%expect_test "StandardScaler" =
  let open Sklearn.Preprocessing in
  let data = matrixi [|[|0; 0|]; [|0; 0|]; [|1; 1|]; [|1; 1|]|] in
  let scaler = StandardScaler.create () in
  print StandardScaler.pp @@ StandardScaler.fit scaler ~x:data;
  [%expect {|
            StandardScaler(copy=True, with_mean=True, with_std=True)
    |}];
  print_ndarray @@ StandardScaler.mean_ scaler;
  [%expect {|
            [0.5 0.5]
    |}];
  print_ndarray @@ StandardScaler.transform scaler ~x:data;
  [%expect {|
            [[-1. -1.]
             [-1. -1.]
             [ 1.  1.]
             [ 1.  1.]]
    |}];
  print_ndarray @@ StandardScaler.transform scaler ~x:(matrixi [|[|2; 2|]|]);
  [%expect {|
            [[3. 3.]]
    |}]

(* add_dummy_feature *)
(*
>>> from sklearn.preprocessing import add_dummy_feature
>>> add_dummy_feature([[0, 1], [1, 0]])
array([[1., 0., 1.],

*)

(* TEST TODO
let%expect_test "add_dummy_feature" =
  let open Sklearn.Preprocessing in
  print_ndarray @@ add_dummy_feature((matrixi [|[|0; 1|]; [|1; 0|]|]));  
  [%expect {|
      array([[1., 0., 1.],      
  |}]

*)

(* label_binarize *)
(*
>>> from sklearn.preprocessing import label_binarize
>>> label_binarize([1, 6], classes=[1, 2, 4, 6])
array([[1, 0, 0, 0],
       [0, 0, 0, 1]])


*)

let%expect_test "label_binarize" =
  let open Sklearn.Preprocessing in
  print_ndarray @@ label_binarize ~y:(vectori [|1; 6|]) ~classes:(vectori [|1; 2; 4; 6|]) ();
  [%expect {|
            [[1 0 0 0]
             [0 0 0 1]]
    |}]


(* label_binarize *)
(*
>>> label_binarize([1, 6], classes=[1, 6, 4, 2])
array([[1, 0, 0, 0],
       [0, 1, 0, 0]])


*)

let%expect_test "label_binarize" =
  let open Sklearn.Preprocessing in
  print_ndarray @@ label_binarize ~y:(vectori [|1; 6|]) ~classes:(vectori [|1; 6; 4; 2|]) ();
  [%expect {|
            [[1 0 0 0]
             [0 1 0 0]]
    |}]


(* label_binarize *)
(*
>>> label_binarize(['yes', 'no', 'no', 'yes'], classes=['no', 'yes'])
array([[1],
       [0],
       [0],
       [1]])


*)

let%expect_test "label_binarize" =
  let open Sklearn.Preprocessing in
  print_ndarray @@ label_binarize ~y:(vectors [|"yes"; "no"; "no"; "yes"|]) ~classes:(vectors [|"no"; "yes"|]) ();
  [%expect {|
            [[1]
             [0]
             [0]
             [1]]
    |}]


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

let%expect_test "power_transform" =
  let open Sklearn.Preprocessing in
  let data = matrixi [|[|1; 2|]; [|3; 2|]; [|4; 5|]|] in
  print_ndarray @@ power_transform ~x:data ~method_:`Box_cox ();
  [%expect {|
            [[-1.33269291 -0.70710678]
             [ 0.25653283 -0.70710678]
             [ 1.07616008  1.41421356]]
    |}]


(* quantile_transform *)
(*
>>> import numpy as np
>>> from sklearn.preprocessing import quantile_transform
>>> rng = np.random.RandomState(0)
>>> X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
>>> quantile_transform(X, n_quantiles=10, random_state=0, copy=True)
array([...])


*)

let%expect_test "quantile_transform" =
  let open Sklearn.Preprocessing in
  let module Matrix = Owl.Dense.Matrix.D in
  Owl_stats_prng.init 0;
  let x = Matrix.(gaussian ~mu:0.5 ~sigma:0.25 25 1 |> sort) |> Sklearn.Arr.of_bigarray in
  print_ndarray @@ quantile_transform ~x ~n_quantiles:10 ~random_state:0 ~copy:true ();
  [%expect {|
            [[0.        ]
             [0.0465588 ]
             [0.07236635]
             [0.13435383]
             [0.19038255]
             [0.20744256]
             [0.27077313]
             [0.3072948 ]
             [0.33333333]
             [0.40459044]
             [0.4263402 ]
             [0.46393428]
             [0.52235714]
             [0.54881137]
             [0.59860807]
             [0.60721673]
             [0.66666667]
             [0.69638456]
             [0.74844166]
             [0.78237337]
             [0.80823856]
             [0.86774379]
             [0.90414956]
             [0.95992288]
             [1.        ]]
    |}]
