(* KNNImputer *)
(*
>>> import numpy as np
>>> from sklearn.impute import KNNImputer
>>> X = [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]
>>> imputer = KNNImputer(n_neighbors=2)
>>> imputer.fit_transform(X)
array([[1. , 2. , 4. ],
       [3. , 4. , 3. ],
       [5.5, 6. , 5. ],

*)

(* TEST TODO
let%expect_test "KNNImputer" =
  let open Sklearn.Impute in
  let x = (matrix [|[|1; 2; np.nan|]; [|3; 4; 3|]; [|np.nan; 6; 5|]; [|8; 8; 7|]|]) in  
  let imputer = KNNImputer.create ~n_neighbors:2 () in  
  print_ndarray @@ KNNImputer.fit_transform ~x imputer;  
  [%expect {|
      array([[1. , 2. , 4. ],      
             [3. , 4. , 3. ],      
             [5.5, 6. , 5. ],      
  |}]

*)



(* MissingIndicator *)
(*
>>> import numpy as np
>>> from sklearn.impute import MissingIndicator
>>> X1 = np.array([[np.nan, 1, 3],
...                [4, 0, np.nan],
...                [8, 1, 0]])
>>> X2 = np.array([[5, 1, np.nan],
...                [np.nan, 2, 3],
...                [2, 4, 0]])
>>> indicator = MissingIndicator()
>>> indicator.fit(X1)
MissingIndicator()
>>> X2_tr = indicator.transform(X2)
>>> X2_tr
array([[False,  True],
       [ True, False],

*)

(* TEST TODO
let%expect_test "MissingIndicator" =
  let open Sklearn.Impute in
  let X1 = .array [[np.nan ~1 3] [4 ~0 np.nan] (vectori [|8; 1; 0|])] np in  
  let X2 = .array [[5 ~1 np.nan] [np.nan ~2 3] (vectori [|2; 4; 0|])] np in  
  let indicator = MissingIndicator.create () in  
  print MissingIndicator.pp @@ MissingIndicator.fit ~X1 indicator;  
  [%expect {|
      MissingIndicator()      
  |}]
  let X2_tr = MissingIndicator.transform ~X2 indicator in  
  print_ndarray @@ X2_tr;  
  [%expect {|
      array([[False,  True],      
             [ True, False],      
  |}]

*)



(* SimpleImputer *)
(*
>>> import numpy as np
>>> from sklearn.impute import SimpleImputer
>>> imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
>>> imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
SimpleImputer()
>>> X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
>>> print(imp_mean.transform(X))
[[ 7.   2.   3. ]
 [ 4.   3.5  6. ]
 [10.   3.5  9. ]]

*)

(* TEST TODO
let%expect_test "SimpleImputer" =
  let open Sklearn.Impute in
  let imp_mean = SimpleImputer.create ~missing_values:np.nan ~strategy:'mean' () in  
  print SimpleImputer.pp @@ SimpleImputer.fit (matrix [|[|7; 2; 3|]; [|4; np.nan; 6|]; [|10; 5; 9|]|]) imp_mean;  
  [%expect {|
      SimpleImputer()      
  |}]
  let x = (matrix [|[|np.nan; 2; 3|]; [|4; np.nan; 6|]; [|10; np.nan; 9|]|]) in  
  print_ndarray @@ print SimpleImputer.transform x () imp_mean;  
  [%expect {|
      [[ 7.   2.   3. ]      
       [ 4.   3.5  6. ]      
       [10.   3.5  9. ]]      
  |}]

*)



