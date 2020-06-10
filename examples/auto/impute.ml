module Np = Np.Numpy

let print f x = Format.printf "%a" f x

let print_py x = Format.printf "%s" (Py.Object.to_string x)

let print_ndarray = Np.Obj.print

let print_float = Format.printf "%g\n"

let print_string = Format.printf "%s\n"

let print_int = Format.printf "%d\n"

module Arr = Sklearn.Arr

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


let%expect_test "KNNImputer" =
  let open Sklearn.Impute in
  let x = Np.Ndarray.matrixf [|[|1.; 2.; nan|]; [|3.; 4.; 3.|]; [|nan; 6.; 5.|]; [|8.; 8.; 7.|]|] in
  let imputer = KNNImputer.create ~n_neighbors:2 () in
  print_ndarray @@ KNNImputer.fit_transform ~x imputer;
  [%expect {|
      [[1.  2.  4. ]
       [3.  4.  3. ]
       [5.5 6.  5. ]
       [8.  8.  7. ]]
  |}]


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

let%expect_test "MissingIndicator" =
  let open Sklearn.Impute in
  let x1 = Np.matrixf [|[|nan; 1.; 3.|]; [|4.; 0.; nan|]; [|8.; 1.; 0.|]|] in
  let x2 = Np.matrixf [|[|5.; 1.; nan|]; [|nan; 2.; 3.|]; [|2.; 4.; 0.|]|] in
  let indicator = MissingIndicator.create () in
  print MissingIndicator.pp @@ MissingIndicator.fit ~x:x1 indicator;
  [%expect {|
      MissingIndicator(error_on_new=True, features='missing-only', missing_values=nan,
                       sparse='auto')
  |}];
  let x2_tr = MissingIndicator.transform ~x:x2 indicator in
  print_ndarray @@ x2_tr;
  [%expect {|
      [[False  True]
       [ True False]
       [False False]]
  |}]


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

let%expect_test "SimpleImputer" =
  let open Sklearn.Impute in
  let imp_mean = SimpleImputer.create ~missing_values:(`F nan) ~strategy:`Mean () in
  print SimpleImputer.pp @@ SimpleImputer.fit ~x:(Np.matrixf [|[|7.; 2.; 3.|]; [|4.; nan; 6.|]; [|10.; 5.; 9.|]|]) imp_mean;
  [%expect {|
      SimpleImputer(add_indicator=False, copy=True, fill_value=None,
                    missing_values=nan, strategy='mean', verbose=0)
  |}];
  let x = Np.matrixf [|[|nan; 2.; 3.|]; [|4.; nan; 6.|]; [|10.; nan; 9.|]|] in
  print_ndarray @@ SimpleImputer.transform ~x imp_mean;
  [%expect {|
      [[ 7.   2.   3. ]
       [ 4.   3.5  6. ]
       [10.   3.5  9. ]]
  |}]
