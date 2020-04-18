(* LinearDiscriminantAnalysis *)
(*
>>> import numpy as np
>>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
>>> y = np.array([1, 1, 1, 2, 2, 2])
>>> clf = LinearDiscriminantAnalysis()
>>> clf.fit(X, y)
LinearDiscriminantAnalysis()
>>> print(clf.predict([[-0.8, -1]]))

*)

(* TEST TODO
let%expect_test "LinearDiscriminantAnalysis" =
  let open Sklearn.Discriminant_analysis in
  let x = .array (matrixi [|[|-1; -1|]; [|-2; -1|]; [|-3; -2|]; [|1; 1|]; [|2; 1|]; [|3; 2|]|]) np in  
  let y = .array (vectori [|1; 1; 1; 2; 2; 2|]) np in  
  let clf = LinearDiscriminantAnalysis.create () in  
  print LinearDiscriminantAnalysis.pp @@ LinearDiscriminantAnalysis.fit ~x y clf;  
  [%expect {|
      LinearDiscriminantAnalysis()      
  |}]
  print_ndarray @@ print(LinearDiscriminantAnalysis.predict (matrix [|[|-0.8; -1|]|])) clf;  
  [%expect {|
  |}]

*)



(* QuadraticDiscriminantAnalysis *)
(*
>>> from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
>>> import numpy as np
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
>>> y = np.array([1, 1, 1, 2, 2, 2])
>>> clf = QuadraticDiscriminantAnalysis()
>>> clf.fit(X, y)
QuadraticDiscriminantAnalysis()
>>> print(clf.predict([[-0.8, -1]]))
[1]

*)

(* TEST TODO
let%expect_test "QuadraticDiscriminantAnalysis" =
  let open Sklearn.Discriminant_analysis in
  let x = .array (matrixi [|[|-1; -1|]; [|-2; -1|]; [|-3; -2|]; [|1; 1|]; [|2; 1|]; [|3; 2|]|]) np in  
  let y = .array (vectori [|1; 1; 1; 2; 2; 2|]) np in  
  let clf = QuadraticDiscriminantAnalysis.create () in  
  print QuadraticDiscriminantAnalysis.pp @@ QuadraticDiscriminantAnalysis.fit ~x y clf;  
  [%expect {|
      QuadraticDiscriminantAnalysis()      
  |}]
  print_ndarray @@ print(QuadraticDiscriminantAnalysis.predict (matrix [|[|-0.8; -1|]|])) clf;  
  [%expect {|
      [1]      
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
  let open Sklearn.Discriminant_analysis in
  let data = (matrixi [|[|0; 0|]; [|0; 0|]; [|1; 1|]; [|1; 1|]|]) in  
  let scaler = StandardScaler.create () in  
  print_ndarray @@ print StandardScaler.fit data () scaler;  
  [%expect {|
      StandardScaler()      
  |}]
  print_ndarray @@ print scaler.mean_ ();  
  [%expect {|
      [0.5 0.5]      
  |}]
  print_ndarray @@ print StandardScaler.transform data () scaler;  
  [%expect {|
      [[-1. -1.]      
       [-1. -1.]      
       [ 1.  1.]      
       [ 1.  1.]]      
  |}]
  print_ndarray @@ print(StandardScaler.transform (matrixi [|[|2; 2|]|])) scaler;  
  [%expect {|
      [[3. 3.]]      
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
  let open Sklearn.Discriminant_analysis in
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



