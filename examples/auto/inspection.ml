(* partial_dependence *)
(*
>>> X = [[0, 0, 2], [1, 0, 0]]
>>> y = [0, 1]
>>> from sklearn.ensemble import GradientBoostingClassifier
>>> gb = GradientBoostingClassifier(random_state=0).fit(X, y)
>>> partial_dependence(gb, features=[0], X=X, percentiles=(0, 1),
...                    grid_resolution=2) # doctest: +SKIP
(array([[-4.52...,  4.52...]]), [array([ 0.,  1.])])

*)

(* TEST TODO
let%expect_test "partial_dependence" =
  let open Sklearn.Inspection in
  let x = (matrixi [|[|0; 0; 2|]; [|1; 0; 0|]|]) in  
  let y = (vectori [|0; 1|]) in  
  let gb = GradientBoostingClassifier(random_state=0).fit ~x y () in  
  print_ndarray @@ partial_dependence(gb, features=(vectori [|0|]), x=x, percentiles=(0, 1),grid_resolution=2) # doctest: +SKIP;  
  [%expect {|
      (array([[-4.52...,  4.52...]]), [array([ 0.,  1.])])      
  |}]

*)



(* plot_partial_dependence *)
(*
>>> from sklearn.datasets import make_friedman1
>>> from sklearn.ensemble import GradientBoostingRegressor
>>> X, y = make_friedman1()
>>> clf = GradientBoostingRegressor(n_estimators=10).fit(X, y)
>>> plot_partial_dependence(clf, X, [0, (0, 1)]) #doctest: +SKIP

*)

(* TEST TODO
let%expect_test "plot_partial_dependence" =
  let open Sklearn.Inspection in
  let x, y = make_friedman1() in  
  let clf = GradientBoostingRegressor(n_estimators=10).fit ~x y () in  
  print_ndarray @@ plot_partial_dependence(clf, x, [0, (0, 1)]) #doctest: +SKIP;  
  [%expect {|
  |}]

*)



