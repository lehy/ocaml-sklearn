module Np = Np.Numpy

let print f x = Format.printf "%a" f x

let print_py x = Format.printf "%s" (Py.Object.to_string x)

let print_ndarray = Np.Obj.print

let print_float = Format.printf "%g\n"

let print_string = Format.printf "%s\n"

let print_int = Format.printf "%d\n"

let matrixi = Np.Ndarray.matrixi

let matrixf = Np.Ndarray.matrixf

(* IsotonicRegression *)
(*
>>> from sklearn.datasets import make_regression
>>> from sklearn.isotonic import IsotonicRegression
>>> X, y = make_regression(n_samples=10, n_features=1, random_state=41)
>>> iso_reg = IsotonicRegression().fit(X.flatten(), y)
>>> iso_reg.predict([.1, .2])

*)

let%expect_test "IsotonicRegression" =
  let open Sklearn.Calibration in
  let x, y, _ = Sklearn.Datasets.make_regression ~n_samples:10 ~n_features:1 ~random_state:41 () in  
  let iso_reg = IsotonicRegression.(create () |> fit ~x:(Np.Ndarray.flatten x) ~y) in
  print_ndarray @@ IsotonicRegression.predict ~t:(Np.Ndarray.vectorf [|0.1; 0.2|]) iso_reg;  
  [%expect {| [1.86282267 3.72564535] |}]
