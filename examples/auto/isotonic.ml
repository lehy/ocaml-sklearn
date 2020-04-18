(* IsotonicRegression *)
(*
>>> from sklearn.datasets import make_regression
>>> from sklearn.isotonic import IsotonicRegression
>>> X, y = make_regression(n_samples=10, n_features=1, random_state=41)
>>> iso_reg = IsotonicRegression().fit(X.flatten(), y)
>>> iso_reg.predict([.1, .2])

*)

(* TEST TODO
let%expect_test "IsotonicRegression" =
  let open Sklearn.Isotonic in
  let x, y = make_regression ~n_samples:10 ~n_features:1 ~random_state:(`Int 41) () in  
  let iso_reg = IsotonicRegression().fit(x.flatten (), y) in  
  print_ndarray @@ IsotonicRegression.predict [.1 .2] iso_reg;  
  [%expect {|
  |}]

*)



(* spearmanr *)
(*
>>> from scipy import stats
>>> stats.spearmanr([1,2,3,4,5], [5,6,7,8,7])
(0.82078268166812329, 0.088587005313543798)
>>> np.random.seed(1234321)
>>> x2n = np.random.randn(100, 2)
>>> y2n = np.random.randn(100, 2)
>>> stats.spearmanr(x2n)
(0.059969996999699973, 0.55338590803773591)
>>> stats.spearmanr(x2n[:,0], x2n[:,1])
(0.059969996999699973, 0.55338590803773591)
>>> rho, pval = stats.spearmanr(x2n, y2n)
>>> rho
array([[ 1.        ,  0.05997   ,  0.18569457,  0.06258626],
       [ 0.05997   ,  1.        ,  0.110003  ,  0.02534653],
       [ 0.18569457,  0.110003  ,  1.        ,  0.03488749],
       [ 0.06258626,  0.02534653,  0.03488749,  1.        ]])
>>> pval
array([[ 0.        ,  0.55338591,  0.06435364,  0.53617935],
       [ 0.55338591,  0.        ,  0.27592895,  0.80234077],
       [ 0.06435364,  0.27592895,  0.        ,  0.73039992],
       [ 0.53617935,  0.80234077,  0.73039992,  0.        ]])
>>> rho, pval = stats.spearmanr(x2n.T, y2n.T, axis=1)
>>> rho
array([[ 1.        ,  0.05997   ,  0.18569457,  0.06258626],
       [ 0.05997   ,  1.        ,  0.110003  ,  0.02534653],
       [ 0.18569457,  0.110003  ,  1.        ,  0.03488749],
       [ 0.06258626,  0.02534653,  0.03488749,  1.        ]])
>>> stats.spearmanr(x2n, y2n, axis=None)
(0.10816770419260482, 0.1273562188027364)
>>> stats.spearmanr(x2n.ravel(), y2n.ravel())
(0.10816770419260482, 0.1273562188027364)

*)

(* TEST TODO
let%expect_test "spearmanr" =
  let open Sklearn.Isotonic in
  print_ndarray @@ .spearmanr (vectori [|1;2;3;4;5|]) (vectori [|5;6;7;8;7|]) stats;  
  [%expect {|
      (0.82078268166812329, 0.088587005313543798)      
  |}]
  print_ndarray @@ np..seed ~1234321 random;  
  let x2n = np..randn ~100 2 random in  
  let y2n = np..randn ~100 2 random in  
  print_ndarray @@ .spearmanr ~x2n stats;  
  [%expect {|
      (0.059969996999699973, 0.55338590803773591)      
  |}]
  print_ndarray @@ .spearmanr x2n(vectori [|:;0|]) x2n(vectori [|:;1|]) stats;  
  [%expect {|
      (0.059969996999699973, 0.55338590803773591)      
  |}]
  let rho, pval = .spearmanr ~x2n y2n stats in  
  print_ndarray @@ rho;  
  [%expect {|
      array([[ 1.        ,  0.05997   ,  0.18569457,  0.06258626],      
             [ 0.05997   ,  1.        ,  0.110003  ,  0.02534653],      
             [ 0.18569457,  0.110003  ,  1.        ,  0.03488749],      
             [ 0.06258626,  0.02534653,  0.03488749,  1.        ]])      
  |}]
  print_ndarray @@ pval;  
  [%expect {|
      array([[ 0.        ,  0.55338591,  0.06435364,  0.53617935],      
             [ 0.55338591,  0.        ,  0.27592895,  0.80234077],      
             [ 0.06435364,  0.27592895,  0.        ,  0.73039992],      
             [ 0.53617935,  0.80234077,  0.73039992,  0.        ]])      
  |}]
  let rho, pval = .spearmanr x2n.T y2n.T ~axis:1 stats in  
  print_ndarray @@ rho;  
  [%expect {|
      array([[ 1.        ,  0.05997   ,  0.18569457,  0.06258626],      
             [ 0.05997   ,  1.        ,  0.110003  ,  0.02534653],      
             [ 0.18569457,  0.110003  ,  1.        ,  0.03488749],      
             [ 0.06258626,  0.02534653,  0.03488749,  1.        ]])      
  |}]
  print_ndarray @@ .spearmanr ~x2n y2n ~axis:None stats;  
  [%expect {|
      (0.10816770419260482, 0.1273562188027364)      
  |}]
  print_ndarray @@ .spearmanr x2n.ravel () y2n.ravel () stats;  
  [%expect {|
      (0.10816770419260482, 0.1273562188027364)      
  |}]

*)



(* spearmanr *)
(*
>>> xint = np.random.randint(10, size=(100, 2))
>>> stats.spearmanr(xint)

*)

(* TEST TODO
let%expect_test "spearmanr" =
  let open Sklearn.Isotonic in
  let xint = np..randint 10 ~size:(100 2) random in  
  print_ndarray @@ .spearmanr ~xint stats;  
  [%expect {|
  |}]

*)



