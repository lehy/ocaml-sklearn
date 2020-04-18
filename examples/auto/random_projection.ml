(* GaussianRandomProjection *)
(*
>>> import numpy as np
>>> from sklearn.random_projection import GaussianRandomProjection
>>> rng = np.random.RandomState(42)
>>> X = rng.rand(100, 10000)
>>> transformer = GaussianRandomProjection(random_state=rng)
>>> X_new = transformer.fit_transform(X)
>>> X_new.shape
(100, 3947)

*)

(* TEST TODO
let%expect_test "GaussianRandomProjection" =
  let open Sklearn.Random_projection in
  let rng = np..randomState ~42 random in  
  let x = .rand ~100 10000 rng in  
  let transformer = GaussianRandomProjection.create ~random_state:rng () in  
  let X_new = GaussianRandomProjection.fit_transform ~x transformer in  
  print_ndarray @@ X_new.shape;  
  [%expect {|
      (100, 3947)      
  |}]

*)



(* SparseRandomProjection *)
(*
>>> import numpy as np
>>> from sklearn.random_projection import SparseRandomProjection
>>> rng = np.random.RandomState(42)
>>> X = rng.rand(100, 10000)
>>> transformer = SparseRandomProjection(random_state=rng)
>>> X_new = transformer.fit_transform(X)
>>> X_new.shape
(100, 3947)
>>> # very few components are non-zero
>>> np.mean(transformer.components_ != 0)
0.0100...

*)

(* TEST TODO
let%expect_test "SparseRandomProjection" =
  let open Sklearn.Random_projection in
  let rng = np..randomState ~42 random in  
  let x = .rand ~100 10000 rng in  
  let transformer = SparseRandomProjection.create ~random_state:rng () in  
  let X_new = SparseRandomProjection.fit_transform ~x transformer in  
  print_ndarray @@ X_new.shape;  
  [%expect {|
      (100, 3947)      
  |}]
  print_ndarray @@ # very few components are non-zero;  
  print_ndarray @@ .mean transformer.components_ != 0 np;  
  [%expect {|
      0.0100...      
  |}]

*)



(* deprecated *)
(*
>>> from sklearn.utils import deprecated
>>> deprecated()
<sklearn.utils.deprecation.deprecated object at ...>

*)

(* TEST TODO
let%expect_test "deprecated" =
  let open Sklearn.Random_projection in
  print_ndarray @@ deprecated ();  
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
let%expect_test "deprecated" =
  let open Sklearn.Random_projection in
  print_ndarray @@ @deprecated ()def some_function (): pass;  
  [%expect {|
  |}]

*)



(* johnson_lindenstrauss_min_dim *)
(*
>>> johnson_lindenstrauss_min_dim(1e6, eps=0.5)
663

*)

(* TEST TODO
let%expect_test "johnson_lindenstrauss_min_dim" =
  let open Sklearn.Random_projection in
  print_ndarray @@ johnson_lindenstrauss_min_dim 1e6 ~eps:0.5 ();  
  [%expect {|
      663      
  |}]

*)



(* johnson_lindenstrauss_min_dim *)
(*
>>> johnson_lindenstrauss_min_dim(1e6, eps=[0.5, 0.1, 0.01])
array([    663,   11841, 1112658])

*)

(* TEST TODO
let%expect_test "johnson_lindenstrauss_min_dim" =
  let open Sklearn.Random_projection in
  print_ndarray @@ johnson_lindenstrauss_min_dim 1e6 ~eps:[0.5 0.1 0.01] ();  
  [%expect {|
      array([    663,   11841, 1112658])      
  |}]

*)



(* johnson_lindenstrauss_min_dim *)
(*
>>> johnson_lindenstrauss_min_dim([1e4, 1e5, 1e6], eps=0.1)
array([ 7894,  9868, 11841])

*)

(* TEST TODO
let%expect_test "johnson_lindenstrauss_min_dim" =
  let open Sklearn.Random_projection in
  print_ndarray @@ johnson_lindenstrauss_min_dim [1e4 ~1e5 1e6] ~eps:0.1 ();  
  [%expect {|
      array([ 7894,  9868, 11841])      
  |}]

*)



