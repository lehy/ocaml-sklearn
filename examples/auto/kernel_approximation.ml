(* AdditiveChi2Sampler *)
(*
>>> from sklearn.datasets import load_digits
>>> from sklearn.linear_model import SGDClassifier
>>> from sklearn.kernel_approximation import AdditiveChi2Sampler
>>> X, y = load_digits(return_X_y=True)
>>> chi2sampler = AdditiveChi2Sampler(sample_steps=2)
>>> X_transformed = chi2sampler.fit_transform(X, y)
>>> clf = SGDClassifier(max_iter=5, random_state=0, tol=1e-3)
>>> clf.fit(X_transformed, y)
SGDClassifier(max_iter=5, random_state=0)
>>> clf.score(X_transformed, y)
0.9499...

*)

(* TEST TODO
let%expect_test "AdditiveChi2Sampler" =
  let open Sklearn.Kernel_approximation in
  let x, y = load_digits ~return_X_y:true () in  
  let chi2sampler = AdditiveChi2Sampler(sample_steps=2) in  
  let X_transformed = chi2sampler.fit_transform ~x y () in  
  let clf = SGDClassifier.create ~max_iter:5 ~random_state:(`Int 0) ~tol:1e-3 () in  
  print SGDClassifier.pp @@ SGDClassifier.fit ~X_transformed y clf;  
  [%expect {|
      SGDClassifier(max_iter=5, random_state=0)      
  |}]
  print_ndarray @@ SGDClassifier.score ~X_transformed y clf;  
  [%expect {|
      0.9499...      
  |}]

*)



(* Nystroem *)
(*
>>> from sklearn import datasets, svm
>>> from sklearn.kernel_approximation import Nystroem
>>> X, y = datasets.load_digits(n_class=9, return_X_y=True)
>>> data = X / 16.
>>> clf = svm.LinearSVC()
>>> feature_map_nystroem = Nystroem(gamma=.2,
...                                 random_state=1,
...                                 n_components=300)
>>> data_transformed = feature_map_nystroem.fit_transform(data)
>>> clf.fit(data_transformed, y)
LinearSVC()
>>> clf.score(data_transformed, y)
0.9987...

*)

(* TEST TODO
let%expect_test "Nystroem" =
  let open Sklearn.Kernel_approximation in
  let x, y = .load_digits ~n_class:9 ~return_X_y:true datasets in  
  let data = x / 16. in  
  let clf = .linearSVC svm in  
  let feature_map_nystroem = Nystroem.create ~gamma:.2 ~random_state:(`Int 1) ~n_components:300 () in  
  let data_transformed = Nystroem.fit_transform ~data feature_map_nystroem in  
  print_ndarray @@ .fit ~data_transformed y clf;  
  [%expect {|
      LinearSVC()      
  |}]
  print_ndarray @@ .score ~data_transformed y clf;  
  [%expect {|
      0.9987...      
  |}]

*)



(* RBFSampler *)
(*
>>> from sklearn.kernel_approximation import RBFSampler
>>> from sklearn.linear_model import SGDClassifier
>>> X = [[0, 0], [1, 1], [1, 0], [0, 1]]
>>> y = [0, 0, 1, 1]
>>> rbf_feature = RBFSampler(gamma=1, random_state=1)
>>> X_features = rbf_feature.fit_transform(X)
>>> clf = SGDClassifier(max_iter=5, tol=1e-3)
>>> clf.fit(X_features, y)
SGDClassifier(max_iter=5)
>>> clf.score(X_features, y)
1.0

*)

(* TEST TODO
let%expect_test "RBFSampler" =
  let open Sklearn.Kernel_approximation in
  let x = (matrixi [|[|0; 0|]; [|1; 1|]; [|1; 0|]; [|0; 1|]|]) in  
  let y = (vectori [|0; 0; 1; 1|]) in  
  let rbf_feature = RBFSampler.create ~gamma:1 ~random_state:(`Int 1) () in  
  let X_features = RBFSampler.fit_transform ~x rbf_feature in  
  let clf = SGDClassifier.create ~max_iter:5 ~tol:1e-3 () in  
  print SGDClassifier.pp @@ SGDClassifier.fit ~X_features y clf;  
  [%expect {|
      SGDClassifier(max_iter=5)      
  |}]
  print_ndarray @@ SGDClassifier.score ~X_features y clf;  
  [%expect {|
      1.0      
  |}]

*)



(* SkewedChi2Sampler *)
(*
>>> from sklearn.kernel_approximation import SkewedChi2Sampler
>>> from sklearn.linear_model import SGDClassifier
>>> X = [[0, 0], [1, 1], [1, 0], [0, 1]]
>>> y = [0, 0, 1, 1]
>>> chi2_feature = SkewedChi2Sampler(skewedness=.01,
...                                  n_components=10,
...                                  random_state=0)
>>> X_features = chi2_feature.fit_transform(X, y)
>>> clf = SGDClassifier(max_iter=10, tol=1e-3)
>>> clf.fit(X_features, y)
SGDClassifier(max_iter=10)
>>> clf.score(X_features, y)
1.0

*)

(* TEST TODO
let%expect_test "SkewedChi2Sampler" =
  let open Sklearn.Kernel_approximation in
  let x = (matrixi [|[|0; 0|]; [|1; 1|]; [|1; 0|]; [|0; 1|]|]) in  
  let y = (vectori [|0; 0; 1; 1|]) in  
  let chi2_feature = SkewedChi2Sampler(skewedness=.01,n_components=10,random_state=0) in  
  let X_features = chi2_feature.fit_transform ~x y () in  
  let clf = SGDClassifier.create ~max_iter:10 ~tol:1e-3 () in  
  print SGDClassifier.pp @@ SGDClassifier.fit ~X_features y clf;  
  [%expect {|
      SGDClassifier(max_iter=10)      
  |}]
  print_ndarray @@ SGDClassifier.score ~X_features y clf;  
  [%expect {|
      1.0      
  |}]

*)



(* svd *)
(*
>>> from scipy import linalg
>>> m, n = 9, 6
>>> a = np.random.randn(m, n) + 1.j*np.random.randn(m, n)
>>> U, s, Vh = linalg.svd(a)
>>> U.shape,  s.shape, Vh.shape
((9, 9), (6,), (6, 6))

*)

(* TEST TODO
let%expect_test "svd" =
  let open Sklearn.Kernel_approximation in
  let m, n = 9, 6 in  
  let a = np..randn ~m n) + 1.j*np.random.randn(m ~n random in  
  let U, s, Vh = .svd ~a linalg in  
  print_ndarray @@ U.shape, s.shape, Vh.shape;  
  [%expect {|
      ((9, 9), (6,), (6, 6))      
  |}]

*)



(* svd *)
(*
>>> sigma = np.zeros((m, n))
>>> for i in range(min(m, n)):
...     sigma[i, i] = s[i]
>>> a1 = np.dot(U, np.dot(sigma, Vh))
>>> np.allclose(a, a1)
True

*)

(* TEST TODO
let%expect_test "svd" =
  let open Sklearn.Kernel_approximation in
  let sigma = .zeros (m n) np in  
  print_ndarray @@ for i in range(min ~m n ()):sigma vectori [|i; i|] () = s(vectori [|i|]);  
  let a1 = .dot ~U np.dot sigma Vh () np in  
  print_ndarray @@ .allclose ~a a1 np;  
  [%expect {|
      True      
  |}]

*)



(* svd *)
(*
>>> U, s, Vh = linalg.svd(a, full_matrices=False)
>>> U.shape, s.shape, Vh.shape
((9, 6), (6,), (6, 6))
>>> S = np.diag(s)
>>> np.allclose(a, np.dot(U, np.dot(S, Vh)))
True

*)

(* TEST TODO
let%expect_test "svd" =
  let open Sklearn.Kernel_approximation in
  let U, s, Vh = .svd a ~full_matrices:false linalg in  
  print_ndarray @@ U.shape, s.shape, Vh.shape;  
  [%expect {|
      ((9, 6), (6,), (6, 6))      
  |}]
  let S = .diag ~s np in  
  print_ndarray @@ .allclose ~a np.dot(U np.dot S Vh ()) np;  
  [%expect {|
      True      
  |}]

*)



(* svd *)
(*
>>> s2 = linalg.svd(a, compute_uv=False)
>>> np.allclose(s, s2)

*)

(* TEST TODO
let%expect_test "svd" =
  let open Sklearn.Kernel_approximation in
  let s2 = .svd a ~compute_uv:false linalg in  
  print_ndarray @@ .allclose ~s s2 np;  
  [%expect {|
  |}]

*)



