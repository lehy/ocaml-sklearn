(* FactorAnalysis *)
(*
>>> from sklearn.datasets import load_digits
>>> from sklearn.decomposition import FactorAnalysis
>>> X, _ = load_digits(return_X_y=True)
>>> transformer = FactorAnalysis(n_components=7, random_state=0)
>>> X_transformed = transformer.fit_transform(X)
>>> X_transformed.shape
(1797, 7)

*)

(* TEST TODO
let%expect_test "FactorAnalysis" =
  let open Sklearn.Decomposition in
  let x, _ = load_digits ~return_X_y:true () in  
  let transformer = FactorAnalysis.create ~n_components:7 ~random_state:0 () in  
  let X_transformed = FactorAnalysis.fit_transform ~x transformer in  
  print_ndarray @@ X_transformed.shape;  
  [%expect {|
      (1797, 7)      
  |}]

*)



(* FastICA *)
(*
>>> from sklearn.datasets import load_digits
>>> from sklearn.decomposition import FastICA
>>> X, _ = load_digits(return_X_y=True)
>>> transformer = FastICA(n_components=7,
...         random_state=0)
>>> X_transformed = transformer.fit_transform(X)
>>> X_transformed.shape
(1797, 7)

*)

(* TEST TODO
let%expect_test "FastICA" =
  let open Sklearn.Decomposition in
  let x, _ = load_digits ~return_X_y:true () in  
  let transformer = FastICA.create ~n_components:7 ~random_state:0 () in  
  let X_transformed = FastICA.fit_transform ~x transformer in  
  print_ndarray @@ X_transformed.shape;  
  [%expect {|
      (1797, 7)      
  |}]

*)



(* IncrementalPCA *)
(*
>>> from sklearn.datasets import load_digits
>>> from sklearn.decomposition import IncrementalPCA
>>> from scipy import sparse
>>> X, _ = load_digits(return_X_y=True)
>>> transformer = IncrementalPCA(n_components=7, batch_size=200)
>>> # either partially fit on smaller batches of data
>>> transformer.partial_fit(X[:100, :])
IncrementalPCA(batch_size=200, n_components=7)
>>> # or let the fit function itself divide the data into batches
>>> X_sparse = sparse.csr_matrix(X)
>>> X_transformed = transformer.fit_transform(X_sparse)
>>> X_transformed.shape
(1797, 7)

*)

(* TEST TODO
let%expect_test "IncrementalPCA" =
  let open Sklearn.Decomposition in
  let x, _ = load_digits ~return_X_y:true () in  
  let transformer = IncrementalPCA.create ~n_components:7 ~batch_size:200 () in  
  print_ndarray @@ # either partially fit on smaller batches of data;  
  print_ndarray @@ IncrementalPCA.partial_fit x[:100 :] transformer;  
  [%expect {|
      IncrementalPCA(batch_size=200, n_components=7)      
  |}]
  print_ndarray @@ # or let the fit function itself divide the data into batches;  
  let X_sparse = .csr_matrix ~x sparse in  
  let X_transformed = IncrementalPCA.fit_transform ~X_sparse transformer in  
  print_ndarray @@ X_transformed.shape;  
  [%expect {|
      (1797, 7)      
  |}]

*)



(* transform *)
(*
>>> import numpy as np
>>> from sklearn.decomposition import IncrementalPCA
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2],
...               [1, 1], [2, 1], [3, 2]])
>>> ipca = IncrementalPCA(n_components=2, batch_size=3)
>>> ipca.fit(X)
IncrementalPCA(batch_size=3, n_components=2)

*)

(* TEST TODO
let%expect_test "IncrementalPCA.transform" =
  let open Sklearn.Decomposition in
  let x = .array [[-1 -1] [-2 -1] [-3 -2] (vectori [|1; 1|]) (vectori [|2; 1|]) (vectori [|3; 2|])] np in  
  let ipca = IncrementalPCA.create ~n_components:2 ~batch_size:3 () in  
  print IncrementalPCA.pp @@ IncrementalPCA.fit ~x ipca;  
  [%expect {|
      IncrementalPCA(batch_size=3, n_components=2)      
  |}]

*)



(* KernelPCA *)
(*
>>> from sklearn.datasets import load_digits
>>> from sklearn.decomposition import KernelPCA
>>> X, _ = load_digits(return_X_y=True)
>>> transformer = KernelPCA(n_components=7, kernel='linear')
>>> X_transformed = transformer.fit_transform(X)
>>> X_transformed.shape
(1797, 7)

*)

(* TEST TODO
let%expect_test "KernelPCA" =
  let open Sklearn.Decomposition in
  let x, _ = load_digits ~return_X_y:true () in  
  let transformer = KernelPCA.create ~n_components:7 ~kernel:'linear' () in  
  let X_transformed = KernelPCA.fit_transform ~x transformer in  
  print_ndarray @@ X_transformed.shape;  
  [%expect {|
      (1797, 7)      
  |}]

*)



(* LatentDirichletAllocation *)
(*
>>> from sklearn.decomposition import LatentDirichletAllocation
>>> from sklearn.datasets import make_multilabel_classification
>>> # This produces a feature matrix of token counts, similar to what
>>> # CountVectorizer would produce on text.
>>> X, _ = make_multilabel_classification(random_state=0)
>>> lda = LatentDirichletAllocation(n_components=5,
...     random_state=0)
>>> lda.fit(X)
LatentDirichletAllocation(...)
>>> # get topics for some given samples:
>>> lda.transform(X[-2:])
array([[0.00360392, 0.25499205, 0.0036211 , 0.64236448, 0.09541846],
       [0.15297572, 0.00362644, 0.44412786, 0.39568399, 0.003586  ]])

*)

(* TEST TODO
let%expect_test "LatentDirichletAllocation" =
  let open Sklearn.Decomposition in
  print_ndarray @@ # This produces a feature matrix of token counts, similar to what;  
  print_ndarray @@ # CountVectorizer would produce on text.;  
  let x, _ = make_multilabel_classification ~random_state:0 () in  
  let lda = LatentDirichletAllocation.create ~n_components:5 ~random_state:0 () in  
  print LatentDirichletAllocation.pp @@ LatentDirichletAllocation.fit ~x lda;  
  [%expect {|
      LatentDirichletAllocation(...)      
  |}]
  print_ndarray @@ # get topics for some given samples:;  
  print_ndarray @@ LatentDirichletAllocation.transform x[-2:] lda;  
  [%expect {|
      array([[0.00360392, 0.25499205, 0.0036211 , 0.64236448, 0.09541846],      
             [0.15297572, 0.00362644, 0.44412786, 0.39568399, 0.003586  ]])      
  |}]

*)



(* MiniBatchSparsePCA *)
(*
>>> import numpy as np
>>> from sklearn.datasets import make_friedman1
>>> from sklearn.decomposition import MiniBatchSparsePCA
>>> X, _ = make_friedman1(n_samples=200, n_features=30, random_state=0)
>>> transformer = MiniBatchSparsePCA(n_components=5, batch_size=50,
...                                  random_state=0)
>>> transformer.fit(X)
MiniBatchSparsePCA(...)
>>> X_transformed = transformer.transform(X)
>>> X_transformed.shape
(200, 5)
>>> # most values in the components_ are zero (sparsity)
>>> np.mean(transformer.components_ == 0)
0.94

*)

(* TEST TODO
let%expect_test "MiniBatchSparsePCA" =
  let open Sklearn.Decomposition in
  let x, _ = make_friedman1(n_samples=200, n_features=30, random_state=0) in  
  let transformer = MiniBatchSparsePCA.create ~n_components:5 ~batch_size:50 ~random_state:0 () in  
  print MiniBatchSparsePCA.pp @@ MiniBatchSparsePCA.fit ~x transformer;  
  [%expect {|
      MiniBatchSparsePCA(...)      
  |}]
  let X_transformed = MiniBatchSparsePCA.transform ~x transformer in  
  print_ndarray @@ X_transformed.shape;  
  [%expect {|
      (200, 5)      
  |}]
  print_ndarray @@ # most values in the components_ are zero (sparsity);  
  print_ndarray @@ .mean transformer.components_ == 0 np;  
  [%expect {|
      0.94      
  |}]

*)



(* NMF *)
(*
>>> import numpy as np
>>> X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
>>> from sklearn.decomposition import NMF
>>> model = NMF(n_components=2, init='random', random_state=0)
>>> W = model.fit_transform(X)
>>> H = model.components_

*)

(* TEST TODO
let%expect_test "NMF" =
  let open Sklearn.Decomposition in
  let x = .array (matrix [|[|1; 1|]; [|2; 1|]; [|3; 1.2|]; [|4; 1|]; [|5; 0.8|]; [|6; 1|]|]) np in  
  let model = NMF.create ~n_components:2 ~init:'random' ~random_state:0 () in  
  let W = NMF.fit_transform ~x model in  
  let H = NMF.components_ model in  
  [%expect {|
  |}]

*)



(* PCA *)
(*
>>> import numpy as np
>>> from sklearn.decomposition import PCA
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
>>> pca = PCA(n_components=2)
>>> pca.fit(X)
PCA(n_components=2)
>>> print(pca.explained_variance_ratio_)
[0.9924... 0.0075...]
>>> print(pca.singular_values_)
[6.30061... 0.54980...]

*)

(* TEST TODO
let%expect_test "PCA" =
  let open Sklearn.Decomposition in
  let x = .array (matrixi [|[|-1; -1|]; [|-2; -1|]; [|-3; -2|]; [|1; 1|]; [|2; 1|]; [|3; 2|]|]) np in  
  let pca = PCA.create ~n_components:2 () in  
  print PCA.pp @@ PCA.fit ~x pca;  
  [%expect {|
      PCA(n_components=2)      
  |}]
  print_ndarray @@ print pca.explained_variance_ratio_ ();  
  [%expect {|
      [0.9924... 0.0075...]      
  |}]
  print_ndarray @@ print pca.singular_values_ ();  
  [%expect {|
      [6.30061... 0.54980...]      
  |}]

*)



(* PCA *)
(*
>>> pca = PCA(n_components=2, svd_solver='full')
>>> pca.fit(X)
PCA(n_components=2, svd_solver='full')
>>> print(pca.explained_variance_ratio_)
[0.9924... 0.00755...]
>>> print(pca.singular_values_)
[6.30061... 0.54980...]

*)

(* TEST TODO
let%expect_test "PCA" =
  let open Sklearn.Decomposition in
  let pca = PCA.create ~n_components:2 ~svd_solver:'full' () in  
  print PCA.pp @@ PCA.fit ~x pca;  
  [%expect {|
      PCA(n_components=2, svd_solver='full')      
  |}]
  print_ndarray @@ print pca.explained_variance_ratio_ ();  
  [%expect {|
      [0.9924... 0.00755...]      
  |}]
  print_ndarray @@ print pca.singular_values_ ();  
  [%expect {|
      [6.30061... 0.54980...]      
  |}]

*)



(* PCA *)
(*
>>> pca = PCA(n_components=1, svd_solver='arpack')
>>> pca.fit(X)
PCA(n_components=1, svd_solver='arpack')
>>> print(pca.explained_variance_ratio_)
[0.99244...]
>>> print(pca.singular_values_)

*)

(* TEST TODO
let%expect_test "PCA" =
  let open Sklearn.Decomposition in
  let pca = PCA.create ~n_components:1 ~svd_solver:'arpack' () in  
  print PCA.pp @@ PCA.fit ~x pca;  
  [%expect {|
      PCA(n_components=1, svd_solver='arpack')      
  |}]
  print_ndarray @@ print pca.explained_variance_ratio_ ();  
  [%expect {|
      [0.99244...]      
  |}]
  print_ndarray @@ print pca.singular_values_ ();  
  [%expect {|
  |}]

*)



(* transform *)
(*
>>> import numpy as np
>>> from sklearn.decomposition import IncrementalPCA
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
>>> ipca = IncrementalPCA(n_components=2, batch_size=3)
>>> ipca.fit(X)
IncrementalPCA(batch_size=3, n_components=2)

*)

(* TEST TODO
let%expect_test "_BasePCA.transform" =
  let open Sklearn.Decomposition in
  let x = .array (matrixi [|[|-1; -1|]; [|-2; -1|]; [|-3; -2|]; [|1; 1|]; [|2; 1|]; [|3; 2|]|]) np in  
  let ipca = IncrementalPCA.create ~n_components:2 ~batch_size:3 () in  
  print IncrementalPCA.pp @@ IncrementalPCA.fit ~x ipca;  
  [%expect {|
      IncrementalPCA(batch_size=3, n_components=2)      
  |}]

*)



(* SparsePCA *)
(*
>>> import numpy as np
>>> from sklearn.datasets import make_friedman1
>>> from sklearn.decomposition import SparsePCA
>>> X, _ = make_friedman1(n_samples=200, n_features=30, random_state=0)
>>> transformer = SparsePCA(n_components=5, random_state=0)
>>> transformer.fit(X)
SparsePCA(...)
>>> X_transformed = transformer.transform(X)
>>> X_transformed.shape
(200, 5)
>>> # most values in the components_ are zero (sparsity)
>>> np.mean(transformer.components_ == 0)
0.9666...

*)

(* TEST TODO
let%expect_test "SparsePCA" =
  let open Sklearn.Decomposition in
  let x, _ = make_friedman1(n_samples=200, n_features=30, random_state=0) in  
  let transformer = SparsePCA.create ~n_components:5 ~random_state:0 () in  
  print SparsePCA.pp @@ SparsePCA.fit ~x transformer;  
  [%expect {|
      SparsePCA(...)      
  |}]
  let X_transformed = SparsePCA.transform ~x transformer in  
  print_ndarray @@ X_transformed.shape;  
  [%expect {|
      (200, 5)      
  |}]
  print_ndarray @@ # most values in the components_ are zero (sparsity);  
  print_ndarray @@ .mean transformer.components_ == 0 np;  
  [%expect {|
      0.9666...      
  |}]

*)



(* TruncatedSVD *)
(*
>>> from sklearn.decomposition import TruncatedSVD
>>> from scipy.sparse import random as sparse_random
>>> from sklearn.random_projection import sparse_random_matrix
>>> X = sparse_random(100, 100, density=0.01, format='csr',
...                   random_state=42)
>>> svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
>>> svd.fit(X)
TruncatedSVD(n_components=5, n_iter=7, random_state=42)
>>> print(svd.explained_variance_ratio_)
[0.0646... 0.0633... 0.0639... 0.0535... 0.0406...]
>>> print(svd.explained_variance_ratio_.sum())
0.286...
>>> print(svd.singular_values_)
[1.553... 1.512...  1.510... 1.370... 1.199...]

*)

(* TEST TODO
let%expect_test "TruncatedSVD" =
  let open Sklearn.Decomposition in
  let x = sparse_random ~100 100 ~density:0.01 ~format:'csr' ~random_state:42 () in  
  let svd = TruncatedSVD.create ~n_components:5 ~n_iter:7 ~random_state:42 () in  
  print TruncatedSVD.pp @@ TruncatedSVD.fit ~x svd;  
  [%expect {|
      TruncatedSVD(n_components=5, n_iter=7, random_state=42)      
  |}]
  print_ndarray @@ print svd.explained_variance_ratio_ ();  
  [%expect {|
      [0.0646... 0.0633... 0.0639... 0.0535... 0.0406...]      
  |}]
  print_ndarray @@ print svd..sum () explained_variance_ratio_;  
  [%expect {|
      0.286...      
  |}]
  print_ndarray @@ print svd.singular_values_ ();  
  [%expect {|
      [1.553... 1.512...  1.510... 1.370... 1.199...]      
  |}]

*)



(* non_negative_factorization *)
(*
>>> import numpy as np
>>> X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
>>> from sklearn.decomposition import non_negative_factorization
>>> W, H, n_iter = non_negative_factorization(X, n_components=2,
... init='random', random_state=0)

*)

(* TEST TODO
let%expect_test "non_negative_factorization" =
  let open Sklearn.Decomposition in
  let x = .array (matrix [|[|1;1|]; [|2; 1|]; [|3; 1.2|]; [|4; 1|]; [|5; 0.8|]; [|6; 1|]|]) np in  
  let W, H, n_iter = non_negative_factorization x ~n_components:2 ~init:'random' ~random_state:0 () in  
  [%expect {|
  |}]

*)



