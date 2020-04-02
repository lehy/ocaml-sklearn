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
let%expect_text "FactorAnalysis" =
    let load_digits = Sklearn.Datasets.load_digits in
    let factorAnalysis = Sklearn.Decomposition.factorAnalysis in
    let x, _ = load_digits return_X_y=True in
    transformer = FactorAnalysis(n_components=7, random_state=0)    
    X_transformed = transformer.fit_transform(X)    
    X_transformed.shape    
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
let%expect_text "FastICA" =
    let load_digits = Sklearn.Datasets.load_digits in
    let fastICA = Sklearn.Decomposition.fastICA in
    let x, _ = load_digits return_X_y=True in
    transformer = FastICA(n_components=7,random_state=0)    
    X_transformed = transformer.fit_transform(X)    
    X_transformed.shape    
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
let%expect_text "IncrementalPCA" =
    let load_digits = Sklearn.Datasets.load_digits in
    let incrementalPCA = Sklearn.Decomposition.incrementalPCA in
    let sparse = Scipy.sparse in
    let x, _ = load_digits return_X_y=True in
    transformer = IncrementalPCA(n_components=7, batch_size=200)    
    # either partially fit on smaller batches of data    
    print @@ partial_fit transformer x[:100 :]
    [%expect {|
            IncrementalPCA(batch_size=200, n_components=7)            
    |}]
    # or let the fit function itself divide the data into batches    
    X_sparse = sparse.csr_matrix(X)    
    X_transformed = transformer.fit_transform(X_sparse)    
    X_transformed.shape    
    [%expect {|
            (1797, 7)            
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
let%expect_text "KernelPCA" =
    let load_digits = Sklearn.Datasets.load_digits in
    let kernelPCA = Sklearn.Decomposition.kernelPCA in
    let x, _ = load_digits return_X_y=True in
    transformer = KernelPCA(n_components=7, kernel='linear')    
    X_transformed = transformer.fit_transform(X)    
    X_transformed.shape    
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
let%expect_text "LatentDirichletAllocation" =
    let latentDirichletAllocation = Sklearn.Decomposition.latentDirichletAllocation in
    let make_multilabel_classification = Sklearn.Datasets.make_multilabel_classification in
    # This produces a feature matrix of token counts, similar to what    
    # CountVectorizer would produce on text.    
    let x, _ = make_multilabel_classification random_state=0 in
    lda = LatentDirichletAllocation(n_components=5,random_state=0)    
    print @@ fit lda x
    [%expect {|
            LatentDirichletAllocation(...)            
    |}]
    # get topics for some given samples:    
    print @@ transform lda x[-2:]
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
let%expect_text "MiniBatchSparsePCA" =
    import numpy as np    
    let make_friedman1 = Sklearn.Datasets.make_friedman1 in
    let miniBatchSparsePCA = Sklearn.Decomposition.miniBatchSparsePCA in
    let x, _ = make_friedman1 n_samples=200 n_features=30 random_state=0 in
    transformer = MiniBatchSparsePCA(n_components=5, batch_size=50,random_state=0)    
    print @@ fit transformer x
    [%expect {|
            MiniBatchSparsePCA(...)            
    |}]
    X_transformed = transformer.transform(X)    
    X_transformed.shape    
    [%expect {|
            (200, 5)            
    |}]
    # most values in the components_ are zero (sparsity)    
    print @@ mean np transformer.components_ == 0
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
let%expect_text "NMF" =
    import numpy as np    
    X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])    
    let nmf = Sklearn.Decomposition.nmf in
    model = NMF(n_components=2, init='random', random_state=0)    
    W = model.fit_transform(X)    
    H = model.components_    
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
let%expect_text "PCA" =
    import numpy as np    
    let pca = Sklearn.Decomposition.pca in
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])    
    pca = PCA(n_components=2)    
    print @@ fit pca x
    [%expect {|
            PCA(n_components=2)            
    |}]
    print(pca.explained_variance_ratio_)    
    [%expect {|
            [0.9924... 0.0075...]            
    |}]
    print(pca.singular_values_)    
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
let%expect_text "PCA" =
    pca = PCA(n_components=2, svd_solver='full')    
    print @@ fit pca x
    [%expect {|
            PCA(n_components=2, svd_solver='full')            
    |}]
    print(pca.explained_variance_ratio_)    
    [%expect {|
            [0.9924... 0.00755...]            
    |}]
    print(pca.singular_values_)    
    [%expect {|
            [6.30061... 0.54980...]            
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
let%expect_text "SparsePCA" =
    import numpy as np    
    let make_friedman1 = Sklearn.Datasets.make_friedman1 in
    let sparsePCA = Sklearn.Decomposition.sparsePCA in
    let x, _ = make_friedman1 n_samples=200 n_features=30 random_state=0 in
    transformer = SparsePCA(n_components=5, random_state=0)    
    print @@ fit transformer x
    [%expect {|
            SparsePCA(...)            
    |}]
    X_transformed = transformer.transform(X)    
    X_transformed.shape    
    [%expect {|
            (200, 5)            
    |}]
    # most values in the components_ are zero (sparsity)    
    print @@ mean np transformer.components_ == 0
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
let%expect_text "TruncatedSVD" =
    let truncatedSVD = Sklearn.Decomposition.truncatedSVD in
    from scipy.sparse import random as sparse_random    
    let sparse_random_matrix = Sklearn.Random_projection.sparse_random_matrix in
    X = sparse_random(100, 100, density=0.01, format='csr',random_state=42)    
    svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)    
    print @@ fit svd x
    [%expect {|
            TruncatedSVD(n_components=5, n_iter=7, random_state=42)            
    |}]
    print(svd.explained_variance_ratio_)    
    [%expect {|
            [0.0646... 0.0633... 0.0639... 0.0535... 0.0406...]            
    |}]
    print(svd.explained_variance_ratio_.sum())    
    [%expect {|
            0.286...            
    |}]
    print(svd.singular_values_)    
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
let%expect_text "non_negative_factorization" =
    import numpy as np    
    X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])    
    let non_negative_factorization = Sklearn.Decomposition.non_negative_factorization in
    W, H, n_iter = non_negative_factorization(X, n_components=2,init='random', random_state=0)    
    [%expect {|
    |}]

*)



