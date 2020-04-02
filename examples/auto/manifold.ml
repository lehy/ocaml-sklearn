(* Isomap *)
(*
>>> from sklearn.datasets import load_digits
>>> from sklearn.manifold import Isomap
>>> X, _ = load_digits(return_X_y=True)
>>> X.shape
(1797, 64)
>>> embedding = Isomap(n_components=2)
>>> X_transformed = embedding.fit_transform(X[:100])
>>> X_transformed.shape
(100, 2)


*)

(* TEST TODO
let%expect_text "Isomap" =
    let load_digits = Sklearn.Datasets.load_digits in
    let isomap = Sklearn.Manifold.isomap in
    let x, _ = load_digits return_X_y=True in
    X.shape    
    [%expect {|
            (1797, 64)            
    |}]
    embedding = Isomap(n_components=2)    
    X_transformed = embedding.fit_transform(X[:100])    
    X_transformed.shape    
    [%expect {|
            (100, 2)            
    |}]

*)



(* LocallyLinearEmbedding *)
(*
>>> from sklearn.datasets import load_digits
>>> from sklearn.manifold import LocallyLinearEmbedding
>>> X, _ = load_digits(return_X_y=True)
>>> X.shape
(1797, 64)
>>> embedding = LocallyLinearEmbedding(n_components=2)
>>> X_transformed = embedding.fit_transform(X[:100])
>>> X_transformed.shape
(100, 2)


*)

(* TEST TODO
let%expect_text "LocallyLinearEmbedding" =
    let load_digits = Sklearn.Datasets.load_digits in
    let locallyLinearEmbedding = Sklearn.Manifold.locallyLinearEmbedding in
    let x, _ = load_digits return_X_y=True in
    X.shape    
    [%expect {|
            (1797, 64)            
    |}]
    embedding = LocallyLinearEmbedding(n_components=2)    
    X_transformed = embedding.fit_transform(X[:100])    
    X_transformed.shape    
    [%expect {|
            (100, 2)            
    |}]

*)



(* MDS *)
(*
>>> from sklearn.datasets import load_digits
>>> from sklearn.manifold import MDS
>>> X, _ = load_digits(return_X_y=True)
>>> X.shape
(1797, 64)
>>> embedding = MDS(n_components=2)
>>> X_transformed = embedding.fit_transform(X[:100])
>>> X_transformed.shape
(100, 2)


*)

(* TEST TODO
let%expect_text "MDS" =
    let load_digits = Sklearn.Datasets.load_digits in
    let mds = Sklearn.Manifold.mds in
    let x, _ = load_digits return_X_y=True in
    X.shape    
    [%expect {|
            (1797, 64)            
    |}]
    embedding = MDS(n_components=2)    
    X_transformed = embedding.fit_transform(X[:100])    
    X_transformed.shape    
    [%expect {|
            (100, 2)            
    |}]

*)



(* SpectralEmbedding *)
(*
>>> from sklearn.datasets import load_digits
>>> from sklearn.manifold import SpectralEmbedding
>>> X, _ = load_digits(return_X_y=True)
>>> X.shape
(1797, 64)
>>> embedding = SpectralEmbedding(n_components=2)
>>> X_transformed = embedding.fit_transform(X[:100])
>>> X_transformed.shape
(100, 2)


*)

(* TEST TODO
let%expect_text "SpectralEmbedding" =
    let load_digits = Sklearn.Datasets.load_digits in
    let spectralEmbedding = Sklearn.Manifold.spectralEmbedding in
    let x, _ = load_digits return_X_y=True in
    X.shape    
    [%expect {|
            (1797, 64)            
    |}]
    embedding = SpectralEmbedding(n_components=2)    
    X_transformed = embedding.fit_transform(X[:100])    
    X_transformed.shape    
    [%expect {|
            (100, 2)            
    |}]

*)



(* TSNE *)
(*
>>> import numpy as np
>>> from sklearn.manifold import TSNE
>>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
>>> X_embedded = TSNE(n_components=2).fit_transform(X)
>>> X_embedded.shape
(4, 2)


*)

(* TEST TODO
let%expect_text "TSNE" =
    import numpy as np    
    let tsne = Sklearn.Manifold.tsne in
    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])    
    X_embedded = TSNE(n_components=2).fit_transform(X)    
    X_embedded.shape    
    [%expect {|
            (4, 2)            
    |}]

*)



