(* AffinityPropagation *)
(*
>>> from sklearn.cluster import AffinityPropagation
>>> import numpy as np
>>> X = np.array([[1, 2], [1, 4], [1, 0],
...               [4, 2], [4, 4], [4, 0]])
>>> clustering = AffinityPropagation().fit(X)
>>> clustering
AffinityPropagation()
>>> clustering.labels_
array([0, 0, 0, 1, 1, 1])
>>> clustering.predict([[0, 0], [4, 4]])
array([0, 1])
>>> clustering.cluster_centers_
array([[1, 2],
       [4, 2]])


*)

(* TEST TODO
let%expect_text "AffinityPropagation" =
    let affinityPropagation = Sklearn.Cluster.affinityPropagation in
    import numpy as np    
    X = np.array([[1, 2], [1, 4], [1, 0],[4, 2], [4, 4], [4, 0]])    
    clustering = AffinityPropagation().fit(X)    
    clustering    
    [%expect {|
            AffinityPropagation()            
    |}]
    clustering.labels_    
    [%expect {|
            array([0, 0, 0, 1, 1, 1])            
    |}]
    print @@ predict clustering [[0 0] [4 4]]
    [%expect {|
            array([0, 1])            
    |}]
    clustering.cluster_centers_    
    [%expect {|
            array([[1, 2],            
                   [4, 2]])            
    |}]

*)



(* DBSCAN *)
(*
>>> from sklearn.cluster import DBSCAN
>>> import numpy as np
>>> X = np.array([[1, 2], [2, 2], [2, 3],
...               [8, 7], [8, 8], [25, 80]])
>>> clustering = DBSCAN(eps=3, min_samples=2).fit(X)
>>> clustering.labels_
array([ 0,  0,  0,  1,  1, -1])
>>> clustering
DBSCAN(eps=3, min_samples=2)


*)

(* TEST TODO
let%expect_text "DBSCAN" =
    let dbscan = Sklearn.Cluster.dbscan in
    import numpy as np    
    X = np.array([[1, 2], [2, 2], [2, 3],[8, 7], [8, 8], [25, 80]])    
    clustering = DBSCAN(eps=3, min_samples=2).fit(X)    
    clustering.labels_    
    [%expect {|
            array([ 0,  0,  0,  1,  1, -1])            
    |}]
    clustering    
    [%expect {|
            DBSCAN(eps=3, min_samples=2)            
    |}]

*)



(* MeanShift *)
(*
>>> from sklearn.cluster import MeanShift
>>> import numpy as np
>>> X = np.array([[1, 1], [2, 1], [1, 0],
...               [4, 7], [3, 5], [3, 6]])
>>> clustering = MeanShift(bandwidth=2).fit(X)
>>> clustering.labels_
array([1, 1, 1, 0, 0, 0])
>>> clustering.predict([[0, 0], [5, 5]])
array([1, 0])
>>> clustering
MeanShift(bandwidth=2)


*)

(* TEST TODO
let%expect_text "MeanShift" =
    let meanShift = Sklearn.Cluster.meanShift in
    import numpy as np    
    X = np.array([[1, 1], [2, 1], [1, 0],[4, 7], [3, 5], [3, 6]])    
    clustering = MeanShift(bandwidth=2).fit(X)    
    clustering.labels_    
    [%expect {|
            array([1, 1, 1, 0, 0, 0])            
    |}]
    print @@ predict clustering [[0 0] [5 5]]
    [%expect {|
            array([1, 0])            
    |}]
    clustering    
    [%expect {|
            MeanShift(bandwidth=2)            
    |}]

*)



(* SpectralBiclustering *)
(*
>>> from sklearn.cluster import SpectralBiclustering
>>> import numpy as np
>>> X = np.array([[1, 1], [2, 1], [1, 0],
...               [4, 7], [3, 5], [3, 6]])
>>> clustering = SpectralBiclustering(n_clusters=2, random_state=0).fit(X)
>>> clustering.row_labels_
array([1, 1, 1, 0, 0, 0], dtype=int32)
>>> clustering.column_labels_
array([0, 1], dtype=int32)
>>> clustering
SpectralBiclustering(n_clusters=2, random_state=0)


*)

(* TEST TODO
let%expect_text "SpectralBiclustering" =
    let spectralBiclustering = Sklearn.Cluster.spectralBiclustering in
    import numpy as np    
    X = np.array([[1, 1], [2, 1], [1, 0],[4, 7], [3, 5], [3, 6]])    
    clustering = SpectralBiclustering(n_clusters=2, random_state=0).fit(X)    
    clustering.row_labels_    
    [%expect {|
            array([1, 1, 1, 0, 0, 0], dtype=int32)            
    |}]
    clustering.column_labels_    
    [%expect {|
            array([0, 1], dtype=int32)            
    |}]
    clustering    
    [%expect {|
            SpectralBiclustering(n_clusters=2, random_state=0)            
    |}]

*)



(* SpectralClustering *)
(*
>>> from sklearn.cluster import SpectralClustering
>>> import numpy as np
>>> X = np.array([[1, 1], [2, 1], [1, 0],
...               [4, 7], [3, 5], [3, 6]])
>>> clustering = SpectralClustering(n_clusters=2,
...         assign_labels="discretize",
...         random_state=0).fit(X)
>>> clustering.labels_
array([1, 1, 1, 0, 0, 0])
>>> clustering
SpectralClustering(assign_labels='discretize', n_clusters=2,
    random_state=0)


*)

(* TEST TODO
let%expect_text "SpectralClustering" =
    let spectralClustering = Sklearn.Cluster.spectralClustering in
    import numpy as np    
    X = np.array([[1, 1], [2, 1], [1, 0],[4, 7], [3, 5], [3, 6]])    
    clustering = SpectralClustering(n_clusters=2,assign_labels="discretize",random_state=0).fit(X)    
    clustering.labels_    
    [%expect {|
            array([1, 1, 1, 0, 0, 0])            
    |}]
    clustering    
    [%expect {|
            SpectralClustering(assign_labels='discretize', n_clusters=2,            
                random_state=0)            
    |}]

*)



(* SpectralCoclustering *)
(*
>>> from sklearn.cluster import SpectralCoclustering
>>> import numpy as np
>>> X = np.array([[1, 1], [2, 1], [1, 0],
...               [4, 7], [3, 5], [3, 6]])
>>> clustering = SpectralCoclustering(n_clusters=2, random_state=0).fit(X)
>>> clustering.row_labels_ #doctest: +SKIP
array([0, 1, 1, 0, 0, 0], dtype=int32)
>>> clustering.column_labels_ #doctest: +SKIP
array([0, 0], dtype=int32)
>>> clustering
SpectralCoclustering(n_clusters=2, random_state=0)


*)

(* TEST TODO
let%expect_text "SpectralCoclustering" =
    let spectralCoclustering = Sklearn.Cluster.spectralCoclustering in
    import numpy as np    
    X = np.array([[1, 1], [2, 1], [1, 0],[4, 7], [3, 5], [3, 6]])    
    clustering = SpectralCoclustering(n_clusters=2, random_state=0).fit(X)    
    clustering.row_labels_ #doctest: +SKIP    
    [%expect {|
            array([0, 1, 1, 0, 0, 0], dtype=int32)            
    |}]
    clustering.column_labels_ #doctest: +SKIP    
    [%expect {|
            array([0, 0], dtype=int32)            
    |}]
    clustering    
    [%expect {|
            SpectralCoclustering(n_clusters=2, random_state=0)            
    |}]

*)



