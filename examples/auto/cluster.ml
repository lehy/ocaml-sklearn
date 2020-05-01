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
let%expect_test "AffinityPropagation" =
  let open Sklearn.Cluster in
  let x = .array [(vectori [|1; 2|]) (vectori [|1; 4|]) (vectori [|1; 0|]) (vectori [|4; 2|]) (vectori [|4; 4|]) (vectori [|4; 0|])] np in  
  let clustering = AffinityPropagation().fit ~x () in  
  print_ndarray @@ clustering;  
  [%expect {|
      AffinityPropagation()      
  |}]
  print_ndarray @@ AffinityPropagation.labels_ clustering;  
  [%expect {|
      array([0, 0, 0, 1, 1, 1])      
  |}]
  print_ndarray @@ AffinityPropagation.predict (matrixi [|[|0; 0|]; [|4; 4|]|]) clustering;  
  [%expect {|
      array([0, 1])      
  |}]
  print_ndarray @@ AffinityPropagation.cluster_centers_ clustering;  
  [%expect {|
      array([[1, 2],      
             [4, 2]])      
  |}]

*)



(* AgglomerativeClustering *)
(*
>>> from sklearn.cluster import AgglomerativeClustering
>>> import numpy as np
>>> X = np.array([[1, 2], [1, 4], [1, 0],
...               [4, 2], [4, 4], [4, 0]])
>>> clustering = AgglomerativeClustering().fit(X)
>>> clustering
AgglomerativeClustering()
>>> clustering.labels_

*)

(* TEST TODO
let%expect_test "AgglomerativeClustering" =
  let open Sklearn.Cluster in
  let x = .array [(vectori [|1; 2|]) (vectori [|1; 4|]) (vectori [|1; 0|]) (vectori [|4; 2|]) (vectori [|4; 4|]) (vectori [|4; 0|])] np in  
  let clustering = AgglomerativeClustering().fit ~x () in  
  print_ndarray @@ clustering;  
  [%expect {|
      AgglomerativeClustering()      
  |}]
  print_ndarray @@ AgglomerativeClustering.labels_ clustering;  
  [%expect {|
  |}]

*)



(* Birch *)
(*
>>> from sklearn.cluster import Birch
>>> X = [[0, 1], [0.3, 1], [-0.3, 1], [0, -1], [0.3, -1], [-0.3, -1]]
>>> brc = Birch(n_clusters=None)
>>> brc.fit(X)
Birch(n_clusters=None)
>>> brc.predict(X)

*)

(* TEST TODO
let%expect_test "Birch" =
  let open Sklearn.Cluster in
  let x = (matrix [|[|0; 1|]; [|0.3; 1|]; [|-0.3; 1|]; [|0; -1|]; [|0.3; -1|]; [|-0.3; -1|]|]) in  
  let brc = Birch.create ~n_clusters:None () in  
  print Birch.pp @@ Birch.fit ~x brc;  
  [%expect {|
      Birch(n_clusters=None)      
  |}]
  print_ndarray @@ Birch.predict ~x brc;  
  [%expect {|
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
let%expect_test "DBSCAN" =
  let open Sklearn.Cluster in
  let x = .array [(vectori [|1; 2|]) (vectori [|2; 2|]) (vectori [|2; 3|]) (vectori [|8; 7|]) (vectori [|8; 8|]) [25 80]] np in  
  let clustering = DBSCAN(eps=3, min_samples=2).fit ~x () in  
  print_ndarray @@ DBSCAN.labels_ clustering;  
  [%expect {|
      array([ 0,  0,  0,  1,  1, -1])      
  |}]
  print_ndarray @@ clustering;  
  [%expect {|
      DBSCAN(eps=3, min_samples=2)      
  |}]

*)



(* FeatureAgglomeration *)
(*
>>> import numpy as np
>>> from sklearn import datasets, cluster
>>> digits = datasets.load_digits()
>>> images = digits.images
>>> X = np.reshape(images, (len(images), -1))
>>> agglo = cluster.FeatureAgglomeration(n_clusters=32)
>>> agglo.fit(X)
FeatureAgglomeration(n_clusters=32)
>>> X_reduced = agglo.transform(X)
>>> X_reduced.shape

*)

(* TEST TODO
let%expect_test "FeatureAgglomeration" =
  let open Sklearn.Cluster in
  let digits = .load_digits datasets in  
  let images = .images digits in  
  let x = .reshape ~images (len ~images () -1) np in  
  let agglo = .featureAgglomeration ~n_clusters:32 cluster in  
  print_ndarray @@ .fit ~x agglo;  
  [%expect {|
      FeatureAgglomeration(n_clusters=32)      
  |}]
  let X_reduced = .transform ~x agglo in  
  print_ndarray @@ X_reduced.shape;  
  [%expect {|
  |}]

*)



(* KMeans *)
(*
>>> from sklearn.cluster import KMeans
>>> import numpy as np
>>> X = np.array([[1, 2], [1, 4], [1, 0],
...               [10, 2], [10, 4], [10, 0]])
>>> kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
>>> kmeans.labels_
array([1, 1, 1, 0, 0, 0], dtype=int32)
>>> kmeans.predict([[0, 0], [12, 3]])
array([1, 0], dtype=int32)
>>> kmeans.cluster_centers_
array([[10.,  2.],

*)

(* TEST TODO
let%expect_test "KMeans" =
  let open Sklearn.Cluster in
  let x = .array [(vectori [|1; 2|]) (vectori [|1; 4|]) (vectori [|1; 0|]) [10 2] [10 4] [10 0]] np in  
  let kmeans = KMeans(n_clusters=2, random_state=0).fit ~x () in  
  print_ndarray @@ KMeans.labels_ kmeans;  
  [%expect {|
      array([1, 1, 1, 0, 0, 0], dtype=int32)      
  |}]
  print_ndarray @@ KMeans.predict (matrixi [|[|0; 0|]; [|12; 3|]|]) kmeans;  
  [%expect {|
      array([1, 0], dtype=int32)      
  |}]
  print_ndarray @@ KMeans.cluster_centers_ kmeans;  
  [%expect {|
      array([[10.,  2.],      
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
let%expect_test "MeanShift" =
  let open Sklearn.Cluster in
  let x = .array [(vectori [|1; 1|]) (vectori [|2; 1|]) (vectori [|1; 0|]) (vectori [|4; 7|]) (vectori [|3; 5|]) (vectori [|3; 6|])] np in  
  let clustering = MeanShift(bandwidth=2).fit ~x () in  
  print_ndarray @@ MeanShift.labels_ clustering;  
  [%expect {|
      array([1, 1, 1, 0, 0, 0])      
  |}]
  print_ndarray @@ MeanShift.predict (matrixi [|[|0; 0|]; [|5; 5|]|]) clustering;  
  [%expect {|
      array([1, 0])      
  |}]
  print_ndarray @@ clustering;  
  [%expect {|
      MeanShift(bandwidth=2)      
  |}]

*)



(* MiniBatchKMeans *)
(*
>>> from sklearn.cluster import MiniBatchKMeans
>>> import numpy as np
>>> X = np.array([[1, 2], [1, 4], [1, 0],
...               [4, 2], [4, 0], [4, 4],
...               [4, 5], [0, 1], [2, 2],
...               [3, 2], [5, 5], [1, -1]])
>>> # manually fit on batches
>>> kmeans = MiniBatchKMeans(n_clusters=2,
...                          random_state=0,
...                          batch_size=6)
>>> kmeans = kmeans.partial_fit(X[0:6,:])
>>> kmeans = kmeans.partial_fit(X[6:12,:])
>>> kmeans.cluster_centers_
array([[2. , 1. ],
       [3.5, 4.5]])
>>> kmeans.predict([[0, 0], [4, 4]])
array([0, 1], dtype=int32)
>>> # fit on the whole data
>>> kmeans = MiniBatchKMeans(n_clusters=2,
...                          random_state=0,
...                          batch_size=6,
...                          max_iter=10).fit(X)
>>> kmeans.cluster_centers_
array([[3.95918367, 2.40816327],
       [1.12195122, 1.3902439 ]])
>>> kmeans.predict([[0, 0], [4, 4]])

*)

(* TEST TODO
let%expect_test "MiniBatchKMeans" =
  let open Sklearn.Cluster in
  let x = .array [(vectori [|1; 2|]) (vectori [|1; 4|]) (vectori [|1; 0|]) (vectori [|4; 2|]) (vectori [|4; 0|]) (vectori [|4; 4|]) (vectori [|4; 5|]) (vectori [|0; 1|]) (vectori [|2; 2|]) (vectori [|3; 2|]) (vectori [|5; 5|]) [1 -1]] np in  
  print_ndarray @@ # manually fit on batches;  
  let kmeans = MiniBatchKMeans.create ~n_clusters:2 ~random_state:0 ~batch_size:6 () in  
  let kmeans = MiniBatchKMeans.partial_fit x[0:6 :] kmeans in  
  let kmeans = MiniBatchKMeans.partial_fit x[6:12 :] kmeans in  
  print_ndarray @@ MiniBatchKMeans.cluster_centers_ kmeans;  
  [%expect {|
      array([[2. , 1. ],      
             [3.5, 4.5]])      
  |}]
  print_ndarray @@ MiniBatchKMeans.predict (matrixi [|[|0; 0|]; [|4; 4|]|]) kmeans;  
  [%expect {|
      array([0, 1], dtype=int32)      
  |}]
  print_ndarray @@ # fit on the whole data;  
  let kmeans = MiniBatchKMeans(n_clusters=2,random_state=0,batch_size=6,max_iter=10).fit ~x () in  
  print_ndarray @@ MiniBatchKMeans.cluster_centers_ kmeans;  
  [%expect {|
      array([[3.95918367, 2.40816327],      
             [1.12195122, 1.3902439 ]])      
  |}]
  print_ndarray @@ MiniBatchKMeans.predict (matrixi [|[|0; 0|]; [|4; 4|]|]) kmeans;  
  [%expect {|
  |}]

*)



(* OPTICS *)
(*
>>> from sklearn.cluster import OPTICS
>>> import numpy as np
>>> X = np.array([[1, 2], [2, 5], [3, 6],
...               [8, 7], [8, 8], [7, 3]])
>>> clustering = OPTICS(min_samples=2).fit(X)
>>> clustering.labels_

*)

(* TEST TODO
let%expect_test "OPTICS" =
  let open Sklearn.Cluster in
  let x = .array [(vectori [|1; 2|]) (vectori [|2; 5|]) (vectori [|3; 6|]) (vectori [|8; 7|]) (vectori [|8; 8|]) (vectori [|7; 3|])] np in  
  let clustering = OPTICS(min_samples=2).fit ~x () in  
  print_ndarray @@ OPTICS.labels_ clustering;  
  [%expect {|
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
let%expect_test "SpectralBiclustering" =
  let open Sklearn.Cluster in
  let x = .array [(vectori [|1; 1|]) (vectori [|2; 1|]) (vectori [|1; 0|]) (vectori [|4; 7|]) (vectori [|3; 5|]) (vectori [|3; 6|])] np in  
  let clustering = SpectralBiclustering(n_clusters=2, random_state=0).fit ~x () in  
  print_ndarray @@ SpectralBiclustering.row_labels_ clustering;  
  [%expect {|
      array([1, 1, 1, 0, 0, 0], dtype=int32)      
  |}]
  print_ndarray @@ SpectralBiclustering.column_labels_ clustering;  
  [%expect {|
      array([0, 1], dtype=int32)      
  |}]
  print_ndarray @@ clustering;  
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
let%expect_test "SpectralClustering" =
  let open Sklearn.Cluster in
  let x = .array [(vectori [|1; 1|]) (vectori [|2; 1|]) (vectori [|1; 0|]) (vectori [|4; 7|]) (vectori [|3; 5|]) (vectori [|3; 6|])] np in  
  let clustering = SpectralClustering(n_clusters=2,assign_labels="discretize",random_state=0).fit ~x () in  
  print_ndarray @@ SpectralClustering.labels_ clustering;  
  [%expect {|
      array([1, 1, 1, 0, 0, 0])      
  |}]
  print_ndarray @@ clustering;  
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
let%expect_test "SpectralCoclustering" =
  let open Sklearn.Cluster in
  let x = .array [(vectori [|1; 1|]) (vectori [|2; 1|]) (vectori [|1; 0|]) (vectori [|4; 7|]) (vectori [|3; 5|]) (vectori [|3; 6|])] np in  
  let clustering = SpectralCoclustering(n_clusters=2, random_state=0).fit ~x () in  
  print_ndarray @@ clustering.row_labels_ #doctest: +SKIP;  
  [%expect {|
      array([0, 1, 1, 0, 0, 0], dtype=int32)      
  |}]
  print_ndarray @@ clustering.column_labels_ #doctest: +SKIP;  
  [%expect {|
      array([0, 0], dtype=int32)      
  |}]
  print_ndarray @@ clustering;  
  [%expect {|
      SpectralCoclustering(n_clusters=2, random_state=0)      
  |}]

*)



