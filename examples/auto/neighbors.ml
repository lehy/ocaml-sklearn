(* KNeighborsClassifier *)
(*
>>> X = [[0], [1], [2], [3]]
>>> y = [0, 0, 1, 1]
>>> from sklearn.neighbors import KNeighborsClassifier
>>> neigh = KNeighborsClassifier(n_neighbors=3)
>>> neigh.fit(X, y)
KNeighborsClassifier(...)
>>> print(neigh.predict([[1.1]]))
[0]
>>> print(neigh.predict_proba([[0.9]]))
[[0.66666667 0.33333333]]


*)

let print f x = Format.printf "%a" f x
let print_py x = Format.printf "%s" (Py.Object.to_string x)
let print_ndarray = print Sklearn.Arr.pp

let matrix = Sklearn.Arr.Float.matrix
let vector = Sklearn.Arr.Float.vector
let matrixi = Sklearn.Arr.Int.matrix
let vectori = Sklearn.Arr.Int.vector

let get x = match x with
  | None -> failwith "Option.get"
  | Some x -> x

let%expect_test "KNeighborsClassifier" =
    let x = matrix [|[|0.|]; [|1.|]; [|2.|]; [|3.|]|] in
    let y = vector [|0.; 0.; 1.; 1.|] in
    let open Sklearn.Neighbors in
    let neigh = KNeighborsClassifier.create ~n_neighbors:3 () in
    print KNeighborsClassifier.pp @@ KNeighborsClassifier.fit neigh ~x:(`Arr x) ~y;
    [%expect {|
            KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                 metric_params=None, n_jobs=None, n_neighbors=3, p=2,
                                 weights='uniform')
    |}];
    print_ndarray @@ KNeighborsClassifier.predict neigh ~x:(matrix [|[|1.1|]|]);
    [%expect {|
            [0.]
    |}];
    print_ndarray @@ KNeighborsClassifier.predict_proba neigh ~x:(matrix [|[|0.9|]|]);
    [%expect {|
            [[0.66666667 0.33333333]]
    |}]


(* kneighbors *)
(*
>>> samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
>>> from sklearn.neighbors import NearestNeighbors
>>> neigh = NearestNeighbors(n_neighbors=1)
>>> neigh.fit(samples)
NearestNeighbors(n_neighbors=1)
>>> print(neigh.kneighbors([[1., 1., 1.]]))
(array([[0.5]]), array([[2]]))


*)

let%expect_test "KNeighborsMixin.kneighbors" =
  let samples = matrix [|[|0.; 0.; 0.|]; [|0.; 0.5; 0.|]; [|1.; 1.; 0.5|]|] in
  let open Sklearn.Neighbors in
  let neigh = NearestNeighbors.create ~n_neighbors:1 () in
  print NearestNeighbors.pp @@ NearestNeighbors.fit neigh ~x:(`Arr samples);
  [%expect {|
            NearestNeighbors(algorithm='auto', leaf_size=30, metric='minkowski',
                             metric_params=None, n_jobs=None, n_neighbors=1, p=2,
                             radius=1.0)
    |}];
  let neigh_dist, neigh_ind = NearestNeighbors.kneighbors neigh ~x:(matrix [|[|1.; 1.; 1.|]|]) in
  Format.printf "(%a, %a)" Sklearn.Arr.pp neigh_dist Sklearn.Arr.pp neigh_ind;
  [%expect {|
            ([[0.5]], [[2]])
    |}]


(* kneighbors_graph *)
(*
>>> X = [[0], [3], [1]]
>>> from sklearn.neighbors import NearestNeighbors
>>> neigh = NearestNeighbors(n_neighbors=2)
>>> neigh.fit(X)
NearestNeighbors(n_neighbors=2)
>>> A = neigh.kneighbors_graph(X)
>>> A.toarray()
array([[1., 0., 1.],
       [0., 1., 1.],
       [1., 0., 1.]])

*)

let%expect_test "KNeighborsMixin.kneighbors_graph" =
  let x = matrix [|[|0.|]; [|3.|]; [|1.|]|] in
  let open Sklearn.Neighbors in
  let neigh = NearestNeighbors.create ~n_neighbors:2 () in
  print NearestNeighbors.pp @@ NearestNeighbors.fit neigh ~x:(`Arr x);
  [%expect {|
            NearestNeighbors(algorithm='auto', leaf_size=30, metric='minkowski',
                             metric_params=None, n_jobs=None, n_neighbors=2, p=2,
                             radius=1.0)
    |}];
  let a = NearestNeighbors.kneighbors_graph neigh ~x in
  Sklearn.Csr_matrix.pp Format.std_formatter @@ a;
  [%expect {|
            (0, 0)	1.0
            (0, 2)	1.0
            (1, 1)	1.0
            (1, 2)	1.0
            (2, 2)	1.0
            (2, 0)	1.0
    |}];
  print Sklearn.Ndarray.pp @@ Sklearn.Csr_matrix.toarray a;
  [%expect {|
            [[1. 0. 1.]
             [0. 1. 1.]
             [1. 0. 1.]]
    |}]

(* KNeighborsRegressor *)
(*
>>> X = [[0], [1], [2], [3]]
>>> y = [0, 0, 1, 1]
>>> from sklearn.neighbors import KNeighborsRegressor
>>> neigh = KNeighborsRegressor(n_neighbors=2)
>>> neigh.fit(X, y)
KNeighborsRegressor(...)
>>> print(neigh.predict([[1.5]]))
[0.5]


*)

let%expect_test "KNeighborsRegressor" =
    let x = matrix [|[|0.|]; [|1.|]; [|2.|]; [|3.|]|] in
    let y = vector [|0.; 0.; 1.; 1.|] in
    let open Sklearn.Neighbors in
    let neigh = KNeighborsRegressor.create ~n_neighbors:2 () in
    print KNeighborsRegressor.pp @@ KNeighborsRegressor.fit neigh ~x:(`Arr x) ~y;
    [%expect {|
            KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
                                metric_params=None, n_jobs=None, n_neighbors=2, p=2,
                                weights='uniform')
    |}];
    print_ndarray @@ KNeighborsRegressor.predict neigh ~x:(matrix [|[|1.5|]|]);
    [%expect {|
            [0.5]
    |}]



(* LocalOutlierFactor *)
(*
>>> import numpy as np
>>> from sklearn.neighbors import LocalOutlierFactor
>>> X = [[-1.1], [0.2], [101.1], [0.3]]
>>> clf = LocalOutlierFactor(n_neighbors=2)
>>> clf.fit_predict(X)
array([ 1,  1, -1,  1])
>>> clf.negative_outlier_factor_
array([ -0.9821...,  -1.0370..., -73.3697...,  -0.9821...])


*)

let%expect_test "LocalOutlierFactor" =
  let open Sklearn.Neighbors in
  let x = matrix [|[|-1.1|]; [|0.2|]; [|101.1|]; [|0.3|]|] in
  let clf = LocalOutlierFactor.create ~n_neighbors:2 () in
  print Sklearn.Arr.pp @@ LocalOutlierFactor.fit_predict clf ~x;
  [%expect {|
            [ 1  1 -1  1]
    |}];
  print_ndarray @@ LocalOutlierFactor.negative_outlier_factor_ clf;
  [%expect {|
            [ -0.98214286  -1.03703704 -73.36970899  -0.98214286]
    |}]





(* NearestCentroid *)
(*
>>> from sklearn.neighbors import NearestCentroid
>>> import numpy as np
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
>>> y = np.array([1, 1, 1, 2, 2, 2])
>>> clf = NearestCentroid()
>>> clf.fit(X, y)
NearestCentroid()
>>> print(clf.predict([[-0.8, -1]]))
[1]


*)

let%expect_test "NearestCentroid" =
  let open Sklearn.Neighbors in
  let x = matrix [|[|-1.; -1.|]; [|-2.; -1.|]; [|-3.; -2.|]; [|1.; 1.|]; [|2.; 1.|]; [|3.; 2.|]|] in
  let y = vectori [|1; 1; 1; 2; 2; 2|] in
  let clf = NearestCentroid.create () in
  print NearestCentroid.pp @@ NearestCentroid.fit clf ~x ~y;
  [%expect {|
            NearestCentroid(metric='euclidean', shrink_threshold=None)
    |}];
  print Sklearn.Arr.pp @@ NearestCentroid.predict clf ~x:(matrix [|[|-0.8; -1.|]|]);
  [%expect {|
            [1]
    |}]


(* radius_neighbors *)
(*
>>> import numpy as np
>>> samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
>>> from sklearn.neighbors import NearestNeighbors
>>> neigh = NearestNeighbors(radius=1.6)
>>> neigh.fit(samples)
NearestNeighbors(radius=1.6)
>>> rng = neigh.radius_neighbors([[1., 1., 1.]])
>>> print(np.asarray(rng[0][0]))
[1.5 0.5]
>>> print(np.asarray(rng[1][0]))
[1 2]


*)

let%expect_test "RadiusNeighborsMixin.radius_neighbors" =
    let samples = matrix [|[|0.; 0.; 0.|]; [|0.; 0.5; 0.|]; [|1.; 1.; 0.5|]|] in
    let open Sklearn.Neighbors in
    let neigh = NearestNeighbors.create ~radius:1.6 () in
    print NearestNeighbors.pp @@ NearestNeighbors.fit neigh ~x:(`Arr samples);
    [%expect {|
            NearestNeighbors(algorithm='auto', leaf_size=30, metric='minkowski',
                             metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                             radius=1.6)
    |}];
    let dist, ind = NearestNeighbors.radius_neighbors neigh ~x:(`Arr (matrix [|[|1.; 1.; 1.|]|])) in
    print Sklearn.Arr.List.pp @@ dist;
    [%expect {|
            [array([1.5, 0.5])]
    |}];
    print Sklearn.Arr.List.pp @@ ind;
    [%expect {|
            [array([1, 2])]
    |}]

(* radius_neighbors_graph *)
(*
>>> X = [[0], [3], [1]]
>>> from sklearn.neighbors import NearestNeighbors
>>> neigh = NearestNeighbors(radius=1.5)
>>> neigh.fit(X)
NearestNeighbors(radius=1.5)
>>> A = neigh.radius_neighbors_graph(X)
>>> A.toarray()
array([[1., 0., 1.],
       [0., 1., 0.],
       [1., 0., 1.]])


*)

let%expect_test "RadiusNeighborsMixin.radius_neighbors_graph" =
  let x = matrix [|[|0.|]; [|3.|]; [|1.|]|] in
  let open Sklearn.Neighbors in
  let neigh = NearestNeighbors.create ~radius:1.5 () in
  print NearestNeighbors.pp @@ NearestNeighbors.fit neigh ~x:(`Arr x);
  [%expect {|
            NearestNeighbors(algorithm='auto', leaf_size=30, metric='minkowski',
                             metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                             radius=1.5)
    |}];
  let a = NearestNeighbors.radius_neighbors_graph neigh ~x in
  Sklearn.Ndarray.pp Format.std_formatter @@ Sklearn.Csr_matrix.toarray a;
  [%expect {|
            [[1. 0. 1.]
             [0. 1. 0.]
             [1. 0. 1.]]
    |}]

(* NeighborhoodComponentsAnalysis *)
(*
>>> from sklearn.neighbors import NeighborhoodComponentsAnalysis
>>> from sklearn.neighbors import KNeighborsClassifier
>>> from sklearn.datasets import load_iris
>>> from sklearn.model_selection import train_test_split
>>> X, y = load_iris(return_X_y=True)
>>> X_train, X_test, y_train, y_test = train_test_split(X, y,
... stratify=y, test_size=0.7, random_state=42)
>>> nca = NeighborhoodComponentsAnalysis(random_state=42)
>>> nca.fit(X_train, y_train)
NeighborhoodComponentsAnalysis(...)
>>> knn = KNeighborsClassifier(n_neighbors=3)
>>> knn.fit(X_train, y_train)
KNeighborsClassifier(...)
>>> print(knn.score(X_test, y_test))
0.933333...
>>> knn.fit(nca.transform(X_train), y_train)
KNeighborsClassifier(...)
>>> print(knn.score(nca.transform(X_test), y_test))
0.961904...


*)

let%expect_test "NeighborhoodComponentsAnalysis" =
  let open Sklearn.Neighbors in
  let open Sklearn.Datasets in
  let open Sklearn.Model_selection in
  let iris = load_iris () in
  let x, y = iris#data, iris#target in
  let [@ocaml.warning "-8"] [x_train; x_test; y_train; y_test] =
    train_test_split [x; y] ~stratify:y ~test_size:(`Float 0.7) ~random_state:42
  in
  let nca = NeighborhoodComponentsAnalysis.create ~random_state:42 () in
  print NeighborhoodComponentsAnalysis.pp @@ NeighborhoodComponentsAnalysis.fit nca ~x:x_train ~y:y_train;
  [%expect {|
            NeighborhoodComponentsAnalysis(callback=None, init='auto', max_iter=50,
                                           n_components=None, random_state=42, tol=1e-05,
                                           verbose=0, warm_start=False)
    |}];
  let knn = KNeighborsClassifier.create ~n_neighbors:3 () in
  print KNeighborsClassifier.pp @@ KNeighborsClassifier.fit knn ~x:(`Arr x_train) ~y:y_train;
  [%expect {|
            KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                 metric_params=None, n_jobs=None, n_neighbors=3, p=2,
                                 weights='uniform')
    |}];
  Format.printf "%g" @@ KNeighborsClassifier.score knn ~x:x_test ~y:y_test;
    [%expect {|
            0.933333
    |}];
  print KNeighborsClassifier.pp @@
  KNeighborsClassifier.fit knn ~x:(`Arr (NeighborhoodComponentsAnalysis.transform nca ~x:x_train)) ~y:y_train;
  [%expect {|
            KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                 metric_params=None, n_jobs=None, n_neighbors=3, p=2,
                                 weights='uniform')
    |}];
  Format.printf "%g" @@ KNeighborsClassifier.score knn ~x:(NeighborhoodComponentsAnalysis.transform nca ~x:x_test) ~y:y_test;
  [%expect {|
            0.961905
    |}]


(* RadiusNeighborsClassifier *)
(*
>>> X = [[0], [1], [2], [3]]
>>> y = [0, 0, 1, 1]
>>> from sklearn.neighbors import RadiusNeighborsClassifier
>>> neigh = RadiusNeighborsClassifier(radius=1.0)
>>> neigh.fit(X, y)
RadiusNeighborsClassifier(...)
>>> print(neigh.predict([[1.5]]))
[0]
>>> print(neigh.predict_proba([[1.0]]))
[[0.66666667 0.33333333]]


*)

let%expect_test "RadiusNeighborsClassifier" =
   let x = matrixi [|[|0|]; [|1|]; [|2|]; [|3|]|] in
   let y = vectori [|0; 0; 1; 1|] in
   let open Sklearn.Neighbors in
   let neigh = RadiusNeighborsClassifier.create ~radius:1.0 () in
   print RadiusNeighborsClassifier.pp @@ RadiusNeighborsClassifier.fit neigh ~x:(`Arr x) ~y;
   [%expect {|
            RadiusNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                      metric_params=None, n_jobs=None, outlier_label=None,
                                      p=2, radius=1.0, weights='uniform')
    |}];
   print_ndarray @@ RadiusNeighborsClassifier.predict neigh ~x:(matrix [|[|1.5|]|]);
   [%expect {|
            [0]
    |}];
   print_ndarray @@ RadiusNeighborsClassifier.predict_proba neigh ~x:(matrix [|[|1.0|]|]);
   [%expect {|
            [[0.66666667 0.33333333]]
    |}]
   

(* RadiusNeighborsRegressor *)
(*
>>> X = [[0], [1], [2], [3]]
>>> y = [0, 0, 1, 1]
>>> from sklearn.neighbors import RadiusNeighborsRegressor
>>> neigh = RadiusNeighborsRegressor(radius=1.0)
>>> neigh.fit(X, y)
RadiusNeighborsRegressor(...)
>>> print(neigh.predict([[1.5]]))
[0.5]


*)

let%expect_test "RadiusNeighborsRegressor" =
  let x = matrixi [|[|0|]; [|1|]; [|2|]; [|3|]|] in
  let y = vectori [|0; 0; 1; 1|] in
  let open Sklearn.Neighbors in
  let neigh = RadiusNeighborsRegressor.create ~radius:1.0 () in
  print RadiusNeighborsRegressor.pp @@ RadiusNeighborsRegressor.fit neigh ~x:(`Arr x) ~y;
  [%expect {|
            RadiusNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
                                     metric_params=None, n_jobs=None, p=2, radius=1.0,
                                     weights='uniform')
    |}];
  print_ndarray @@ RadiusNeighborsRegressor.predict neigh ~x:(matrix [|[|1.5|]|]);
  [%expect {|
            [0.5]
    |}]


(* kneighbors_graph *)
(*
>>> X = [[0], [3], [1]]
>>> from sklearn.neighbors import kneighbors_graph
>>> A = kneighbors_graph(X, 2, mode='connectivity', include_self=True)
>>> A.toarray()
array([[1., 0., 1.],
       [0., 1., 1.],
       [1., 0., 1.]])


*)

let%expect_test "kneighbors_graph" =
  let x = matrixi [|[|0|]; [|3|]; [|1|]|] in
  let a =
    Sklearn.Neighbors.kneighbors_graph ~x:(`Arr x) ~n_neighbors:2
      ~mode:`Connectivity ~include_self:(`Bool true) ()
  in
  Sklearn.Ndarray.pp Format.std_formatter @@ Sklearn.Csr_matrix.toarray @@ a;
  [%expect {|
            [[1. 0. 1.]
             [0. 1. 1.]
             [1. 0. 1.]]
    |}]


(* radius_neighbors_graph *)
(*
>>> X = [[0], [3], [1]]
>>> from sklearn.neighbors import radius_neighbors_graph
>>> A = radius_neighbors_graph(X, 1.5, mode='connectivity',
...                            include_self=True)
>>> A.toarray()
array([[1., 0., 1.],
       [0., 1., 0.],
       [1., 0., 1.]])


*)

let%expect_test "radius_neighbors_graph" =
  let x = matrixi [|[|0|]; [|3|]; [|1|]|] in
  let a = Sklearn.Neighbors.radius_neighbors_graph ~x:(`Arr x) ~radius:1.5
      ~mode:`Connectivity ~include_self:(`Bool true) ()
  in
  Sklearn.Ndarray.pp Format.std_formatter @@ Sklearn.Csr_matrix.toarray a;
  [%expect {|
            [[1. 0. 1.]
             [0. 1. 0.]
             [1. 0. 1.]]
    |}]
