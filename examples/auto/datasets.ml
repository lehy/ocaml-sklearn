(* load_boston *)
(*
>>> from sklearn.datasets import load_boston
>>> X, y = load_boston(return_X_y=True)
>>> print(X.shape)

*)

(* TEST TODO
let%expect_test "load_boston" =
  let open Sklearn.Datasets in
  let x, y = load_boston ~return_X_y:true () in  
  print_ndarray @@ print x.shape ();  
  [%expect {|
  |}]

*)



(* load_breast_cancer *)
(*
>>> from sklearn.datasets import load_breast_cancer
>>> data = load_breast_cancer()
>>> data.target[[10, 50, 85]]
array([0, 1, 0])
>>> list(data.target_names)

*)

(* TEST TODO
let%expect_test "load_breast_cancer" =
  let open Sklearn.Datasets in
  let data = load_breast_cancer () in  
  print_ndarray @@ .target matrixi [|[|10; 50; 85|]|] data;  
  [%expect {|
      array([0, 1, 0])      
  |}]
  print_ndarray @@ list data.target_names ();  
  [%expect {|
  |}]

*)



(* load_iris *)
(*
>>> from sklearn.datasets import load_iris
>>> data = load_iris()
>>> data.target[[10, 25, 50]]
array([0, 0, 1])
>>> list(data.target_names)

*)

(* TEST TODO
let%expect_test "load_iris" =
  let open Sklearn.Datasets in
  let data = load_iris () in  
  print_ndarray @@ .target matrixi [|[|10; 25; 50|]|] data;  
  [%expect {|
      array([0, 0, 1])      
  |}]
  print_ndarray @@ list data.target_names ();  
  [%expect {|
  |}]

*)



(* load_sample_image *)
(*
>>> from sklearn.datasets import load_sample_image
>>> china = load_sample_image('china.jpg')   # doctest: +SKIP
>>> china.dtype                              # doctest: +SKIP
dtype('uint8')
>>> china.shape                              # doctest: +SKIP
(427, 640, 3)
>>> flower = load_sample_image('flower.jpg') # doctest: +SKIP
>>> flower.dtype                             # doctest: +SKIP
dtype('uint8')
>>> flower.shape                             # doctest: +SKIP

*)

(* TEST TODO
let%expect_test "load_sample_image" =
  let open Sklearn.Datasets in
  let china = load_sample_image 'china.jpg' () # doctest: +SKIP in  
  print_ndarray @@ china.dtype # doctest: +SKIP;  
  [%expect {|
      dtype('uint8')      
  |}]
  print_ndarray @@ china.shape # doctest: +SKIP;  
  [%expect {|
      (427, 640, 3)      
  |}]
  let flower = load_sample_image 'flower.jpg' () # doctest: +SKIP in  
  print_ndarray @@ flower.dtype # doctest: +SKIP;  
  [%expect {|
      dtype('uint8')      
  |}]
  print_ndarray @@ flower.shape # doctest: +SKIP;  
  [%expect {|
  |}]

*)



(* load_sample_images *)
(*
>>> from sklearn.datasets import load_sample_images
>>> dataset = load_sample_images()     #doctest: +SKIP
>>> len(dataset.images)                #doctest: +SKIP
2
>>> first_img_data = dataset.images[0] #doctest: +SKIP
>>> first_img_data.shape               #doctest: +SKIP
(427, 640, 3)
>>> first_img_data.dtype               #doctest: +SKIP

*)

(* TEST TODO
let%expect_test "load_sample_images" =
  let open Sklearn.Datasets in
  let dataset = load_sample_images () #doctest: +SKIP in  
  print_ndarray @@ len dataset.images () #doctest: +SKIP;  
  [%expect {|
      2      
  |}]
  let first_img_data = .images vectori [|0|] dataset #doctest: +SKIP in  
  print_ndarray @@ first_img_data.shape #doctest: +SKIP;  
  [%expect {|
      (427, 640, 3)      
  |}]
  print_ndarray @@ first_img_data.dtype #doctest: +SKIP;  
  [%expect {|
  |}]

*)



(* load_wine *)
(*
>>> from sklearn.datasets import load_wine
>>> data = load_wine()
>>> data.target[[10, 80, 140]]
array([0, 1, 2])
>>> list(data.target_names)

*)

(* TEST TODO
let%expect_test "load_wine" =
  let open Sklearn.Datasets in
  let data = load_wine () in  
  print_ndarray @@ .target matrixi [|[|10; 80; 140|]|] data;  
  [%expect {|
      array([0, 1, 2])      
  |}]
  print_ndarray @@ list data.target_names ();  
  [%expect {|
  |}]

*)



(* make_blobs *)
(*
>>> from sklearn.datasets import make_blobs
>>> X, y = make_blobs(n_samples=10, centers=3, n_features=2,
...                   random_state=0)
>>> print(X.shape)
(10, 2)
>>> y
array([0, 0, 1, 0, 2, 2, 2, 1, 1, 0])
>>> X, y = make_blobs(n_samples=[3, 3, 4], centers=None, n_features=2,
...                   random_state=0)
>>> print(X.shape)
(10, 2)
>>> y
array([0, 1, 2, 0, 2, 2, 2, 1, 1, 0])

*)

(* TEST TODO
let%expect_test "make_blobs" =
  let open Sklearn.Datasets in
  let x, y = make_blobs ~n_samples:10 ~centers:3 ~n_features:2 ~random_state:0 () in  
  print_ndarray @@ print x.shape ();  
  [%expect {|
      (10, 2)      
  |}]
  print_ndarray @@ y;  
  [%expect {|
      array([0, 0, 1, 0, 2, 2, 2, 1, 1, 0])      
  |}]
  let x, y = make_blobs(n_samples=(vectori [|3; 3; 4|]), centers=None, n_features=2,random_state=0) in  
  print_ndarray @@ print x.shape ();  
  [%expect {|
      (10, 2)      
  |}]
  print_ndarray @@ y;  
  [%expect {|
      array([0, 1, 2, 0, 2, 2, 2, 1, 1, 0])      
  |}]

*)



