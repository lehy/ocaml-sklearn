module Np = Np.Numpy

let print f x = Format.printf "%a" f x

let print_py x = Format.printf "%s" (Py.Object.to_string x)

let print_ndarray = Np.Obj.print

let print_float = Format.printf "%g\n"

let print_string = Format.printf "%s\n"

let print_int = Format.printf "%d\n"

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

let%expect_test "Isomap" =
  let open Sklearn.Manifold in
  let digits = Sklearn.Datasets.load_digits () in
  let x = digits#data in
  print_ndarray @@ Np.(shape x |> vectori);
  [%expect {|
      [1797   64]
  |}];
  let embedding = Isomap.create ~n_components:2 () in
  let x_transformed = Isomap.fit_transform ~x:Np.(Ndarray.get x ~key:[slice ~j:100 ()]) embedding in
  print_ndarray @@ Np.(shape x_transformed |> vectori);
  [%expect {|
      [100   2]
  |}]


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

let%expect_test "LocallyLinearEmbedding" =
  let open Sklearn.Manifold in
  let digits = Sklearn.Datasets.load_digits () in
  let x = digits#data in
  print_ndarray @@ Np.(shape x |> vectori);
  [%expect {|
      [1797   64]
  |}];
  let embedding = LocallyLinearEmbedding.create ~n_components:2 () in
  let x_transformed = LocallyLinearEmbedding.fit_transform ~x:Np.(Ndarray.get x ~key:[slice ~j:100 ()]) embedding in
  print_ndarray @@ Np.(shape x_transformed |> vectori);
  [%expect {|
      [100   2]
  |}]


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

let%expect_test "MDS" =
  let open Sklearn.Manifold in
  let digits = Sklearn.Datasets.load_digits () in
  let x = digits#data in
  print_ndarray @@ Np.(shape x |> vectori);
  [%expect {|
      [1797   64]
  |}];
  let embedding = MDS.create ~n_components:2 () in
  let x_transformed = MDS.fit_transform ~x:Np.(Ndarray.get x ~key:[slice ~j:100 ()]) embedding in
  print_ndarray @@ Np.(shape x_transformed |> vectori);
  [%expect {|
      [100   2]
  |}]


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


let%expect_test "SpectralEmbedding" =
  let open Sklearn.Manifold in
  let digits = Sklearn.Datasets.load_digits () in
  let x = digits#data in
  print_ndarray @@ Np.(shape x |> vectori);
  [%expect {|
      [1797   64]
  |}];
  let embedding = SpectralEmbedding.create ~n_components:2 () in
  let x_transformed = SpectralEmbedding.fit_transform ~x:Np.(Ndarray.get x ~key:[slice ~j:100 ()]) embedding in
  print_ndarray @@ Np.(shape x_transformed |> vectori);
  [%expect {|
      [100   2]
  |}]


(* TSNE *)
(*
>>> import numpy as np
>>> from sklearn.manifold import TSNE
>>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
>>> X_embedded = TSNE(n_components=2).fit_transform(X)
>>> X_embedded.shape
(4, 2)

*)

let%expect_test "TSNE" =
  let open Sklearn.Manifold in
  let x = Np.matrixi [|[|0; 0; 0|]; [|0; 1; 1|]; [|1; 0; 1|]; [|1; 1; 1|]|] in
  let x_embedded = TSNE.(create ~n_components:2 () |> fit_transform ~x) in
  print_ndarray @@ Np.(shape x_embedded |> vectori);
  [%expect {|
      [4 2]
  |}]
