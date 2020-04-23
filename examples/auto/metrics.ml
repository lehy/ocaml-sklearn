let print f x = Format.printf "%a" f x
let print_py x = Format.printf "%s" (Py.Object.to_string x)
let print_ndarray = print Sklearn.Arr.pp
let print_float = Format.printf "%g\n"

let matrix = Sklearn.Arr.Float.matrix
let vector = Sklearn.Arr.Float.vector
let matrixi = Sklearn.Arr.Int.matrix
let vectori = Sklearn.Arr.Int.vector
let vectors = Sklearn.Arr.String.vector

(* RocCurveDisplay *)
(*
>>> import matplotlib.pyplot as plt  # doctest: +SKIP
>>> import numpy as np
>>> from sklearn import metrics
>>> y = np.array([0, 0, 1, 1])
>>> pred = np.array([0.1, 0.4, 0.35, 0.8])
>>> fpr, tpr, thresholds = metrics.roc_curve(y, pred)
>>> roc_auc = metrics.auc(fpr, tpr)
>>> display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,                                          estimator_name='example estimator')
>>> display.plot()  # doctest: +SKIP

*)

let%expect_test "RocCurveDisplay" =
  let open Sklearn.Metrics in
  let y = vectori [|0; 0; 1; 1|] in
  let pred = vector [|0.1; 0.4; 0.35; 0.8|] in
  let fpr, tpr, _thresholds = roc_curve ~y_true:y ~y_score:pred () in
  print_ndarray fpr;
  [%expect {| [0.  0.  0.5 0.5 1. ] |}];
  print_ndarray tpr;
  [%expect {| [0.  0.5 0.5 1.  1. ] |}];
  let roc_auc = auc ~x:fpr ~y:tpr () in
  Format.printf "%g\n" roc_auc;
  [%expect {| 0.75 |}];
  let display = RocCurveDisplay.create ~fpr:fpr ~tpr:tpr ~roc_auc:roc_auc ~estimator_name:"example estimator" () in
  (*  plot() using TkAgg needs a non-empty sys.argv else all we get is an exception :( *)
  let _ = Py.Run.eval ~start:Py.File "import sys\nif not sys.argv:\n    sys.argv = ['']" in
  begin try
      let _ = RocCurveDisplay.plot display in ();
    with Py.E _ ->
      Sklearn.Wrap_utils.print_python_traceback ()
  end;
  (* This is empty to check that no exception is thrown. Adding
     Matplotlib.Mpl.show () after this in utop after #require
     "matplotlib" works :). *)
  [%expect {| |}]


(* accuracy_score *)
(*
>>> from sklearn.metrics import accuracy_score
>>> y_pred = [0, 2, 1, 3]
>>> y_true = [0, 1, 2, 3]
>>> accuracy_score(y_true, y_pred)
0.5
>>> accuracy_score(y_true, y_pred, normalize=False)
2

*)

let%expect_test "accuracy_score" =
  let open Sklearn.Metrics in
  let y_pred = vectori [|0; 2; 1; 3|] in
  let y_true = vectori [|0; 1; 2; 3|] in
  Format.printf "%g\n" @@ accuracy_score ~y_true ~y_pred ();
  [%expect {|
      0.5
  |}];
  Format.printf "%g\n" @@ accuracy_score ~y_true ~y_pred ~normalize:false ();
  [%expect {|
      2
  |}]


(* accuracy_score *)
(*
>>> import numpy as np
>>> accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))

*)

let%expect_test "accuracy_score" =
  let open Sklearn.Metrics in
  Format.printf "%g\n" @@
  accuracy_score ~y_true:(matrixi [|[|0; 1|]; [|1; 1|]|]) ~y_pred:(Sklearn.Arr.ones [2; 2]) ();
  [%expect {| 0.5 |}]

(* auc *)
(*
>>> import numpy as np
>>> from sklearn import metrics
>>> y = np.array([1, 1, 2, 2])
>>> pred = np.array([0.1, 0.4, 0.35, 0.8])
>>> fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
>>> metrics.auc(fpr, tpr)
0.75

*)

let%expect_test "auc" =
  let open Sklearn.Metrics in
  let y = vectori [|1; 1; 2; 2|] in
  let pred = vector [|0.1; 0.4; 0.35; 0.8|] in
  let fpr, tpr, _thresholds = roc_curve ~y_true:y ~y_score:pred ~pos_label:(`I 2) () in
  Format.printf "%g\n" @@ auc ~x:fpr ~y:tpr ();
  [%expect {|
      0.75
  |}]

(* average_precision_score *)
(*
>>> import numpy as np
>>> from sklearn.metrics import average_precision_score
>>> y_true = np.array([0, 0, 1, 1])
>>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
>>> average_precision_score(y_true, y_scores)
0.83...

*)

let%expect_test "average_precision_score" =
  let open Sklearn.Metrics in
  let y_true = vectori [|0; 0; 1; 1|] in
  let y_score = vector [|0.1; 0.4; 0.35; 0.8|]in
  Printf.printf "%g" @@ average_precision_score ~y_true ~y_score ();
  [%expect {|
      0.833333
  |}]


(* balanced_accuracy_score *)
(*
>>> from sklearn.metrics import balanced_accuracy_score
>>> y_true = [0, 1, 0, 0, 1, 0]
>>> y_pred = [0, 1, 0, 0, 0, 1]
>>> balanced_accuracy_score(y_true, y_pred)

*)

let%expect_test "balanced_accuracy_score" =
  let open Sklearn.Metrics in
  let y_true = vectori [|0; 1; 0; 0; 1; 0|] in
  let y_pred = vectori [|0; 1; 0; 0; 0; 1|] in
  Format.printf "%g" @@ balanced_accuracy_score ~y_true ~y_pred ();
  [%expect {| 0.625 |}]


(* brier_score_loss *)
(*
>>> import numpy as np
>>> from sklearn.metrics import brier_score_loss
>>> y_true = np.array([0, 1, 1, 0])
>>> y_true_categorical = np.array(["spam", "ham", "ham", "spam"])
>>> y_prob = np.array([0.1, 0.9, 0.8, 0.3])
>>> brier_score_loss(y_true, y_prob)
0.037...
>>> brier_score_loss(y_true, 1-y_prob, pos_label=0)
0.037...
>>> brier_score_loss(y_true_categorical, y_prob, pos_label="ham")
0.037...
>>> brier_score_loss(y_true, np.array(y_prob) > 0.5)
0.0

*)

let%expect_test "brier_score_loss" =
  let open Sklearn.Metrics in
  let y_true = vectori [|0; 1; 1; 0|] in
  let y_true_categorical = vectors [|"spam"; "ham"; "ham"; "spam"|] in
  let y_prob = vector [|0.1; 0.9; 0.8; 0.3|] in
  Printf.printf "%g" @@ brier_score_loss ~y_true ~y_prob ();
  [%expect {|
      0.0375
  |}];
  print_float @@ brier_score_loss ~y_true ~y_prob:Sklearn.Arr.Ops.((int 1)-y_prob) ~pos_label:(`I 0) ();
  [%expect {|
      0.0375
  |}];
  print_float @@ brier_score_loss ~y_true:y_true_categorical ~y_prob ~pos_label:(`S "ham") ();
  [%expect {|
      0.0375
  |}];
  print_float @@ brier_score_loss ~y_true ~y_prob:Sklearn.Arr.Ops.(y_prob > float 0.5) ();
  [%expect {|
      0
  |}]


(* classification_report *)
(*
>>> from sklearn.metrics import classification_report
>>> y_true = [0, 1, 2, 2, 2]
>>> y_pred = [0, 0, 2, 2, 1]
>>> target_names = ['class 0', 'class 1', 'class 2']
>>> print(classification_report(y_true, y_pred, target_names=target_names))
              precision    recall  f1-score   support
<BLANKLINE>
     class 0       0.50      1.00      0.67         1
     class 1       0.00      0.00      0.00         1
     class 2       1.00      0.67      0.80         3
<BLANKLINE>
    accuracy                           0.60         5
   macro avg       0.50      0.56      0.49         5
weighted avg       0.70      0.60      0.61         5
<BLANKLINE>
>>> y_pred = [1, 1, 0]
>>> y_true = [1, 1, 1]
>>> print(classification_report(y_true, y_pred, labels=[1, 2, 3]))
              precision    recall  f1-score   support
<BLANKLINE>
           1       1.00      0.67      0.80         3
           2       0.00      0.00      0.00         0
           3       0.00      0.00      0.00         0
<BLANKLINE>
   micro avg       1.00      0.67      0.80         3
   macro avg       0.33      0.22      0.27         3
weighted avg       1.00      0.67      0.80         3

*)

let print_report report =
  let print_values v =
    Format.printf "%g %g %g %g\n" v#precision v#recall v#f1_score v#support
  in
  let print_line (k, v) =
    Format.printf "%s " k;
    print_values v;
  in
  List.iter print_line report

let%expect_test "classification_report" =
  let open Sklearn.Metrics in
  let y_true = vectori [|0; 1; 2; 2; 2|] in
  let y_pred = vectori [|0; 0; 2; 2; 1|] in
  let target_names = vectors [|"class 0"; "class 1"; "class 2"|] in
  begin match classification_report ~y_true ~y_pred ~target_names () with
    | `Dict _ -> assert false
    | `S report -> print_endline report
  end;
  [%expect {|
                    precision    recall  f1-score   support

           class 0       0.50      1.00      0.67         1
           class 1       0.00      0.00      0.00         1
           class 2       1.00      0.67      0.80         3

          accuracy                           0.60         5
         macro avg       0.50      0.56      0.49         5
      weighted avg       0.70      0.60      0.61         5
  |}];
  let y_pred = vectori [|1; 1; 0|] in
  let y_true = vectori [|1; 1; 1|] in
  begin match classification_report ~y_true ~y_pred ~labels:(vectori [|1; 2; 3|]) () with
  | `Dict _ -> assert false
  | `S report -> print_endline report
  end;
  [%expect {|
                    precision    recall  f1-score   support

                 1       1.00      0.67      0.80         3
                 2       0.00      0.00      0.00         0
                 3       0.00      0.00      0.00         0

         micro avg       1.00      0.67      0.80         3
         macro avg       0.33      0.22      0.27         3
      weighted avg       1.00      0.67      0.80         3
  |}];
  begin match classification_report ~output_dict:true ~y_true ~y_pred ~labels:(vectori [|1; 2; 3|]) () with
    | `Dict report -> print_report report
    | `S _ -> assert false
  end;
  [%expect {|
    weighted avg 1 0.666667 0.8 3
    macro avg 0.333333 0.222222 0.266667 3
    micro avg 1 0.666667 0.8 3
    3 0 0 0 0
    2 0 0 0 0
    1 1 0.666667 0.8 3
  |}]


(*--------- Examples for module Sklearn.Metrics.Cluster ----------*)
(* confusion_matrix *)
(*
>>> from sklearn.metrics import confusion_matrix
>>> y_true = [2, 0, 2, 2, 0, 1]
>>> y_pred = [0, 0, 2, 2, 0, 2]
>>> confusion_matrix(y_true, y_pred)
array([[2, 0, 0],
       [0, 0, 1],
       [1, 0, 2]])

*)

let%expect_test "confusion_matrix" =
  let open Sklearn.Metrics in
  let y_true = vectori [|2; 0; 2; 2; 0; 1|] in
  let y_pred = vectori [|0; 0; 2; 2; 0; 2|] in
  print_ndarray @@ confusion_matrix ~y_true ~y_pred ();
  [%expect {|
      [[2 0 0]
       [0 0 1]
       [1 0 2]]
  |}]


(* confusion_matrix *)
(*
>>> y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
>>> y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
>>> confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
array([[2, 0, 0],
       [0, 0, 1],
       [1, 0, 2]])

*)

let%expect_test "confusion_matrix" =
  let open Sklearn.Metrics in
  let y_true = vectors [|"cat"; "ant"; "cat"; "cat"; "ant"; "bird"|] in
  let y_pred = vectors [|"ant"; "ant"; "cat"; "cat"; "ant"; "cat"|] in
  print_ndarray @@ confusion_matrix ~y_true ~y_pred ~labels:(vectors [|"ant"; "bird"; "cat"|]) ();
  [%expect {|
      [[2 0 0]
       [0 0 1]
       [1 0 2]]
  |}]

(* confusion_matrix *)
(*
>>> tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
>>> (tn, fp, fn, tp)

*)

let%expect_test "confusion_matrix" =
  let open Sklearn.Metrics in
  let conf = confusion_matrix ~y_true:(vectori [|0; 1; 0; 1|]) ~y_pred:(vectori [|1; 1; 1; 0|]) () in
  let [@ocaml.warning "-8"] [|tn; fp; fn; tp|] =
    conf |> Sklearn.Arr.get_ndarray |> Sklearn.Ndarray.to_int_array
  in
  print_ndarray conf;
  [%expect {|
    [[0 2]
     [1 1]]
  |}];
  Format.printf "%d, %d, %d, %d" tn fp fn tp;
  [%expect {| 0, 2, 1, 1 |}]


(* dcg_score *)
(*
>>> from sklearn.metrics import dcg_score
>>> # we have groud-truth relevance of some answers to a query:
>>> true_relevance = np.asarray([[10, 0, 0, 1, 5]])
>>> # we predict scores for the answers
>>> scores = np.asarray([[.1, .2, .3, 4, 70]])
>>> dcg_score(true_relevance, scores) # doctest: +ELLIPSIS
9.49...
>>> # we can set k to truncate the sum; only top k answers contribute
>>> dcg_score(true_relevance, scores, k=2) # doctest: +ELLIPSIS
5.63...
>>> # now we have some ties in our prediction
>>> scores = np.asarray([[1, 0, 0, 0, 1]])
>>> # by default ties are averaged, so here we get the average true
>>> # relevance of our top predictions: (10 + 5) / 2 = 7.5
>>> dcg_score(true_relevance, scores, k=1) # doctest: +ELLIPSIS
7.5
>>> # we can choose to ignore ties for faster results, but only
>>> # if we know there aren't ties in our scores, otherwise we get
>>> # wrong results:
>>> dcg_score(true_relevance,
...           scores, k=1, ignore_ties=True) # doctest: +ELLIPSIS

*)

let%expect_test "dcg_score" =
  let open Sklearn.Metrics in
  (* we have groud-truth relevance of some answers to a query *)
  let true_relevance = matrixi [|[|10; 0; 0; 1; 5|]|] in
  (* we predict scores for the answers *)
  let scores = matrix [|[|0.1; 0.2; 0.3; 4.; 70.|]|] in
  print_float @@ dcg_score ~y_true:true_relevance ~y_score:scores ();
  [%expect {|
      9.49946
  |}];
  (* we can set k to truncate the sum; only top k answers contribute *)
  print_float @@ dcg_score ~y_true:true_relevance ~y_score:scores ~k:2 ();
  [%expect {|
      5.63093
  |}];
  (* now we have some ties in our prediction *)
  let scores = matrixi [|[|1; 0; 0; 0; 1|]|] in
  (* by default ties are averaged, so here we get the average true *)
  (* relevance of our top predictions: (10 + 5) / 2 = 7.5 *)
  print_float @@ dcg_score ~y_true:true_relevance ~y_score:scores ~k:1 ();
  [%expect {|
      7.5
  |}];
  (* we can choose to ignore ties for faster results, but only *)
  (* if we know there aren't ties in our scores, otherwise we get *)
  (* wrong results: *)
  print_float @@ dcg_score ~y_true:true_relevance ~y_score:scores ~k:1 ~ignore_ties:true ();
  [%expect {| 5 |}]


(* euclidean_distances *)
(*
>>> from sklearn.metrics.pairwise import euclidean_distances
>>> X = [[0, 1], [1, 1]]
>>> # distance between rows of X
>>> euclidean_distances(X, X)
array([[0., 1.],
       [1., 0.]])
>>> # get distance to origin
>>> euclidean_distances(X, [[0, 0]])
array([[1.        ],
       [1.41421356]])

*)

let%expect_test "euclidean_distances" =
  let open Sklearn.Metrics in
  let x = matrixi [|[|0; 1|]; [|1; 1|]|] in
  (* distance between rows of x *)
  print_ndarray @@ euclidean_distances ~x ~y:x ();
  [%expect {|
      [[0. 1.]
       [1. 0.]]
  |}];
  (* get distance to origin *)
  print_ndarray @@ euclidean_distances ~x ~y:(matrixi [|[|0; 0|]|]) ();
  [%expect {|
      [[1.        ]
       [1.41421356]]
  |}]


(* explained_variance_score *)
(*
>>> from sklearn.metrics import explained_variance_score
>>> y_true = [3, -0.5, 2, 7]
>>> y_pred = [2.5, 0.0, 2, 8]
>>> explained_variance_score(y_true, y_pred)
0.957...
>>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
>>> y_pred = [[0, 2], [-1, 2], [8, -5]]
>>> explained_variance_score(y_true, y_pred, multioutput='uniform_average')

*)

let%expect_test "explained_variance_score" =
  let open Sklearn.Metrics in
  let y_true = vector [|3.; -0.5; 2.; 7.|] in
  let y_pred = vector [|2.5; 0.0; 2.; 8.|] in
  begin match explained_variance_score ~y_true ~y_pred () with
  | `Arr _ -> assert false
  | `F f -> print_float f
  end;
  [%expect {|
      0.957173
  |}];
  let y_true = matrix [|[|0.5; 1.|]; [|-1.; 1.|]; [|7.; -6.|]|] in
  let y_pred = matrixi [|[|0; 2|]; [|-1; 2|]; [|8; -5|]|] in
  begin match explained_variance_score ~y_true ~y_pred ~multioutput:`Uniform_average () with
  | `Arr _ -> assert false
  | `F f -> print_float f
  end;
  [%expect {| 0.983871 |}]

(* f1_score *)
(*
>>> from sklearn.metrics import f1_score
>>> y_true = [0, 1, 2, 0, 1, 2]
>>> y_pred = [0, 2, 1, 0, 0, 1]
>>> f1_score(y_true, y_pred, average='macro')
0.26...
>>> f1_score(y_true, y_pred, average='micro')
0.33...
>>> f1_score(y_true, y_pred, average='weighted')
0.26...
>>> f1_score(y_true, y_pred, average=None)
array([0.8, 0. , 0. ])
>>> y_true = [0, 0, 0, 0, 0, 0]
>>> y_pred = [0, 0, 0, 0, 0, 0]
>>> f1_score(y_true, y_pred, zero_division=1)
1.0...

*)

let print_f = function
  | `F x -> print_float x
  | `Arr _ -> assert false

let print_arr = function
  | `F _ -> assert false
  | `Arr x -> print_ndarray x

let%expect_test "f1_score" =
  let open Sklearn.Metrics in
  let y_true = vectori [|0; 1; 2; 0; 1; 2|] in
  let y_pred = vectori [|0; 2; 1; 0; 0; 1|] in
  print_f @@ f1_score ~y_true ~y_pred ~average:`Macro ();
  [%expect {|
      0.266667
  |}];
  print_f @@ f1_score ~y_true ~y_pred ~average:`Micro ();
  [%expect {|
      0.333333
  |}];
  print_f @@ f1_score ~y_true ~y_pred ~average:`Weighted ();
  [%expect {|
      0.266667
  |}];
  print_arr @@ f1_score ~y_true ~y_pred ~average:`None ();
  [%expect {|
      [0.8 0.  0. ]
  |}];
  let y_true = vectori [|0; 0; 0; 0; 0; 0|] in
  let y_pred = vectori [|0; 0; 0; 0; 0; 0|] in
  print_f @@ f1_score ~y_true ~y_pred ~zero_division:`One ();
  [%expect {|
      1
  |}]


(* fbeta_score *)
(*
>>> from sklearn.metrics import fbeta_score
>>> y_true = [0, 1, 2, 0, 1, 2]
>>> y_pred = [0, 2, 1, 0, 0, 1]
>>> fbeta_score(y_true, y_pred, average='macro', beta=0.5)
0.23...
>>> fbeta_score(y_true, y_pred, average='micro', beta=0.5)
0.33...
>>> fbeta_score(y_true, y_pred, average='weighted', beta=0.5)
0.23...
>>> fbeta_score(y_true, y_pred, average=None, beta=0.5)
array([0.71..., 0.        , 0.        ])

*)

(* TEST TODO
let%expect_test "fbeta_score" =
  let open Sklearn.Metrics in
  let y_true = (vectori [|0; 1; 2; 0; 1; 2|]) in
  let y_pred = (vectori [|0; 2; 1; 0; 0; 1|]) in
  print_ndarray @@ fbeta_score ~y_true y_pred ~average:'macro' ~beta:0.5 ();
  [%expect {|
      0.23...
  |}]
  print_ndarray @@ fbeta_score ~y_true y_pred ~average:'micro' ~beta:0.5 ();
  [%expect {|
      0.33...
  |}]
  print_ndarray @@ fbeta_score ~y_true y_pred ~average:'weighted' ~beta:0.5 ();
  [%expect {|
      0.23...
  |}]
  print_ndarray @@ fbeta_score ~y_true y_pred ~average:None ~beta:0.5 ();
  [%expect {|
      array([0.71..., 0.        , 0.        ])
  |}]

*)



(* hamming_loss *)
(*
>>> from sklearn.metrics import hamming_loss
>>> y_pred = [1, 2, 3, 4]
>>> y_true = [2, 2, 3, 4]
>>> hamming_loss(y_true, y_pred)
0.25

*)

(* TEST TODO
let%expect_test "hamming_loss" =
  let open Sklearn.Metrics in
  let y_pred = (vectori [|1; 2; 3; 4|]) in
  let y_true = (vectori [|2; 2; 3; 4|]) in
  print_ndarray @@ hamming_loss ~y_true y_pred ();
  [%expect {|
      0.25
  |}]

*)



(* hamming_loss *)
(*
>>> import numpy as np
>>> hamming_loss(np.array([[0, 1], [1, 1]]), np.zeros((2, 2)))

*)

(* TEST TODO
let%expect_test "hamming_loss" =
  let open Sklearn.Metrics in
  print_ndarray @@ hamming_loss(.array (matrixi [|[|0; 1|]; [|1; 1|]|])) np.zeros((2 2)) np;
  [%expect {|
  |}]

*)



(* hinge_loss *)
(*
>>> from sklearn import svm
>>> from sklearn.metrics import hinge_loss
>>> X = [[0], [1]]
>>> y = [-1, 1]
>>> est = svm.LinearSVC(random_state=0)
>>> est.fit(X, y)
LinearSVC(random_state=0)
>>> pred_decision = est.decision_function([[-2], [3], [0.5]])
>>> pred_decision
array([-2.18...,  2.36...,  0.09...])
>>> hinge_loss([-1, 1, 1], pred_decision)
0.30...

*)

(* TEST TODO
let%expect_test "hinge_loss" =
  let open Sklearn.Metrics in
  let x = (matrixi [|[|0|]; [|1|]|]) in
  let y = [-1, 1] in
  let est = .linearSVC ~random_state:0 svm in
  print_ndarray @@ .fit ~x y est;
  [%expect {|
      LinearSVC(random_state=0)
  |}]
  let pred_decision = .decision_function (matrix [|[|-2|]; [|3|]; [|0.5|]|]) est in
  print_ndarray @@ pred_decision;
  [%expect {|
      array([-2.18...,  2.36...,  0.09...])
  |}]
  print_ndarray @@ hinge_loss [-1 ~1 1] ~pred_decision ();
  [%expect {|
      0.30...
  |}]

*)



(* hinge_loss *)
(*
>>> import numpy as np
>>> X = np.array([[0], [1], [2], [3]])
>>> Y = np.array([0, 1, 2, 3])
>>> labels = np.array([0, 1, 2, 3])
>>> est = svm.LinearSVC()
>>> est.fit(X, Y)
LinearSVC()
>>> pred_decision = est.decision_function([[-1], [2], [3]])
>>> y_true = [0, 2, 3]
>>> hinge_loss(y_true, pred_decision, labels)

*)

(* TEST TODO
let%expect_test "hinge_loss" =
  let open Sklearn.Metrics in
  let x = .array (matrixi [|[|0|]; [|1|]; [|2|]; [|3|]|]) np in
  let Y = .array (vectori [|0; 1; 2; 3|]) np in
  let labels = .array (vectori [|0; 1; 2; 3|]) np in
  let est = .linearSVC svm in
  print_ndarray @@ .fit ~x Y est;
  [%expect {|
      LinearSVC()
  |}]
  let pred_decision = .decision_function (matrixi [|[|-1|]; [|2|]; [|3|]|]) est in
  let y_true = (vectori [|0; 2; 3|]) in
  print_ndarray @@ hinge_loss ~y_true pred_decision ~labels ();
  [%expect {|
  |}]

*)



(* jaccard_score *)
(*
>>> import numpy as np
>>> from sklearn.metrics import jaccard_score
>>> y_true = np.array([[0, 1, 1],
...                    [1, 1, 0]])
>>> y_pred = np.array([[1, 1, 1],
...                    [1, 0, 0]])

*)

(* TEST TODO
let%expect_test "jaccard_score" =
  let open Sklearn.Metrics in
  let y_true = .array [(vectori [|0; 1; 1|]) (vectori [|1; 1; 0|])] np in
  let y_pred = .array [(vectori [|1; 1; 1|]) (vectori [|1; 0; 0|])] np in
  [%expect {|
  |}]

*)



(* jaccard_score *)
(*
>>> jaccard_score(y_true[0], y_pred[0])
0.6666...

*)

(* TEST TODO
let%expect_test "jaccard_score" =
  let open Sklearn.Metrics in
  print_ndarray @@ jaccard_score(y_true vectori [|0|] (), y_pred vectori [|0|] ());
  [%expect {|
      0.6666...
  |}]

*)



(* jaccard_score *)
(*
>>> jaccard_score(y_true, y_pred, average='samples')
0.5833...
>>> jaccard_score(y_true, y_pred, average='macro')
0.6666...
>>> jaccard_score(y_true, y_pred, average=None)
array([0.5, 0.5, 1. ])

*)

(* TEST TODO
let%expect_test "jaccard_score" =
  let open Sklearn.Metrics in
  print_ndarray @@ jaccard_score ~y_true y_pred ~average:'samples' ();
  [%expect {|
      0.5833...
  |}]
  print_ndarray @@ jaccard_score ~y_true y_pred ~average:'macro' ();
  [%expect {|
      0.6666...
  |}]
  print_ndarray @@ jaccard_score ~y_true y_pred ~average:None ();
  [%expect {|
      array([0.5, 0.5, 1. ])
  |}]

*)



(* jaccard_score *)
(*
>>> y_pred = [0, 2, 1, 2]
>>> y_true = [0, 1, 2, 2]
>>> jaccard_score(y_true, y_pred, average=None)

*)

(* TEST TODO
let%expect_test "jaccard_score" =
  let open Sklearn.Metrics in
  let y_pred = (vectori [|0; 2; 1; 2|]) in
  let y_true = (vectori [|0; 1; 2; 2|]) in
  print_ndarray @@ jaccard_score ~y_true y_pred ~average:None ();
  [%expect {|
  |}]

*)



(* label_ranking_average_precision_score *)
(*
>>> import numpy as np
>>> from sklearn.metrics import label_ranking_average_precision_score
>>> y_true = np.array([[1, 0, 0], [0, 0, 1]])
>>> y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
>>> label_ranking_average_precision_score(y_true, y_score)

*)

(* TEST TODO
let%expect_test "label_ranking_average_precision_score" =
  let open Sklearn.Metrics in
  let y_true = .array (matrixi [|[|1; 0; 0|]; [|0; 0; 1|]|]) np in
  let y_score = .array (matrix [|[|0.75; 0.5; 1|]; [|1; 0.2; 0.1|]|]) np in
  print_ndarray @@ label_ranking_average_precision_score ~y_true y_score ();
  [%expect {|
  |}]

*)



(* log_loss *)
(*
>>> from sklearn.metrics import log_loss
>>> log_loss(["spam", "ham", "ham", "spam"],
...          [[.1, .9], [.9, .1], [.8, .2], [.35, .65]])
0.21616...

*)

(* TEST TODO
let%expect_test "log_loss" =
  let open Sklearn.Metrics in
  print_ndarray @@ log_loss(["spam", "ham", "ham", "spam"],(matrix [|[|.1; .9|]; [|.9; .1|]; [|.8; .2|]; [|.35; .65|]|]));
  [%expect {|
      0.21616...
  |}]

*)



(* make_scorer *)
(*
>>> from sklearn.metrics import fbeta_score, make_scorer
>>> ftwo_scorer = make_scorer(fbeta_score, beta=2)
>>> ftwo_scorer
make_scorer(fbeta_score, beta=2)
>>> from sklearn.model_selection import GridSearchCV
>>> from sklearn.svm import LinearSVC
>>> grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
...                     scoring=ftwo_scorer)

*)

(* TEST TODO
let%expect_test "make_scorer" =
  let open Sklearn.Metrics in
  let ftwo_scorer = make_scorer fbeta_score ~beta:2 () in
  print_ndarray @@ ftwo_scorer;
  [%expect {|
      make_scorer(fbeta_score, beta=2)
  |}]
  let grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},scoring=ftwo_scorer) in
  [%expect {|
  |}]

*)



(* matthews_corrcoef *)
(*
>>> from sklearn.metrics import matthews_corrcoef
>>> y_true = [+1, +1, +1, -1]
>>> y_pred = [+1, -1, +1, +1]
>>> matthews_corrcoef(y_true, y_pred)

*)

(* TEST TODO
let%expect_test "matthews_corrcoef" =
  let open Sklearn.Metrics in
  let y_true = [+1, +1, +1, -1] in
  let y_pred = [+1, -1, +1, +1] in
  print_ndarray @@ matthews_corrcoef ~y_true y_pred ();
  [%expect {|
  |}]

*)



(* max_error *)
(*
>>> from sklearn.metrics import max_error
>>> y_true = [3, 2, 7, 1]
>>> y_pred = [4, 2, 7, 1]
>>> max_error(y_true, y_pred)

*)

(* TEST TODO
let%expect_test "max_error" =
  let open Sklearn.Metrics in
  let y_true = (vectori [|3; 2; 7; 1|]) in
  let y_pred = (vectori [|4; 2; 7; 1|]) in
  print_ndarray @@ max_error ~y_true y_pred ();
  [%expect {|
  |}]

*)



(* mean_absolute_error *)
(*
>>> from sklearn.metrics import mean_absolute_error
>>> y_true = [3, -0.5, 2, 7]
>>> y_pred = [2.5, 0.0, 2, 8]
>>> mean_absolute_error(y_true, y_pred)
0.5
>>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
>>> y_pred = [[0, 2], [-1, 2], [8, -5]]
>>> mean_absolute_error(y_true, y_pred)
0.75
>>> mean_absolute_error(y_true, y_pred, multioutput='raw_values')
array([0.5, 1. ])
>>> mean_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7])

*)

(* TEST TODO
let%expect_test "mean_absolute_error" =
  let open Sklearn.Metrics in
  let y_true = [3, -0.5, 2, 7] in
  let y_pred = [2.5, 0.0, 2, 8] in
  print_ndarray @@ mean_absolute_error ~y_true y_pred ();
  [%expect {|
      0.5
  |}]
  let y_true = (matrix [|[|0.5; 1|]; [|-1; 1|]; [|7; -6|]|]) in
  let y_pred = (matrixi [|[|0; 2|]; [|-1; 2|]; [|8; -5|]|]) in
  print_ndarray @@ mean_absolute_error ~y_true y_pred ();
  [%expect {|
      0.75
  |}]
  print_ndarray @@ mean_absolute_error ~y_true y_pred ~multioutput:'raw_values' ();
  [%expect {|
      array([0.5, 1. ])
  |}]
  print_ndarray @@ mean_absolute_error ~y_true y_pred ~multioutput:[0.3 0.7] ();
  [%expect {|
  |}]

*)



(* mean_gamma_deviance *)
(*
>>> from sklearn.metrics import mean_gamma_deviance
>>> y_true = [2, 0.5, 1, 4]
>>> y_pred = [0.5, 0.5, 2., 2.]
>>> mean_gamma_deviance(y_true, y_pred)

*)

(* TEST TODO
let%expect_test "mean_gamma_deviance" =
  let open Sklearn.Metrics in
  let y_true = [2, 0.5, 1, 4] in
  let y_pred = [0.5, 0.5, 2., 2.] in
  print_ndarray @@ mean_gamma_deviance ~y_true y_pred ();
  [%expect {|
  |}]

*)



(* mean_poisson_deviance *)
(*
>>> from sklearn.metrics import mean_poisson_deviance
>>> y_true = [2, 0, 1, 4]
>>> y_pred = [0.5, 0.5, 2., 2.]
>>> mean_poisson_deviance(y_true, y_pred)

*)

(* TEST TODO
let%expect_test "mean_poisson_deviance" =
  let open Sklearn.Metrics in
  let y_true = (vectori [|2; 0; 1; 4|]) in
  let y_pred = [0.5, 0.5, 2., 2.] in
  print_ndarray @@ mean_poisson_deviance ~y_true y_pred ();
  [%expect {|
  |}]

*)



(* mean_squared_error *)
(*
>>> from sklearn.metrics import mean_squared_error
>>> y_true = [3, -0.5, 2, 7]
>>> y_pred = [2.5, 0.0, 2, 8]
>>> mean_squared_error(y_true, y_pred)
0.375
>>> y_true = [3, -0.5, 2, 7]
>>> y_pred = [2.5, 0.0, 2, 8]
>>> mean_squared_error(y_true, y_pred, squared=False)
0.612...
>>> y_true = [[0.5, 1],[-1, 1],[7, -6]]
>>> y_pred = [[0, 2],[-1, 2],[8, -5]]
>>> mean_squared_error(y_true, y_pred)
0.708...
>>> mean_squared_error(y_true, y_pred, multioutput='raw_values')
array([0.41666667, 1.        ])
>>> mean_squared_error(y_true, y_pred, multioutput=[0.3, 0.7])

*)

(* TEST TODO
let%expect_test "mean_squared_error" =
  let open Sklearn.Metrics in
  let y_true = [3, -0.5, 2, 7] in
  let y_pred = [2.5, 0.0, 2, 8] in
  print_ndarray @@ mean_squared_error ~y_true y_pred ();
  [%expect {|
      0.375
  |}]
  let y_true = [3, -0.5, 2, 7] in
  let y_pred = [2.5, 0.0, 2, 8] in
  print_ndarray @@ mean_squared_error ~y_true y_pred ~squared:false ();
  [%expect {|
      0.612...
  |}]
  let y_true = [[0.5, 1],[-1, 1],[7, -6]] in
  let y_pred = [(vectori [|0; 2|]),[-1, 2],[8, -5]] in
  print_ndarray @@ mean_squared_error ~y_true y_pred ();
  [%expect {|
      0.708...
  |}]
  print_ndarray @@ mean_squared_error ~y_true y_pred ~multioutput:'raw_values' ();
  [%expect {|
      array([0.41666667, 1.        ])
  |}]
  print_ndarray @@ mean_squared_error ~y_true y_pred ~multioutput:[0.3 0.7] ();
  [%expect {|
  |}]

*)



(* mean_squared_log_error *)
(*
>>> from sklearn.metrics import mean_squared_log_error
>>> y_true = [3, 5, 2.5, 7]
>>> y_pred = [2.5, 5, 4, 8]
>>> mean_squared_log_error(y_true, y_pred)
0.039...
>>> y_true = [[0.5, 1], [1, 2], [7, 6]]
>>> y_pred = [[0.5, 2], [1, 2.5], [8, 8]]
>>> mean_squared_log_error(y_true, y_pred)
0.044...
>>> mean_squared_log_error(y_true, y_pred, multioutput='raw_values')
array([0.00462428, 0.08377444])
>>> mean_squared_log_error(y_true, y_pred, multioutput=[0.3, 0.7])

*)

(* TEST TODO
let%expect_test "mean_squared_log_error" =
  let open Sklearn.Metrics in
  let y_true = [3, 5, 2.5, 7] in
  let y_pred = [2.5, 5, 4, 8] in
  print_ndarray @@ mean_squared_log_error ~y_true y_pred ();
  [%expect {|
      0.039...
  |}]
  let y_true = (matrix [|[|0.5; 1|]; [|1; 2|]; [|7; 6|]|]) in
  let y_pred = (matrix [|[|0.5; 2|]; [|1; 2.5|]; [|8; 8|]|]) in
  print_ndarray @@ mean_squared_log_error ~y_true y_pred ();
  [%expect {|
      0.044...
  |}]
  print_ndarray @@ mean_squared_log_error ~y_true y_pred ~multioutput:'raw_values' ();
  [%expect {|
      array([0.00462428, 0.08377444])
  |}]
  print_ndarray @@ mean_squared_log_error ~y_true y_pred ~multioutput:[0.3 0.7] ();
  [%expect {|
  |}]

*)



(* mean_tweedie_deviance *)
(*
>>> from sklearn.metrics import mean_tweedie_deviance
>>> y_true = [2, 0, 1, 4]
>>> y_pred = [0.5, 0.5, 2., 2.]
>>> mean_tweedie_deviance(y_true, y_pred, power=1)

*)

(* TEST TODO
let%expect_test "mean_tweedie_deviance" =
  let open Sklearn.Metrics in
  let y_true = (vectori [|2; 0; 1; 4|]) in
  let y_pred = [0.5, 0.5, 2., 2.] in
  print_ndarray @@ mean_tweedie_deviance ~y_true y_pred ~power:1 ();
  [%expect {|
  |}]

*)



(* median_absolute_error *)
(*
>>> from sklearn.metrics import median_absolute_error
>>> y_true = [3, -0.5, 2, 7]
>>> y_pred = [2.5, 0.0, 2, 8]
>>> median_absolute_error(y_true, y_pred)
0.5
>>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
>>> y_pred = [[0, 2], [-1, 2], [8, -5]]
>>> median_absolute_error(y_true, y_pred)
0.75
>>> median_absolute_error(y_true, y_pred, multioutput='raw_values')
array([0.5, 1. ])
>>> median_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7])

*)

(* TEST TODO
let%expect_test "median_absolute_error" =
  let open Sklearn.Metrics in
  let y_true = [3, -0.5, 2, 7] in
  let y_pred = [2.5, 0.0, 2, 8] in
  print_ndarray @@ median_absolute_error ~y_true y_pred ();
  [%expect {|
      0.5
  |}]
  let y_true = (matrix [|[|0.5; 1|]; [|-1; 1|]; [|7; -6|]|]) in
  let y_pred = (matrixi [|[|0; 2|]; [|-1; 2|]; [|8; -5|]|]) in
  print_ndarray @@ median_absolute_error ~y_true y_pred ();
  [%expect {|
      0.75
  |}]
  print_ndarray @@ median_absolute_error ~y_true y_pred ~multioutput:'raw_values' ();
  [%expect {|
      array([0.5, 1. ])
  |}]
  print_ndarray @@ median_absolute_error ~y_true y_pred ~multioutput:[0.3 0.7] ();
  [%expect {|
  |}]

*)



(* multilabel_confusion_matrix *)
(*
>>> import numpy as np
>>> from sklearn.metrics import multilabel_confusion_matrix
>>> y_true = np.array([[1, 0, 1],
...                    [0, 1, 0]])
>>> y_pred = np.array([[1, 0, 0],
...                    [0, 1, 1]])
>>> multilabel_confusion_matrix(y_true, y_pred)
array([[[1, 0],
        [0, 1]],
<BLANKLINE>
       [[1, 0],
        [0, 1]],
<BLANKLINE>
       [[0, 1],
        [1, 0]]])

*)

(* TEST TODO
let%expect_test "multilabel_confusion_matrix" =
  let open Sklearn.Metrics in
  let y_true = .array [(vectori [|1; 0; 1|]) (vectori [|0; 1; 0|])] np in
  let y_pred = .array [(vectori [|1; 0; 0|]) (vectori [|0; 1; 1|])] np in
  print_ndarray @@ multilabel_confusion_matrix ~y_true y_pred ();
  [%expect {|
      array([[[1, 0],
              [0, 1]],
      <BLANKLINE>
             [[1, 0],
              [0, 1]],
      <BLANKLINE>
             [[0, 1],
              [1, 0]]])
  |}]

*)



(* multilabel_confusion_matrix *)
(*
>>> y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
>>> y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
>>> multilabel_confusion_matrix(y_true, y_pred,
...                             labels=["ant", "bird", "cat"])
array([[[3, 1],
        [0, 2]],
<BLANKLINE>
       [[5, 0],
        [1, 0]],
<BLANKLINE>
       [[2, 1],

*)

(* TEST TODO
let%expect_test "multilabel_confusion_matrix" =
  let open Sklearn.Metrics in
  let y_true = ["cat", "ant", "cat", "cat", "ant", "bird"] in
  let y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"] in
  print_ndarray @@ multilabel_confusion_matrix ~y_true y_pred ~labels:["ant" "bird" "cat"] ();
  [%expect {|
      array([[[3, 1],
              [0, 2]],
      <BLANKLINE>
             [[5, 0],
              [1, 0]],
      <BLANKLINE>
             [[2, 1],
  |}]

*)



(* nan_euclidean_distances *)
(*
>>> from sklearn.metrics.pairwise import nan_euclidean_distances
>>> nan = float("NaN")
>>> X = [[0, 1], [1, nan]]
>>> nan_euclidean_distances(X, X) # distance between rows of X
array([[0.        , 1.41421356],
       [1.41421356, 0.        ]])

*)

(* TEST TODO
let%expect_test "nan_euclidean_distances" =
  let open Sklearn.Metrics in
  let nan = float "NaN" () in
  let x = (matrixi [|[|0; 1|]; [|1; nan|]|]) in
  print_ndarray @@ nan_euclidean_distances ~x x () # distance between rows of x;
  [%expect {|
      array([[0.        , 1.41421356],
             [1.41421356, 0.        ]])
  |}]

*)



(* nan_euclidean_distances *)
(*
>>> # get distance to origin
>>> nan_euclidean_distances(X, [[0, 0]])
array([[1.        ],
       [1.41421356]])

*)

(* TEST TODO
let%expect_test "nan_euclidean_distances" =
  let open Sklearn.Metrics in
  # get distance to origin
  print_ndarray @@ nan_euclidean_distances(x, (matrixi [|[|0; 0|]|]));
  [%expect {|
      array([[1.        ],
             [1.41421356]])
  |}]

*)



(* ndcg_score *)
(*
>>> from sklearn.metrics import ndcg_score
>>> # we have groud-truth relevance of some answers to a query:
>>> true_relevance = np.asarray([[10, 0, 0, 1, 5]])
>>> # we predict some scores (relevance) for the answers
>>> scores = np.asarray([[.1, .2, .3, 4, 70]])
>>> ndcg_score(true_relevance, scores) # doctest: +ELLIPSIS
0.69...
>>> scores = np.asarray([[.05, 1.1, 1., .5, .0]])
>>> ndcg_score(true_relevance, scores) # doctest: +ELLIPSIS
0.49...
>>> # we can set k to truncate the sum; only top k answers contribute.
>>> ndcg_score(true_relevance, scores, k=4) # doctest: +ELLIPSIS
0.35...
>>> # the normalization takes k into account so a perfect answer
>>> # would still get 1.0
>>> ndcg_score(true_relevance, true_relevance, k=4) # doctest: +ELLIPSIS
1.0
>>> # now we have some ties in our prediction
>>> scores = np.asarray([[1, 0, 0, 0, 1]])
>>> # by default ties are averaged, so here we get the average (normalized)
>>> # true relevance of our top predictions: (10 / 10 + 5 / 10) / 2 = .75
>>> ndcg_score(true_relevance, scores, k=1) # doctest: +ELLIPSIS
0.75
>>> # we can choose to ignore ties for faster results, but only
>>> # if we know there aren't ties in our scores, otherwise we get
>>> # wrong results:
>>> ndcg_score(true_relevance,
...           scores, k=1, ignore_ties=True) # doctest: +ELLIPSIS

*)

(* TEST TODO
let%expect_test "ndcg_score" =
  let open Sklearn.Metrics in
  print_ndarray @@ # we have groud-truth relevance of some answers to a query:;
  let true_relevance = .asarray (matrixi [|[|10; 0; 0; 1; 5|]|]) np in
  print_ndarray @@ # we predict some scores (relevance) for the answers;
  let scores = .asarray (matrix [|[|.1; .2; .3; 4; 70|]|]) np in
  print_ndarray @@ ndcg_score ~true_relevance scores () # doctest: +ELLIPSIS;
  [%expect {|
      0.69...
  |}]
  let scores = .asarray (matrix [|[|.05; 1.1; 1.; .5; .0|]|]) np in
  print_ndarray @@ ndcg_score ~true_relevance scores () # doctest: +ELLIPSIS;
  [%expect {|
      0.49...
  |}]
  print_ndarray @@ # we can set k to truncate the sum; only top k answers contribute.;
  print_ndarray @@ ndcg_score ~true_relevance scores ~k:4 () # doctest: +ELLIPSIS;
  [%expect {|
      0.35...
  |}]
  print_ndarray @@ # the normalization takes k into account so a perfect answer;
  print_ndarray @@ # would still get 1.0;
  print_ndarray @@ ndcg_score ~true_relevance true_relevance ~k:4 () # doctest: +ELLIPSIS;
  [%expect {|
      1.0
  |}]
  print_ndarray @@ # now we have some ties in our prediction;
  let scores = .asarray (matrixi [|[|1; 0; 0; 0; 1|]|]) np in
  print_ndarray @@ # by default ties are averaged, so here we get the average (normalized);
  print_ndarray @@ # true relevance of our top predictions: (10 / 10 + 5 / 10) / 2 = .75;
  print_ndarray @@ ndcg_score ~true_relevance scores ~k:1 () # doctest: +ELLIPSIS;
  [%expect {|
      0.75
  |}]
  print_ndarray @@ # we can choose to ignore ties for faster results, but only;
  print_ndarray @@ # if we know there aren't ties in our scores, otherwise we get;
  print_ndarray @@ # wrong results:;
  print_ndarray @@ ndcg_score ~true_relevance scores ~k:1 ~ignore_ties:true () # doctest: +ELLIPSIS;
  [%expect {|
  |}]

*)



(*--------- Examples for module Sklearn.Metrics.Pairwise ----------*)
(* Parallel *)
(*
>>> from math import sqrt
>>> from joblib import Parallel, delayed
>>> Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10))
[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

*)

(* TEST TODO
let%expect_test "Parallel" =
  let open Sklearn.Metrics in
  print_ndarray @@ Parallel(n_jobs=1)(delayed ~sqrt ()(i**2) for i in range ~10 ());
  [%expect {|
      [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
  |}]

*)



(* Parallel *)
(*
>>> from math import modf
>>> from joblib import Parallel, delayed
>>> r = Parallel(n_jobs=1)(delayed(modf)(i/2.) for i in range(10))
>>> res, i = zip( *r)
>>> res
(0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5)
>>> i
(0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0)

*)

(* TEST TODO
let%expect_test "Parallel" =
  let open Sklearn.Metrics in
  let r = Parallel(n_jobs=1)(delayed ~modf ()(i/2.) for i in range ~10 ()) in
  let res, i = zip *r () in
  print_ndarray @@ res;
  [%expect {|
      (0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5)
  |}]
  print_ndarray @@ i;
  [%expect {|
      (0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0)
  |}]

*)



(* Parallel *)
(*
>>> from time import sleep
>>> from joblib import Parallel, delayed
>>> r = Parallel(n_jobs=2, verbose=10)(delayed(sleep)(.2) for _ in range(10)) #doctest: +SKIP
[Parallel(n_jobs=2)]: Done   1 tasks      | elapsed:    0.6s
[Parallel(n_jobs=2)]: Done   4 tasks      | elapsed:    0.8s
[Parallel(n_jobs=2)]: Done  10 out of  10 | elapsed:    1.4s finished

*)

(* TEST TODO
let%expect_test "Parallel" =
  let open Sklearn.Metrics in
  let r = Parallel(n_jobs=2, verbose=10)(delayed ~sleep ()(.2) for _ in range ~10 ()) #doctest: +SKIP in
  [%expect {|
      [Parallel(n_jobs=2)]: Done   1 tasks      | elapsed:    0.6s
      [Parallel(n_jobs=2)]: Done   4 tasks      | elapsed:    0.8s
      [Parallel(n_jobs=2)]: Done  10 out of  10 | elapsed:    1.4s finished
  |}]

*)



(* Parallel *)
(*
>>> from heapq import nlargest
>>> from joblib import Parallel, delayed
>>> Parallel(n_jobs=2)(delayed(nlargest)(2, n) for n in (range(4), 'abcde', 3)) #doctest: +SKIP
#...
---------------------------------------------------------------------------
Sub-process traceback:
---------------------------------------------------------------------------
TypeError                                          Mon Nov 12 11:37:46 2012
PID: 12934                                    Python 2.7.3: /usr/bin/python
...........................................................................
/usr/lib/python2.7/heapq.pyc in nlargest(n=2, iterable=3, key=None)
    419         if n >= size:
    420             return sorted(iterable, key=key, reverse=True)[:n]
    421
    422     # When key is none, use simpler decoration
    423     if key is None:
--> 424         it = izip(iterable, count(0,-1))                    # decorate
    425         result = _nlargest(n, it)
    426         return map(itemgetter(0), result)                   # undecorate
    427
    428     # General case, slowest method
 TypeError: izip argument #1 must support iteration
___________________________________________________________________________

*)

(* TEST TODO
let%expect_test "Parallel" =
  let open Sklearn.Metrics in
  print_ndarray @@ Parallel(n_jobs=2)(delayed ~nlargest ()(2, n) for n in (range ~4 (), 'abcde', 3)) #doctest: +SKIP;
  [%expect {|
      #...
      ---------------------------------------------------------------------------
      Sub-process traceback:
      ---------------------------------------------------------------------------
      TypeError                                          Mon Nov 12 11:37:46 2012
      PID: 12934                                    Python 2.7.3: /usr/bin/python
      ...........................................................................
      /usr/lib/python2.7/heapq.pyc in nlargest(n=2, iterable=3, key=None)
          419         if n >= size:
          420             return sorted(iterable, key=key, reverse=True)[:n]
          421
          422     # When key is none, use simpler decoration
          423     if key is None:
      --> 424         it = izip(iterable, count(0,-1))                    # decorate
          425         result = _nlargest(n, it)
          426         return map(itemgetter(0), result)                   # undecorate
          427
          428     # General case, slowest method
       TypeError: izip argument #1 must support iteration
      ___________________________________________________________________________
  |}]

*)



(* Parallel *)
(*
>>> from math import sqrt
>>> from joblib import Parallel, delayed
>>> def producer():
...     for i in range(6):
...         print('Produced %s' % i)
...         yield i
>>> out = Parallel(n_jobs=2, verbose=100, pre_dispatch='1.5*n_jobs')(
...                delayed(sqrt)(i) for i in producer()) #doctest: +SKIP
Produced 0
Produced 1
Produced 2
[Parallel(n_jobs=2)]: Done 1 jobs     | elapsed:  0.0s
Produced 3
[Parallel(n_jobs=2)]: Done 2 jobs     | elapsed:  0.0s
Produced 4
[Parallel(n_jobs=2)]: Done 3 jobs     | elapsed:  0.0s
Produced 5
[Parallel(n_jobs=2)]: Done 4 jobs     | elapsed:  0.0s
[Parallel(n_jobs=2)]: Done 6 out of 6 | elapsed:  0.0s remaining: 0.0s

*)

(* TEST TODO
let%expect_test "Parallel" =
  let open Sklearn.Metrics in
  print_ndarray @@ def producer ():for i in range ~6 ():print 'Produced %s' % i ()yield i;
  let out = Parallel(n_jobs=2, verbose=100, pre_dispatch='1.5*n_jobs')(delayed ~sqrt ()(i) for i in producer ()) #doctest: +SKIP in
  [%expect {|
      Produced 0
      Produced 1
      Produced 2
      [Parallel(n_jobs=2)]: Done 1 jobs     | elapsed:  0.0s
      Produced 3
      [Parallel(n_jobs=2)]: Done 2 jobs     | elapsed:  0.0s
      Produced 4
      [Parallel(n_jobs=2)]: Done 3 jobs     | elapsed:  0.0s
      Produced 5
      [Parallel(n_jobs=2)]: Done 4 jobs     | elapsed:  0.0s
      [Parallel(n_jobs=2)]: Done 6 out of 6 | elapsed:  0.0s remaining: 0.0s
  |}]

*)



(* euclidean_distances *)
(*
>>> from sklearn.metrics.pairwise import euclidean_distances
>>> X = [[0, 1], [1, 1]]
>>> # distance between rows of X
>>> euclidean_distances(X, X)
array([[0., 1.],
       [1., 0.]])
>>> # get distance to origin
>>> euclidean_distances(X, [[0, 0]])
array([[1.        ],
       [1.41421356]])

*)

(* TEST TODO
let%expect_test "euclidean_distances" =
  let open Sklearn.Metrics in
  let x = (matrixi [|[|0; 1|]; [|1; 1|]|]) in
  print_ndarray @@ # distance between rows of x;
  print_ndarray @@ euclidean_distances ~x x ();
  [%expect {|
      array([[0., 1.],
             [1., 0.]])
  |}]
  # get distance to origin
  print_ndarray @@ euclidean_distances(x, (matrixi [|[|0; 0|]|]));
  [%expect {|
      array([[1.        ],
             [1.41421356]])
  |}]

*)



(* gen_batches *)
(*
>>> from sklearn.utils import gen_batches
>>> list(gen_batches(7, 3))
[slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
>>> list(gen_batches(6, 3))
[slice(0, 3, None), slice(3, 6, None)]
>>> list(gen_batches(2, 3))
[slice(0, 2, None)]
>>> list(gen_batches(7, 3, min_batch_size=0))
[slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
>>> list(gen_batches(7, 3, min_batch_size=2))

*)

(* TEST TODO
let%expect_test "gen_batches" =
  let open Sklearn.Metrics in
  print_ndarray @@ list(gen_batches ~7 3 ());
  [%expect {|
      [slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
  |}]
  print_ndarray @@ list(gen_batches ~6 3 ());
  [%expect {|
      [slice(0, 3, None), slice(3, 6, None)]
  |}]
  print_ndarray @@ list(gen_batches ~2 3 ());
  [%expect {|
      [slice(0, 2, None)]
  |}]
  print_ndarray @@ list(gen_batches ~7 3 ~min_batch_size:0 ());
  [%expect {|
      [slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
  |}]
  print_ndarray @@ list(gen_batches ~7 3 ~min_batch_size:2 ());
  [%expect {|
  |}]

*)



(* gen_even_slices *)
(*
>>> from sklearn.utils import gen_even_slices
>>> list(gen_even_slices(10, 1))
[slice(0, 10, None)]
>>> list(gen_even_slices(10, 10))
[slice(0, 1, None), slice(1, 2, None), ..., slice(9, 10, None)]
>>> list(gen_even_slices(10, 5))
[slice(0, 2, None), slice(2, 4, None), ..., slice(8, 10, None)]
>>> list(gen_even_slices(10, 3))

*)

(* TEST TODO
let%expect_test "gen_even_slices" =
  let open Sklearn.Metrics in
  print_ndarray @@ list(gen_even_slices ~10 1 ());
  [%expect {|
      [slice(0, 10, None)]
  |}]
  print_ndarray @@ list(gen_even_slices ~10 10 ());
  [%expect {|
      [slice(0, 1, None), slice(1, 2, None), ..., slice(9, 10, None)]
  |}]
  print_ndarray @@ list(gen_even_slices ~10 5 ());
  [%expect {|
      [slice(0, 2, None), slice(2, 4, None), ..., slice(8, 10, None)]
  |}]
  print_ndarray @@ list(gen_even_slices ~10 3 ());
  [%expect {|
  |}]

*)



(* haversine_distances *)
(*
>>> from sklearn.metrics.pairwise import haversine_distances
>>> from math import radians
>>> bsas = [-34.83333, -58.5166646]
>>> paris = [49.0083899664, 2.53844117956]
>>> bsas_in_radians = [radians(_) for _ in bsas]
>>> paris_in_radians = [radians(_) for _ in paris]
>>> result = haversine_distances([bsas_in_radians, paris_in_radians])
>>> result * 6371000/1000  # multiply by Earth radius to get kilometers
array([[    0.        , 11099.54035582],

*)

(* TEST TODO
let%expect_test "haversine_distances" =
  let open Sklearn.Metrics in
  let bsas = [-34.83333, -58.5166646] in
  let paris = [49.0083899664, 2.53844117956] in
  let bsas_in_radians = [radians ~_ () for _ in bsas] in
  let paris_in_radians = [radians ~_ () for _ in paris] in
  let result = haversine_distances [bsas_in_radians paris_in_radians] () in
  print_ndarray @@ result * 6371000/1000 # multiply by Earth radius to get kilometers;
  [%expect {|
      array([[    0.        , 11099.54035582],
  |}]

*)



(* is_scalar_nan *)
(*
>>> is_scalar_nan(np.nan)
True
>>> is_scalar_nan(float("nan"))
True
>>> is_scalar_nan(None)
False
>>> is_scalar_nan("")
False
>>> is_scalar_nan([np.nan])

*)

(* TEST TODO
let%expect_test "is_scalar_nan" =
  let open Sklearn.Metrics in
  print_ndarray @@ is_scalar_nan np.nan ();
  [%expect {|
      True
  |}]
  print_ndarray @@ is_scalar_nan(float "nan" ());
  [%expect {|
      True
  |}]
  print_ndarray @@ is_scalar_nan ~None ();
  [%expect {|
      False
  |}]
  print_ndarray @@ is_scalar_nan "" ();
  [%expect {|
      False
  |}]
  print_ndarray @@ is_scalar_nan [np.nan] ();
  [%expect {|
  |}]

*)



(* isspmatrix *)
(*
>>> from scipy.sparse import csr_matrix, isspmatrix
>>> isspmatrix(csr_matrix([[5]]))
True

*)

(* TEST TODO
let%expect_test "isspmatrix" =
  let open Sklearn.Metrics in
  print_ndarray @@ isspmatrix(csr_matrix((matrixi [|[|5|]|])));
  [%expect {|
      True
  |}]

*)



(* isspmatrix *)
(*
>>> from scipy.sparse import isspmatrix
>>> isspmatrix(5)

*)

(* TEST TODO
let%expect_test "isspmatrix" =
  let open Sklearn.Metrics in
  print_ndarray @@ isspmatrix ~5 ();
  [%expect {|
  |}]

*)



(* manhattan_distances *)
(*
>>> from sklearn.metrics.pairwise import manhattan_distances
>>> manhattan_distances([[3]], [[3]])
array([[0.]])
>>> manhattan_distances([[3]], [[2]])
array([[1.]])
>>> manhattan_distances([[2]], [[3]])
array([[1.]])
>>> manhattan_distances([[1, 2], [3, 4]],         [[1, 2], [0, 3]])
array([[0., 2.],
       [4., 4.]])
>>> import numpy as np
>>> X = np.ones((1, 2))
>>> y = np.full((2, 2), 2.)
>>> manhattan_distances(X, y, sum_over_features=False)
array([[1., 1.],

*)

(* TEST TODO
let%expect_test "manhattan_distances" =
  let open Sklearn.Metrics in
  print_ndarray @@ manhattan_distances((matrixi [|[|3|]|]), (matrixi [|[|3|]|]));
  [%expect {|
      array([[0.]])
  |}]
  print_ndarray @@ manhattan_distances((matrixi [|[|3|]|]), (matrixi [|[|2|]|]));
  [%expect {|
      array([[1.]])
  |}]
  print_ndarray @@ manhattan_distances((matrixi [|[|2|]|]), (matrixi [|[|3|]|]));
  [%expect {|
      array([[1.]])
  |}]
  print_ndarray @@ manhattan_distances((matrixi [|[|1; 2|]; [|3; 4|]|]), (matrixi [|[|1; 2|]; [|0; 3|]|]));
  [%expect {|
      array([[0., 2.],
             [4., 4.]])
  |}]
  let x = .ones (1 2) np in
  let y = .full (2 2) 2. np in
  print_ndarray @@ manhattan_distances ~x y ~sum_over_features:false ();
  [%expect {|
      array([[1., 1.],
  |}]

*)



(* nan_euclidean_distances *)
(*
>>> from sklearn.metrics.pairwise import nan_euclidean_distances
>>> nan = float("NaN")
>>> X = [[0, 1], [1, nan]]
>>> nan_euclidean_distances(X, X) # distance between rows of X
array([[0.        , 1.41421356],
       [1.41421356, 0.        ]])

*)

(* TEST TODO
let%expect_test "nan_euclidean_distances" =
  let open Sklearn.Metrics in
  let nan = float "NaN" () in
  let x = (matrixi [|[|0; 1|]; [|1; nan|]|]) in
  print_ndarray @@ nan_euclidean_distances ~x x () # distance between rows of x;
  [%expect {|
      array([[0.        , 1.41421356],
             [1.41421356, 0.        ]])
  |}]

*)



(* nan_euclidean_distances *)
(*
>>> # get distance to origin
>>> nan_euclidean_distances(X, [[0, 0]])
array([[1.        ],
       [1.41421356]])

*)

(* TEST TODO
let%expect_test "nan_euclidean_distances" =
  let open Sklearn.Metrics in
  # get distance to origin
  print_ndarray @@ nan_euclidean_distances(x, (matrixi [|[|0; 0|]|]));
  [%expect {|
      array([[1.        ],
             [1.41421356]])
  |}]

*)



(* paired_distances *)
(*
>>> from sklearn.metrics.pairwise import paired_distances
>>> X = [[0, 1], [1, 1]]
>>> Y = [[0, 1], [2, 1]]
>>> paired_distances(X, Y)
array([0., 1.])

*)

(* TEST TODO
let%expect_test "paired_distances" =
  let open Sklearn.Metrics in
  let x = (matrixi [|[|0; 1|]; [|1; 1|]|]) in
  let Y = (matrixi [|[|0; 1|]; [|2; 1|]|]) in
  print_ndarray @@ paired_distances ~x Y ();
  [%expect {|
      array([0., 1.])
  |}]

*)



(* pairwise_distances_chunked *)
(*
>>> import numpy as np
>>> from sklearn.metrics import pairwise_distances_chunked
>>> X = np.random.RandomState(0).rand(5, 3)
>>> D_chunk = next(pairwise_distances_chunked(X))
>>> D_chunk
array([[0.  ..., 0.29..., 0.41..., 0.19..., 0.57...],
       [0.29..., 0.  ..., 0.57..., 0.41..., 0.76...],
       [0.41..., 0.57..., 0.  ..., 0.44..., 0.90...],
       [0.19..., 0.41..., 0.44..., 0.  ..., 0.51...],
       [0.57..., 0.76..., 0.90..., 0.51..., 0.  ...]])

*)

(* TEST TODO
let%expect_test "pairwise_distances_chunked" =
  let open Sklearn.Metrics in
  let x = np..randomState 0).rand(5 ~3 random in
  let D_chunk = next(pairwise_distances_chunked ~x ()) in
  print_ndarray @@ D_chunk;
  [%expect {|
      array([[0.  ..., 0.29..., 0.41..., 0.19..., 0.57...],
             [0.29..., 0.  ..., 0.57..., 0.41..., 0.76...],
             [0.41..., 0.57..., 0.  ..., 0.44..., 0.90...],
             [0.19..., 0.41..., 0.44..., 0.  ..., 0.51...],
             [0.57..., 0.76..., 0.90..., 0.51..., 0.  ...]])
  |}]

*)



(* pairwise_distances_chunked *)
(*
>>> r = .2
>>> def reduce_func(D_chunk, start):
...     neigh = [np.flatnonzero(d < r) for d in D_chunk]
...     avg_dist = (D_chunk * (D_chunk < r)).mean(axis=1)
...     return neigh, avg_dist
>>> gen = pairwise_distances_chunked(X, reduce_func=reduce_func)
>>> neigh, avg_dist = next(gen)
>>> neigh
[array([0, 3]), array([1]), array([2]), array([0, 3]), array([4])]
>>> avg_dist
array([0.039..., 0.        , 0.        , 0.039..., 0.        ])

*)

(* TEST TODO
let%expect_test "pairwise_distances_chunked" =
  let open Sklearn.Metrics in
  let r = .2 in
  print_ndarray @@ def reduce_func ~D_chunk start ():neigh = [.flatnonzero d < r) for d in D_chunk]avg_dist = (D_chunk * (D_chunk < r)).mean( ~axis:1 npreturn neigh, avg_dist;
  let gen = pairwise_distances_chunked x ~reduce_func:reduce_func () in
  let neigh, avg_dist = next ~gen () in
  print_ndarray @@ neigh;
  [%expect {|
      [array([0, 3]), array([1]), array([2]), array([0, 3]), array([4])]
  |}]
  print_ndarray @@ avg_dist;
  [%expect {|
      array([0.039..., 0.        , 0.        , 0.039..., 0.        ])
  |}]

*)



(* pairwise_distances_chunked *)
(*
>>> r = [.2, .4, .4, .3, .1]
>>> def reduce_func(D_chunk, start):
...     neigh = [np.flatnonzero(d < r[i])
...              for i, d in enumerate(D_chunk, start)]
...     return neigh
>>> neigh = next(pairwise_distances_chunked(X, reduce_func=reduce_func))
>>> neigh
[array([0, 3]), array([0, 1]), array([2]), array([0, 3]), array([4])]

*)

(* TEST TODO
let%expect_test "pairwise_distances_chunked" =
  let open Sklearn.Metrics in
  let r = [.2, .4, .4, .3, .1] in
  print_ndarray @@ def reduce_func ~D_chunk start ():neigh = [.flatnonzero d < r(vectori [|i|]))for i d in enumerate(D_chunk ~start np]return neigh;
  let neigh = next(pairwise_distances_chunked x ~reduce_func:reduce_func ()) in
  print_ndarray @@ neigh;
  [%expect {|
      [array([0, 3]), array([0, 1]), array([2]), array([0, 3]), array([4])]
  |}]

*)



(* pairwise_distances_chunked *)
(*
>>> gen = pairwise_distances_chunked(X, reduce_func=reduce_func,
...                                  working_memory=0)
>>> next(gen)
[array([0, 3])]
>>> next(gen)

*)

(* TEST TODO
let%expect_test "pairwise_distances_chunked" =
  let open Sklearn.Metrics in
  let gen = pairwise_distances_chunked x ~reduce_func:reduce_func ~working_memory:0 () in
  print_ndarray @@ next ~gen ();
  [%expect {|
      [array([0, 3])]
  |}]
  print_ndarray @@ next ~gen ();
  [%expect {|
  |}]

*)



(* pairwise_distances_chunked *)
(*
>>> import numpy as np
>>> from sklearn.metrics import pairwise_distances_chunked
>>> X = np.random.RandomState(0).rand(5, 3)
>>> D_chunk = next(pairwise_distances_chunked(X))
>>> D_chunk
array([[0.  ..., 0.29..., 0.41..., 0.19..., 0.57...],
       [0.29..., 0.  ..., 0.57..., 0.41..., 0.76...],
       [0.41..., 0.57..., 0.  ..., 0.44..., 0.90...],
       [0.19..., 0.41..., 0.44..., 0.  ..., 0.51...],
       [0.57..., 0.76..., 0.90..., 0.51..., 0.  ...]])

*)

(* TEST TODO
let%expect_test "pairwise_distances_chunked" =
  let open Sklearn.Metrics in
  let x = np..randomState 0).rand(5 ~3 random in
  let D_chunk = next(pairwise_distances_chunked ~x ()) in
  print_ndarray @@ D_chunk;
  [%expect {|
      array([[0.  ..., 0.29..., 0.41..., 0.19..., 0.57...],
             [0.29..., 0.  ..., 0.57..., 0.41..., 0.76...],
             [0.41..., 0.57..., 0.  ..., 0.44..., 0.90...],
             [0.19..., 0.41..., 0.44..., 0.  ..., 0.51...],
             [0.57..., 0.76..., 0.90..., 0.51..., 0.  ...]])
  |}]

*)



(* pairwise_distances_chunked *)
(*
>>> r = .2
>>> def reduce_func(D_chunk, start):
...     neigh = [np.flatnonzero(d < r) for d in D_chunk]
...     avg_dist = (D_chunk * (D_chunk < r)).mean(axis=1)
...     return neigh, avg_dist
>>> gen = pairwise_distances_chunked(X, reduce_func=reduce_func)
>>> neigh, avg_dist = next(gen)
>>> neigh
[array([0, 3]), array([1]), array([2]), array([0, 3]), array([4])]
>>> avg_dist
array([0.039..., 0.        , 0.        , 0.039..., 0.        ])

*)

(* TEST TODO
let%expect_test "pairwise_distances_chunked" =
  let open Sklearn.Metrics in
  let r = .2 in
  print_ndarray @@ def reduce_func ~D_chunk start ():neigh = [.flatnonzero d < r) for d in D_chunk]avg_dist = (D_chunk * (D_chunk < r)).mean( ~axis:1 npreturn neigh, avg_dist;
  let gen = pairwise_distances_chunked x ~reduce_func:reduce_func () in
  let neigh, avg_dist = next ~gen () in
  print_ndarray @@ neigh;
  [%expect {|
      [array([0, 3]), array([1]), array([2]), array([0, 3]), array([4])]
  |}]
  print_ndarray @@ avg_dist;
  [%expect {|
      array([0.039..., 0.        , 0.        , 0.039..., 0.        ])
  |}]

*)



(* pairwise_distances_chunked *)
(*
>>> r = [.2, .4, .4, .3, .1]
>>> def reduce_func(D_chunk, start):
...     neigh = [np.flatnonzero(d < r[i])
...              for i, d in enumerate(D_chunk, start)]
...     return neigh
>>> neigh = next(pairwise_distances_chunked(X, reduce_func=reduce_func))
>>> neigh
[array([0, 3]), array([0, 1]), array([2]), array([0, 3]), array([4])]

*)

(* TEST TODO
let%expect_test "pairwise_distances_chunked" =
  let open Sklearn.Metrics in
  let r = [.2, .4, .4, .3, .1] in
  print_ndarray @@ def reduce_func ~D_chunk start ():neigh = [.flatnonzero d < r(vectori [|i|]))for i d in enumerate(D_chunk ~start np]return neigh;
  let neigh = next(pairwise_distances_chunked x ~reduce_func:reduce_func ()) in
  print_ndarray @@ neigh;
  [%expect {|
      [array([0, 3]), array([0, 1]), array([2]), array([0, 3]), array([4])]
  |}]

*)



(* pairwise_distances_chunked *)
(*
>>> gen = pairwise_distances_chunked(X, reduce_func=reduce_func,
...                                  working_memory=0)
>>> next(gen)
[array([0, 3])]
>>> next(gen)

*)

(* TEST TODO
let%expect_test "pairwise_distances_chunked" =
  let open Sklearn.Metrics in
  let gen = pairwise_distances_chunked x ~reduce_func:reduce_func ~working_memory:0 () in
  print_ndarray @@ next ~gen ();
  [%expect {|
      [array([0, 3])]
  |}]
  print_ndarray @@ next ~gen ();
  [%expect {|
  |}]

*)



(* plot_roc_curve *)
(*
>>> import matplotlib.pyplot as plt  # doctest: +SKIP
>>> from sklearn import datasets, metrics, model_selection, svm
>>> X, y = datasets.make_classification(random_state=0)
>>> X_train, X_test, y_train, y_test = model_selection.train_test_split(            X, y, random_state=0)
>>> clf = svm.SVC(random_state=0)
>>> clf.fit(X_train, y_train)
SVC(random_state=0)
>>> metrics.plot_roc_curve(clf, X_test, y_test)  # doctest: +SKIP

*)

(* TEST TODO
let%expect_test "plot_roc_curve" =
  let open Sklearn.Metrics in
  let x, y = .make_classification ~random_state:0 datasets in
  let X_train, X_test, y_train, y_test = .train_test_split x y ~random_state:0 model_selection in
  let clf = .svc ~random_state:0 svm in
  print_ndarray @@ .fit ~X_train y_train clf;
  [%expect {|
      SVC(random_state=0)
  |}]
  print_ndarray @@ .plot_roc_curve ~clf X_test ~y_test metrics # doctest: +SKIP;
  [%expect {|
  |}]

*)



(* precision_recall_curve *)
(*
>>> import numpy as np
>>> from sklearn.metrics import precision_recall_curve
>>> y_true = np.array([0, 0, 1, 1])
>>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
>>> precision, recall, thresholds = precision_recall_curve(
...     y_true, y_scores)
>>> precision
array([0.66666667, 0.5       , 1.        , 1.        ])
>>> recall
array([1. , 0.5, 0.5, 0. ])
>>> thresholds

*)

(* TEST TODO
let%expect_test "precision_recall_curve" =
  let open Sklearn.Metrics in
  let y_true = .array (vectori [|0; 0; 1; 1|]) np in
  let y_scores = .array [0.1 0.4 0.35 0.8] np in
  let precision, recall, thresholds = precision_recall_curve ~y_true y_scores () in
  print_ndarray @@ precision;
  [%expect {|
      array([0.66666667, 0.5       , 1.        , 1.        ])
  |}]
  print_ndarray @@ recall;
  [%expect {|
      array([1. , 0.5, 0.5, 0. ])
  |}]
  print_ndarray @@ thresholds;
  [%expect {|
  |}]

*)



(* precision_recall_fscore_support *)
(*
>>> import numpy as np
>>> from sklearn.metrics import precision_recall_fscore_support
>>> y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
>>> y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])
>>> precision_recall_fscore_support(y_true, y_pred, average='macro')
(0.22..., 0.33..., 0.26..., None)
>>> precision_recall_fscore_support(y_true, y_pred, average='micro')
(0.33..., 0.33..., 0.33..., None)
>>> precision_recall_fscore_support(y_true, y_pred, average='weighted')
(0.22..., 0.33..., 0.26..., None)

*)

(* TEST TODO
let%expect_test "precision_recall_fscore_support" =
  let open Sklearn.Metrics in
  let y_true = .array ['cat' 'dog' 'pig' 'cat' 'dog' 'pig'] np in
  let y_pred = .array ['cat' 'pig' 'dog' 'cat' 'cat' 'dog'] np in
  print_ndarray @@ precision_recall_fscore_support ~y_true y_pred ~average:'macro' ();
  [%expect {|
      (0.22..., 0.33..., 0.26..., None)
  |}]
  print_ndarray @@ precision_recall_fscore_support ~y_true y_pred ~average:'micro' ();
  [%expect {|
      (0.33..., 0.33..., 0.33..., None)
  |}]
  print_ndarray @@ precision_recall_fscore_support ~y_true y_pred ~average:'weighted' ();
  [%expect {|
      (0.22..., 0.33..., 0.26..., None)
  |}]

*)



(* precision_recall_fscore_support *)
(*
>>> precision_recall_fscore_support(y_true, y_pred, average=None,
... labels=['pig', 'dog', 'cat'])
(array([0.        , 0.        , 0.66...]),
 array([0., 0., 1.]), array([0. , 0. , 0.8]),
 array([2, 2, 2]))

*)

(* TEST TODO
let%expect_test "precision_recall_fscore_support" =
  let open Sklearn.Metrics in
  print_ndarray @@ precision_recall_fscore_support ~y_true y_pred ~average:None ~labels:['pig' 'dog' 'cat'] ();
  [%expect {|
      (array([0.        , 0.        , 0.66...]),
       array([0., 0., 1.]), array([0. , 0. , 0.8]),
       array([2, 2, 2]))
  |}]

*)



(* precision_score *)
(*
>>> from sklearn.metrics import precision_score
>>> y_true = [0, 1, 2, 0, 1, 2]
>>> y_pred = [0, 2, 1, 0, 0, 1]
>>> precision_score(y_true, y_pred, average='macro')
0.22...
>>> precision_score(y_true, y_pred, average='micro')
0.33...
>>> precision_score(y_true, y_pred, average='weighted')
0.22...
>>> precision_score(y_true, y_pred, average=None)
array([0.66..., 0.        , 0.        ])
>>> y_pred = [0, 0, 0, 0, 0, 0]
>>> precision_score(y_true, y_pred, average=None)
array([0.33..., 0.        , 0.        ])
>>> precision_score(y_true, y_pred, average=None, zero_division=1)
array([0.33..., 1.        , 1.        ])

*)

(* TEST TODO
let%expect_test "precision_score" =
  let open Sklearn.Metrics in
  let y_true = (vectori [|0; 1; 2; 0; 1; 2|]) in
  let y_pred = (vectori [|0; 2; 1; 0; 0; 1|]) in
  print_ndarray @@ precision_score ~y_true y_pred ~average:'macro' ();
  [%expect {|
      0.22...
  |}]
  print_ndarray @@ precision_score ~y_true y_pred ~average:'micro' ();
  [%expect {|
      0.33...
  |}]
  print_ndarray @@ precision_score ~y_true y_pred ~average:'weighted' ();
  [%expect {|
      0.22...
  |}]
  print_ndarray @@ precision_score ~y_true y_pred ~average:None ();
  [%expect {|
      array([0.66..., 0.        , 0.        ])
  |}]
  let y_pred = (vectori [|0; 0; 0; 0; 0; 0|]) in
  print_ndarray @@ precision_score ~y_true y_pred ~average:None ();
  [%expect {|
      array([0.33..., 0.        , 0.        ])
  |}]
  print_ndarray @@ precision_score ~y_true y_pred ~average:None ~zero_division:1 ();
  [%expect {|
      array([0.33..., 1.        , 1.        ])
  |}]

*)



(* r2_score *)
(*
>>> from sklearn.metrics import r2_score
>>> y_true = [3, -0.5, 2, 7]
>>> y_pred = [2.5, 0.0, 2, 8]
>>> r2_score(y_true, y_pred)
0.948...
>>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
>>> y_pred = [[0, 2], [-1, 2], [8, -5]]
>>> r2_score(y_true, y_pred,
...          multioutput='variance_weighted')
0.938...
>>> y_true = [1, 2, 3]
>>> y_pred = [1, 2, 3]
>>> r2_score(y_true, y_pred)
1.0
>>> y_true = [1, 2, 3]
>>> y_pred = [2, 2, 2]
>>> r2_score(y_true, y_pred)
0.0
>>> y_true = [1, 2, 3]
>>> y_pred = [3, 2, 1]
>>> r2_score(y_true, y_pred)

*)

(* TEST TODO
let%expect_test "r2_score" =
  let open Sklearn.Metrics in
  let y_true = [3, -0.5, 2, 7] in
  let y_pred = [2.5, 0.0, 2, 8] in
  print_ndarray @@ r2_score(y_true, y_pred);
  [%expect {|
      0.948...
  |}]
  let y_true = (matrix [|[|0.5; 1|]; [|-1; 1|]; [|7; -6|]|]) in
  let y_pred = (matrixi [|[|0; 2|]; [|-1; 2|]; [|8; -5|]|]) in
  print_ndarray @@ r2_score(y_true, y_pred,multioutput='variance_weighted');
  [%expect {|
      0.938...
  |}]
  let y_true = (vectori [|1; 2; 3|]) in
  let y_pred = (vectori [|1; 2; 3|]) in
  print_ndarray @@ r2_score(y_true, y_pred);
  [%expect {|
      1.0
  |}]
  let y_true = (vectori [|1; 2; 3|]) in
  let y_pred = (vectori [|2; 2; 2|]) in
  print_ndarray @@ r2_score(y_true, y_pred);
  [%expect {|
      0.0
  |}]
  let y_true = (vectori [|1; 2; 3|]) in
  let y_pred = (vectori [|3; 2; 1|]) in
  print_ndarray @@ r2_score(y_true, y_pred);
  [%expect {|
  |}]

*)



(* recall_score *)
(*
>>> from sklearn.metrics import recall_score
>>> y_true = [0, 1, 2, 0, 1, 2]
>>> y_pred = [0, 2, 1, 0, 0, 1]
>>> recall_score(y_true, y_pred, average='macro')
0.33...
>>> recall_score(y_true, y_pred, average='micro')
0.33...
>>> recall_score(y_true, y_pred, average='weighted')
0.33...
>>> recall_score(y_true, y_pred, average=None)
array([1., 0., 0.])
>>> y_true = [0, 0, 0, 0, 0, 0]
>>> recall_score(y_true, y_pred, average=None)
array([0.5, 0. , 0. ])
>>> recall_score(y_true, y_pred, average=None, zero_division=1)
array([0.5, 1. , 1. ])

*)

(* TEST TODO
let%expect_test "recall_score" =
  let open Sklearn.Metrics in
  let y_true = (vectori [|0; 1; 2; 0; 1; 2|]) in
  let y_pred = (vectori [|0; 2; 1; 0; 0; 1|]) in
  print_ndarray @@ recall_score ~y_true y_pred ~average:'macro' ();
  [%expect {|
      0.33...
  |}]
  print_ndarray @@ recall_score ~y_true y_pred ~average:'micro' ();
  [%expect {|
      0.33...
  |}]
  print_ndarray @@ recall_score ~y_true y_pred ~average:'weighted' ();
  [%expect {|
      0.33...
  |}]
  print_ndarray @@ recall_score ~y_true y_pred ~average:None ();
  [%expect {|
      array([1., 0., 0.])
  |}]
  let y_true = (vectori [|0; 0; 0; 0; 0; 0|]) in
  print_ndarray @@ recall_score ~y_true y_pred ~average:None ();
  [%expect {|
      array([0.5, 0. , 0. ])
  |}]
  print_ndarray @@ recall_score ~y_true y_pred ~average:None ~zero_division:1 ();
  [%expect {|
      array([0.5, 1. , 1. ])
  |}]

*)



(* roc_auc_score *)
(*
>>> import numpy as np
>>> from sklearn.metrics import roc_auc_score
>>> y_true = np.array([0, 0, 1, 1])
>>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
>>> roc_auc_score(y_true, y_scores)

*)

(* TEST TODO
let%expect_test "roc_auc_score" =
  let open Sklearn.Metrics in
  let y_true = .array (vectori [|0; 0; 1; 1|]) np in
  let y_scores = .array [0.1 0.4 0.35 0.8] np in
  print_ndarray @@ roc_auc_score ~y_true y_scores ();
  [%expect {|
  |}]

*)



(* roc_curve *)
(*
>>> import numpy as np
>>> from sklearn import metrics
>>> y = np.array([1, 1, 2, 2])
>>> scores = np.array([0.1, 0.4, 0.35, 0.8])
>>> fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
>>> fpr
array([0. , 0. , 0.5, 0.5, 1. ])
>>> tpr
array([0. , 0.5, 0.5, 1. , 1. ])
>>> thresholds

*)

(* TEST TODO
let%expect_test "roc_curve" =
  let open Sklearn.Metrics in
  let y = .array (vectori [|1; 1; 2; 2|]) np in
  let scores = .array [0.1 0.4 0.35 0.8] np in
  let fpr, tpr, thresholds = .roc_curve ~y scores ~pos_label:2 metrics in
  print_ndarray @@ fpr;
  [%expect {|
      array([0. , 0. , 0.5, 0.5, 1. ])
  |}]
  print_ndarray @@ tpr;
  [%expect {|
      array([0. , 0.5, 0.5, 1. , 1. ])
  |}]
  print_ndarray @@ thresholds;
  [%expect {|
  |}]

*)



(* zero_one_loss *)
(*
>>> from sklearn.metrics import zero_one_loss
>>> y_pred = [1, 2, 3, 4]
>>> y_true = [2, 2, 3, 4]
>>> zero_one_loss(y_true, y_pred)
0.25
>>> zero_one_loss(y_true, y_pred, normalize=False)
1

*)

(* TEST TODO
let%expect_test "zero_one_loss" =
  let open Sklearn.Metrics in
  let y_pred = (vectori [|1; 2; 3; 4|]) in
  let y_true = (vectori [|2; 2; 3; 4|]) in
  print_ndarray @@ zero_one_loss ~y_true y_pred ();
  [%expect {|
      0.25
  |}]
  print_ndarray @@ zero_one_loss ~y_true y_pred ~normalize:false ();
  [%expect {|
      1
  |}]

*)



(* zero_one_loss *)
(*
>>> import numpy as np
>>> zero_one_loss(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))

*)

(* TEST TODO
let%expect_test "zero_one_loss" =
  let open Sklearn.Metrics in
  print_ndarray @@ zero_one_loss(.array (matrixi [|[|0; 1|]; [|1; 1|]|])) np.ones((2 2)) np;
  [%expect {|
  |}]

*)
