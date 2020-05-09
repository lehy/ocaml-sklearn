let print f x = Format.printf "%a" f x
let print_py x = Format.printf "%s" (Py.Object.to_string x)
let print_ndarray = print Sklearn.Arr.pp
let print_float = Format.printf "%g\n"
let print_string = Format.printf "%s\n"

let matrix = Sklearn.Arr.Float.matrix
let vector = Sklearn.Arr.Float.vector
let matrixi = Sklearn.Arr.Int.matrix
let vectori = Sklearn.Arr.Int.vector
let vectors = Sklearn.Arr.String.vector

let option_get = function Some x -> x | None -> invalid_arg "option_get: None"

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
  print_float @@ brier_score_loss ~y_true ~y_prob:Sklearn.Arr.((int 1)-y_prob) ~pos_label:(`I 0) ();
  [%expect {|
      0.0375
  |}];
  print_float @@ brier_score_loss ~y_true:y_true_categorical ~y_prob ~pos_label:(`S "ham") ();
  [%expect {|
      0.0375
  |}];
  print_float @@ brier_score_loss ~y_true ~y_prob:Sklearn.Arr.(y_prob > float 0.5) ();
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
    conf |> Sklearn.Arr.to_int_array
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
  print_ndarray @@ explained_variance_score ~y_true ~y_pred ();
  [%expect {|
      0.9571734475374732
  |}];
  let y_true = matrix [|[|0.5; 1.|]; [|-1.; 1.|]; [|7.; -6.|]|] in
  let y_pred = matrixi [|[|0; 2|]; [|-1; 2|]; [|8; -5|]|] in
  print_ndarray @@ explained_variance_score ~y_true ~y_pred ~multioutput:`Uniform_average ();
  [%expect {| 0.9838709677419355 |}]

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

let%expect_test "f1_score" =
  let open Sklearn.Metrics in
  let y_true = vectori [|0; 1; 2; 0; 1; 2|] in
  let y_pred = vectori [|0; 2; 1; 0; 0; 1|] in
  print_ndarray @@ f1_score ~y_true ~y_pred ~average:`Macro ();
  [%expect {|
      0.26666666666666666
  |}];
  print_ndarray @@ f1_score ~y_true ~y_pred ~average:`Micro ();
  [%expect {|
      0.3333333333333333
  |}];
  print_ndarray @@ f1_score ~y_true ~y_pred ~average:`Weighted ();
  [%expect {|
      0.26666666666666666
  |}];
  print_ndarray @@ f1_score ~y_true ~y_pred ~average:`None ();
  [%expect {|
      [0.8 0.  0. ]
  |}];
  let y_true = vectori [|0; 0; 0; 0; 0; 0|] in
  let y_pred = vectori [|0; 0; 0; 0; 0; 0|] in
  print_ndarray @@ f1_score ~y_true ~y_pred ~zero_division:`One ();
  [%expect {|
      1.0
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

let%expect_test "fbeta_score" =
  let open Sklearn.Metrics in
  let y_true = vectori [|0; 1; 2; 0; 1; 2|] in
  let y_pred = vectori [|0; 2; 1; 0; 0; 1|] in
  print_ndarray @@ fbeta_score ~y_true ~y_pred ~average:`Macro ~beta:0.5 ();
  [%expect {|
      0.23809523809523805
  |}];
  print_ndarray @@ fbeta_score ~y_true ~y_pred ~average:`Micro ~beta:0.5 ();
  [%expect {|
      0.3333333333333333
  |}];
  print_ndarray @@ fbeta_score ~y_true ~y_pred ~average:`Weighted ~beta:0.5 ();
  [%expect {|
      0.23809523809523805
  |}];
  print_ndarray @@ fbeta_score ~y_true ~y_pred ~average:`None ~beta:0.5 ();
  [%expect {|
      [0.71428571 0.         0.        ]
  |}]


(* hamming_loss *)
(*
>>> from sklearn.metrics import hamming_loss
>>> y_pred = [1, 2, 3, 4]
>>> y_true = [2, 2, 3, 4]
>>> hamming_loss(y_true, y_pred)
0.25

*)

let%expect_test "hamming_loss" =
  let open Sklearn.Metrics in
  let y_pred = vectori [|1; 2; 3; 4|] in
  let y_true = vectori [|2; 2; 3; 4|] in
  print_float @@ hamming_loss ~y_true ~y_pred ();
  [%expect {|
      0.25
  |}]


(* hamming_loss *)
(*
>>> import numpy as np
>>> hamming_loss(np.array([[0, 1], [1, 1]]), np.zeros((2, 2)))

*)

let%expect_test "hamming_loss" =
  let open Sklearn.Metrics in
  print_float @@ hamming_loss ~y_true:(matrixi [|[|0; 1|]; [|1; 1|]|]) ~y_pred:(Sklearn.Arr.zeros [2; 2]) ();
  [%expect {| 0.75 |}]


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

let%expect_test "hinge_loss" =
  let open Sklearn.Metrics in
  let open Sklearn.Svm in
  let x = matrixi [|[|0|]; [|1|]|] in
  let y = vectori [|-1; 1|] in
  let est = LinearSVC.create ~random_state:0 () in
  print LinearSVC.pp @@ LinearSVC.fit ~x ~y est;
  [%expect {|
      LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
                verbose=0)
  |}];
  let pred_decision = LinearSVC.decision_function ~x:(matrix [|[|-2.|]; [|3.|]; [|0.5|]|]) est in
  print_ndarray @@ pred_decision;
  [%expect {|
      [-2.18177944  2.36355888  0.09088972]
  |}];
  print_float @@ hinge_loss ~y_true:(vectori [|-1; 1; 1|]) ~pred_decision ();
  [%expect {|
      0.303037
  |}]


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

let%expect_test "hinge_loss" =
  let open Sklearn.Metrics in
  let open Sklearn.Svm in
  let x = matrixi [|[|0|]; [|1|]; [|2|]; [|3|]|] in
  let y = vectori [|0; 1; 2; 3|] in
  let labels = vectori [|0; 1; 2; 3|] in
  let est = LinearSVC.create ~random_state:0 () in
  print LinearSVC.pp @@ LinearSVC.fit ~x ~y est;
  [%expect {|
      LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
                verbose=0)
  |}];
  let pred_decision = LinearSVC.decision_function ~x:(matrixi [|[|-1|]; [|2|]; [|3|]|]) est in
  let y_true = vectori [|0; 2; 3|] in
  print_float @@ hinge_loss ~y_true ~pred_decision ~labels ();
  [%expect {| 0.564119 |}]


(* jaccard_score *)
(*
>>> import numpy as np
>>> from sklearn.metrics import jaccard_score
>>> y_true = np.array([[0, 1, 1],
...                    [1, 1, 0]])
>>> y_pred = np.array([[1, 1, 1],
...                    [1, 0, 0]])
>>> jaccard_score(y_true[0], y_pred[0])
0.6666...
>>> jaccard_score(y_true, y_pred, average='samples')
0.5833...
>>> jaccard_score(y_true, y_pred, average='macro')
0.6666...
>>> jaccard_score(y_true, y_pred, average=None)
array([0.5, 0.5, 1. ])

*)

let%expect_test "jaccard_score" =
  let open Sklearn.Metrics in
  let y_true = matrixi [|[|0; 1; 1|]; [|1; 1; 0|]|] in
  let y_pred = matrixi [|[|1; 1; 1|]; [|1; 0; 0|]|] in
  print_ndarray @@ jaccard_score ~y_true:(Sklearn.Arr.get ~i:[`I 0] y_true) ~y_pred:(Sklearn.Arr.get ~i:[`I 0] y_pred) (); 
  [%expect {|
      0.6666666666666666
   |}];
  print_ndarray @@ jaccard_score ~y_true ~y_pred ~average:`Samples ();
  [%expect {|
      0.5833333333333333
  |}];
  print_ndarray @@ jaccard_score ~y_true ~y_pred ~average:`Macro ();
  [%expect {|
      0.6666666666666666
  |}];
  print_ndarray @@ jaccard_score ~y_true ~y_pred ~average:`None ();
  [%expect {|
      [0.5 0.5 1. ]
  |}]


(* jaccard_score *)
(*
>>> y_pred = [0, 2, 1, 2]
>>> y_true = [0, 1, 2, 2]
>>> jaccard_score(y_true, y_pred, average=None)

*)

let%expect_test "jaccard_score" =
  let open Sklearn.Metrics in
  let y_pred = vectori [|0; 2; 1; 2|] in
  let y_true = vectori [|0; 1; 2; 2|] in
  print_ndarray @@ jaccard_score ~y_true ~y_pred ~average:`None ();
  [%expect {| [1.         0.         0.33333333] |}]


(* label_ranking_average_precision_score *)
(*
>>> import numpy as np
>>> from sklearn.metrics import label_ranking_average_precision_score
>>> y_true = np.array([[1, 0, 0], [0, 0, 1]])
>>> y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
>>> label_ranking_average_precision_score(y_true, y_score)

*)

let%expect_test "label_ranking_average_precision_score" =
  let open Sklearn.Metrics in
  let y_true = matrixi [|[|1; 0; 0|]; [|0; 0; 1|]|] in
  let y_score = matrix [|[|0.75; 0.5; 1.|]; [|1.; 0.2; 0.1|]|] in
  print_float @@ label_ranking_average_precision_score ~y_true ~y_score ();
  [%expect {| 0.416667 |}]


(* log_loss *)
(*
>>> from sklearn.metrics import log_loss
>>> log_loss(["spam", "ham", "ham", "spam"],
...          [[.1, .9], [.9, .1], [.8, .2], [.35, .65]])
0.21616...

*)

let%expect_test "log_loss" =
  let open Sklearn.Metrics in
  print_float @@ log_loss
    ~y_true:(vectors [|"spam"; "ham"; "ham"; "spam"|])
    ~y_pred:(matrix [|[|0.1; 0.9|]; [|0.9; 0.1|]; [|0.8; 0.2|]; [|0.35; 0.65|]|]) ();
  [%expect {|
      0.216162
  |}]


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

let%expect_test "make_scorer" =
  let open Sklearn.Metrics in
  let ftwo_scorer = make_scorer ~score_func:(get_py "fbeta_score") ~kwargs:["beta", Py.Int.of_int 2] () in
  print_py @@ ftwo_scorer;
  [%expect {|
      make_scorer(fbeta_score, beta=2)
   |}];
  let grid =
    Sklearn.Model_selection.GridSearchCV.create
      ~estimator:Sklearn.Svm.LinearSVC.(create ~random_state:0 ())
      ~param_grid:(`Grid ["C", `Ints [1; 10]]) ~scoring:(`Callable ftwo_scorer) ~cv:(`I 2) ()
  in
  print Sklearn.Model_selection.GridSearchCV.pp @@
  Sklearn.Model_selection.GridSearchCV.fit
    ~x:(matrixi [|[|1;2|]; [|3;4|]; [|5;6|]; [|7;8|]; [|9;10|]; [|11;12|]|])
    ~y:(vectori [|0; 1; 1; 0; 0; 1|]) grid;
  [%expect {|
     GridSearchCV(cv=2, error_score=nan,
                  estimator=LinearSVC(C=1.0, class_weight=None, dual=True,
                                      fit_intercept=True, intercept_scaling=1,
                                      loss='squared_hinge', max_iter=1000,
                                      multi_class='ovr', penalty='l2',
                                      random_state=0, tol=0.0001, verbose=0),
                  iid='deprecated', n_jobs=None, param_grid={'C': [1, 10]},
                  pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                  scoring=make_scorer(fbeta_score, beta=2), verbose=0)
   |}];
  print Sklearn.Dict.pp @@ Sklearn.Model_selection.GridSearchCV.cv_results_ grid;
  let output = Re.replace (Re.Perl.compile_pat {|('\w+time': array)\([^()]+\)|})
      ~f:(fun group -> Printf.sprintf "%s(...)" (Re.Group.get group 1)) [%expect.output]
  in
  let output = Re.replace (Re.Perl.compile_pat {|([)\]],)|})
      ~f:(fun group -> Printf.sprintf "%s\n" (Re.Group.get group 1)) output
  in
  print_string output;
  [%expect {|
    {'mean_fit_time': array(...),
     'std_fit_time': array(...),
     'mean_score_time': array(...),
     'std_score_time': array(...),
     'param_C': masked_array(data=[1, 10],

                 mask=[False, False],

           fill_value='?',
                dtype=object),
     'params': [{'C': 1}, {'C': 10}],
     'split0_test_score': array([0.71428571, 0.83333333]),
     'split1_test_score': array([0., 0.]),
     'mean_test_score': array([0.35714286, 0.41666667]),
     'std_test_score': array([0.35714286, 0.41666667]),
     'rank_test_score': array([2, 1],
     dtype=int32)} |}]

(* matthews_corrcoef *)
(*
>>> from sklearn.metrics import matthews_corrcoef
>>> y_true = [+1, +1, +1, -1]
>>> y_pred = [+1, -1, +1, +1]
>>> matthews_corrcoef(y_true, y_pred)

*)

let%expect_test "matthews_corrcoef" =
  let open Sklearn.Metrics in
  let y_true = vectori [|+1; +1; +1; -1|] in
  let y_pred = vectori [|+1; -1; +1; +1|] in
  print_float @@ matthews_corrcoef ~y_true ~y_pred ();
  [%expect {| -0.333333 |}]


(* max_error *)
(*
>>> from sklearn.metrics import max_error
>>> y_true = [3, 2, 7, 1]
>>> y_pred = [4, 2, 7, 1]
>>> max_error(y_true, y_pred)

*)

let%expect_test "max_error" =
  let open Sklearn.Metrics in
  let y_true = vectori [|3; 2; 7; 1|] in
  let y_pred = vectori [|4; 2; 7; 1|] in
  print_float @@ max_error ~y_true ~y_pred ();
  [%expect {| 1 |}]


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

let%expect_test "mean_absolute_error" =
  let open Sklearn.Metrics in
  let y_true = vector [|3.; -0.5; 2.; 7.|] in
  let y_pred = vector [|2.5; 0.0; 2.; 8.|] in
  print_ndarray @@ mean_absolute_error ~y_true ~y_pred ();
  [%expect {|
      0.5
   |}];
  let y_true = matrix [|[|0.5; 1.|]; [|-1.; 1.|]; [|7.; -6.|]|] in
  let y_pred = matrixi [|[|0; 2|]; [|-1; 2|]; [|8; -5|]|] in
  print_ndarray @@ mean_absolute_error ~y_true ~y_pred ();
  [%expect {|
      0.75
   |}];
  print_ndarray @@ mean_absolute_error ~y_true ~y_pred ~multioutput:`Raw_values ();
  [%expect {|
      [0.5 1. ]
   |}];
  print_ndarray @@ mean_absolute_error ~y_true ~y_pred ~multioutput:(`Arr (vector [|0.3; 0.7|])) ();
  [%expect {| 0.85 |}]


(* mean_gamma_deviance *)
(*
>>> from sklearn.metrics import mean_gamma_deviance
>>> y_true = [2, 0.5, 1, 4]
>>> y_pred = [0.5, 0.5, 2., 2.]
>>> mean_gamma_deviance(y_true, y_pred)

*)

let%expect_test "mean_gamma_deviance" =
  let open Sklearn.Metrics in
  let y_true = vector [|2.; 0.5; 1.; 4.|] in
  let y_pred = vector [|0.5; 0.5; 2.; 2.|] in
  print_float @@ mean_gamma_deviance ~y_true ~y_pred ();
  [%expect {| 1.05685 |}]


(* mean_poisson_deviance *)
(*
>>> from sklearn.metrics import mean_poisson_deviance
>>> y_true = [2, 0, 1, 4]
>>> y_pred = [0.5, 0.5, 2., 2.]
>>> mean_poisson_deviance(y_true, y_pred)

*)

let%expect_test "mean_poisson_deviance" =
  let open Sklearn.Metrics in
  let y_true = vectori [|2; 0; 1; 4|] in
  let y_pred = vector [|0.5; 0.5; 2.; 2.|] in
  print_float @@ mean_poisson_deviance ~y_true ~y_pred ();
  [%expect {| 1.42602 |}]


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

let%expect_test "mean_squared_error" =
  let open Sklearn.Metrics in
  let y_true = vector [|3.; -0.5; 2.; 7.|] in
  let y_pred = vector [|2.5; 0.0; 2.; 8.|] in
  print_ndarray @@ mean_squared_error ~y_true ~y_pred ();
  [%expect {|
      0.375
   |}];
  let y_true = vector [|3.; -0.5; 2.; 7.|] in
  let y_pred = vector [|2.5; 0.0; 2.; 8.|] in
  print_ndarray @@ mean_squared_error ~y_true ~y_pred ~squared:false ();
  [%expect {|
      0.6123724356957945
   |}];
  let y_true = matrix  [|[|0.5; 1.|]; [|-1.; 1.|]; [|7.;  -6.|]|] in
  let y_pred = matrixi [|[|0;   2 |]; [|-1;  2 |]; [|8;   -5 |]|] in
  print_ndarray @@ mean_squared_error ~y_true ~y_pred ();
  [%expect {|
      0.7083333333333334
   |}];
  print_ndarray @@ mean_squared_error ~y_true ~y_pred ~multioutput:`Raw_values ();
  [%expect {|
      [0.41666667 1.        ]
   |}];
  print_ndarray @@ mean_squared_error ~y_true ~y_pred ~multioutput:(`Arr (vector [|0.3; 0.7|])) ();
  [%expect {| 0.825 |}]


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

let%expect_test "mean_squared_log_error" =
  let open Sklearn.Metrics in
  let y_true = vector [|3.; 5.; 2.5; 7.|] in
  let y_pred = vector [|2.5; 5.; 4.; 8.|] in
  print_ndarray @@ mean_squared_log_error ~y_true ~y_pred ();
  [%expect {|
      0.03973012298459379
   |}];
  let y_true = matrix [|[|0.5; 1.|]; [|1.; 2.|]; [|7.; 6.|]|] in
  let y_pred = matrix [|[|0.5; 2.|]; [|1.; 2.5|]; [|8.; 8.|]|] in
  print_ndarray @@ mean_squared_log_error ~y_true ~y_pred ();
  [%expect {|
      0.044199361889160516
   |}];
  print_ndarray @@ mean_squared_log_error ~y_true ~y_pred ~multioutput:`Raw_values ();
  [%expect {|
      [0.00462428 0.08377444]
   |}];
  print_ndarray @@ mean_squared_log_error ~y_true ~y_pred ~multioutput:(`Arr (vector [|0.3; 0.7|])) ();
  [%expect {| 0.06002939417970032 |}]


(* mean_tweedie_deviance *)
(*
>>> from sklearn.metrics import mean_tweedie_deviance
>>> y_true = [2, 0, 1, 4]
>>> y_pred = [0.5, 0.5, 2., 2.]
>>> mean_tweedie_deviance(y_true, y_pred, power=1)

*)

let%expect_test "mean_tweedie_deviance" =
  let open Sklearn.Metrics in
  let y_true = vectori [|2; 0; 1; 4|] in
  let y_pred = vector [|0.5; 0.5; 2.; 2.|] in
  print_float @@ mean_tweedie_deviance ~y_true ~y_pred ~power:1. ();
  [%expect {| 1.42602 |}]


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

let%expect_test "median_absolute_error" =
  let open Sklearn.Metrics in
  let y_true = vector [|3.; -0.5; 2.; 7.|] in
  let y_pred = vector [|2.5; 0.0; 2.; 8.|] in
  print_ndarray @@ median_absolute_error ~y_true ~y_pred ();
  [%expect {|
      0.5
   |}];
  let y_true = matrix [|[|0.5; 1.|]; [|-1.; 1.|]; [|7.; -6.|]|] in
  let y_pred = matrixi [|[|0; 2|]; [|-1; 2|]; [|8; -5|]|] in
  print_ndarray @@ median_absolute_error ~y_true ~y_pred ();
  [%expect {|
      0.75
   |}];
  print_ndarray @@ median_absolute_error ~y_true ~y_pred ~multioutput:`Raw_values ();
  [%expect {|
      [0.5 1. ]
   |}];
  print_ndarray @@ median_absolute_error ~y_true ~y_pred ~multioutput:(`Arr (vector [|0.3; 0.7|])) ();
  [%expect {| 0.85 |}]


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

let%expect_test "multilabel_confusion_matrix" =
  let open Sklearn.Metrics in
  let y_true = matrixi [|[|1; 0; 1|]; [|0; 1; 0|]|] in
  let y_pred = matrixi [|[|1; 0; 0|]; [|0; 1; 1|]|] in
  print_ndarray @@ multilabel_confusion_matrix ~y_true ~y_pred ();
  [%expect {|
      [[[1 0]
        [0 1]]

       [[1 0]
        [0 1]]

       [[0 1]
        [1 0]]]
   |}]

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

let%expect_test "multilabel_confusion_matrix" =
  let open Sklearn.Metrics in
  let y_true = vectors [|"cat"; "ant"; "cat"; "cat"; "ant"; "bird"|] in
  let y_pred = vectors [|"ant"; "ant"; "cat"; "cat"; "ant"; "cat"|] in
  print_ndarray @@ multilabel_confusion_matrix ~y_true ~y_pred ~labels:(vectors [|"ant"; "bird"; "cat"|]) ();
  [%expect {|
      [[[3 1]
        [0 2]]

       [[5 0]
        [1 0]]

       [[2 1]
        [1 2]]]
   |}]


(* nan_euclidean_distances *)
(*
>>> from sklearn.metrics.pairwise import nan_euclidean_distances
>>> nan = float("NaN")
>>> X = [[0, 1], [1, nan]]
>>> nan_euclidean_distances(X, X) # distance between rows of X
array([[0.        , 1.41421356],
       [1.41421356, 0.        ]])

*)

let%expect_test "nan_euclidean_distances" =
  let open Sklearn.Metrics in
  let x = matrix [|[|0.; 1.|]; [|1.; nan|]|] in
  print_ndarray @@ nan_euclidean_distances ~x ~y:x () (* distance between rows of x *);
  [%expect {|
      [[0.         1.41421356]
       [1.41421356 0.        ]]
   |}]


(* nan_euclidean_distances *)
(*
>>> # get distance to origin
>>> nan_euclidean_distances(X, [[0, 0]])
array([[1.        ],
       [1.41421356]])

*)

let%expect_test "nan_euclidean_distances" =
  let open Sklearn.Metrics in
  let x = matrix [|[|0.; 1.|]; [|1.; nan|]|] in
  (* get distance to origin *)
  print_ndarray @@ nan_euclidean_distances ~x ~y:(matrixi [|[|0; 0|]|]) ();
  [%expect {|
      [[1.        ]
       [1.41421356]]
   |}]


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

let%expect_test "ndcg_score" =
  let open Sklearn.Metrics in
  (*  we have groud-truth relevance of some answers to a query:; *)
  let true_relevance = matrixi [|[|10; 0; 0; 1; 5|]|] in
  (*  we predict some scores (relevance) for the answers; *)
  let scores = matrix [|[|0.1; 0.2; 0.3; 4.; 70.|]|] in
  print_float @@ ndcg_score ~y_true:true_relevance ~y_score:scores ();
  [%expect {|
      0.695694
   |}];
  let scores = matrix [|[|0.05; 1.1; 1.; 0.5; 0.0|]|] in
  print_float @@ ndcg_score ~y_true:true_relevance ~y_score:scores ();
  [%expect {|
      0.49368
   |}];
  (*  we can set k to truncate the sum; only top k answers contribute.; *)
  print_float @@ ndcg_score ~y_true:true_relevance ~y_score:scores ~k:4 ();
  [%expect {|
      0.352024
   |}];
  (*  the normalization takes k into account so a perfect answer; *)
  (*  would still get 1.0; *)
  print_float @@ ndcg_score ~y_true:true_relevance ~y_score:true_relevance ~k:4 ();
  [%expect {|
      1
   |}];
  (*  now we have some ties in our prediction; *)
  let scores = matrixi [|[|1; 0; 0; 0; 1|]|] in
  (*  by default ties are averaged, so here we get the average (normalized); *)
  (*  true relevance of our top predictions: (10 / 10 + 5 / 10) / 2 = .75; *)
  print_float @@ ndcg_score ~y_true:true_relevance ~y_score:scores ~k:1 ();
  [%expect {|
      0.75
   |}];
  (*  we can choose to ignore ties for faster results, but only; *)
  (*  if we know there aren't ties in our scores, otherwise we get; *)
  (*  wrong results:; *)
  print_float @@ ndcg_score ~y_true:true_relevance ~y_score:scores ~k:1 ~ignore_ties:true ();
  [%expect {| 0.5 |}]


(*--------- Examples for module Sklearn.Metrics.Pairwise ----------*)


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

let%expect_test "haversine_distances" =
  let open Sklearn.Metrics.Pairwise in
  let radians deg = Sklearn.Arr.(deg * (float Stdlib.Float.pi) / (float 180.)) in
  let bsas = vector [|-34.83333; -58.5166646|] in
  let paris = vector [|49.0083899664; 2.53844117956|] in
  let bsas_in_radians = radians bsas in
  let paris_in_radians = radians paris in
  let result = haversine_distances ~x:(Sklearn.Arr.vstack [bsas_in_radians; paris_in_radians]) () in
  print_ndarray @@ Sklearn.Arr.(result * (float 6371000.) / (float 1000.)) (* multiply by Earth radius to get kilometers *);
  [%expect {|
      [[    0.         11099.54035582]
       [11099.54035582     0.        ]]
   |}]


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

(*
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

(*
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

(*
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

let%expect_test "manhattan_distances" =
  let open Sklearn.Metrics.Pairwise in
  print_ndarray @@ manhattan_distances ~x:(matrixi [|[|3|]|]) ~y:(matrixi [|[|3|]|]) ();
  [%expect {|
      [[0.]]
   |}];
  print_ndarray @@ manhattan_distances ~x:(matrixi [|[|3|]|]) ~y:(matrixi [|[|2|]|]) ();
  [%expect {|
      [[1.]]
   |}];
  print_ndarray @@ manhattan_distances ~x:(matrixi [|[|2|]|]) ~y:(matrixi [|[|3|]|]) ();
  [%expect {|
      [[1.]]
   |}];
  print_ndarray @@ manhattan_distances ~x:(matrixi [|[|1; 2|]; [|3; 4|]|]) ~y:(matrixi [|[|1; 2|]; [|0; 3|]|]) ();
  [%expect {|
      [[0. 2.]
       [4. 4.]]
   |}];
  let x = Sklearn.Arr.ones [1; 2] in
  let y = Sklearn.Arr.full ~shape:[2; 2] (`F 2.) in
  print_ndarray @@ manhattan_distances ~x ~y ~sum_over_features:false ();
  [%expect {|
      [[1. 1.]
       [1. 1.]]
   |}]


(* paired_distances *)
(*
>>> from sklearn.metrics.pairwise import paired_distances
>>> X = [[0, 1], [1, 1]]
>>> Y = [[0, 1], [2, 1]]
>>> paired_distances(X, Y)
array([0., 1.])

*)

let%expect_test "paired_distances" =
  let open Sklearn.Metrics.Pairwise in
  let x = matrixi [|[|0; 1|]; [|1; 1|]|] in
  let y = matrixi [|[|0; 1|]; [|2; 1|]|] in
  print_ndarray @@ paired_distances ~x ~y ();
  [%expect {|
      [0. 1.]
   |}]


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

let next_exn seq =
  let open Seq in
  match seq () with
  | Nil -> invalid_arg "next_exn: empty Seq.t"
  | Cons(t, qseq) -> t, qseq

let%expect_test "pairwise_distances_chunked" =
  let open Sklearn.Metrics in
  Sklearn.Arr.Random.seed 0;
  let x = Sklearn.Arr.Random.random_sample [5; 3] in
  let d_chunk, _ = next_exn @@ pairwise_distances_chunked ~x () in
  print_ndarray @@ d_chunk;
  [%expect {|
      [[0.         0.29473397 0.41689499 0.19662693 0.57216693]
       [0.29473397 0.         0.57586803 0.41860234 0.76350759]
       [0.41689499 0.57586803 0.         0.44940452 0.90274337]
       [0.19662693 0.41860234 0.44940452 0.         0.51150232]
       [0.57216693 0.76350759 0.90274337 0.51150232 0.        ]]
   |}]


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

let%expect_test "pairwise_distances_chunked" =
  let open Sklearn.Metrics.Pairwise in
  let module Arr = Sklearn.Arr in
  let r = 0.2 in
  let reduce_func d_chunk =
    let neigh = Arr.iter d_chunk |> Seq.fold_left (fun acc row ->
        Arr.List.append acc Arr.(flatnonzero (row < (float r)));
        acc
      ) (Arr.List.create ())
    in
    let avg_dist = Arr.(d_chunk * (d_chunk < (float r)) |> mean ~axis:[1]) in
    neigh, avg_dist
  in
  Sklearn.Arr.Random.seed 0;
  let x = Sklearn.Arr.Random.random_sample [5; 3] in
  let gen = pairwise_distances_chunked ~x () in
  let gen = Seq.map reduce_func gen in
  let (neigh, avg_dist), _ = next_exn gen in
  print Sklearn.Arr.List.pp neigh;
  [%expect {|
      [array([0, 3]), array([1]), array([2]), array([0, 3]), array([4])]
   |}];
  print_ndarray @@ avg_dist;
  [%expect {|
      [0.03932539 0.         0.         0.03932539 0.        ]
   |}]


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
>>> gen = pairwise_distances_chunked(X, reduce_func=reduce_func,
...                                  working_memory=0)
>>> next(gen)
[array([0, 3])]
>>> next(gen)

*)

let chunk_mapi ~f seq =
  let i = ref 0 in
  Seq.map (fun x -> let ret = f x !i in i := !i + (Sklearn.Arr.shape x).(0); ret) seq

let%expect_test "pairwise_distances_chunked" =
  let open Sklearn.Metrics in
  let module Arr = Sklearn.Arr in
  let r = vector [|0.2; 0.4; 0.4; 0.3; 0.1|] in
  let reduce_func d_chunk i =
    let neigh, _ = Arr.iter d_chunk |> Seq.fold_left (fun (acc, i) row ->
        Arr.List.append acc Arr.(flatnonzero (row < (get ~i:[`I i] r)));
        acc, succ i
      ) (Arr.List.create (), i)
    in
    neigh
  in
  Sklearn.Arr.Random.seed 0;
  let x = Sklearn.Arr.Random.random_sample [5; 3] in

  let gen = pairwise_distances_chunked ~x () in
  let gen = chunk_mapi ~f:reduce_func gen in

  print Sklearn.Arr.List.pp (fst @@ next_exn gen);
  [%expect {|
      [array([0, 3]), array([0, 1]), array([2]), array([0, 3]), array([4])]
   |}];

  let gen = pairwise_distances_chunked ~x ~working_memory:0 () in
  let gen = chunk_mapi ~f:reduce_func gen in
  let neigh, gen = next_exn gen in
  print Sklearn.Arr.List.pp neigh;
  [%expect {|
      [array([0, 3])]
   |}];
  let neigh, _gen = next_exn gen in
  print Sklearn.Arr.List.pp neigh;
  [%expect {|
      [array([0, 1])]
   |}]

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

let%expect_test "plot_roc_curve" =
  let x, y = Sklearn.Datasets.make_classification ~random_state:0 () in
  let [@ocaml.warning "-8"] [x_train; x_test; y_train; y_test] =
    Sklearn.Model_selection.train_test_split [x; y] ~random_state:0
  in
  let module SVC = Sklearn.Svm.SVC in
  let clf = SVC.create ~random_state:0 () in
  print SVC.pp @@ SVC.fit ~x:x_train ~y:y_train clf;
  [%expect {|
      SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
          decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
          max_iter=-1, probability=False, random_state=0, shrinking=True, tol=0.001,
          verbose=False)
   |}];
  print_py @@ Sklearn.Metrics.plot_roc_curve ~estimator:clf ~x:x_test ~y:y_test ();
  let output = Re.replace_string (Re.Perl.compile_pat {|0x[a-f0-9]+|}) ~by:"0x..." [%expect.output]
  in
  print_string output;
  [%expect {| <sklearn.metrics._plot.roc_curve.RocCurveDisplay object at 0x...> |}]


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

let%expect_test "precision_recall_curve" =
  let open Sklearn.Metrics in
  let y_true = vectori [|0; 0; 1; 1|] in
  let y_scores = vector [|0.1; 0.4; 0.35; 0.8|] in
  let precision, recall, thresholds = precision_recall_curve ~y_true ~probas_pred:y_scores () in
  print_ndarray precision;
  [%expect {|
      [0.66666667 0.5        1.         1.        ]
   |}];
  print_ndarray recall;
  [%expect {|
      [1.  0.5 0.5 0. ]
   |}];
  print_ndarray thresholds;
  [%expect {| [0.35 0.4  0.8 ] |}]


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

let print_tuple4 x =
  let module Arr = Sklearn.Arr in
  match x with
  | (a, b, c, None) ->
    Format.printf "(%a, %a, %a, None)" Arr.pp a Arr.pp b Arr.pp c
  | (a, b, c, Some d) ->
    Format.printf "(%a, %a, %a, %a)" Arr.pp a Arr.pp b Arr.pp c Arr.pp d

let%expect_test "precision_recall_fscore_support" =
  let open Sklearn.Metrics in
  let y_true = vectors [|"cat"; "dog"; "pig"; "cat"; "dog"; "pig"|] in
  let y_pred = vectors [|"cat"; "pig"; "dog"; "cat"; "cat"; "dog"|] in
  print_tuple4 @@ precision_recall_fscore_support ~y_true ~y_pred ~average:`Macro ();
  [%expect {|
      (0.2222222222222222, 0.3333333333333333, 0.26666666666666666, None)
   |}];
  print_tuple4 @@ precision_recall_fscore_support ~y_true ~y_pred ~average:`Micro ();
  [%expect {|
      (0.3333333333333333, 0.3333333333333333, 0.3333333333333333, None)
   |}];
  print_tuple4 @@ precision_recall_fscore_support ~y_true ~y_pred ~average:`Weighted ();
  [%expect {|
      (0.2222222222222222, 0.3333333333333333, 0.26666666666666666, None)
   |}]


(* precision_recall_fscore_support *)
(*
>>> precision_recall_fscore_support(y_true, y_pred, average=None,
... labels=['pig', 'dog', 'cat'])
(array([0.        , 0.        , 0.66...]),
 array([0., 0., 1.]), array([0. , 0. , 0.8]),
 array([2, 2, 2]))

*)

let%expect_test "precision_recall_fscore_support" =
  let open Sklearn.Metrics in
  let y_true = vectors [|"cat"; "dog"; "pig"; "cat"; "dog"; "pig"|] in
  let y_pred = vectors [|"cat"; "pig"; "dog"; "cat"; "cat"; "dog"|] in
  print_tuple4 @@ precision_recall_fscore_support ~y_true ~y_pred ~labels:(vectors [|"pig"; "dog"; "cat"|]) ();
  [%expect {|
      ([0.         0.         0.66666667], [0. 0. 1.], [0.  0.  0.8], [2 2 2])
   |}]


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

let%expect_test "precision_score" =
  let open Sklearn.Metrics in
  let y_true = vectori [|0; 1; 2; 0; 1; 2|] in
  let y_pred = vectori [|0; 2; 1; 0; 0; 1|] in
  print_ndarray @@ precision_score ~y_true ~y_pred ~average:`Macro ();
  [%expect {|
      0.2222222222222222
   |}];
  print_ndarray @@ precision_score ~y_true ~y_pred ~average:`Micro ();
  [%expect {|
      0.3333333333333333
   |}];
  print_ndarray @@ precision_score ~y_true ~y_pred ~average:`Weighted ();
  [%expect {|
      0.2222222222222222
   |}];
  print_ndarray @@ precision_score ~y_true ~y_pred ~average:`None ();
  [%expect {|
      [0.66666667 0.         0.        ]
   |}];
  let y_pred = vectori [|0; 0; 0; 0; 0; 0|] in
  print_ndarray @@ precision_score ~y_true ~y_pred ~average:`None ();
  [%expect {|
      [0.33333333 0.         0.        ]
   |}];
  print_ndarray @@ precision_score ~y_true ~y_pred ~average:`None ~zero_division:`One ();
  [%expect {|
      [0.33333333 1.         1.        ]
   |}]


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

let%expect_test "r2_score" =
  let open Sklearn.Metrics in
  let y_true = vector [|3.; -0.5; 2.; 7.|] in
  let y_pred = vector [|2.5; 0.0; 2.; 8.|] in
  print_ndarray @@ r2_score ~y_true ~y_pred ();
  [%expect {|
      0.9486081370449679
   |}];
  let y_true = (matrix [|[|0.5; 1.|]; [|-1.; 1.|]; [|7.; -6.|]|]) in
  let y_pred = (matrixi [|[|0; 2|]; [|-1; 2|]; [|8; -5|]|]) in
  print_ndarray @@ r2_score ~y_true ~y_pred ~multioutput:`Variance_weighted ();
  [%expect {|
      0.9382566585956417
   |}];
  let y_true = vectori [|1; 2; 3|] in
  let y_pred = vectori [|1; 2; 3|] in
  print_ndarray @@ r2_score ~y_true ~y_pred ();
  [%expect {|
      1.0
   |}];
  let y_true = vectori [|1; 2; 3|] in
  let y_pred = vectori [|2; 2; 2|] in
  print_ndarray @@ r2_score ~y_true ~y_pred ();
  [%expect {|
      0.0
   |}];
  let y_true = vectori [|1; 2; 3|] in
  let y_pred = vectori [|3; 2; 1|] in
  print_ndarray @@ r2_score ~y_true ~y_pred ();
  [%expect {| -3.0 |}]


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

let%expect_test "recall_score" =
  let open Sklearn.Metrics in
  let y_true = vectori [|0; 1; 2; 0; 1; 2|] in
  let y_pred = vectori [|0; 2; 1; 0; 0; 1|] in
  print_ndarray @@ recall_score ~y_true ~y_pred ~average:`Macro ();
  [%expect {|
      0.3333333333333333
   |}];
  print_ndarray @@ recall_score ~y_true ~y_pred ~average:`Micro ();
  [%expect {|
      0.3333333333333333
   |}];
  print_ndarray @@ recall_score ~y_true ~y_pred ~average:`Weighted ();
  [%expect {|
      0.3333333333333333
   |}];
  print_ndarray @@ recall_score ~y_true ~y_pred ~average:`None ();
  [%expect {|
      [1. 0. 0.]
   |}];
  let y_true = vectori [|0; 0; 0; 0; 0; 0|] in
  print_ndarray @@ recall_score ~y_true ~y_pred ~average:`None ();
  [%expect {|
      [0.5 0.  0. ]
   |}];
  print_ndarray @@ recall_score ~y_true ~y_pred ~average:`None ~zero_division:`One ();
  [%expect {|
      [0.5 1.  1. ]
   |}]

(* roc_auc_score *)
(*
>>> import numpy as np
>>> from sklearn.metrics import roc_auc_score
>>> y_true = np.array([0, 0, 1, 1])
>>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
>>> roc_auc_score(y_true, y_scores)

*)

let%expect_test "roc_auc_score" =
  let open Sklearn.Metrics in
  let y_true = vectori [|0; 0; 1; 1|] in
  let y_score = vector [|0.1; 0.4; 0.35; 0.8|] in
  print_float @@ roc_auc_score ~y_true ~y_score ();
  [%expect {| 0.75 |}]


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

let%expect_test "roc_curve" =
  let open Sklearn.Metrics in
  let y = vectori [|1; 1; 2; 2|] in
  let scores = vector [|0.1; 0.4; 0.35; 0.8|] in
  let fpr, tpr, thresholds = roc_curve ~y_true:y ~y_score:scores ~pos_label:(`I 2) () in
  print_ndarray fpr;
  [%expect {|
      [0.  0.  0.5 0.5 1. ]
   |}];
  print_ndarray tpr;
  [%expect {|
      [0.  0.5 0.5 1.  1. ]
   |}];
  print_ndarray thresholds;
  [%expect {| [1.8  0.8  0.4  0.35 0.1 ] |}]


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

let print_zol x = match x with
  | `I x -> Format.printf "%d\n" x
  | `F x -> Format.printf "%g\n" x

let%expect_test "zero_one_loss" =
  let open Sklearn.Metrics in
  let y_pred = vectori [|1; 2; 3; 4|] in
  let y_true = vectori [|2; 2; 3; 4|] in
  print_zol @@ zero_one_loss ~y_true ~y_pred ();
  [%expect {|
      0.25
   |}];
  print_zol @@ zero_one_loss ~y_true ~y_pred ~normalize:false ();
  [%expect {|
      1
   |}]


(* zero_one_loss *)
(*
>>> import numpy as np
>>> zero_one_loss(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))

*)

let%expect_test "zero_one_loss" =
  let open Sklearn.Metrics in
  let module Arr = Sklearn.Arr in
  print_zol @@ zero_one_loss ~y_true:(matrixi [|[|0; 1|]; [|1; 1|]|]) ~y_pred:(Arr.ones [2;2]) ();
  [%expect {| 0.5 |}]
