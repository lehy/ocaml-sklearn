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

(* Parallel *)
(*
>>> from math import sqrt
>>> from joblib import Parallel, delayed
>>> Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10))
[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


*)

(* TEST TODO
let%expect_test "Parallel" =
    let sqrt = Math.sqrt in
    from joblib import Parallel, delayed    
    Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10))    
    [%expect {|
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]            
    |}]

*)
(* let%expect_test "Parallel" =
 *   let sqrt args = args.(0) |> Py.Number.to_float |> sqrt |> Py.Float.of_float in
 *   let par = Sklearn.Parallel.create ~n_jobs:1 () in
 *   let powi x n = Float.pow (Float.of_int x) n in
 *   let res = Sklearn.Parallel.call par ~f:sqrt ~args:(Array.init 10 (fun i -> Py.Number.of_float (powi i 2.)) in
 *   Format.printf "%s" (Py.Object.to_string res);
 *   [%expect {|
 *             [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]            
 *     |}] *)



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
    let modf = Math.modf in
    from joblib import Parallel, delayed    
    r = Parallel(n_jobs=1)(delayed(modf)(i/2.) for i in range(10))    
    let res, i = zip *r in
    res    
    [%expect {|
            (0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5)            
    |}]
    i    
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
    let sleep = Time.sleep in
    from joblib import Parallel, delayed    
    r = Parallel(n_jobs=2, verbose=10)(delayed(sleep)(.2) for _ in range(10)) #doctest: +SKIP    
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
    let nlargest = Heapq.nlargest in
    from joblib import Parallel, delayed    
    Parallel(n_jobs=2)(delayed(nlargest)(2, n) for n in (range(4), 'abcde', 3)) #doctest: +SKIP    
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



(* make_pipeline *)
(*
>>> from sklearn.naive_bayes import GaussianNB
>>> from sklearn.preprocessing import StandardScaler
>>> make_pipeline(StandardScaler(), GaussianNB(priors=None))
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('gaussiannb', GaussianNB())])


*)

let%expect_test "make_pipeline" =
  let open Sklearn in
  let pipe = Pipeline.make_pipeline [Preprocessing.StandardScaler.(create () |> to_pyobject);
                                     Naive_bayes.GaussianNB.(create () |> to_pyobject)] in
  print Pipeline.Pipeline.pp pipe;
    [%expect {|
            Pipeline(memory=None,
                     steps=[('standardscaler',
                             StandardScaler(copy=True, with_mean=True, with_std=True)),
                            ('gaussiannb', GaussianNB(priors=None, var_smoothing=1e-09))],
                     verbose=False)
    |}]

(* >>> from sklearn.decomposition import PCA, TruncatedSVD
 * >>> from sklearn.pipeline import make_union
 * >>> make_union(PCA(), TruncatedSVD())
 *   FeatureUnion(transformer_list=[('pca', PCA()),
 *                                  ('truncatedsvd', TruncatedSVD())]) *)

let%expect_test "make_union" =
  let open Sklearn in
  let union = Pipeline.make_union [Sklearn.Decomposition.PCA.(create () |> to_pyobject);
                                   Sklearn.Decomposition.TruncatedSVD.(create () |> to_pyobject)]
  in
  print Pipeline.FeatureUnion.pp union;
  [%expect {|
    FeatureUnion(n_jobs=None,
                 transformer_list=[('pca',
                                    PCA(copy=True, iterated_power='auto',
                                        n_components=None, random_state=None,
                                        svd_solver='auto', tol=0.0, whiten=False)),
                                   ('truncatedsvd',
                                    TruncatedSVD(algorithm='randomized',
                                                 n_components=2, n_iter=5,
                                                 random_state=None, tol=0.0))],
                 transformer_weights=None, verbose=False) |}]

(* >>> from sklearn.pipeline import FeatureUnion
 * >>> from sklearn.decomposition import PCA, TruncatedSVD
 * >>> union = FeatureUnion([("pca", PCA(n_components=1)),
 *                           ...                       ("svd", TruncatedSVD(n_components=2))])
 * >>> X = [[0., 1., 3], [2., 2., 5]]
 * >>> union.fit_transform(X)
 *   array([[ 1.5       ,  3.0...,  0.8...],
 *          [-1.5       ,  5.7..., -0.4...]]) *)

let%expect_test "FeatureUnion" =
  let open Sklearn in
  let open Pipeline in
  let union = FeatureUnion.create
      ~transformer_list:["pca", Sklearn.Decomposition.PCA.(create ~n_components:(`I 1) () |> to_pyobject);
                         "svd", Sklearn.Decomposition.TruncatedSVD.(create ~n_components:2 () |> to_pyobject)] ()
  in
  let x = matrix [|[|0.; 1.; 3.|]; [|2.; 2.; 5.|]|] in
  print_ndarray @@ FeatureUnion.fit_transform union ~x;
  [%expect {|
    [[ 1.5         3.03954967  0.87243213]
     [-1.5         5.72586357 -0.46312679]] |}]


let%expect_test "complex_pipeline" =
  let open Sklearn in
  let open Sklearn.Pipeline in
  let x, y = Datasets.make_classification ~n_informative:5 ~n_redundant:0 ~random_state:42 () in
  (*  ANOVA SVM-C  *)
  let f_regression =
    let _ = Py.Run.eval ~start:Py.File "import sklearn.feature_selection" in
    Py.Run.eval "sklearn.feature_selection.f_regression"
  in
  let anova_filter = Feature_selection.SelectKBest.create ~score_func:f_regression ~k:(`I 5) () in
  let clf = Svm.SVC.create ~kernel:"linear" () in
  let anova_svm = Pipeline.create ~steps:["anova", Feature_selection.SelectKBest.to_pyobject anova_filter;
                                          "svc", Svm.SVC.to_pyobject clf] ()
  in
  let anova_svm = Pipeline.(
      set_params anova_svm ~kwargs:["anova__k", (Py.Int.of_int 10); "svc__C", (Py.Float.of_float 0.1)]
      |> fit ~x ~y) in
  print Pipeline.pp anova_svm;
  (*  filtering out the address printed for f_regression, which is a problem for this test  *)
  print_string @@ Re.replace_string (Re.Perl.compile_pat "0x[a-fA-F0-9]+") ~by:"0x..." [%expect.output];
  [%expect {|
    Pipeline(memory=None,
             steps=[('anova',
                     SelectKBest(k=10,
                                 score_func=<function f_regression at 0x...>)),
                    ('svc',
                     SVC(C=0.1, break_ties=False, cache_size=200, class_weight=None,
                         coef0=0.0, decision_function_shape='ovr', degree=3,
                         gamma='scale', kernel='linear', max_iter=-1,
                         probability=False, random_state=None, shrinking=True,
                         tol=0.001, verbose=False))],
             verbose=False) |}];
  let prediction = Pipeline.predict anova_svm ~x in
  print_ndarray prediction;
  [%expect {|
    [1 0 0 1 1 1 0 1 0 0 1 0 1 0 0 1 0 1 0 1 0 1 1 0 0 0 0 1 0 1 0 0 1 1 1 1 1
     1 1 1 1 1 0 0 0 0 0 1 1 0 0 1 0 1 0 0 1 1 1 0 1 0 1 0 0 1 0 1 0 1 1 1 1 1
     1 0 1 1 0 1 0 1 1 0 0 0 1 1 0 0 1 0 1 1 1 0 1 0 0 0] |}];
  Format.printf "%g" @@ Pipeline.score anova_svm ~x ~y;
  [%expect {| 0.83 |}];
  (*  ouch, this is very ugly compared to anova_svm['anova']  *)
  let anova = Pipeline.get_item anova_svm ~ind:(`S "anova") |> Feature_selection.SelectKBest.of_pyobject in
  print_ndarray @@ Feature_selection.SelectKBest.get_support anova;
  [%expect {|
    [False False  True  True False False  True  True False  True False  True
      True False  True False  True  True False False] |}];
  let anova = Pipeline.named_steps anova_svm |> Sklearn.Dict.get (module Feature_selection.SelectKBest) ~name:"anova" in
  print_ndarray @@ Feature_selection.SelectKBest.get_support anova;
  [%expect {|
    [False False  True  True False False  True  True False  True False  True
      True False  True False  True  True False False] |}];
  (*  anova_svm.names_steps.anova: not wrapping that, won't be easier than the above  *)
  let sub_pipeline = Pipeline.get_item anova_svm ~ind:(`Slice(`None, `I 1, `None)) |> Pipeline.of_pyobject in
  let svc = Pipeline.get_item anova_svm ~ind:(`I (-1)) |> Svm.SVC.of_pyobject in
  let coef = Svm.SVC.coef_ svc in
  Sklearn.Arr.(shape coef |> Int.vector |> print pp);
  [%expect {| [ 1 10] |}];
  Arr.(Pipeline.inverse_transform sub_pipeline ~x:coef
       |> shape |> Int.vector |> print pp);
  [%expect {| [ 1 20] |}]
  
(* >>> from sklearn import svm
 * >>> from sklearn.datasets import make_classification
 * >>> from sklearn.feature_selection import SelectKBest
 * >>> from sklearn.feature_selection import f_regression
 * >>> from sklearn.pipeline import Pipeline
 * >>> # generate some data to play with
 * >>> X, y = make_classification(
 *     ...     n_informative=5, n_redundant=0, random_state=42)
 * >>> # ANOVA SVM-C
 * >>> anova_filter = SelectKBest(f_regression, k=5)
 * >>> clf = svm.SVC(kernel='linear')
 * >>> anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])
 * >>> # You can set the parameters using the names issued
 * >>> # For instance, fit using a k of 10 in the SelectKBest
 * >>> # and a parameter 'C' of the svm
 * >>> anova_svm.set_params(anova__k=10, svc__C=.1).fit(X, y)
 *   Pipeline(steps=[('anova', SelectKBest(...)), ('svc', SVC(...))])
 * >>> prediction = anova_svm.predict(X)
 * >>> anova_svm.score(X, y)
 *   0.83
 * >>> # getting the selected features chosen by anova_filter
 * >>> anova_svm['anova'].get_support()
 *   array([False, False,  True,  True, False, False,  True,  True, False,
 *          True, False,  True,  True, False,  True, False,  True,  True,
 *          False, False])
 * >>> # Another way to get selected features chosen by anova_filter
 * >>> anova_svm.named_steps.anova.get_support()
 *  array([False, False,  True,  True, False, False,  True,  True, False,
 *         True, False,  True,  True, False,  True, False,  True,  True,
 *         False, False])
 * >>> # Indexing can also be used to extract a sub-pipeline.
 * >>> sub_pipeline = anova_svm[:1]
 * >>> sub_pipeline
 *   Pipeline(steps=[('anova', SelectKBest(...))])
 * >>> coef = anova_svm[-1].coef_
 * >>> anova_svm['svc'] is anova_svm[-1]
 *   True
 * >>> coef.shape
 *   (1, 10)
 * >>> sub_pipeline.inverse_transform(coef).shape
 *   (1, 20) *)
