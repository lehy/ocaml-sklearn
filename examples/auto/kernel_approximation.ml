(* AdditiveChi2Sampler *)
(*
>>> from sklearn.datasets import load_digits
>>> from sklearn.linear_model import SGDClassifier
>>> from sklearn.kernel_approximation import AdditiveChi2Sampler
>>> X, y = load_digits(return_X_y=True)
>>> chi2sampler = AdditiveChi2Sampler(sample_steps=2)
>>> X_transformed = chi2sampler.fit_transform(X, y)
>>> clf = SGDClassifier(max_iter=5, random_state=0, tol=1e-3)
>>> clf.fit(X_transformed, y)
SGDClassifier(max_iter=5, random_state=0)
>>> clf.score(X_transformed, y)
0.9499...


*)

(* TEST TODO
let%expect_text "AdditiveChi2Sampler" =
    let load_digits = Sklearn.Datasets.load_digits in
    let sGDClassifier = Sklearn.Linear_model.sGDClassifier in
    let additiveChi2Sampler = Sklearn.Kernel_approximation.additiveChi2Sampler in
    let x, y = load_digits return_X_y=True in
    chi2sampler = AdditiveChi2Sampler(sample_steps=2)    
    X_transformed = chi2sampler.fit_transform(X, y)    
    clf = SGDClassifier(max_iter=5, random_state=0, tol=1e-3)    
    print @@ fit clf x_transformed y
    [%expect {|
            SGDClassifier(max_iter=5, random_state=0)            
    |}]
    print @@ score clf x_transformed y
    [%expect {|
            0.9499...            
    |}]

*)



(* Nystroem *)
(*
>>> from sklearn import datasets, svm
>>> from sklearn.kernel_approximation import Nystroem
>>> X, y = datasets.load_digits(n_class=9, return_X_y=True)
>>> data = X / 16.
>>> clf = svm.LinearSVC()
>>> feature_map_nystroem = Nystroem(gamma=.2,
...                                 random_state=1,
...                                 n_components=300)
>>> data_transformed = feature_map_nystroem.fit_transform(data)
>>> clf.fit(data_transformed, y)
LinearSVC()
>>> clf.score(data_transformed, y)
0.9987...


*)

(* TEST TODO
let%expect_text "Nystroem" =
    from sklearn import datasets, svm    
    let nystroem = Sklearn.Kernel_approximation.nystroem in
    X, y = datasets.load_digits(n_class=9, return_X_y=True)    
    data = X / 16.    
    clf = svm.LinearSVC()    
    feature_map_nystroem = Nystroem(gamma=.2,random_state=1,n_components=300)    
    data_transformed = feature_map_nystroem.fit_transform(data)    
    print @@ fit clf data_transformed y
    [%expect {|
            LinearSVC()            
    |}]
    print @@ score clf data_transformed y
    [%expect {|
            0.9987...            
    |}]

*)



(* RBFSampler *)
(*
>>> from sklearn.kernel_approximation import RBFSampler
>>> from sklearn.linear_model import SGDClassifier
>>> X = [[0, 0], [1, 1], [1, 0], [0, 1]]
>>> y = [0, 0, 1, 1]
>>> rbf_feature = RBFSampler(gamma=1, random_state=1)
>>> X_features = rbf_feature.fit_transform(X)
>>> clf = SGDClassifier(max_iter=5, tol=1e-3)
>>> clf.fit(X_features, y)
SGDClassifier(max_iter=5)
>>> clf.score(X_features, y)
1.0


*)

(* TEST TODO
let%expect_text "RBFSampler" =
    let rBFSampler = Sklearn.Kernel_approximation.rBFSampler in
    let sGDClassifier = Sklearn.Linear_model.sGDClassifier in
    X = [[0, 0], [1, 1], [1, 0], [0, 1]]    
    y = [0, 0, 1, 1]    
    rbf_feature = RBFSampler(gamma=1, random_state=1)    
    X_features = rbf_feature.fit_transform(X)    
    clf = SGDClassifier(max_iter=5, tol=1e-3)    
    print @@ fit clf x_features y
    [%expect {|
            SGDClassifier(max_iter=5)            
    |}]
    print @@ score clf x_features y
    [%expect {|
            1.0            
    |}]

*)



(* SkewedChi2Sampler *)
(*
>>> from sklearn.kernel_approximation import SkewedChi2Sampler
>>> from sklearn.linear_model import SGDClassifier
>>> X = [[0, 0], [1, 1], [1, 0], [0, 1]]
>>> y = [0, 0, 1, 1]
>>> chi2_feature = SkewedChi2Sampler(skewedness=.01,
...                                  n_components=10,
...                                  random_state=0)
>>> X_features = chi2_feature.fit_transform(X, y)
>>> clf = SGDClassifier(max_iter=10, tol=1e-3)
>>> clf.fit(X_features, y)
SGDClassifier(max_iter=10)
>>> clf.score(X_features, y)
1.0


*)

(* TEST TODO
let%expect_text "SkewedChi2Sampler" =
    let skewedChi2Sampler = Sklearn.Kernel_approximation.skewedChi2Sampler in
    let sGDClassifier = Sklearn.Linear_model.sGDClassifier in
    X = [[0, 0], [1, 1], [1, 0], [0, 1]]    
    y = [0, 0, 1, 1]    
    chi2_feature = SkewedChi2Sampler(skewedness=.01,n_components=10,random_state=0)    
    X_features = chi2_feature.fit_transform(X, y)    
    clf = SGDClassifier(max_iter=10, tol=1e-3)    
    print @@ fit clf x_features y
    [%expect {|
            SGDClassifier(max_iter=10)            
    |}]
    print @@ score clf x_features y
    [%expect {|
            1.0            
    |}]

*)



(* svd *)
(*
>>> from scipy import linalg
>>> m, n = 9, 6
>>> a = np.random.randn(m, n) + 1.j*np.random.randn(m, n)
>>> U, s, Vh = linalg.svd(a)
>>> U.shape,  s.shape, Vh.shape
((9, 9), (6,), (6, 6))


*)

(* TEST TODO
let%expect_text "svd" =
    let linalg = Scipy.linalg in
    m, n = 9, 6    
    a = np.random.randn(m, n) + 1.j*np.random.randn(m, n)    
    U, s, Vh = linalg.svd(a)    
    U.shape,  s.shape, Vh.shape    
    [%expect {|
            ((9, 9), (6,), (6, 6))            
    |}]

*)



(* svd *)
(*
>>> sigma = np.zeros((m, n))
>>> for i in range(min(m, n)):
...     sigma[i, i] = s[i]
>>> a1 = np.dot(U, np.dot(sigma, Vh))
>>> np.allclose(a, a1)
True


*)

(* TEST TODO
let%expect_text "svd" =
    sigma = np.zeros((m, n))    
    for i in range(min(m, n)):sigma[i, i] = s[i]    
    a1 = np.dot(U, np.dot(sigma, Vh))    
    print @@ allclose np a a1
    [%expect {|
            True            
    |}]

*)



(* svd *)
(*
>>> U, s, Vh = linalg.svd(a, full_matrices=False)
>>> U.shape, s.shape, Vh.shape
((9, 6), (6,), (6, 6))
>>> S = np.diag(s)
>>> np.allclose(a, np.dot(U, np.dot(S, Vh)))
True


*)

(* TEST TODO
let%expect_text "svd" =
    U, s, Vh = linalg.svd(a, full_matrices=False)    
    U.shape, s.shape, Vh.shape    
    [%expect {|
            ((9, 6), (6,), (6, 6))            
    |}]
    S = np.diag(s)    
    np.allclose(a, np.dot(U, np.dot(S, Vh)))    
    [%expect {|
            True            
    |}]

*)



