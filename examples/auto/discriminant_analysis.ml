(* QuadraticDiscriminantAnalysis *)
(*
>>> from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
>>> import numpy as np
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
>>> y = np.array([1, 1, 1, 2, 2, 2])
>>> clf = QuadraticDiscriminantAnalysis()
>>> clf.fit(X, y)
QuadraticDiscriminantAnalysis()
>>> print(clf.predict([[-0.8, -1]]))
[1]


*)

(* TEST TODO
let%expect_text "QuadraticDiscriminantAnalysis" =
    let quadraticDiscriminantAnalysis = Sklearn.Discriminant_analysis.quadraticDiscriminantAnalysis in
    import numpy as np    
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])    
    y = np.array([1, 1, 1, 2, 2, 2])    
    clf = QuadraticDiscriminantAnalysis()    
    print @@ fit clf x y
    [%expect {|
            QuadraticDiscriminantAnalysis()            
    |}]
    print(clf.predict([[-0.8, -1]]))    
    [%expect {|
            [1]            
    |}]

*)



(* StandardScaler *)
(*
>>> from sklearn.preprocessing import StandardScaler
>>> data = [[0, 0], [0, 0], [1, 1], [1, 1]]
>>> scaler = StandardScaler()
>>> print(scaler.fit(data))
StandardScaler()
>>> print(scaler.mean_)
[0.5 0.5]
>>> print(scaler.transform(data))
[[-1. -1.]
 [-1. -1.]
 [ 1.  1.]
 [ 1.  1.]]
>>> print(scaler.transform([[2, 2]]))
[[3. 3.]]


*)

(* TEST TODO
let%expect_text "StandardScaler" =
    let standardScaler = Sklearn.Preprocessing.standardScaler in
    data = [[0, 0], [0, 0], [1, 1], [1, 1]]    
    scaler = StandardScaler()    
    print(scaler.fit(data))    
    [%expect {|
            StandardScaler()            
    |}]
    print(scaler.mean_)    
    [%expect {|
            [0.5 0.5]            
    |}]
    print(scaler.transform(data))    
    [%expect {|
            [[-1. -1.]            
             [-1. -1.]            
             [ 1.  1.]            
             [ 1.  1.]]            
    |}]
    print(scaler.transform([[2, 2]]))    
    [%expect {|
            [[3. 3.]]            
    |}]

*)



