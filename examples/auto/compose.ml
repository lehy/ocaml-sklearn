(* TransformedTargetRegressor *)
(*
>>> import numpy as np
>>> from sklearn.linear_model import LinearRegression
>>> from sklearn.compose import TransformedTargetRegressor
>>> tt = TransformedTargetRegressor(regressor=LinearRegression(),
...                                 func=np.log, inverse_func=np.exp)
>>> X = np.arange(4).reshape(-1, 1)
>>> y = np.exp(2 * X).ravel()
>>> tt.fit(X, y)
TransformedTargetRegressor(...)
>>> tt.score(X, y)
1.0
>>> tt.regressor_.coef_
array([2.])


*)

(* TEST TODO
let%expect_text "TransformedTargetRegressor" =
    import numpy as np    
    let linearRegression = Sklearn.Linear_model.linearRegression in
    let transformedTargetRegressor = Sklearn.Compose.transformedTargetRegressor in
    tt = TransformedTargetRegressor(regressor=LinearRegression(),func=np.log, inverse_func=np.exp)    
    X = np.arange(4).reshape(-1, 1)    
    y = np.exp(2 * X).ravel()    
    print @@ fit tt x y
    [%expect {|
            TransformedTargetRegressor(...)            
    |}]
    print @@ score tt x y
    [%expect {|
            1.0            
    |}]
    tt.regressor_.coef_    
    [%expect {|
            array([2.])            
    |}]

*)



