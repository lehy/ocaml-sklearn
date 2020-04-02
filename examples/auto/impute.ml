(* SimpleImputer *)
(*
>>> import numpy as np
>>> from sklearn.impute import SimpleImputer
>>> imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
>>> imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
SimpleImputer()
>>> X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
>>> print(imp_mean.transform(X))
[[ 7.   2.   3. ]
 [ 4.   3.5  6. ]
 [10.   3.5  9. ]]


*)

(* TEST TODO
let%expect_text "SimpleImputer" =
    import numpy as np    
    let simpleImputer = Sklearn.Impute.simpleImputer in
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')    
    print @@ fit imp_mean [[7 2 3] [4 np.nan 6] [10 5 9]]
    [%expect {|
            SimpleImputer()            
    |}]
    X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]    
    print(imp_mean.transform(X))    
    [%expect {|
            [[ 7.   2.   3. ]            
             [ 4.   3.5  6. ]            
             [10.   3.5  9. ]]            
    |}]

*)



