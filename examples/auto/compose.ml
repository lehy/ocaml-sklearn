(* ColumnTransformer *)
(*
>>> import numpy as np
>>> from sklearn.compose import ColumnTransformer
>>> from sklearn.preprocessing import Normalizer
>>> ct = ColumnTransformer(
...     [("norm1", Normalizer(norm='l1'), [0, 1]),
...      ("norm2", Normalizer(norm='l1'), slice(2, 4))])
>>> X = np.array([[0., 1., 2., 2.],
...               [1., 1., 0., 1.]])
>>> # Normalizer scales each row of X to unit norm. A separate scaling
>>> # is applied for the two first and two last elements of each
>>> # row independently.
>>> ct.fit_transform(X)
array([[0. , 1. , 0.5, 0.5],

*)

(* TEST TODO
let%expect_test "ColumnTransformer" =
  let open Sklearn.Compose in
  let ct = ColumnTransformer([("norm1", Normalizer(norm='l1'), (vectori [|0; 1|])),("norm2", Normalizer(norm='l1'), slice ~2 4 ())]) in  
  let x = .array [[0. 1. 2. 2.] [1. 1. 0. 1.]] np in  
  print_ndarray @@ # Normalizer scales each row of x to unit norm. A separate scaling;  
  print_ndarray @@ # is applied for the two first and two last elements of each;  
  print_ndarray @@ # row independently.;  
  print_ndarray @@ ColumnTransformer.fit_transform ~x ct;  
  [%expect {|
      array([[0. , 1. , 0.5, 0.5],      
  |}]

*)



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
let%expect_test "TransformedTargetRegressor" =
  let open Sklearn.Compose in
  let tt = TransformedTargetRegressor(regressor=LinearRegression(),func=np.log, inverse_func=np.exp) in  
  let x = .arange 4).reshape(-1 ~1 np in  
  let y = .exp 2 * x).ravel( np in  
  print TransformedTargetRegressor.pp @@ TransformedTargetRegressor.fit ~x y tt;  
  [%expect {|
      TransformedTargetRegressor(...)      
  |}]
  print_ndarray @@ TransformedTargetRegressor.score ~x y tt;  
  [%expect {|
      1.0      
  |}]
  print_ndarray @@ tt..coef_ regressor_;  
  [%expect {|
      array([2.])      
  |}]

*)



(* make_column_selector *)
(*
>>> from sklearn.preprocessing import StandardScaler, OneHotEncoder
>>> from sklearn.compose import make_column_transformer
>>> from sklearn.compose import make_column_selector
>>> import pandas as pd  # doctest: +SKIP
>>> X = pd.DataFrame({'city': ['London', 'London', 'Paris', 'Sallisaw'],
...                   'rating': [5, 3, 4, 5]})  # doctest: +SKIP
>>> ct = make_column_transformer(
...       (StandardScaler(),
...        make_column_selector(dtype_include=np.number)),  # rating
...       (OneHotEncoder(),
...        make_column_selector(dtype_include=object)))  # city
>>> ct.fit_transform(X)  # doctest: +SKIP
array([[ 0.90453403,  1.        ,  0.        ,  0.        ],
       [-1.50755672,  1.        ,  0.        ,  0.        ],
       [-0.30151134,  0.        ,  1.        ,  0.        ],

*)

(* TEST TODO
let%expect_test "make_column_selector" =
  let open Sklearn.Compose in
  let x = .dataFrame {'city': ['London' 'London' 'Paris' 'Sallisaw'] 'rating': (vectori [|5; 3; 4; 5|])} pd # doctest: +SKIP in  
  let ct = make_column_transformer((StandardScaler(),make_column_selector ~dtype_include:np.number ()), # rating(OneHotEncoder(),make_column_selector ~dtype_include:object ())) # city in  
  print_ndarray @@ .fit_transform ~x ct # doctest: +SKIP;  
  [%expect {|
      array([[ 0.90453403,  1.        ,  0.        ,  0.        ],      
             [-1.50755672,  1.        ,  0.        ,  0.        ],      
             [-0.30151134,  0.        ,  1.        ,  0.        ],      
  |}]

*)



(* make_column_transformer *)
(*
>>> from sklearn.preprocessing import StandardScaler, OneHotEncoder
>>> from sklearn.compose import make_column_transformer
>>> make_column_transformer(
...     (StandardScaler(), ['numerical_column']),
...     (OneHotEncoder(), ['categorical_column']))
ColumnTransformer(transformers=[('standardscaler', StandardScaler(...),
                                 ['numerical_column']),
                                ('onehotencoder', OneHotEncoder(...),

*)

(* TEST TODO
let%expect_test "make_column_transformer" =
  let open Sklearn.Compose in
  print_ndarray @@ make_column_transformer((StandardScaler(), ['numerical_column']),(OneHotEncoder(), ['categorical_column']));  
  [%expect {|
      ColumnTransformer(transformers=[('standardscaler', StandardScaler(...),      
                                       ['numerical_column']),      
                                      ('onehotencoder', OneHotEncoder(...),      
  |}]

*)



