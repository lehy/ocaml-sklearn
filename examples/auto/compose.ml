module Np = Np.Numpy

let print f x = Format.printf "%a" f x

let print_py x = Format.printf "%s" (Py.Object.to_string x)

let print_ndarray = Np.Obj.print

let print_float = Format.printf "%g\n"

let print_string = Format.printf "%s\n"

let print_int = Format.printf "%d\n"

let matrixi = Np.Ndarray.matrixi

let matrixf = Np.Ndarray.matrixf

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

(* Enum([String(), Int(), List(String()), List(Int()), Slice(), Arr(), Callable()]) *)
let%expect_test "ColumnTransformer" =
  let open Sklearn.Compose in
  let ct = ColumnTransformer.create
      ~transformers:["norm1", Sklearn.Preprocessing.Normalizer.create ~norm:`L1 (), `Is [0;1];
                     "norm2", Sklearn.Preprocessing.Normalizer.create ~norm:`L1 (), Np.slice ~i:2 ~j:4 ()]
      ()
  in
  let x = Np.Ndarray.matrixf [|[|0.; 1.; 2.; 2.|]; [|1.; 1.; 0.; 1.|]|] in
  (* Normalizer scales each row of x to unit norm. A separate scaling *)
  (* is applied for the two first and two last elements of each *)
  (* row independently. *)
  print_ndarray @@ ColumnTransformer.fit_transform ~x ct;
  [%expect {|
      [[0.  1.  0.5 0.5]
       [0.5 0.5 0.  1. ]]
  |}]


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

let%expect_test "TransformedTargetRegressor" =
  let open Sklearn.Compose in
  let tt = TransformedTargetRegressor.create
      ~regressor:Sklearn.Linear_model.LinearRegression.(create ())
      ~func:(Np.get_py "log") ~inverse_func:(Np.get_py "exp")
      ()
  in
  [%expect {||}];
  let x = Np.(arange (`I 4) |> reshape ~newshape:[-1; 1]) in
  [%expect {||}];
  let y = Np.(exp ((float 2.) * x) |> ravel) in
  [%expect {||}];
  print TransformedTargetRegressor.pp @@ TransformedTargetRegressor.fit ~x ~y tt;
  [%expect {|
      TransformedTargetRegressor(check_inverse=True, func=<ufunc 'log'>,
                                 inverse_func=<ufunc 'exp'>,
                                 regressor=LinearRegression(copy_X=True,
                                                            fit_intercept=True,
                                                            n_jobs=None,
                                                            normalize=False),
                                 transformer=None)
  |}];
  print_float @@ TransformedTargetRegressor.score ~x ~y tt;
  [%expect {|
      1
  |}];
  print_ndarray @@ (TransformedTargetRegressor.regressor_ tt
                    |> fun r -> Sklearn.Linear_model.LinearRegression.(of_pyobject r |> coef_));
  [%expect {|

      [2.]
  |}]



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

(*  TODO, needs a pandas wrapper  *)
(* let%expect_test "make_column_selector" =
 *   let open Sklearn.Compose in
 *   let x = .dataFrame {'city': ['London' 'London' 'Paris' 'Sallisaw'] 'rating': (vectori [|5; 3; 4; 5|])} pd # doctest: +SKIP in
 *   let ct = make_column_transformer((StandardScaler(),make_column_selector ~dtype_include:np.number ()), # rating(OneHotEncoder(),make_column_selector ~dtype_include:object ())) # city in
 *   print_ndarray @@ .fit_transform ~x ct # doctest: +SKIP;
 *   [%expect {|
 *       array([[ 0.90453403,  1.        ,  0.        ,  0.        ],
 *              [-1.50755672,  1.        ,  0.        ,  0.        ],
 *              [-0.30151134,  0.        ,  1.        ,  0.        ],
 *   |}] *)


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

let%expect_test "make_column_transformer" =
  let open Sklearn.Compose in
  print ColumnTransformer.pp @@ make_column_transformer
    [Sklearn.Preprocessing.StandardScaler.(create () |> as_transformer), `Ss ["numerical_column"];
     Sklearn.Preprocessing.OneHotEncoder.(create () |> as_transformer), `Ss ["categorical_column"]];
  [%expect {|
      ColumnTransformer(n_jobs=None, remainder='drop', sparse_threshold=0.3,
                        transformer_weights=None,
                        transformers=[('standardscaler',
                                       StandardScaler(copy=True, with_mean=True,
                                                      with_std=True),
                                       ['numerical_column']),
                                      ('onehotencoder',
                                       OneHotEncoder(categories='auto', drop=None,
                                                     dtype=<class 'numpy.float64'>,
                                                     handle_unknown='error',
                                                     sparse=True),
                                       ['categorical_column'])],
                        verbose=False)
  |}]
