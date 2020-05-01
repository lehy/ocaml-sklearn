(** Get an attribute of this module as a Py.Object.t. This is useful to pass a Python function to another function. *)
val get_py : string -> Py.Object.t

val clear_data_home : ?data_home:string -> unit -> Py.Object.t
(**
Delete all the content of the data home cache.

Parameters
----------
data_home : str | None
    The path to scikit-learn data dir.
*)

val dump_svmlight_file : ?zero_based:bool -> ?comment:string -> ?query_id:Arr.t -> ?multilabel:bool -> x:Arr.t -> y:[`Arr of Arr.t | `PyObject of Py.Object.t] -> f:[`S of string | `File_like_in_binary_mode of Py.Object.t] -> unit -> Py.Object.t
(**
Dump the dataset in svmlight / libsvm file format.

This format is a text-based format, with one sample per line. It does
not store zero valued features hence is suitable for sparse dataset.

The first element of each line can be used to store a target variable
to predict.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

y : {array-like, sparse matrix}, shape = [n_samples (, n_labels)]
    Target values. Class labels must be an
    integer or float, or array-like objects of integer or float for
    multilabel classifications.

f : string or file-like in binary mode
    If string, specifies the path that will contain the data.
    If file-like, data will be written to f. f should be opened in binary
    mode.

zero_based : boolean, optional
    Whether column indices should be written zero-based (True) or one-based
    (False).

comment : string, optional
    Comment to insert at the top of the file. This should be either a
    Unicode string, which will be encoded as UTF-8, or an ASCII byte
    string.
    If a comment is given, then it will be preceded by one that identifies
    the file as having been dumped by scikit-learn. Note that not all
    tools grok comments in SVMlight files.

query_id : array-like of shape (n_samples,)
    Array containing pairwise preference constraints (qid in svmlight
    format).

multilabel : boolean, optional
    Samples may have several labels each (see
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html)

    .. versionadded:: 0.17
       parameter *multilabel* to support multilabel datasets.
*)

val fetch_20newsgroups : ?data_home:Py.Object.t -> ?subset:[`Train | `Test | `All] -> ?categories:[`Collection_of_string of Py.Object.t | `S of string] -> ?shuffle:bool -> ?random_state:int -> ?remove:Py.Object.t -> ?download_if_missing:Py.Object.t -> unit -> < data: Arr.t; target: Arr.t; filenames: string list; descr: Py.Object.t; target_names: Py.Object.t >
(**
Load the filenames and data from the 20 newsgroups dataset (classification).

Download it if necessary.

=================   ==========
Classes                     20
Samples total            18846
Dimensionality               1
Features                  text
=================   ==========

Read more in the :ref:`User Guide <20newsgroups_dataset>`.

Parameters
----------
data_home : optional, default: None
    Specify a download and cache folder for the datasets. If None,
    all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

subset : 'train' or 'test', 'all', optional
    Select the dataset to load: 'train' for the training set, 'test'
    for the test set, 'all' for both, with shuffled ordering.

categories : None or collection of string or unicode
    If None (default), load all the categories.
    If not None, list of category names to load (other categories
    ignored).

shuffle : bool, optional
    Whether or not to shuffle the data: might be important for models that
    make the assumption that the samples are independent and identically
    distributed (i.i.d.), such as stochastic gradient descent.

random_state : int, RandomState instance or None (default)
    Determines random number generation for dataset shuffling. Pass an int
    for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

remove : tuple
    May contain any subset of ('headers', 'footers', 'quotes'). Each of
    these are kinds of text that will be detected and removed from the
    newsgroup posts, preventing classifiers from overfitting on
    metadata.

    'headers' removes newsgroup headers, 'footers' removes blocks at the
    ends of posts that look like signatures, and 'quotes' removes lines
    that appear to be quoting another post.

    'headers' follows an exact standard; the other filters are not always
    correct.

download_if_missing : optional, True by default
    If False, raise an IOError if the data is not locally available
    instead of trying to download the data from the source site.

return_X_y : bool, default=False.
    If True, returns `(data.data, data.target)` instead of a Bunch
    object.

    .. versionadded:: 0.22

Returns
-------
bunch : Bunch object with the following attribute:
    - data: list, length [n_samples]
    - target: array, shape [n_samples]
    - filenames: list, length [n_samples]
    - DESCR: a description of the dataset.
    - target_names: a list of categories of the returned data,
      length [n_classes]. This depends on the `categories` parameter.

(data, target) : tuple if `return_X_y=True`
    .. versionadded:: 0.22
*)

val fetch_20newsgroups_vectorized : ?subset:[`Train | `Test | `All] -> ?remove:Py.Object.t -> ?data_home:Py.Object.t -> ?download_if_missing:Py.Object.t -> ?normalize:bool -> unit -> < data: Csr_matrix.t; target: Arr.t; target_names: Py.Object.t; descr: Py.Object.t >
(**
Load the 20 newsgroups dataset and vectorize it into token counts (classification).

Download it if necessary.

This is a convenience function; the transformation is done using the
default settings for
:class:`sklearn.feature_extraction.text.CountVectorizer`. For more
advanced usage (stopword filtering, n-gram extraction, etc.), combine
fetch_20newsgroups with a custom
:class:`sklearn.feature_extraction.text.CountVectorizer`,
:class:`sklearn.feature_extraction.text.HashingVectorizer`,
:class:`sklearn.feature_extraction.text.TfidfTransformer` or
:class:`sklearn.feature_extraction.text.TfidfVectorizer`.

The resulting counts are normalized using
:func:`sklearn.preprocessing.normalize` unless normalize is set to False.

=================   ==========
Classes                     20
Samples total            18846
Dimensionality          130107
Features                  real
=================   ==========

Read more in the :ref:`User Guide <20newsgroups_dataset>`.

Parameters
----------
subset : 'train' or 'test', 'all', optional
    Select the dataset to load: 'train' for the training set, 'test'
    for the test set, 'all' for both, with shuffled ordering.

remove : tuple
    May contain any subset of ('headers', 'footers', 'quotes'). Each of
    these are kinds of text that will be detected and removed from the
    newsgroup posts, preventing classifiers from overfitting on
    metadata.

    'headers' removes newsgroup headers, 'footers' removes blocks at the
    ends of posts that look like signatures, and 'quotes' removes lines
    that appear to be quoting another post.

data_home : optional, default: None
    Specify an download and cache folder for the datasets. If None,
    all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

download_if_missing : optional, True by default
    If False, raise an IOError if the data is not locally available
    instead of trying to download the data from the source site.

return_X_y : bool, default=False
    If True, returns ``(data.data, data.target)`` instead of a Bunch
    object.

    .. versionadded:: 0.20

normalize : bool, default=True
    If True, normalizes each document's feature vector to unit norm using
    :func:`sklearn.preprocessing.normalize`.

    .. versionadded:: 0.22

Returns
-------
bunch : Bunch object with the following attribute:
    - bunch.data: sparse matrix, shape [n_samples, n_features]
    - bunch.target: array, shape [n_samples]
    - bunch.target_names: a list of categories of the returned data,
      length [n_classes].
    - bunch.DESCR: a description of the dataset.

(data, target) : tuple if ``return_X_y`` is True

    .. versionadded:: 0.20
*)

val fetch_california_housing : ?data_home:Py.Object.t -> ?download_if_missing:Py.Object.t -> unit -> < data: Arr.t; target: Arr.t; feature_names: string list; descr: string >
(**
Load the California housing dataset (regression).

==============   ==============
Samples total             20640
Dimensionality                8
Features                   real
Target           real 0.15 - 5.
==============   ==============

Read more in the :ref:`User Guide <california_housing_dataset>`.

Parameters
----------
data_home : optional, default: None
    Specify another download and cache folder for the datasets. By default
    all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

download_if_missing : optional, default=True
    If False, raise a IOError if the data is not locally available
    instead of trying to download the data from the source site.


return_X_y : boolean, default=False.
    If True, returns ``(data.data, data.target)`` instead of a Bunch
    object.

    .. versionadded:: 0.20

Returns
-------
dataset : dict-like object with the following attributes:

dataset.data : ndarray, shape [20640, 8]
    Each row corresponding to the 8 feature values in order.

dataset.target : numpy array of shape (20640,)
    Each value corresponds to the average house value in units of 100,000.

dataset.feature_names : array of length 8
    Array of ordered feature names used in the dataset.

dataset.DESCR : string
    Description of the California housing dataset.

(data, target) : tuple if ``return_X_y`` is True

    .. versionadded:: 0.20

Notes
-----

This dataset consists of 20,640 samples and 9 features.
*)

val fetch_covtype : ?data_home:string -> ?download_if_missing:bool -> ?random_state:int -> ?shuffle:bool -> unit -> < data: Arr.t; target: Arr.t; descr: string >
(**
Load the covertype dataset (classification).

Download it if necessary.

=================   ============
Classes                        7
Samples total             581012
Dimensionality                54
Features                     int
=================   ============

Read more in the :ref:`User Guide <covtype_dataset>`.

Parameters
----------
data_home : string, optional
    Specify another download and cache folder for the datasets. By default
    all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

download_if_missing : boolean, default=True
    If False, raise a IOError if the data is not locally available
    instead of trying to download the data from the source site.

random_state : int, RandomState instance or None (default)
    Determines random number generation for dataset shuffling. Pass an int
    for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

shuffle : bool, default=False
    Whether to shuffle dataset.

return_X_y : boolean, default=False.
    If True, returns ``(data.data, data.target)`` instead of a Bunch
    object.

    .. versionadded:: 0.20

Returns
-------
dataset : dict-like object with the following attributes:

dataset.data : numpy array of shape (581012, 54)
    Each row corresponds to the 54 features in the dataset.

dataset.target : numpy array of shape (581012,)
    Each value corresponds to one of the 7 forest covertypes with values
    ranging between 1 to 7.

dataset.DESCR : string
    Description of the forest covertype dataset.

(data, target) : tuple if ``return_X_y`` is True

    .. versionadded:: 0.20
*)

val fetch_kddcup99 : ?subset:[`SA | `SF | `Http | `Smtp] -> ?data_home:string -> ?shuffle:bool -> ?random_state:int -> ?percent10:bool -> ?download_if_missing:bool -> unit -> < data: Arr.t; target: Arr.t; descr: string >
(**
Load the kddcup99 dataset (classification).

Download it if necessary.

=================   ====================================
Classes                                               23
Samples total                                    4898431
Dimensionality                                        41
Features            discrete (int) or continuous (float)
=================   ====================================

Read more in the :ref:`User Guide <kddcup99_dataset>`.

.. versionadded:: 0.18

Parameters
----------
subset : None, 'SA', 'SF', 'http', 'smtp'
    To return the corresponding classical subsets of kddcup 99.
    If None, return the entire kddcup 99 dataset.

data_home : string, optional
    Specify another download and cache folder for the datasets. By default
    all scikit-learn data is stored in '~/scikit_learn_data' subfolders.
    .. versionadded:: 0.19

shuffle : bool, default=False
    Whether to shuffle dataset.

random_state : int, RandomState instance or None (default)
    Determines random number generation for dataset shuffling and for
    selection of abnormal samples if `subset='SA'`. Pass an int for
    reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

percent10 : bool, default=True
    Whether to load only 10 percent of the data.

download_if_missing : bool, default=True
    If False, raise a IOError if the data is not locally available
    instead of trying to download the data from the source site.

return_X_y : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object. See
    below for more information about the `data` and `target` object.

    .. versionadded:: 0.20

Returns
-------
data : Bunch
    Dictionary-like object, the interesting attributes are:
     - 'data', the data to learn.
     - 'target', the regression target for each sample.
     - 'DESCR', a description of the dataset.

(data, target) : tuple if ``return_X_y`` is True

    .. versionadded:: 0.20
*)

val fetch_lfw_pairs : ?subset:Py.Object.t -> ?data_home:Py.Object.t -> ?funneled:bool -> ?resize:float -> ?color:bool -> ?slice_:Py.Object.t -> ?download_if_missing:Py.Object.t -> unit -> < data: Py.Object.t; pairs: Py.Object.t; target: Py.Object.t; descr: string >
(**
Load the Labeled Faces in the Wild (LFW) pairs dataset (classification).

Download it if necessary.

=================   =======================
Classes                                5749
Samples total                         13233
Dimensionality                         5828
Features            real, between 0 and 255
=================   =======================

In the official `README.txt`_ this task is described as the
"Restricted" task.  As I am not sure as to implement the
"Unrestricted" variant correctly, I left it as unsupported for now.

  .. _`README.txt`: http://vis-www.cs.umass.edu/lfw/README.txt

The original images are 250 x 250 pixels, but the default slice and resize
arguments reduce them to 62 x 47.

Read more in the :ref:`User Guide <labeled_faces_in_the_wild_dataset>`.

Parameters
----------
subset : optional, default: 'train'
    Select the dataset to load: 'train' for the development training
    set, 'test' for the development test set, and '10_folds' for the
    official evaluation set that is meant to be used with a 10-folds
    cross validation.

data_home : optional, default: None
    Specify another download and cache folder for the datasets. By
    default all scikit-learn data is stored in '~/scikit_learn_data'
    subfolders.

funneled : boolean, optional, default: True
    Download and use the funneled variant of the dataset.

resize : float, optional, default 0.5
    Ratio used to resize the each face picture.

color : boolean, optional, default False
    Keep the 3 RGB channels instead of averaging them to a single
    gray level channel. If color is True the shape of the data has
    one more dimension than the shape with color = False.

slice_ : optional
    Provide a custom 2D slice (height, width) to extract the
    'interesting' part of the jpeg files and avoid use statistical
    correlation from the background

download_if_missing : optional, True by default
    If False, raise a IOError if the data is not locally available
    instead of trying to download the data from the source site.

Returns
-------
The data is returned as a Bunch object with the following attributes:

data : numpy array of shape (2200, 5828). Shape depends on ``subset``.
    Each row corresponds to 2 ravel'd face images of original size 62 x 47
    pixels. Changing the ``slice_``, ``resize`` or ``subset`` parameters
    will change the shape of the output.

pairs : numpy array of shape (2200, 2, 62, 47). Shape depends on ``subset``
    Each row has 2 face images corresponding to same or different person
    from the dataset containing 5749 people. Changing the ``slice_``,
    ``resize`` or ``subset`` parameters will change the shape of the
    output.

target : numpy array of shape (2200,). Shape depends on ``subset``.
    Labels associated to each pair of images. The two label values being
    different persons or the same person.

DESCR : string
    Description of the Labeled Faces in the Wild (LFW) dataset.
*)

val fetch_lfw_people : ?data_home:Py.Object.t -> ?funneled:bool -> ?resize:float -> ?min_faces_per_person:int -> ?color:bool -> ?slice_:Py.Object.t -> ?download_if_missing:Py.Object.t -> unit -> < data: Arr.t; images: Arr.t; target: Arr.t; descr: string >
(**
Load the Labeled Faces in the Wild (LFW) people dataset (classification).

Download it if necessary.

=================   =======================
Classes                                5749
Samples total                         13233
Dimensionality                         5828
Features            real, between 0 and 255
=================   =======================

Read more in the :ref:`User Guide <labeled_faces_in_the_wild_dataset>`.

Parameters
----------
data_home : optional, default: None
    Specify another download and cache folder for the datasets. By default
    all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

funneled : boolean, optional, default: True
    Download and use the funneled variant of the dataset.

resize : float, optional, default 0.5
    Ratio used to resize the each face picture.

min_faces_per_person : int, optional, default None
    The extracted dataset will only retain pictures of people that have at
    least `min_faces_per_person` different pictures.

color : boolean, optional, default False
    Keep the 3 RGB channels instead of averaging them to a single
    gray level channel. If color is True the shape of the data has
    one more dimension than the shape with color = False.

slice_ : optional
    Provide a custom 2D slice (height, width) to extract the
    'interesting' part of the jpeg files and avoid use statistical
    correlation from the background

download_if_missing : optional, True by default
    If False, raise a IOError if the data is not locally available
    instead of trying to download the data from the source site.

return_X_y : boolean, default=False.
    If True, returns ``(dataset.data, dataset.target)`` instead of a Bunch
    object. See below for more information about the `dataset.data` and
    `dataset.target` object.

    .. versionadded:: 0.20

Returns
-------
dataset : dict-like object with the following attributes:

dataset.data : numpy array of shape (13233, 2914)
    Each row corresponds to a ravelled face image of original size 62 x 47
    pixels. Changing the ``slice_`` or resize parameters will change the
    shape of the output.

dataset.images : numpy array of shape (13233, 62, 47)
    Each row is a face image corresponding to one of the 5749 people in
    the dataset. Changing the ``slice_`` or resize parameters will change
    the shape of the output.

dataset.target : numpy array of shape (13233,)
    Labels associated to each face image. Those labels range from 0-5748
    and correspond to the person IDs.

dataset.DESCR : string
    Description of the Labeled Faces in the Wild (LFW) dataset.

(data, target) : tuple if ``return_X_y`` is True

    .. versionadded:: 0.20
*)

val fetch_olivetti_faces : ?data_home:Py.Object.t -> ?shuffle:bool -> ?random_state:int -> ?download_if_missing:Py.Object.t -> unit -> < data: Py.Object.t; images: Py.Object.t; target: Py.Object.t; descr: Py.Object.t >
(**
Load the Olivetti faces data-set from AT&T (classification).

Download it if necessary.

=================   =====================
Classes                                40
Samples total                         400
Dimensionality                       4096
Features            real, between 0 and 1
=================   =====================

Read more in the :ref:`User Guide <olivetti_faces_dataset>`.

Parameters
----------
data_home : optional, default: None
    Specify another download and cache folder for the datasets. By default
    all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

shuffle : boolean, optional
    If True the order of the dataset is shuffled to avoid having
    images of the same person grouped.

random_state : int, RandomState instance or None (default=0)
    Determines random number generation for dataset shuffling. Pass an int
    for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

download_if_missing : optional, True by default
    If False, raise a IOError if the data is not locally available
    instead of trying to download the data from the source site.

return_X_y : boolean, default=False.
    If True, returns `(data, target)` instead of a `Bunch` object. See
    below for more information about the `data` and `target` object.

    .. versionadded:: 0.22

Returns
-------
bunch : Bunch object with the following attributes:
    - data: ndarray, shape (400, 4096). Each row corresponds to a ravelled
      face image of original size 64 x 64 pixels.
    - images : ndarray, shape (400, 64, 64). Each row is a face image
      corresponding to one of the 40 subjects of the dataset.
    - target : ndarray, shape (400,). Labels associated to each face image.
      Those labels are ranging from 0-39 and correspond to the
      Subject IDs.
    - DESCR : string. Description of the modified Olivetti Faces Dataset.

(data, target) : tuple if `return_X_y=True`
    .. versionadded:: 0.22
*)

val fetch_openml : ?name:string -> ?version:[`I of int | `Active] -> ?data_id:int -> ?data_home:string -> ?target_column:[`S of string | `Arr of Arr.t | `None] -> ?cache:bool -> ?as_frame:bool -> unit -> < data: Py.Object.t; target: Py.Object.t; descr: string; feature_names: string list; target_names: string list; categories: Py.Object.t; details: Dict.t; frame: Py.Object.t >
(**
Fetch dataset from openml by name or dataset id.

Datasets are uniquely identified by either an integer ID or by a
combination of name and version (i.e. there might be multiple
versions of the 'iris' dataset). Please give either name or data_id
(not both). In case a name is given, a version can also be
provided.

Read more in the :ref:`User Guide <openml>`.

.. note:: EXPERIMENTAL

    The API is experimental (particularly the return value structure),
    and might have small backward-incompatible changes in future releases.

Parameters
----------
name : str or None
    String identifier of the dataset. Note that OpenML can have multiple
    datasets with the same name.

version : integer or 'active', default='active'
    Version of the dataset. Can only be provided if also ``name`` is given.
    If 'active' the oldest version that's still active is used. Since
    there may be more than one active version of a dataset, and those
    versions may fundamentally be different from one another, setting an
    exact version is highly recommended.

data_id : int or None
    OpenML ID of the dataset. The most specific way of retrieving a
    dataset. If data_id is not given, name (and potential version) are
    used to obtain a dataset.

data_home : string or None, default None
    Specify another download and cache folder for the data sets. By default
    all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

target_column : string, list or None, default 'default-target'
    Specify the column name in the data to use as target. If
    'default-target', the standard target column a stored on the server
    is used. If ``None``, all columns are returned as data and the
    target is ``None``. If list (of strings), all columns with these names
    are returned as multi-target (Note: not all scikit-learn classifiers
    can handle all types of multi-output combinations)

cache : boolean, default=True
    Whether to cache downloaded datasets using joblib.

return_X_y : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object. See
    below for more information about the `data` and `target` objects.

as_frame : boolean, default=False
    If True, the data is a pandas DataFrame including columns with
    appropriate dtypes (numeric, string or categorical). The target is
    a pandas DataFrame or Series depending on the number of target_columns.
    The Bunch will contain a ``frame`` attribute with the target and the
    data. If ``return_X_y`` is True, then ``(data, target)`` will be pandas
    DataFrames or Series as describe above.

Returns
-------

data : Bunch
    Dictionary-like object, with attributes:

    data : np.array, scipy.sparse.csr_matrix of floats, or pandas DataFrame
        The feature matrix. Categorical features are encoded as ordinals.
    target : np.array, pandas Series or DataFrame
        The regression target or classification labels, if applicable.
        Dtype is float if numeric, and object if categorical. If
        ``as_frame`` is True, ``target`` is a pandas object.
    DESCR : str
        The full description of the dataset
    feature_names : list
        The names of the dataset columns
    target_names: list
        The names of the target columns

    .. versionadded:: 0.22

    categories : dict or None
        Maps each categorical feature name to a list of values, such
        that the value encoded as i is ith in the list. If ``as_frame``
        is True, this is None.
    details : dict
        More metadata from OpenML
    frame : pandas DataFrame
        Only present when `as_frame=True`. DataFrame with ``data`` and
        ``target``.

(data, target) : tuple if ``return_X_y`` is True

    .. note:: EXPERIMENTAL

        This interface is **experimental** and subsequent releases may
        change attributes without notice (although there should only be
        minor changes to ``data`` and ``target``).

    Missing values in the 'data' are represented as NaN's. Missing values
    in 'target' are represented as NaN's (numerical target) or None
    (categorical target)
*)

val fetch_rcv1 : ?data_home:string -> ?subset:[`Train | `Test | `All] -> ?download_if_missing:bool -> ?random_state:int -> ?shuffle:bool -> unit -> < data: Py.Object.t; target: Py.Object.t; sample_id: Py.Object.t; target_names: Py.Object.t; descr: string >
(**
Load the RCV1 multilabel dataset (classification).

Download it if necessary.

Version: RCV1-v2, vectors, full sets, topics multilabels.

=================   =====================
Classes                               103
Samples total                      804414
Dimensionality                      47236
Features            real, between 0 and 1
=================   =====================

Read more in the :ref:`User Guide <rcv1_dataset>`.

.. versionadded:: 0.17

Parameters
----------
data_home : string, optional
    Specify another download and cache folder for the datasets. By default
    all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

subset : string, 'train', 'test', or 'all', default='all'
    Select the dataset to load: 'train' for the training set
    (23149 samples), 'test' for the test set (781265 samples),
    'all' for both, with the training samples first if shuffle is False.
    This follows the official LYRL2004 chronological split.

download_if_missing : boolean, default=True
    If False, raise a IOError if the data is not locally available
    instead of trying to download the data from the source site.

random_state : int, RandomState instance or None (default)
    Determines random number generation for dataset shuffling. Pass an int
    for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

shuffle : bool, default=False
    Whether to shuffle dataset.

return_X_y : boolean, default=False.
    If True, returns ``(dataset.data, dataset.target)`` instead of a Bunch
    object. See below for more information about the `dataset.data` and
    `dataset.target` object.

    .. versionadded:: 0.20

Returns
-------
dataset : dict-like object with the following attributes:

dataset.data : scipy csr array, dtype np.float64, shape (804414, 47236)
    The array has 0.16% of non zero values.

dataset.target : scipy csr array, dtype np.uint8, shape (804414, 103)
    Each sample has a value of 1 in its categories, and 0 in others.
    The array has 3.15% of non zero values.

dataset.sample_id : numpy array, dtype np.uint32, shape (804414,)
    Identification number of each sample, as ordered in dataset.data.

dataset.target_names : numpy array, dtype object, length (103)
    Names of each target (RCV1 topics), as ordered in dataset.target.

dataset.DESCR : string
    Description of the RCV1 dataset.

(data, target) : tuple if ``return_X_y`` is True

    .. versionadded:: 0.20
*)

val fetch_species_distributions : ?data_home:Py.Object.t -> ?download_if_missing:Py.Object.t -> unit -> < coverages: Arr.t; train: Py.Object.t; test: Py.Object.t; nx: Py.Object.t; x_left_lower_corner: Py.Object.t; grid_size: float >
(**
Loader for species distribution dataset from Phillips et. al. (2006)

Read more in the :ref:`User Guide <datasets>`.

Parameters
----------
data_home : optional, default: None
    Specify another download and cache folder for the datasets. By default
    all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

download_if_missing : optional, True by default
    If False, raise a IOError if the data is not locally available
    instead of trying to download the data from the source site.

Returns
-------
The data is returned as a Bunch object with the following attributes:

coverages : array, shape = [14, 1592, 1212]
    These represent the 14 features measured at each point of the map grid.
    The latitude/longitude values for the grid are discussed below.
    Missing data is represented by the value -9999.

train : record array, shape = (1624,)
    The training points for the data.  Each point has three fields:

    - train['species'] is the species name
    - train['dd long'] is the longitude, in degrees
    - train['dd lat'] is the latitude, in degrees

test : record array, shape = (620,)
    The test points for the data.  Same format as the training data.

Nx, Ny : integers
    The number of longitudes (x) and latitudes (y) in the grid

x_left_lower_corner, y_left_lower_corner : floats
    The (x,y) position of the lower-left corner, in degrees

grid_size : float
    The spacing between points of the grid, in degrees

References
----------

* `"Maximum entropy modeling of species geographic distributions"
  <http://rob.schapire.net/papers/ecolmod.pdf>`_
  S. J. Phillips, R. P. Anderson, R. E. Schapire - Ecological Modelling,
  190:231-259, 2006.

Notes
-----

This dataset represents the geographic distribution of species.
The dataset is provided by Phillips et. al. (2006).

The two species are:

- `"Bradypus variegatus"
  <http://www.iucnredlist.org/details/3038/0>`_ ,
  the Brown-throated Sloth.

- `"Microryzomys minutus"
  <http://www.iucnredlist.org/details/13408/0>`_ ,
  also known as the Forest Small Rice Rat, a rodent that lives in Peru,
  Colombia, Ecuador, Peru, and Venezuela.

- For an example of using this dataset with scikit-learn, see
  :ref:`examples/applications/plot_species_distribution_modeling.py
  <sphx_glr_auto_examples_applications_plot_species_distribution_modeling.py>`.
*)

val get_data_home : ?data_home:string -> unit -> Py.Object.t
(**
Return the path of the scikit-learn data dir.

This folder is used by some large dataset loaders to avoid downloading the
data several times.

By default the data dir is set to a folder named 'scikit_learn_data' in the
user home folder.

Alternatively, it can be set by the 'SCIKIT_LEARN_DATA' environment
variable or programmatically by giving an explicit folder path. The '~'
symbol is expanded to the user home folder.

If the folder does not already exist, it is automatically created.

Parameters
----------
data_home : str | None
    The path to scikit-learn data dir.
*)

val load_boston : unit -> < data: Arr.t; target: Arr.t; feature_names: Arr.t; descr: string; filename: string >
(**
Load and return the boston house-prices dataset (regression).

==============   ==============
Samples total               506
Dimensionality               13
Features         real, positive
Targets           real 5. - 50.
==============   ==============

Read more in the :ref:`User Guide <boston_dataset>`.

Parameters
----------
return_X_y : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object.
    See below for more information about the `data` and `target` object.

    .. versionadded:: 0.18

Returns
-------
data : Bunch
    Dictionary-like object, the interesting attributes are:
    'data', the data to learn, 'target', the regression targets,
    'DESCR', the full description of the dataset,
    and 'filename', the physical location of boston
    csv dataset (added in version `0.20`).

(data, target) : tuple if ``return_X_y`` is True

    .. versionadded:: 0.18

Notes
-----
    .. versionchanged:: 0.20
        Fixed a wrong data point at [445, 0].

Examples
--------
>>> from sklearn.datasets import load_boston
>>> X, y = load_boston(return_X_y=True)
>>> print(X.shape)
(506, 13)
*)

val load_breast_cancer : unit -> < data: Arr.t; target: Arr.t; target_names: Arr.t; feature_names: Arr.t; descr: string; filename: string >
(**
Load and return the breast cancer wisconsin dataset (classification).

The breast cancer dataset is a classic and very easy binary classification
dataset.

=================   ==============
Classes                          2
Samples per class    212(M),357(B)
Samples total                  569
Dimensionality                  30
Features            real, positive
=================   ==============

Read more in the :ref:`User Guide <breast_cancer_dataset>`.

Parameters
----------
return_X_y : boolean, default=False
    If True, returns ``(data, target)`` instead of a Bunch object.
    See below for more information about the `data` and `target` object.

    .. versionadded:: 0.18

Returns
-------
data : Bunch
    Dictionary-like object, the interesting attributes are:
    'data', the data to learn, 'target', the classification labels,
    'target_names', the meaning of the labels, 'feature_names', the
    meaning of the features, and 'DESCR', the full description of
    the dataset, 'filename', the physical location of
    breast cancer csv dataset (added in version `0.20`).

(data, target) : tuple if ``return_X_y`` is True

    .. versionadded:: 0.18

The copy of UCI ML Breast Cancer Wisconsin (Diagnostic) dataset is
downloaded from:
https://goo.gl/U2Uwz2

Examples
--------
Let's say you are interested in the samples 10, 50, and 85, and want to
know their class name.

>>> from sklearn.datasets import load_breast_cancer
>>> data = load_breast_cancer()
>>> data.target[[10, 50, 85]]
array([0, 1, 0])
>>> list(data.target_names)
['malignant', 'benign']
*)

val load_diabetes : unit -> < data: Arr.t; target: Arr.t; data_filename: string; target_filename: string >
(**
Load and return the diabetes dataset (regression).

==============   ==================
Samples total    442
Dimensionality   10
Features         real, -.2 < x < .2
Targets          integer 25 - 346
==============   ==================

Read more in the :ref:`User Guide <diabetes_dataset>`.

Parameters
----------
return_X_y : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object.
    See below for more information about the `data` and `target` object.

    .. versionadded:: 0.18

Returns
-------
data : Bunch
    Dictionary-like object, the interesting attributes are:
    'data', the data to learn, 'target', the regression target for each
    sample, 'data_filename', the physical location
    of diabetes data csv dataset, and 'target_filename', the physical
    location of diabetes targets csv datataset (added in version `0.20`).

(data, target) : tuple if ``return_X_y`` is True

    .. versionadded:: 0.18
*)

val load_digits : ?n_class:[`I of int | `Between_0_and_10 of Py.Object.t] -> unit -> < data: Arr.t; images: Arr.t; target: Arr.t; target_names: Arr.t; descr: string >
(**
Load and return the digits dataset (classification).

Each datapoint is a 8x8 image of a digit.

=================   ==============
Classes                         10
Samples per class             ~180
Samples total                 1797
Dimensionality                  64
Features             integers 0-16
=================   ==============

Read more in the :ref:`User Guide <digits_dataset>`.

Parameters
----------
n_class : integer, between 0 and 10, optional (default=10)
    The number of classes to return.

return_X_y : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object.
    See below for more information about the `data` and `target` object.

    .. versionadded:: 0.18

Returns
-------
data : Bunch
    Dictionary-like object, the interesting attributes are:
    'data', the data to learn, 'images', the images corresponding
    to each sample, 'target', the classification labels for each
    sample, 'target_names', the meaning of the labels, and 'DESCR',
    the full description of the dataset.

(data, target) : tuple if ``return_X_y`` is True

    .. versionadded:: 0.18

This is a copy of the test set of the UCI ML hand-written digits datasets
https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

Examples
--------
To load the data and visualize the images::

    >>> from sklearn.datasets import load_digits
    >>> digits = load_digits()
    >>> print(digits.data.shape)
    (1797, 64)
    >>> import matplotlib.pyplot as plt #doctest: +SKIP
    >>> plt.gray() #doctest: +SKIP
    >>> plt.matshow(digits.images[0]) #doctest: +SKIP
    >>> plt.show() #doctest: +SKIP
*)

val load_files : ?description:string -> ?categories:Py.Object.t -> ?load_content:bool -> ?shuffle:bool -> ?encoding:[`S of string | `None_ of Py.Object.t] -> ?decode_error:[`Strict | `Ignore | `Replace] -> ?random_state:int -> container_path:string -> unit -> < filenames: Arr.t; target: Arr.t; target_names: Arr.t; descr: string >
(**
Load text files with categories as subfolder names.

Individual samples are assumed to be files stored a two levels folder
structure such as the following:

    container_folder/
        category_1_folder/
            file_1.txt
            file_2.txt
            ...
            file_42.txt
        category_2_folder/
            file_43.txt
            file_44.txt
            ...

The folder names are used as supervised signal label names. The individual
file names are not important.

This function does not try to extract features into a numpy array or scipy
sparse matrix. In addition, if load_content is false it does not try to
load the files in memory.

To use text files in a scikit-learn classification or clustering algorithm,
you will need to use the :mod`~sklearn.feature_extraction.text` module to
build a feature extraction transformer that suits your problem.

If you set load_content=True, you should also specify the encoding of the
text using the 'encoding' parameter. For many modern text files, 'utf-8'
will be the correct encoding. If you leave encoding equal to None, then the
content will be made of bytes instead of Unicode, and you will not be able
to use most functions in :mod:`~sklearn.feature_extraction.text`.

Similar feature extractors should be built for other kind of unstructured
data input such as images, audio, video, ...

Read more in the :ref:`User Guide <datasets>`.

Parameters
----------
container_path : string or unicode
    Path to the main folder holding one subfolder per category

description : string or unicode, optional (default=None)
    A paragraph describing the characteristic of the dataset: its source,
    reference, etc.

categories : A collection of strings or None, optional (default=None)
    If None (default), load all the categories. If not None, list of
    category names to load (other categories ignored).

load_content : boolean, optional (default=True)
    Whether to load or not the content of the different files. If true a
    'data' attribute containing the text information is present in the data
    structure returned. If not, a filenames attribute gives the path to the
    files.

shuffle : bool, optional (default=True)
    Whether or not to shuffle the data: might be important for models that
    make the assumption that the samples are independent and identically
    distributed (i.i.d.), such as stochastic gradient descent.

encoding : string or None (default is None)
    If None, do not try to decode the content of the files (e.g. for images
    or other non-text content). If not None, encoding to use to decode text
    files to Unicode if load_content is True.

decode_error : {'strict', 'ignore', 'replace'}, optional
    Instruction on what to do if a byte sequence is given to analyze that
    contains characters not of the given `encoding`. Passed as keyword
    argument 'errors' to bytes.decode.

random_state : int, RandomState instance or None (default=0)
    Determines random number generation for dataset shuffling. Pass an int
    for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

Returns
-------
data : Bunch
    Dictionary-like object, the interesting attributes are: either
    data, the raw text data to learn, or 'filenames', the files
    holding it, 'target', the classification labels (integer index),
    'target_names', the meaning of the labels, and 'DESCR', the full
    description of the dataset.
*)

val load_iris : unit -> < data: Arr.t; target: Arr.t; target_names: Arr.t; feature_names: Arr.t; descr: string; filename: string >
(**
Load and return the iris dataset (classification).

The iris dataset is a classic and very easy multi-class classification
dataset.

=================   ==============
Classes                          3
Samples per class               50
Samples total                  150
Dimensionality                   4
Features            real, positive
=================   ==============

Read more in the :ref:`User Guide <iris_dataset>`.

Parameters
----------
return_X_y : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object. See
    below for more information about the `data` and `target` object.

    .. versionadded:: 0.18

Returns
-------
data : Bunch
    Dictionary-like object, the interesting attributes are:
    'data', the data to learn, 'target', the classification labels,
    'target_names', the meaning of the labels, 'feature_names', the
    meaning of the features, 'DESCR', the full description of
    the dataset, 'filename', the physical location of
    iris csv dataset (added in version `0.20`).

(data, target) : tuple if ``return_X_y`` is True

    .. versionadded:: 0.18

Notes
-----
    .. versionchanged:: 0.20
        Fixed two wrong data points according to Fisher's paper.
        The new version is the same as in R, but not as in the UCI
        Machine Learning Repository.

Examples
--------
Let's say you are interested in the samples 10, 25, and 50, and want to
know their class name.

>>> from sklearn.datasets import load_iris
>>> data = load_iris()
>>> data.target[[10, 25, 50]]
array([0, 0, 1])
>>> list(data.target_names)
['setosa', 'versicolor', 'virginica']
*)

val load_linnerud : unit -> < data: Arr.t; target: Arr.t; feature_names: Arr.t; target_names: Arr.t; data_filename: string; target_filename: string >
(**
Load and return the linnerud dataset (multivariate regression).

==============   ============================
Samples total    20
Dimensionality   3 (for both data and target)
Features         integer
Targets          integer
==============   ============================

Read more in the :ref:`User Guide <linnerrud_dataset>`.

Parameters
----------
return_X_y : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object.
    See below for more information about the `data` and `target` object.

    .. versionadded:: 0.18

Returns
-------
data : Bunch
    Dictionary-like object, the interesting attributes are: 'data' and
    'target', the two multivariate datasets, with 'data' corresponding to
    the exercise and 'target' corresponding to the physiological
    measurements, as well as 'feature_names' and 'target_names'.
    In addition, you will also have access to 'data_filename',
    the physical location of linnerud data csv dataset, and
    'target_filename', the physical location of
    linnerud targets csv datataset (added in version `0.20`).

(data, target) : tuple if ``return_X_y`` is True

    .. versionadded:: 0.18
*)

val load_sample_image : image_name:Py.Object.t -> unit -> < img: Py.Object.t >
(**
Load the numpy array of a single sample image

Read more in the :ref:`User Guide <sample_images>`.

Parameters
----------
image_name : {`china.jpg`, `flower.jpg`}
    The name of the sample image loaded

Returns
-------
img : 3D array
    The image as a numpy array: height x width x color

Examples
--------

>>> from sklearn.datasets import load_sample_image
>>> china = load_sample_image('china.jpg')   # doctest: +SKIP
>>> china.dtype                              # doctest: +SKIP
dtype('uint8')
>>> china.shape                              # doctest: +SKIP
(427, 640, 3)
>>> flower = load_sample_image('flower.jpg') # doctest: +SKIP
>>> flower.dtype                             # doctest: +SKIP
dtype('uint8')
>>> flower.shape                             # doctest: +SKIP
(427, 640, 3)
*)

val load_sample_images : unit -> < data: Py.Object.t >
(**
Load sample images for image manipulation.

Loads both, ``china`` and ``flower``.

Read more in the :ref:`User Guide <sample_images>`.

Returns
-------
data : Bunch
    Dictionary-like object with the following attributes : 'images', the
    two sample images, 'filenames', the file names for the images, and
    'DESCR' the full description of the dataset.

Examples
--------
To load the data and visualize the images:

>>> from sklearn.datasets import load_sample_images
>>> dataset = load_sample_images()     #doctest: +SKIP
>>> len(dataset.images)                #doctest: +SKIP
2
>>> first_img_data = dataset.images[0] #doctest: +SKIP
>>> first_img_data.shape               #doctest: +SKIP
(427, 640, 3)
>>> first_img_data.dtype               #doctest: +SKIP
dtype('uint8')
*)

val load_svmlight_file : ?n_features:int -> ?dtype:Py.Object.t -> ?multilabel:bool -> ?zero_based:[`Bool of bool | `Auto] -> ?query_id:bool -> ?offset:int -> ?length:int -> f:[`S of string | `File_like of Py.Object.t | `I of int] -> unit -> < x: Py.Object.t; y: Py.Object.t; query_id: Arr.t >
(**
Load datasets in the svmlight / libsvm format into sparse CSR matrix

This format is a text-based format, with one sample per line. It does
not store zero valued features hence is suitable for sparse dataset.

The first element of each line can be used to store a target variable
to predict.

This format is used as the default format for both svmlight and the
libsvm command line programs.

Parsing a text based source can be expensive. When working on
repeatedly on the same dataset, it is recommended to wrap this
loader with joblib.Memory.cache to store a memmapped backup of the
CSR results of the first call and benefit from the near instantaneous
loading of memmapped structures for the subsequent calls.

In case the file contains a pairwise preference constraint (known
as "qid" in the svmlight format) these are ignored unless the
query_id parameter is set to True. These pairwise preference
constraints can be used to constraint the combination of samples
when using pairwise loss functions (as is the case in some
learning to rank problems) so that only pairs with the same
query_id value are considered.

This implementation is written in Cython and is reasonably fast.
However, a faster API-compatible loader is also available at:

  https://github.com/mblondel/svmlight-loader

Parameters
----------
f : {str, file-like, int}
    (Path to) a file to load. If a path ends in ".gz" or ".bz2", it will
    be uncompressed on the fly. If an integer is passed, it is assumed to
    be a file descriptor. A file-like or file descriptor will not be closed
    by this function. A file-like object must be opened in binary mode.

n_features : int or None
    The number of features to use. If None, it will be inferred. This
    argument is useful to load several files that are subsets of a
    bigger sliced dataset: each subset might not have examples of
    every feature, hence the inferred shape might vary from one
    slice to another.
    n_features is only required if ``offset`` or ``length`` are passed a
    non-default value.

dtype : numpy data type, default np.float64
    Data type of dataset to be loaded. This will be the data type of the
    output numpy arrays ``X`` and ``y``.

multilabel : boolean, optional, default False
    Samples may have several labels each (see
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html)

zero_based : boolean or "auto", optional, default "auto"
    Whether column indices in f are zero-based (True) or one-based
    (False). If column indices are one-based, they are transformed to
    zero-based to match Python/NumPy conventions.
    If set to "auto", a heuristic check is applied to determine this from
    the file contents. Both kinds of files occur "in the wild", but they
    are unfortunately not self-identifying. Using "auto" or True should
    always be safe when no ``offset`` or ``length`` is passed.
    If ``offset`` or ``length`` are passed, the "auto" mode falls back
    to ``zero_based=True`` to avoid having the heuristic check yield
    inconsistent results on different segments of the file.

query_id : boolean, default False
    If True, will return the query_id array for each file.

offset : integer, optional, default 0
    Ignore the offset first bytes by seeking forward, then
    discarding the following bytes up until the next new line
    character.

length : integer, optional, default -1
    If strictly positive, stop reading any new line of data once the
    position in the file has reached the (offset + length) bytes threshold.

Returns
-------
X : scipy.sparse matrix of shape (n_samples, n_features)

y : ndarray of shape (n_samples,), or, in the multilabel a list of
    tuples of length n_samples.

query_id : array of shape (n_samples,)
   query_id for each sample. Only returned when query_id is set to
   True.

See also
--------
load_svmlight_files: similar function for loading multiple files in this
                     format, enforcing the same number of features/columns
                     on all of them.

Examples
--------
To use joblib.Memory to cache the svmlight file::

    from joblib import Memory
    from .datasets import load_svmlight_file
    mem = Memory("./mycache")

    @mem.cache
    def get_data():
        data = load_svmlight_file("mysvmlightfile")
        return data[0], data[1]

    X, y = get_data()
*)

val load_svmlight_files : ?n_features:int -> ?dtype:Py.Object.t -> ?multilabel:bool -> ?zero_based:[`Bool of bool | `Auto] -> ?query_id:bool -> ?offset:int -> ?length:int -> files:Py.Object.t -> unit -> Py.Object.t
(**
Load dataset from multiple files in SVMlight format

This function is equivalent to mapping load_svmlight_file over a list of
files, except that the results are concatenated into a single, flat list
and the samples vectors are constrained to all have the same number of
features.

In case the file contains a pairwise preference constraint (known
as "qid" in the svmlight format) these are ignored unless the
query_id parameter is set to True. These pairwise preference
constraints can be used to constraint the combination of samples
when using pairwise loss functions (as is the case in some
learning to rank problems) so that only pairs with the same
query_id value are considered.

Parameters
----------
files : iterable over {str, file-like, int}
    (Paths of) files to load. If a path ends in ".gz" or ".bz2", it will
    be uncompressed on the fly. If an integer is passed, it is assumed to
    be a file descriptor. File-likes and file descriptors will not be
    closed by this function. File-like objects must be opened in binary
    mode.

n_features : int or None
    The number of features to use. If None, it will be inferred from the
    maximum column index occurring in any of the files.

    This can be set to a higher value than the actual number of features
    in any of the input files, but setting it to a lower value will cause
    an exception to be raised.

dtype : numpy data type, default np.float64
    Data type of dataset to be loaded. This will be the data type of the
    output numpy arrays ``X`` and ``y``.

multilabel : boolean, optional
    Samples may have several labels each (see
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html)

zero_based : boolean or "auto", optional
    Whether column indices in f are zero-based (True) or one-based
    (False). If column indices are one-based, they are transformed to
    zero-based to match Python/NumPy conventions.
    If set to "auto", a heuristic check is applied to determine this from
    the file contents. Both kinds of files occur "in the wild", but they
    are unfortunately not self-identifying. Using "auto" or True should
    always be safe when no offset or length is passed.
    If offset or length are passed, the "auto" mode falls back
    to zero_based=True to avoid having the heuristic check yield
    inconsistent results on different segments of the file.

query_id : boolean, defaults to False
    If True, will return the query_id array for each file.

offset : integer, optional, default 0
    Ignore the offset first bytes by seeking forward, then
    discarding the following bytes up until the next new line
    character.

length : integer, optional, default -1
    If strictly positive, stop reading any new line of data once the
    position in the file has reached the (offset + length) bytes threshold.

Returns
-------
[X1, y1, ..., Xn, yn]
where each (Xi, yi) pair is the result from load_svmlight_file(files[i]).

If query_id is set to True, this will return instead [X1, y1, q1,
..., Xn, yn, qn] where (Xi, yi, qi) is the result from
load_svmlight_file(files[i])

Notes
-----
When fitting a model to a matrix X_train and evaluating it against a
matrix X_test, it is essential that X_train and X_test have the same
number of features (X_train.shape[1] == X_test.shape[1]). This may not
be the case if you load the files individually with load_svmlight_file.

See also
--------
load_svmlight_file
*)

val load_wine : unit -> < data: Arr.t; target: Arr.t; target_names: Arr.t; feature_names: Arr.t; descr: string >
(**
Load and return the wine dataset (classification).

.. versionadded:: 0.18

The wine dataset is a classic and very easy multi-class classification
dataset.

=================   ==============
Classes                          3
Samples per class        [59,71,48]
Samples total                  178
Dimensionality                  13
Features            real, positive
=================   ==============

Read more in the :ref:`User Guide <wine_dataset>`.

Parameters
----------
return_X_y : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object.
    See below for more information about the `data` and `target` object.

Returns
-------
data : Bunch
    Dictionary-like object, the interesting attributes are: 'data', the
    data to learn, 'target', the classification labels, 'target_names', the
    meaning of the labels, 'feature_names', the meaning of the features,
    and 'DESCR', the full description of the dataset.

(data, target) : tuple if ``return_X_y`` is True

The copy of UCI ML Wine Data Set dataset is downloaded and modified to fit
standard format from:
https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data

Examples
--------
Let's say you are interested in the samples 10, 80, and 140, and want to
know their class name.

>>> from sklearn.datasets import load_wine
>>> data = load_wine()
>>> data.target[[10, 80, 140]]
array([0, 1, 2])
>>> list(data.target_names)
['class_0', 'class_1', 'class_2']
*)

val make_biclusters : ?noise:float -> ?minval:int -> ?maxval:int -> ?shuffle:bool -> ?random_state:int -> shape:int list -> n_clusters:int -> unit -> (Py.Object.t * Py.Object.t * Py.Object.t)
(**
Generate an array with constant block diagonal structure for
biclustering.

Read more in the :ref:`User Guide <sample_generators>`.

Parameters
----------
shape : iterable (n_rows, n_cols)
    The shape of the result.

n_clusters : integer
    The number of biclusters.

noise : float, optional (default=0.0)
    The standard deviation of the gaussian noise.

minval : int, optional (default=10)
    Minimum value of a bicluster.

maxval : int, optional (default=100)
    Maximum value of a bicluster.

shuffle : boolean, optional (default=True)
    Shuffle the samples.

random_state : int, RandomState instance or None (default)
    Determines random number generation for dataset creation. Pass an int
    for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

Returns
-------
X : array of shape `shape`
    The generated array.

rows : array of shape (n_clusters, X.shape[0],)
    The indicators for cluster membership of each row.

cols : array of shape (n_clusters, X.shape[1],)
    The indicators for cluster membership of each column.

References
----------

.. [1] Dhillon, I. S. (2001, August). Co-clustering documents and
    words using bipartite spectral graph partitioning. In Proceedings
    of the seventh ACM SIGKDD international conference on Knowledge
    discovery and data mining (pp. 269-274). ACM.

See also
--------
make_checkerboard
*)

val make_blobs : ?n_samples:[`I of int | `Arr of Arr.t] -> ?n_features:int -> ?centers:[`I of int | `Arr of Arr.t] -> ?cluster_std:[`F of float | `Sequence_of_floats of Py.Object.t] -> ?center_box:Py.Object.t -> ?shuffle:bool -> ?random_state:int -> unit -> (Arr.t * Arr.t)
(**
Generate isotropic Gaussian blobs for clustering.

Read more in the :ref:`User Guide <sample_generators>`.

Parameters
----------
n_samples : int or array-like, optional (default=100)
    If int, it is the total number of points equally divided among
    clusters.
    If array-like, each element of the sequence indicates
    the number of samples per cluster.

n_features : int, optional (default=2)
    The number of features for each sample.

centers : int or array of shape [n_centers, n_features], optional
    (default=None)
    The number of centers to generate, or the fixed center locations.
    If n_samples is an int and centers is None, 3 centers are generated.
    If n_samples is array-like, centers must be
    either None or an array of length equal to the length of n_samples.

cluster_std : float or sequence of floats, optional (default=1.0)
    The standard deviation of the clusters.

center_box : pair of floats (min, max), optional (default=(-10.0, 10.0))
    The bounding box for each cluster center when centers are
    generated at random.

shuffle : boolean, optional (default=True)
    Shuffle the samples.

random_state : int, RandomState instance or None (default)
    Determines random number generation for dataset creation. Pass an int
    for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

Returns
-------
X : array of shape [n_samples, n_features]
    The generated samples.

y : array of shape [n_samples]
    The integer labels for cluster membership of each sample.

Examples
--------
>>> from sklearn.datasets import make_blobs
>>> X, y = make_blobs(n_samples=10, centers=3, n_features=2,
...                   random_state=0)
>>> print(X.shape)
(10, 2)
>>> y
array([0, 0, 1, 0, 2, 2, 2, 1, 1, 0])
>>> X, y = make_blobs(n_samples=[3, 3, 4], centers=None, n_features=2,
...                   random_state=0)
>>> print(X.shape)
(10, 2)
>>> y
array([0, 1, 2, 0, 2, 2, 2, 1, 1, 0])

See also
--------
make_classification: a more intricate variant
*)

val make_checkerboard : ?noise:float -> ?minval:int -> ?maxval:int -> ?shuffle:bool -> ?random_state:int -> shape:int list -> n_clusters:[`I of int | `PyObject of Py.Object.t] -> unit -> (Py.Object.t * Py.Object.t * Py.Object.t)
(**
Generate an array with block checkerboard structure for
biclustering.

Read more in the :ref:`User Guide <sample_generators>`.

Parameters
----------
shape : iterable (n_rows, n_cols)
    The shape of the result.

n_clusters : integer or iterable (n_row_clusters, n_column_clusters)
    The number of row and column clusters.

noise : float, optional (default=0.0)
    The standard deviation of the gaussian noise.

minval : int, optional (default=10)
    Minimum value of a bicluster.

maxval : int, optional (default=100)
    Maximum value of a bicluster.

shuffle : boolean, optional (default=True)
    Shuffle the samples.

random_state : int, RandomState instance or None (default)
    Determines random number generation for dataset creation. Pass an int
    for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

Returns
-------
X : array of shape `shape`
    The generated array.

rows : array of shape (n_clusters, X.shape[0],)
    The indicators for cluster membership of each row.

cols : array of shape (n_clusters, X.shape[1],)
    The indicators for cluster membership of each column.


References
----------

.. [1] Kluger, Y., Basri, R., Chang, J. T., & Gerstein, M. (2003).
    Spectral biclustering of microarray data: coclustering genes
    and conditions. Genome research, 13(4), 703-716.

See also
--------
make_biclusters
*)

val make_circles : ?n_samples:int -> ?shuffle:bool -> ?noise:float -> ?random_state:int -> ?factor:Py.Object.t -> unit -> (Arr.t * Arr.t)
(**
Make a large circle containing a smaller circle in 2d.

A simple toy dataset to visualize clustering and classification
algorithms.

Read more in the :ref:`User Guide <sample_generators>`.

Parameters
----------
n_samples : int, optional (default=100)
    The total number of points generated. If odd, the inner circle will
    have one point more than the outer circle.

shuffle : bool, optional (default=True)
    Whether to shuffle the samples.

noise : double or None (default=None)
    Standard deviation of Gaussian noise added to the data.

random_state : int, RandomState instance or None (default)
    Determines random number generation for dataset shuffling and noise.
    Pass an int for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

factor : 0 < double < 1 (default=.8)
    Scale factor between inner and outer circle.

Returns
-------
X : array of shape [n_samples, 2]
    The generated samples.

y : array of shape [n_samples]
    The integer labels (0 or 1) for class membership of each sample.
*)

val make_classification : ?n_samples:int -> ?n_features:int -> ?n_informative:int -> ?n_redundant:int -> ?n_repeated:int -> ?n_classes:int -> ?n_clusters_per_class:int -> ?weights:[`Arr of Arr.t | `N_classes_1 of Py.Object.t] -> ?flip_y:float -> ?class_sep:float -> ?hypercube:bool -> ?shift:[`F of float | `Arr of Arr.t | `None] -> ?scale:[`F of float | `Arr of Arr.t | `None] -> ?shuffle:bool -> ?random_state:int -> unit -> (Arr.t * Arr.t)
(**
Generate a random n-class classification problem.

This initially creates clusters of points normally distributed (std=1)
about vertices of an ``n_informative``-dimensional hypercube with sides of
length ``2*class_sep`` and assigns an equal number of clusters to each
class. It introduces interdependence between these features and adds
various types of further noise to the data.

Without shuffling, ``X`` horizontally stacks features in the following
order: the primary ``n_informative`` features, followed by ``n_redundant``
linear combinations of the informative features, followed by ``n_repeated``
duplicates, drawn randomly with replacement from the informative and
redundant features. The remaining features are filled with random noise.
Thus, without shuffling, all useful features are contained in the columns
``X[:, :n_informative + n_redundant + n_repeated]``.

Read more in the :ref:`User Guide <sample_generators>`.

Parameters
----------
n_samples : int, optional (default=100)
    The number of samples.

n_features : int, optional (default=20)
    The total number of features. These comprise ``n_informative``
    informative features, ``n_redundant`` redundant features,
    ``n_repeated`` duplicated features and
    ``n_features-n_informative-n_redundant-n_repeated`` useless features
    drawn at random.

n_informative : int, optional (default=2)
    The number of informative features. Each class is composed of a number
    of gaussian clusters each located around the vertices of a hypercube
    in a subspace of dimension ``n_informative``. For each cluster,
    informative features are drawn independently from  N(0, 1) and then
    randomly linearly combined within each cluster in order to add
    covariance. The clusters are then placed on the vertices of the
    hypercube.

n_redundant : int, optional (default=2)
    The number of redundant features. These features are generated as
    random linear combinations of the informative features.

n_repeated : int, optional (default=0)
    The number of duplicated features, drawn randomly from the informative
    and the redundant features.

n_classes : int, optional (default=2)
    The number of classes (or labels) of the classification problem.

n_clusters_per_class : int, optional (default=2)
    The number of clusters per class.

weights : array-like of shape (n_classes,) or (n_classes - 1,),              (default=None)
    The proportions of samples assigned to each class. If None, then
    classes are balanced. Note that if ``len(weights) == n_classes - 1``,
    then the last class weight is automatically inferred.
    More than ``n_samples`` samples may be returned if the sum of
    ``weights`` exceeds 1.

flip_y : float, optional (default=0.01)
    The fraction of samples whose class is assigned randomly. Larger
    values introduce noise in the labels and make the classification
    task harder.

class_sep : float, optional (default=1.0)
    The factor multiplying the hypercube size.  Larger values spread
    out the clusters/classes and make the classification task easier.

hypercube : boolean, optional (default=True)
    If True, the clusters are put on the vertices of a hypercube. If
    False, the clusters are put on the vertices of a random polytope.

shift : float, array of shape [n_features] or None, optional (default=0.0)
    Shift features by the specified value. If None, then features
    are shifted by a random value drawn in [-class_sep, class_sep].

scale : float, array of shape [n_features] or None, optional (default=1.0)
    Multiply features by the specified value. If None, then features
    are scaled by a random value drawn in [1, 100]. Note that scaling
    happens after shifting.

shuffle : boolean, optional (default=True)
    Shuffle the samples and the features.

random_state : int, RandomState instance or None (default)
    Determines random number generation for dataset creation. Pass an int
    for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

Returns
-------
X : array of shape [n_samples, n_features]
    The generated samples.

y : array of shape [n_samples]
    The integer labels for class membership of each sample.

Notes
-----
The algorithm is adapted from Guyon [1] and was designed to generate
the "Madelon" dataset.

References
----------
.. [1] I. Guyon, "Design of experiments for the NIPS 2003 variable
       selection benchmark", 2003.

See also
--------
make_blobs: simplified variant
make_multilabel_classification: unrelated generator for multilabel tasks
*)

val make_friedman1 : ?n_samples:int -> ?n_features:int -> ?noise:float -> ?random_state:int -> unit -> (Arr.t * Arr.t)
(**
Generate the "Friedman #1" regression problem

This dataset is described in Friedman [1] and Breiman [2].

Inputs `X` are independent features uniformly distributed on the interval
[0, 1]. The output `y` is created according to the formula::

    y(X) = 10 * sin(pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4] + noise * N(0, 1).

Out of the `n_features` features, only 5 are actually used to compute
`y`. The remaining features are independent of `y`.

The number of features has to be >= 5.

Read more in the :ref:`User Guide <sample_generators>`.

Parameters
----------
n_samples : int, optional (default=100)
    The number of samples.

n_features : int, optional (default=10)
    The number of features. Should be at least 5.

noise : float, optional (default=0.0)
    The standard deviation of the gaussian noise applied to the output.

random_state : int, RandomState instance or None (default)
    Determines random number generation for dataset noise. Pass an int
    for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

Returns
-------
X : array of shape [n_samples, n_features]
    The input samples.

y : array of shape [n_samples]
    The output values.

References
----------
.. [1] J. Friedman, "Multivariate adaptive regression splines", The Annals
       of Statistics 19 (1), pages 1-67, 1991.

.. [2] L. Breiman, "Bagging predictors", Machine Learning 24,
       pages 123-140, 1996.
*)

val make_friedman2 : ?n_samples:int -> ?noise:float -> ?random_state:int -> unit -> (Arr.t * Arr.t)
(**
Generate the "Friedman #2" regression problem

This dataset is described in Friedman [1] and Breiman [2].

Inputs `X` are 4 independent features uniformly distributed on the
intervals::

    0 <= X[:, 0] <= 100,
    40 * pi <= X[:, 1] <= 560 * pi,
    0 <= X[:, 2] <= 1,
    1 <= X[:, 3] <= 11.

The output `y` is created according to the formula::

    y(X) = (X[:, 0] ** 2 + (X[:, 1] * X[:, 2]  - 1 / (X[:, 1] * X[:, 3])) ** 2) ** 0.5 + noise * N(0, 1).

Read more in the :ref:`User Guide <sample_generators>`.

Parameters
----------
n_samples : int, optional (default=100)
    The number of samples.

noise : float, optional (default=0.0)
    The standard deviation of the gaussian noise applied to the output.

random_state : int, RandomState instance or None (default)
    Determines random number generation for dataset noise. Pass an int
    for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

Returns
-------
X : array of shape [n_samples, 4]
    The input samples.

y : array of shape [n_samples]
    The output values.

References
----------
.. [1] J. Friedman, "Multivariate adaptive regression splines", The Annals
       of Statistics 19 (1), pages 1-67, 1991.

.. [2] L. Breiman, "Bagging predictors", Machine Learning 24,
       pages 123-140, 1996.
*)

val make_friedman3 : ?n_samples:int -> ?noise:float -> ?random_state:int -> unit -> (Arr.t * Arr.t)
(**
Generate the "Friedman #3" regression problem

This dataset is described in Friedman [1] and Breiman [2].

Inputs `X` are 4 independent features uniformly distributed on the
intervals::

    0 <= X[:, 0] <= 100,
    40 * pi <= X[:, 1] <= 560 * pi,
    0 <= X[:, 2] <= 1,
    1 <= X[:, 3] <= 11.

The output `y` is created according to the formula::

    y(X) = arctan((X[:, 1] * X[:, 2] - 1 / (X[:, 1] * X[:, 3])) / X[:, 0]) + noise * N(0, 1).

Read more in the :ref:`User Guide <sample_generators>`.

Parameters
----------
n_samples : int, optional (default=100)
    The number of samples.

noise : float, optional (default=0.0)
    The standard deviation of the gaussian noise applied to the output.

random_state : int, RandomState instance or None (default)
    Determines random number generation for dataset noise. Pass an int
    for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

Returns
-------
X : array of shape [n_samples, 4]
    The input samples.

y : array of shape [n_samples]
    The output values.

References
----------
.. [1] J. Friedman, "Multivariate adaptive regression splines", The Annals
       of Statistics 19 (1), pages 1-67, 1991.

.. [2] L. Breiman, "Bagging predictors", Machine Learning 24,
       pages 123-140, 1996.
*)

val make_gaussian_quantiles : ?mean:Arr.t -> ?cov:float -> ?n_samples:int -> ?n_features:int -> ?n_classes:int -> ?shuffle:bool -> ?random_state:int -> unit -> (Arr.t * Arr.t)
(**
Generate isotropic Gaussian and label samples by quantile

This classification dataset is constructed by taking a multi-dimensional
standard normal distribution and defining classes separated by nested
concentric multi-dimensional spheres such that roughly equal numbers of
samples are in each class (quantiles of the :math:`\chi^2` distribution).

Read more in the :ref:`User Guide <sample_generators>`.

Parameters
----------
mean : array of shape [n_features], optional (default=None)
    The mean of the multi-dimensional normal distribution.
    If None then use the origin (0, 0, ...).

cov : float, optional (default=1.)
    The covariance matrix will be this value times the unit matrix. This
    dataset only produces symmetric normal distributions.

n_samples : int, optional (default=100)
    The total number of points equally divided among classes.

n_features : int, optional (default=2)
    The number of features for each sample.

n_classes : int, optional (default=3)
    The number of classes

shuffle : boolean, optional (default=True)
    Shuffle the samples.

random_state : int, RandomState instance or None (default)
    Determines random number generation for dataset creation. Pass an int
    for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

Returns
-------
X : array of shape [n_samples, n_features]
    The generated samples.

y : array of shape [n_samples]
    The integer labels for quantile membership of each sample.

Notes
-----
The dataset is from Zhu et al [1].

References
----------
.. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.
*)

val make_hastie_10_2 : ?n_samples:int -> ?random_state:int -> unit -> (Arr.t * Arr.t)
(**
Generates data for binary classification used in
Hastie et al. 2009, Example 10.2.

The ten features are standard independent Gaussian and
the target ``y`` is defined by::

  y[i] = 1 if np.sum(X[i] ** 2) > 9.34 else -1

Read more in the :ref:`User Guide <sample_generators>`.

Parameters
----------
n_samples : int, optional (default=12000)
    The number of samples.

random_state : int, RandomState instance or None (default)
    Determines random number generation for dataset creation. Pass an int
    for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

Returns
-------
X : array of shape [n_samples, 10]
    The input samples.

y : array of shape [n_samples]
    The output values.

References
----------
.. [1] T. Hastie, R. Tibshirani and J. Friedman, "Elements of Statistical
       Learning Ed. 2", Springer, 2009.

See also
--------
make_gaussian_quantiles: a generalization of this dataset approach
*)

val make_low_rank_matrix : ?n_samples:int -> ?n_features:int -> ?effective_rank:int -> ?tail_strength:Py.Object.t -> ?random_state:int -> unit -> Arr.t
(**
Generate a mostly low rank matrix with bell-shaped singular values

Most of the variance can be explained by a bell-shaped curve of width
effective_rank: the low rank part of the singular values profile is::

    (1 - tail_strength) * exp(-1.0 * (i / effective_rank) ** 2)

The remaining singular values' tail is fat, decreasing as::

    tail_strength * exp(-0.1 * i / effective_rank).

The low rank part of the profile can be considered the structured
signal part of the data while the tail can be considered the noisy
part of the data that cannot be summarized by a low number of linear
components (singular vectors).

This kind of singular profiles is often seen in practice, for instance:
 - gray level pictures of faces
 - TF-IDF vectors of text documents crawled from the web

Read more in the :ref:`User Guide <sample_generators>`.

Parameters
----------
n_samples : int, optional (default=100)
    The number of samples.

n_features : int, optional (default=100)
    The number of features.

effective_rank : int, optional (default=10)
    The approximate number of singular vectors required to explain most of
    the data by linear combinations.

tail_strength : float between 0.0 and 1.0, optional (default=0.5)
    The relative importance of the fat noisy tail of the singular values
    profile.

random_state : int, RandomState instance or None (default)
    Determines random number generation for dataset creation. Pass an int
    for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

Returns
-------
X : array of shape [n_samples, n_features]
    The matrix.
*)

val make_moons : ?n_samples:int -> ?shuffle:bool -> ?noise:float -> ?random_state:int -> unit -> (Arr.t * Arr.t)
(**
Make two interleaving half circles

A simple toy dataset to visualize clustering and classification
algorithms. Read more in the :ref:`User Guide <sample_generators>`.

Parameters
----------
n_samples : int, optional (default=100)
    The total number of points generated.

shuffle : bool, optional (default=True)
    Whether to shuffle the samples.

noise : double or None (default=None)
    Standard deviation of Gaussian noise added to the data.

random_state : int, RandomState instance or None (default)
    Determines random number generation for dataset shuffling and noise.
    Pass an int for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

Returns
-------
X : array of shape [n_samples, 2]
    The generated samples.

y : array of shape [n_samples]
    The integer labels (0 or 1) for class membership of each sample.
*)

val make_multilabel_classification : ?n_samples:int -> ?n_features:int -> ?n_classes:int -> ?n_labels:int -> ?length:int -> ?allow_unlabeled:bool -> ?sparse:bool -> ?return_indicator:[`Dense | `Sparse | `False] -> ?return_distributions:bool -> ?random_state:int -> unit -> (Arr.t * Py.Object.t * Arr.t * Arr.t)
(**
Generate a random multilabel classification problem.

For each sample, the generative process is:
    - pick the number of labels: n ~ Poisson(n_labels)
    - n times, choose a class c: c ~ Multinomial(theta)
    - pick the document length: k ~ Poisson(length)
    - k times, choose a word: w ~ Multinomial(theta_c)

In the above process, rejection sampling is used to make sure that
n is never zero or more than `n_classes`, and that the document length
is never zero. Likewise, we reject classes which have already been chosen.

Read more in the :ref:`User Guide <sample_generators>`.

Parameters
----------
n_samples : int, optional (default=100)
    The number of samples.

n_features : int, optional (default=20)
    The total number of features.

n_classes : int, optional (default=5)
    The number of classes of the classification problem.

n_labels : int, optional (default=2)
    The average number of labels per instance. More precisely, the number
    of labels per sample is drawn from a Poisson distribution with
    ``n_labels`` as its expected value, but samples are bounded (using
    rejection sampling) by ``n_classes``, and must be nonzero if
    ``allow_unlabeled`` is False.

length : int, optional (default=50)
    The sum of the features (number of words if documents) is drawn from
    a Poisson distribution with this expected value.

allow_unlabeled : bool, optional (default=True)
    If ``True``, some instances might not belong to any class.

sparse : bool, optional (default=False)
    If ``True``, return a sparse feature matrix

    .. versionadded:: 0.17
       parameter to allow *sparse* output.

return_indicator : 'dense' (default) | 'sparse' | False
    If ``dense`` return ``Y`` in the dense binary indicator format. If
    ``'sparse'`` return ``Y`` in the sparse binary indicator format.
    ``False`` returns a list of lists of labels.

return_distributions : bool, optional (default=False)
    If ``True``, return the prior class probability and conditional
    probabilities of features given classes, from which the data was
    drawn.

random_state : int, RandomState instance or None (default)
    Determines random number generation for dataset creation. Pass an int
    for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

Returns
-------
X : array of shape [n_samples, n_features]
    The generated samples.

Y : array or sparse CSR matrix of shape [n_samples, n_classes]
    The label sets.

p_c : array, shape [n_classes]
    The probability of each class being drawn. Only returned if
    ``return_distributions=True``.

p_w_c : array, shape [n_features, n_classes]
    The probability of each feature being drawn given each class.
    Only returned if ``return_distributions=True``.
*)

val make_regression : ?n_samples:int -> ?n_features:int -> ?n_informative:int -> ?n_targets:int -> ?bias:float -> ?effective_rank:int -> ?tail_strength:Py.Object.t -> ?noise:float -> ?shuffle:bool -> ?random_state:int -> unit -> (Arr.t * Arr.t * Arr.t)
(**
Generate a random regression problem.

The input set can either be well conditioned (by default) or have a low
rank-fat tail singular profile. See :func:`make_low_rank_matrix` for
more details.

The output is generated by applying a (potentially biased) random linear
regression model with `n_informative` nonzero regressors to the previously
generated input and some gaussian centered noise with some adjustable
scale.

Read more in the :ref:`User Guide <sample_generators>`.

Parameters
----------
n_samples : int, optional (default=100)
    The number of samples.

n_features : int, optional (default=100)
    The number of features.

n_informative : int, optional (default=10)
    The number of informative features, i.e., the number of features used
    to build the linear model used to generate the output.

n_targets : int, optional (default=1)
    The number of regression targets, i.e., the dimension of the y output
    vector associated with a sample. By default, the output is a scalar.

bias : float, optional (default=0.0)
    The bias term in the underlying linear model.

effective_rank : int or None, optional (default=None)
    if not None:
        The approximate number of singular vectors required to explain most
        of the input data by linear combinations. Using this kind of
        singular spectrum in the input allows the generator to reproduce
        the correlations often observed in practice.
    if None:
        The input set is well conditioned, centered and gaussian with
        unit variance.

tail_strength : float between 0.0 and 1.0, optional (default=0.5)
    The relative importance of the fat noisy tail of the singular values
    profile if `effective_rank` is not None.

noise : float, optional (default=0.0)
    The standard deviation of the gaussian noise applied to the output.

shuffle : boolean, optional (default=True)
    Shuffle the samples and the features.

coef : boolean, optional (default=False)
    If True, the coefficients of the underlying linear model are returned.

random_state : int, RandomState instance or None (default)
    Determines random number generation for dataset creation. Pass an int
    for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

Returns
-------
X : array of shape [n_samples, n_features]
    The input samples.

y : array of shape [n_samples] or [n_samples, n_targets]
    The output values.

coef : array of shape [n_features] or [n_features, n_targets], optional
    The coefficient of the underlying linear model. It is returned only if
    coef is True.
*)

val make_s_curve : ?n_samples:int -> ?noise:float -> ?random_state:int -> unit -> (Arr.t * Arr.t)
(**
Generate an S curve dataset.

Read more in the :ref:`User Guide <sample_generators>`.

Parameters
----------
n_samples : int, optional (default=100)
    The number of sample points on the S curve.

noise : float, optional (default=0.0)
    The standard deviation of the gaussian noise.

random_state : int, RandomState instance or None (default)
    Determines random number generation for dataset creation. Pass an int
    for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

Returns
-------
X : array of shape [n_samples, 3]
    The points.

t : array of shape [n_samples]
    The univariate position of the sample according to the main dimension
    of the points in the manifold.
*)

val make_sparse_coded_signal : ?random_state:int -> n_samples:int -> n_components:int -> n_features:int -> n_nonzero_coefs:int -> unit -> (Arr.t * Arr.t * Arr.t)
(**
Generate a signal as a sparse combination of dictionary elements.

Returns a matrix Y = DX, such as D is (n_features, n_components),
X is (n_components, n_samples) and each column of X has exactly
n_nonzero_coefs non-zero elements.

Read more in the :ref:`User Guide <sample_generators>`.

Parameters
----------
n_samples : int
    number of samples to generate

n_components :  int,
    number of components in the dictionary

n_features : int
    number of features of the dataset to generate

n_nonzero_coefs : int
    number of active (non-zero) coefficients in each sample

random_state : int, RandomState instance or None (default)
    Determines random number generation for dataset creation. Pass an int
    for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

Returns
-------
data : array of shape [n_features, n_samples]
    The encoded signal (Y).

dictionary : array of shape [n_features, n_components]
    The dictionary with normalized components (D).

code : array of shape [n_components, n_samples]
    The sparse code such that each column of this matrix has exactly
    n_nonzero_coefs non-zero items (X).
*)

val make_sparse_spd_matrix : ?dim:int -> ?alpha:Py.Object.t -> ?norm_diag:bool -> ?smallest_coef:Py.Object.t -> ?largest_coef:Py.Object.t -> ?random_state:int -> unit -> Csr_matrix.t
(**
Generate a sparse symmetric definite positive matrix.

Read more in the :ref:`User Guide <sample_generators>`.

Parameters
----------
dim : integer, optional (default=1)
    The size of the random matrix to generate.

alpha : float between 0 and 1, optional (default=0.95)
    The probability that a coefficient is zero (see notes). Larger values
    enforce more sparsity.

norm_diag : boolean, optional (default=False)
    Whether to normalize the output matrix to make the leading diagonal
    elements all 1

smallest_coef : float between 0 and 1, optional (default=0.1)
    The value of the smallest coefficient.

largest_coef : float between 0 and 1, optional (default=0.9)
    The value of the largest coefficient.

random_state : int, RandomState instance or None (default)
    Determines random number generation for dataset creation. Pass an int
    for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

Returns
-------
prec : sparse matrix of shape (dim, dim)
    The generated matrix.

Notes
-----
The sparsity is actually imposed on the cholesky factor of the matrix.
Thus alpha does not translate directly into the filling fraction of
the matrix itself.

See also
--------
make_spd_matrix
*)

val make_sparse_uncorrelated : ?n_samples:int -> ?n_features:int -> ?random_state:int -> unit -> (Arr.t * Arr.t)
(**
Generate a random regression problem with sparse uncorrelated design

This dataset is described in Celeux et al [1]. as::

    X ~ N(0, 1)
    y(X) = X[:, 0] + 2 * X[:, 1] - 2 * X[:, 2] - 1.5 * X[:, 3]

Only the first 4 features are informative. The remaining features are
useless.

Read more in the :ref:`User Guide <sample_generators>`.

Parameters
----------
n_samples : int, optional (default=100)
    The number of samples.

n_features : int, optional (default=10)
    The number of features.

random_state : int, RandomState instance or None (default)
    Determines random number generation for dataset creation. Pass an int
    for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

Returns
-------
X : array of shape [n_samples, n_features]
    The input samples.

y : array of shape [n_samples]
    The output values.

References
----------
.. [1] G. Celeux, M. El Anbari, J.-M. Marin, C. P. Robert,
       "Regularization in regression: comparing Bayesian and frequentist
       methods in a poorly informative situation", 2009.
*)

val make_spd_matrix : ?random_state:int -> n_dim:int -> unit -> Arr.t
(**
Generate a random symmetric, positive-definite matrix.

Read more in the :ref:`User Guide <sample_generators>`.

Parameters
----------
n_dim : int
    The matrix dimension.

random_state : int, RandomState instance or None (default)
    Determines random number generation for dataset creation. Pass an int
    for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

Returns
-------
X : array of shape [n_dim, n_dim]
    The random symmetric, positive-definite matrix.

See also
--------
make_sparse_spd_matrix
*)

val make_swiss_roll : ?n_samples:int -> ?noise:float -> ?random_state:int -> unit -> (Arr.t * Arr.t)
(**
Generate a swiss roll dataset.

Read more in the :ref:`User Guide <sample_generators>`.

Parameters
----------
n_samples : int, optional (default=100)
    The number of sample points on the S curve.

noise : float, optional (default=0.0)
    The standard deviation of the gaussian noise.

random_state : int, RandomState instance or None (default)
    Determines random number generation for dataset creation. Pass an int
    for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

Returns
-------
X : array of shape [n_samples, 3]
    The points.

t : array of shape [n_samples]
    The univariate position of the sample according to the main dimension
    of the points in the manifold.

Notes
-----
The algorithm is from Marsland [1].

References
----------
.. [1] S. Marsland, "Machine Learning: An Algorithmic Perspective",
       Chapter 10, 2009.
       http://seat.massey.ac.nz/personal/s.r.marsland/Code/10/lle.py
*)

