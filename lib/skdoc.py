import re
import pkgutil
import importlib
import inspect
import textwrap
import io
import os


class Section_title:
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.text}')"


class Paragraph:
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.text}')"


class Type:
    def visit(self, f):
        f(self)
        for child in getattr(self, 'elements', []):
            f(child)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        text = getattr(self, 'text', None)
        if text is None:
            text = getattr(self, 'elements', '')
        return f"{self.__class__.__name__}({text})"

    ml_type = 'Py.Object.t'
    wrap = 'Wrap_utils.id'

    ml_type_ret = 'Py.Object.t'
    unwrap = 'Wrap_utils.id'

    is_type = None

    def tag_name(self):
        return tag(self.__class__.__name__)

    def tag(self):
        ta = self.tag_name()
        t = f"`{ta} of {self.ml_type}"
        destruct = f"`{ta} x -> {self.wrap} x"
        if self.is_type is None:
            construct = None
        else:
            construct = f"if {self.is_type} x then `{ta} ({self.unwrap} x)"
        return t, destruct, construct


class Bunch(Type):
    def __init__(self, elements):
        self.elements = elements
        self.ml_type = ("< " +
                        '; '.join(f"{mlid(name)}: {t.ml_type_ret}"
                                  for name, t in self.elements.items()) + " >")
        # for now Bunch is only for returning things
        self.wrap = 'Wrap_utils.id'

        self.ml_type_ret = self.ml_type
        unwrap_methods = [
            f'method {mlid(name)} = {t.unwrap} (Py.Object.get_attr_string bunch "{name}" |> Wrap_utils.Option.get)'
            for name, t in self.elements.items()
        ]
        self.unwrap = f'(fun bunch -> object {" ".join(unwrap_methods)} end)'

    def visit(self, f):
        f(self)
        for child in self.elements.values():
            f(child)


class UnknownType(Type):
    def __init__(self, text):
        self.text = text

    def tag_name(self):
        return tag(self.text)


class StringValue(Type):
    def __init__(self, text):
        assert text != 'None'
        self.text = text.strip("\"'")

    ml_type = 'string'
    wrap = 'Py.String.of_string'

    ml_type_ret = 'string'
    unwrap = 'Py.String.to_string'

    def tag(self):
        ta = tag(self.text)
        t = f"`{ta}"
        destruct = f'`{ta} -> {self.wrap} "{self.text}"'
        construct = None
        return t, destruct, construct


class IntValue(Type):
    def __init__(self, value):
        self.value = int(value)

    ml_type = 'int'
    wrap = 'Py.Int.of_int'

    ml_type_ret = 'int'
    unwrap = 'Py.Int.to_int'

    def tag(self):
        ta = ['Zero', 'One', 'Two'][self.value]
        t = f"`{ta}"
        destruct = f'`{ta} -> {self.wrap} {self.value}'
        construct = None
        return t, destruct, construct

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"


class Enum(Type):
    def can_be_returned(self):
        for elt in self.elements:
            if elt.tag()[2] is None:
                # print(f"WW enum {self} cannot be returned because element {elt} cannot be discriminated")
                return False
        return True

    def __init__(self, elements):
        assert elements, elements
        self.elements = elements
        self.ml_type = "[" + ' | '.join([elt.tag()[0]
                                         for elt in elements]) + "]"
        wrap_cases = [f"| {elt.tag()[1]}\n" for elt in elements]
        self.wrap = f"(function\n{''.join(wrap_cases)})"

        if self.can_be_returned():
            self.ml_type_ret = self.ml_type
            unwrap_cases = [f"{elt.tag()[2]}" for elt in elements]
            self.unwrap = '(fun x -> ' + ' else '.join(
                unwrap_cases
            ) + ' else failwith "could not identify type from Python value")'


class Tuple(Type):
    def __init__(self, elements):
        self.elements = elements
        self.ml_type = '(' + ' * '.join(elt.ml_type for elt in elements) + ')'
        self.ml_type_ret = '(' + ' * '.join(elt.ml_type_ret
                                            for elt in elements) + ')'
        unwrap_elts = [
            f"({elt.unwrap} (Py.Tuple.get x {i}))"
            for i, elt in enumerate(elements)
        ]
        self.unwrap = f"(fun x -> ({', '.join(unwrap_elts)}))"
        split = '(' + ', '.join([f"ml_{i}"
                                 for i in range(len(self.elements))]) + ')'
        wrap_elts = [f"({elt.wrap} ml_{i})" for i, elt in enumerate(elements)]
        self.wrap = f"(fun {split} -> Py.Tuple.of_list [{'; '.join(wrap_elts)}])"


class Int(Type):
    names = ['int', 'integer', 'integer > 0', 'int with']
    ml_type = 'int'
    wrap = 'Py.Int.of_int'
    ml_type_ret = 'int'
    unwrap = 'Py.Int.to_int'
    is_type = 'Py.Int.check'

    def tag_name(self):
        return tag("I")


class Pipeline(Type):
    names = ['Pipeline', 'pipeline']
    ml_type = 'Sklearn.Pipeline.Pipeline.t'
    wrap = 'Sklearn.Pipeline.Pipeline.to_pyobject'
    ml_type_ret = 'Sklearn.Pipeline.Pipeline.t'
    unwrap = 'Sklearn.Pipeline.Pipeline.of_pyobject'


# XXX TODO refactor this, we could process returned classes in a more generic manner maybe?
# (need to figure out a clean way to have these work wherever they are used)
class FeatureUnion(Type):
    names = ['FeatureUnion']
    ml_type = 'Sklearn.Pipeline.FeatureUnion.t'
    wrap = 'Sklearn.Pipeline.FeatureUnion.to_pyobject'
    ml_type_ret = 'Sklearn.Pipeline.FeatureUnion.t'
    unwrap = 'Sklearn.Pipeline.FeatureUnion.of_pyobject'


class Float(Type):
    names = [
        'float', 'floating', 'double', 'positive float',
        'strictly positive float', 'non-negative float', 'numeric',
        'float (upperlimited by 1.0)', 'float in range'
    ]
    ml_type = 'float'
    wrap = 'Py.Float.of_float'
    ml_type_ret = 'float'
    unwrap = 'Py.Float.to_float'
    is_type = 'Py.Float.check'

    def tag_name(self):
        return tag("F")


class Bool(Type):
    names = ['bool', 'boolean', 'Boolean', 'Bool', 'boolean value']
    ml_type = 'bool'
    wrap = 'Py.Bool.of_bool'
    ml_type_ret = 'bool'
    unwrap = 'Py.Bool.to_bool'


class Ndarray(Type):
    names = []
    ml_type = 'Sklearn.Ndarray.t'
    wrap = 'Sklearn.Ndarray.to_pyobject'
    ml_type_ret = 'Sklearn.Ndarray.t'
    unwrap = 'Sklearn.Ndarray.of_pyobject'
    is_type = 'Wrap_utils.isinstance Wrap_utils.ndarray'


class SparseMatrix(Type):
    names = [
        'sparse matrix', 'sparse-matrix', 'CSR matrix', 'CSR matrix with',
        'CSR sparse matrix', 'sparse graph in CSR format',
        'scipy.sparse.csr_matrix', 'CSR', 'scipy.sparse', 'sparse matrix with'
    ]
    ml_type = 'Sklearn.Csr_matrix.t'
    wrap = 'Sklearn.Csr_matrix.to_pyobject'
    ml_type_ret = 'Sklearn.Csr_matrix.t'
    unwrap = 'Sklearn.Csr_matrix.of_pyobject'
    is_type = 'Wrap_utils.isinstance Wrap_utils.csr_matrix'


class Arr(Type):
    """Handles dense or sparse arrays (ie, union of Ndarray and
    Csr_matrix/SparseMatrix).

    We wrap all arrays using Arr, except in Csr_matrix (which would
    cause a dependency cycle).

    """
    names = [
        'ndarray', 'numpy array', 'array of floats', 'nd-array', 'array',
        'float ndarray', 'iterable', 'indexable', 'an iterable',
        'numeric array-like', 'array of float', 'array-like', 'array_like',
        'array like', 'np.matrix', 'numpy.matrix', 'float array with',
        'matrix', '1d array-like', 'int array', 'int array-like',
        'ndarray of floats', 'numpy array of int', 'numpy array of float',
        '(sparse) array-like', 'array of int', 'numpy.ndarray',
        'array-like of float', 'bool array', 'list-like', 'list',
        'label indicator array / sparse matrix', 'label indicator matrix'
    ]
    ml_type = 'Sklearn.Arr.t'
    wrap = 'Sklearn.Arr.to_pyobject'
    ml_type_ret = 'Sklearn.Arr.t'
    unwrap = 'Sklearn.Arr.of_pyobject'
    is_type = f'(fun x -> ({Ndarray.is_type} x) || ({SparseMatrix.is_type} x))'


class ClassificationReport(Type):
    # XXX ml_type is needed even for retuning, since we use it atm to
    # build the return type inside enums (see Type.tag()).
    ml_type = '(string * <precision:float; recall:float; f1_score:float; support:float>) list'
    ml_type_ret = '(string * <precision:float; recall:float; f1_score:float; support:float>) list'
    unwrap = '''(fun py -> Py.Dict.fold (fun kpy vpy acc -> ((Py.String.to_string kpy), object
      method precision = Py.Dict.get_item_string vpy "precision" |> Wrap_utils.Option.get |> Py.Float.to_float
      method recall = Py.Dict.get_item_string vpy "recall" |> Wrap_utils.Option.get |> Py.Float.to_float
      method f1_score = Py.Dict.get_item_string vpy "f1-score" |> Wrap_utils.Option.get |> Py.Float.to_float
      method support = Py.Dict.get_item_string vpy "support" |> Wrap_utils.Option.get |> Py.Float.to_float
    end)::acc) py [])
    '''
    is_type = 'Wrap_utils.isinstance Wrap_utils.dict'

    def tag_name(self):
        return 'Dict'


class Array(Type):
    def __init__(self, t):
        self.t = t
        self.names = []
        self.ml_type = f"{t.ml_type} array"
        self.wrap = f'(fun ml -> Py.Array.of_array {t.wrap} (fun _ -> invalid_arg "read-only") ml)'
        self.ml_type_ret = f"{t.ml_type} array"
        self.unwrap = f"(fun py -> let len = Py.Sequence.length py in Array.init len (fun i -> {t.unwrap} (Py.Sequence.get_item py i)))"

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"Array[{str(self.t)}]"


class ArrayList(Type):
    names = ['iterable of iterables', 'list of arrays', 'list of ndarray']
    ml_type = 'Sklearn.Arr.List.t'
    ml_type_ret = 'Sklearn.Arr.List.t'
    wrap = 'Sklearn.Arr.List.to_pyobject'
    unwrap = 'Sklearn.Arr.List.of_pyobject'


class List(Type):
    def __init__(self, t):
        self.t = t
        self.names = []
        self.ml_type = f"{t.ml_type} list"
        self.wrap = f'(fun ml -> Py.List.of_list_map {t.wrap} ml)'
        # self.ml_type_ret = f"{t.ml_type} array"
        # self.unwrap = f"(fun py -> let len = Py.Sequence.length py in Array.init len (fun i -> {t.unwrap} (Py.Sequence.get_item py i)))"
        self.ml_type_ret = f"{t.ml_type} list"
        self.unwrap = f"(fun py -> Py.List.to_list_map ({t.unwrap}) py)"

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"List[{str(self.t)}]"


class StarStar(Type):
    def __init__(self):
        self.names = []
        self.ml_type = f"(string * Py.Object.t) list"

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"**kwargs"


class FloatList(Type):
    names = ['list of floats']
    ml_type = 'float list'
    wrap = '(Py.List.of_list_map Py.Float.of_float)'


class StringList(Type):
    names = [
        'list of strings', 'list of string', 'list/tuple of strings',
        'tuple of str', 'list of str', 'strings'
    ]
    ml_type = 'string list'
    wrap = '(Py.List.of_list_map Py.String.of_string)'
    ml_type_ret = 'string list'
    unwrap = '(Py.List.to_list_map Py.String.to_string)'


class String(Type):
    names = ['str', 'string', 'unicode']
    ml_type = 'string'
    wrap = 'Py.String.of_string'
    ml_type_ret = 'string'
    unwrap = 'Py.String.to_string'
    is_type = 'Wrap_utils.isinstance Wrap_utils.string'

    def tag_name(self):
        return tag("S")


# class ArrayLike(Type):
#     names = ['list-like', 'list']


class NoneValue(Type):
    names = ['None', 'none']

    def tag(self):
        return '`None', '`None -> Py.none', None


class TrueValue(Type):
    names = ['True', 'true']

    def tag(self):
        return '`True', '`True -> Py.Bool.t', None


class FalseValue(Type):
    names = ['False', 'false']

    def tag(self):
        return '`False', '`False -> Py.Bool.f', None


class RandomState(Type):
    names = ['RandomState instance', 'instance of RandomState', 'RandomState']


class LinearOperator(Type):
    names = ['LinearOperator']


class CrossValGenerator(Type):
    names = ['cross-validation generator']


class Estimator(Type):
    names = [
        'instance BaseEstimator', 'BaseEstimator instance',
        'estimator instance', 'instance estimator', 'estimator object',
        'estimator'
    ]


class JoblibMemory(Type):
    names = ['object with the joblib.Memory interface']


class PyObject(Type):
    names = ['object']


class Self(Type):
    names = ['self']
    ml_type = 't'
    ml_type_ret = 't'


class Dtype(Type):
    names = ['type', 'dtype', 'numpy dtype']


class TypeList(Type):
    names = ['list of type', 'list of types']


class Dict(Type):
    # XXX dict of numpy (masked) ndarrays is the format of a
    # DataFrame, we could have a more appropriate OCaml type for it
    # maybe?
    names = [
        'dict', 'Dict', 'dictionary', 'mapping of string to any',
        'dict of numpy (masked) ndarrays'
    ]
    ml_type = 'Sklearn.Dict.t'
    ml_type_ret = 'Sklearn.Dict.t'
    wrap = 'Sklearn.Dict.to_pyobject'
    unwrap = 'Sklearn.Dict.of_pyobject'
    is_type = 'Wrap_utils.isinstance Wrap_utils.dict'


class ParamGridDict(Type):
    names = []
    ml_type = '(string * [`Ints of int list | `Floats of float list | `Strings of string list]) list'
    wrap = '(fun x -> Sklearn.Dict.(of_param_grid_alist x |> to_pyobject))'

    def tag_name(self):
        return 'Grid'


# emitted as a postprocessing of Dict() based on param name/callable
class DictIntToFloat(Type):
    names = ['dict int to float']
    ml_type = '(int * float) list'
    wrap = '(Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float)'
    is_type = 'Wrap_utils.isinstance Wrap_utils.dict'


class Callable(Type):
    names = ['callable', 'function']


def Slice():
    int_or_none = Enum([NoneValue(), Int()])
    ret = Tuple([int_or_none, int_or_none, int_or_none])
    destruct = "(`Slice _) as s -> Wrap_utils.Slice.of_variant s"
    ret.tag = lambda: (
        f"`Slice of ({int_or_none.ml_type}) * ({int_or_none.ml_type}) * ({int_or_none.ml_type})",
        destruct, None)
    return ret


def deindent(line):
    m = re.match('^( *)(.*)$', line)
    assert m is not None
    return len(m.group(1)), m.group(2)


def parse(doc):
    lines = doc.split("\n")
    elements = []
    element = []
    previous_indent = None
    first_line, *lines = lines
    for line in lines:
        indent, line = deindent(line)
        if previous_indent is None:
            previous_indent = indent

        if not line or indent < previous_indent:
            elements.append(Paragraph("\n".join(element)))
            element = []
            previous_indent = None
            if line:
                element.append(line)
        elif re.match('^-+$', line):
            block, title = element[:-1], element[-1]
            if block:
                elements.append(Paragraph("\n".join(block)))
            elements.append(Section_title(title))
            element = []
            previous_indent = None
        else:
            element.append(line)
    if element:
        elements.append(element)

    return elements


def parse_params(doc, section='Parameters'):
    elements = parse(doc)
    # if section not in doc:
    #     print(f"no section {section} in doc")
    # if section == 'Returns':
    #     print(elements)
    in_section = False
    params = []
    for elt in elements:
        if isinstance(elt, Section_title):
            if elt.text == section:
                in_section = True
                continue
            else:
                if in_section:
                    break
        if in_section:
            params.append(elt)
            # if section == 'Returns':
            #     print(f"params for section {section}: {params}")
    return params


def remove_default(text):
    text = re.sub(r'\s*\(default\)', '', text)
    text = re.sub(r'\s*\(if [^()]+\)', '', text)
    text = re.sub(r'[Dd]efaults\s+to\s+\S+\.?', '', text)
    text = re.sub(r'[Dd]efault\s+is\s+\S+\.?', '', text)
    text = re.sub(r'\(?\s*[Dd]efault\s*[:=]?\s*.+\)?$', '', text)
    text = re.sub(r'\s*optional', '', text)
    text = re.sub(r'(,\s*)?\S+\s+by\s+default', '', text)
    return text


def remove_shape(text):
    # print(f"remove_shape 0: {text}")
    text = re.sub(
        r'(of\s+)?shape\s*[:=]?\s*[([](\S+,\s*)*\S+?\s*[\])](\s*,?\s*or\s*[([](\S+,\s*)*\S+?\s*[\])])*',
        '', text)
    # print(f"remove_shape 1: {text}")
    # two levels of parentheses should be enough for anyone
    text = re.sub(r'(of\s+)?shape\s*[:=]?\s*\([^()]*(\([^()]*\)[^()]*)?\)', '',
                  text)
    text = re.sub(r'(,\s*)?(of\s+)?length \S+', '', text)
    text = re.sub(r"""(,\s*)?\[[^'"[\]()]+,[^'"[\]()]+\]""", '', text)
    text = re.sub(r'if\s+\S+\s*=+\s*\S+\s*', '', text)
    text = re.sub(r'(,\s*)?\s*\d-dimensional\s*', '', text)
    return text


def is_string(x):
    if not x:
        return False
    return (x[0] == x[-1] and x[0] in ("'", '"')) or x in [
        "eigen'"
    ]  # hack for doc bug of RidgeCV


def is_int(x):
    return re.match(r'^\d+$', x)


def parse_enum(t):
    """Love.
    """
    if not t:
        return None
    # print(f"parse_enum: {t}")
    elts = None
    m = re.match(r'^str(?:ing)?\s*(?:,|in|)\s*(\{.*\}|\[.*\]|\(.*\))$', t)
    if m is not None:
        return parse_enum(m.group(1))
    t = re.sub(r'(?:str(?:ing)?\s*in\s*)?(\{[^}]+\})',
               lambda m: re.sub(r'[,|]|\s+or\s+', ' __OR__ ', m.group(1)), t)
    t = re.sub(r'(?:str(?:ing)?\s*in\s*)?(\[[^}]+\])',
               lambda m: re.sub(r'[,|]|\s+or\s+', ' __OR__ ', m.group(1)), t)
    t = re.sub(r'(?:str(?:ing)?\s*in\s*)?(\([^}]+\))',
               lambda m: re.sub(r'[,|]|\s+or\s+', ' __OR__ ', m.group(1)), t)
    # if '__OR__' in t:
    #     print(f"replaced , with __OR__: {t}")
    assert 'str in ' not in t, t
    if (t[0] == '{' and t[-1] == '}') or (t[0] == '[' and t[-1] == ']') or (
            t[0] == '(' and t[-1] == ')'):
        elts = re.split(',|__OR__', t[1:-1])
    elif '|' in t:
        elts = t.split('|')
    elif re.search(r'\bor\s+', t) is not None:
        elts = list(re.split(r',|\bor\s+', t))
    elif ',' in t:
        elts = t.split(',')
    elif ' __OR__ ' in t:
        elts = t.split('__OR__')
    else:
        return None
    elts = [x.strip() for x in elts]
    elts = [x for x in elts if x]
    # print("parse_enum returns:", elts)
    return elts


transformer_list = List(Tuple([String(), PyObject()]))
transformer_list.names = [
    'list of (string, transformer) tuples', 'list of (str, estimator)',
    'list of (str, estimator) tuples'
]


def init_builtins(builtin, builtin_types):
    builtin.clear()
    for t in builtin_types:
        for name in t.names:
            assert name not in builtin
            builtin[name] = t


builtin_types = [
    Int(),
    Float(),
    Bool(),
    Arr(),
    Ndarray(),
    FloatList(),
    StringList(),
    ArrayList(),
    SparseMatrix(),
    String(),
    Dict(),
    DictIntToFloat(),
    Callable(),
    # Iterable(),
    Dtype(),
    TypeList(),
    # ArrayLike(),
    NoneValue(),
    TrueValue(),
    FalseValue(),
    RandomState(),
    LinearOperator(),
    CrossValGenerator(),
    Estimator(),
    JoblibMemory(),
    PyObject(),
    Self(),
    Pipeline(),
    FeatureUnion(),
    transformer_list
]
builtin = {}
init_builtins(builtin, builtin_types)


def partition(l, pred):
    sat = []
    unsat = []
    for x in l:
        if pred(x):
            sat.append(x)
        else:
            unsat.append(x)
    return sat, unsat


# this does not cover all cases it seems, where different instances of
# the same class with the same params are considered different?
# using elt.__class__ is appealing but would break StringValue for instance.
def remove_duplicates(elts):
    got = set()
    ret = []
    for elt in elts:
        if elt not in got:
            ret.append(elt)
            got.add(elt)
    return ret


def simplify_arr(enum):
    arr, not_arr = partition(enum.elements,
                             lambda x: isinstance(x, (Arr, Ndarray)))
    sparse, not_arr_not_sparse = partition(
        not_arr, lambda x: isinstance(x, SparseMatrix))
    if arr and sparse:
        return type(enum)([Arr()] + not_arr_not_sparse)
    else:
        return enum


def simplify_enum(enum):
    # flatten (once should be enough?)
    elts = []
    for elt in enum.elements:
        if isinstance(elt, Enum):
            elts += elt.elements
        else:
            elts.append(elt)
    enum = type(enum)(elts)

    false_true, not_false_true = partition(
        enum.elements, lambda x: isinstance(x, (FalseValue, TrueValue)))
    # assert len(false_true) in [0, 2], enum
    # There are (probably legitimate) cases where False is accepted
    # but not True!
    if len(false_true) == 2:
        enum = Enum([builtin['bool']] + not_false_true)

    none, not_none = partition(enum.elements,
                               lambda x: isinstance(x, NoneValue))
    assert len(none) <= 1, enum
    if none:
        # enum = EnumOption(not_none)
        enum = type(enum)(not_none + [NoneValue()])

    # Enum(String, StringValue, StringValue) should just be
    # Enum(StringValue, StringValue) since that is (probably) a
    # misparsing of "str, 'l1' or 'l2'"
    is_string_value, is_not_string_value = partition(
        enum.elements, lambda x: isinstance(x, StringValue))
    if is_string_value and len(is_not_string_value) == 1 and isinstance(
            is_not_string_value[0], String):
        enum = type(enum)(is_string_value)

    enum = type(enum)(remove_duplicates(enum.elements))

    # Arr | SparseMatrix == Arr
    enum = simplify_arr(enum)

    # There is no point having more than one Py.Object tag in an enum.
    is_obj, is_not_obj = partition(
        enum.elements, lambda x: isinstance(x, (PyObject, UnknownType)))
    if len(is_obj) > 1:
        enum = type(enum)(is_not_obj + [PyObject()])

    # A one-element enum is just the element itself.
    if len(enum.elements) == 1:
        return enum.elements[0]

    return enum


def remove_ranges(t):
    # print(f"remove_ranges: {t}")
    t = re.sub(r'(float|int)\s*(in\s*(range\s*)?)\s*[()[\]][^\]]+[()[\]]',
               r'\1', t)
    t = re.sub(r'(float|int)\s*,?\s*(greater|smaller)\s+than\s+\d+(\.\d+)?',
               r'\1', t)
    t = re.sub(r'(float|int)\s*,?\s*(<=|<|>=|>)\s*\d+(\.\d+)?', r'\1', t)
    t = re.sub(r'(float|int)\s*,?\s*\(\s*(<=|<|>=|>)\s*\d+(\.\d+)?\s*\)',
               r'\1', t)
    t = t.strip(" \n\t,.;")
    return t


def indent(s, w=4):
    has_nl = False
    if s and s[-1] == "\n":
        has_nl = True
        s = s[:-1]
    ret = re.sub(r'^', ' ' * w, s, flags=re.MULTILINE)
    if has_nl:
        ret += "\n"
    return ret


def append(container, ctor, *args, **kwargs):
    try:
        elt = ctor(*args, **kwargs)
    except (NoSignature, OutsideScope, NoDoc, Deprecated) as e:
        # print(f"WW append: caught error building {ctor}: {type(e)}({e})")
        return
    container.append(elt)


class Package:
    def __init__(self, pkg, overrides):
        self.pkg = pkg
        self.ml_name = ucfirst(self.pkg.__name__)
        self.modules = self._list_modules(pkg, overrides)

    def _list_modules(self, pkg, overrides):
        ret = []
        for mod in pkgutil.iter_modules(pkg.__path__):
            name = mod.name
            if name.startswith('_'):
                continue
            full_name = f"{pkg.__name__}.{name}"
            module = importlib.import_module(full_name)
            ret.append(
                Module(module, parent_name=self.ml_name, overrides=overrides))
        return ret

    def __str__(self):
        return repr(self)

    def __repr__(self):
        ret = f"Package({self.pkg.__name__})[\n"
        for mod in self.modules:
            ret += indent(repr(mod) + ",") + "\n"
        ret += "]"
        return ret

    def output_dir(self, path):
        # path / self.pkg.__name__
        return path

    def write(self, path):
        dire = self.output_dir(path)
        print(f"II writing package {self.pkg.__name__} to {dire}")
        dire.mkdir(parents=True, exist_ok=True)
        for mod in self.modules:
            mod.write(dire, self.ml_name)

    def write_doc(self, path):
        dire = self.output_dir(path)
        dire.mkdir(parents=True, exist_ok=True)
        for mod in self.modules:
            mod.write_doc(dire, self.ml_name)

    def write_examples(self, path):
        dire = self.output_dir(path)
        dire.mkdir(parents=True, exist_ok=True)
        for mod in self.modules:
            mod.write_examples(dire)


class OutsideScope(Exception):
    pass


class NoDoc(Exception):
    pass


class Deprecated(Exception):
    pass


def ucfirst(s):
    if not s:
        return s
    return s[0].upper() + s[1:]


def lcfirst(s):
    if not s:
        return s
    return s[0].lower() + s[1:]


# http://caml.inria.fr/pub/docs/manual-ocaml/lex.html#sss:keywords
ml_keywords = set(
    re.split(
        r'\s+', """
      and         as          assert      asr         begin       class
      constraint  do          done        downto      else        end
      exception   external    false       for         fun         function
      functor     if          in          include     inherit     initializer
      land        lazy        let         lor         lsl         lsr
      lxor        match       method      mod         module      mutable
      new         nonrec      object      of          open        or
      private     rec         sig         struct      then        to
      true        try         type        val         virtual     when
      while       with
""".strip()))


def tag(s):
    s = re.sub(r'[^a-zA-Z0-9_]+', '_', s)
    s = re.sub(r'^([^a-zA-Z])', r'T\1', s)
    return ucfirst(s)


def mlid(s):
    if s is None:
        return None
    # DESCR -> descr
    if s == s.upper():
        return s.lower()
    s = lcfirst(s)
    if s in ml_keywords:
        return s + '_'
    return s


def write_generated_header(f):
    f.write(
        "(* This file was generated by lib/skdoc.py, do not edit by hand. *)\n"
    )


class Module:
    def __init__(self, module, parent_name, overrides):
        self.full_python_name = module.__name__
        if not self.full_python_name.startswith('sklearn'):
            raise OutsideScope
        if '.externals.' in self.full_python_name:
            raise OutsideScope

        self.parent_name = parent_name
        self.python_name = module.__name__.split('.')[-1]
        self.ml_name = ucfirst(self.python_name)
        if parent_name:
            self.full_ml_name = f"{parent_name}.{self.ml_name}"
        else:
            self.full_ml_name = self.ml_name
        # print(f"building module {self.full_python_name}")
        self.module = module
        self.elements = self._list_elements(module, overrides)

    def _list_elements(self, module, overrides):
        elts = []
        for name in dir(module):
            qualname = f"{self.full_python_name}.{name}"
            if name.startswith('_'):
                continue
            # emit this one separately, not inside sklearn.metrics
            if name == "csr_matrix":
                continue
            item = getattr(module, name)
            parent_name = self.full_ml_name
            if inspect.ismodule(item):
                append(elts, Module, item, parent_name, overrides)
            elif inspect.isclass(item):
                if not inspect.isabstract(item):
                    append(elts, Class, item, parent_name, overrides)
            elif callable(item):
                append(elts, Function, name, qualname, item, overrides)
        return elts

    def __str__(self):
        return repr(self)

    def __repr__(self):
        ret = f"Module(self.full_python_name)[\n"
        for elt in self.elements:
            ret += indent(repr(elt) + ",") + "\n"
        ret += "]"
        return ret

    def has_callables(self):
        for elt in self.elements:
            if isinstance(elt, (Function, Class)):
                return True
        return False

    def write_header(self, f):
        # if self.has_callables():
        f.write("let () = Wrap_utils.init ();;\n")
        f.write(f'let ns = Py.import "{self.module.__name__}"\n\n')
        # else:
        #     f.write(
        #         "(* this module has no callables, skipping init and ns *)\n")

    def write(self, path, module_path):
        module_path = f"{module_path}.{self.ml_name}"
        ml = f"{path / self.python_name}.ml"
        with open(ml, 'w') as f:
            self.write_ml_inside(f, module_path)
        mli = f"{path / self.python_name}.mli"
        with open(mli, 'w') as f:
            self.write_mli_inside(f, module_path)

    def write_ml_inside(self, f, module_path):
        self.write_header(f)
        f.write("let get_py name = Py.Module.get ns name\n")
        for element in self.elements:
            element.write_to_ml(f, module_path)

    def write_mli_inside(self, f, module_path):
        f.write(
            "(** Get an attribute of this module as a Py.Object.t. This is useful to pass a Python function to another function. *)\n"
        )
        f.write("val get_py : string -> Py.Object.t\n\n")
        for element in self.elements:
            element.write_to_mli(f, module_path)

    def write_doc(self, path, module_path):
        if module_path:
            module_path = f"{module_path}.{self.ml_name}"
        else:
            module_path = self.ml_name
        md = f"{path / self.python_name}.md"
        with open(md, 'w') as f:
            for element in self.elements:
                element.write_to_md(f, module_path)

    def write_examples(self, path):
        md = f"{path / self.python_name}.ml"
        with open(md, 'w') as f:
            for element in self.elements:
                element.write_examples_to(f)

    def write_to_ml(self, f, module_path):
        module_path = f"{module_path}.{self.ml_name}"
        f.write(f"module {self.ml_name} = struct\n")
        self.write_ml_inside(f, module_path)
        f.write("\nend\n")

    def write_to_mli(self, f, module_path):
        module_path = f"{module_path}.{self.ml_name}"
        f.write(f"module {self.ml_name} : sig\n")
        self.write_mli_inside(f, module_path)
        f.write("\nend\n\n")

    def write_to_md(self, f, module_path):
        if module_path:
            module_path = f"{module_path}.{self.ml_name}"
        else:
            module_path = self.ml_name
        if self.parent_name:
            full_name = f"{self.parent_name}.{self.ml_name}"
        else:
            full_name = self.ml_name
        full_name = re.sub(r'\.', ".\u200b", full_name)
        f.write(f"## module {full_name}\n")
        for element in self.elements:
            element.write_to_md(f, module_path)

    def write_examples_to(self, f):
        full_name = f"{self.parent_name}.{self.ml_name}"
        f.write(f"(*--------- Examples for module {full_name} ----------*)\n")
        for element in self.elements:
            element.write_examples_to(f)


def is_hashable(x):
    try:
        hash(x)
    except TypeError:
        return False
    return True


# Major Major
class Class:
    def __init__(self, klass, parent_name, overrides):
        self.klass = klass
        self.parent_name = parent_name
        self.constructor = Ctor(self.klass.__name__, self.klass.__name__,
                                self.klass, overrides)
        self.elements = self._list_elements(overrides)
        self.ml_name = ucfirst(self.klass.__name__)

    def _list_elements(self, overrides):
        elts = []

        # Parse attributes first, so that we avoid warning about them
        # when listing methods.
        attributes = parse_types_simple(self.klass.__doc__,
                                        section="Attributes")
        attributes = overrides.types(self.klass.__name__, attributes)

        # There may be several times the same function behind the same
        # name. Track elements already seen.
        callables = set()

        # Some classes don't have all the methods ready, you
        # need to instantiate an object. For instance,
        # sklearn.neighbors.LocalOutlierFactor has fit_predict as a
        # property (not yet a method), so taking its signature does
        # not work.  So we attempt to instantiate an object and work
        # on that, and fallback on the class if that fails.
        proto = overrides.proto(self.klass)
        if proto is None:
            try:
                proto = self.klass()
            except FutureWarning:
                raise Deprecated()
            except Exception as e:
                # print(f"WW cannot create proto for {self.klass.__name__}: {e}")
                # print(f"WW could not create proto for {self.klass.__name__}")
                proto = self.klass

        for name in dir(proto):
            if name in attributes:
                continue
            # Build our own qualname instead of relying on
            # item.__name__/item.__qualname__, which are unreliable.
            qualname = f"{self.klass.__name__}.{name}"
            if name.startswith('_') and name not in ['__getitem__']:
                continue
            try:
                item = getattr(proto, name, None)
            except FutureWarning:
                # This is deprecated, skip it.
                continue
            if item is None:
                continue
            # print(f"evaluating possible method {name}: {item}")
            if callable(item):
                if inspect.isclass(item):
                    # print(f"WW skipping class member {item} of proto {proto}")
                    continue
                if is_hashable(item):
                    if item not in callables:
                        append(elts, Method, name, qualname, item, overrides)
                        callables.add(item)
                else:
                    append(elts, Method, name, qualname, item, overrides)
            elif overrides.has_complete_spec(qualname):
                # Some methods show up as properties on the class, and
                # are present only in some instantiations (for example
                # predict, inverse_transform on Pipeline). This is a
                # way to have them show up: let overrides specify
                # everything.
                # print(f"II wrapping non-callable method {name}: {item}")
                elts.append(Method(name, qualname, item, overrides))
            elif isinstance(item, property):
                print(f"WW not wrapping member {qualname}: {item}")
            else:
                pass

        for name, ty in attributes.items():
            append(elts, Attribute, name, ty)
        return elts

    def __str__(self):
        return repr(self)

    def __repr__(self):
        ret = f"class({self.klass.__name__})[\n"
        ret += indent(repr(self.constructor) + ",") + "\n"
        for elt in self.elements:
            ret += indent(repr(elt) + ",") + "\n"
        ret += "]"
        return ret

    def write_header(self, f):
        f.write("type t = Py.Object.t\n")
        f.write("let of_pyobject x = x\n")
        f.write("let to_pyobject x = x\n")

    def write(self, path, module_path, ns):
        module_path = f"{module_path}.{self.ml_name}"
        name = self.klass.__name__
        ml = f"{path / name}.ml"
        with open(ml, 'w') as f:
            f.write("let () = Wrap_utils.init ();;\n")
            f.write(f'let ns = Py.import "{ns}"\n\n')

            self.write_to_ml(f, module_path, wrap=False)

        mli = f"{path / name}.mli"
        with open(mli, 'w') as f:
            self.write_to_mli(f, module_path, wrap=False)

    def write_doc(self, path, module_path):
        module_path = f"{module_path}.{self.ml_name}"
        name = self.klass.__name__
        md = f"{path / name}.md"
        with open(md, 'w') as f:
            self.constructor.write_to_md(f, module_path)
            for element in self.elements:
                element.write_to_md(f, module_path)

    def write_to_ml(self, f, module_path, wrap=True):
        if wrap:
            f.write(f"module {self.ml_name} = struct\n")
            module_path = f"{module_path}.{self.ml_name}"
        self.write_header(f)
        self.constructor.write_to_ml(f, module_path)
        for element in self.elements:
            element.write_to_ml(f, module_path)
        f.write("let to_string self = Py.Object.to_string self\n")
        f.write('let show self = to_string self\n')
        f.write(
            'let pp formatter self = Format.fprintf formatter "%s" (show self)\n'
        )
        if wrap:
            f.write("\nend\n")

    def write_to_mli(self, f, module_path, wrap=True):
        if wrap:
            f.write(f"module {self.ml_name} : sig\n")
            module_path = f"{module_path}.{self.ml_name}"
        f.write("type t\n")
        f.write("val of_pyobject : Py.Object.t -> t\n")
        f.write("val to_pyobject : t -> Py.Object.t\n\n")
        self.constructor.write_to_mli(f, module_path)
        for element in self.elements:
            element.write_to_mli(f, module_path)
        f.write(
            "\n(** Print the object to a human-readable representation. *)\n")
        f.write("val to_string : t -> string\n\n")
        f.write(
            "\n(** Print the object to a human-readable representation. *)\n")
        f.write("val show : t -> string\n\n")
        f.write("(** Pretty-print the object to a formatter. *)\n")
        f.write(
            "val pp : Format.formatter -> t -> unit [@@ocaml.toplevel_printer]\n\n"
        )
        if wrap:
            f.write("\nend\n\n")

    def _write_fun_md(self, f, name, sig, doc):
        valsig = indent(f"val {name}: {sig}")
        doc = indent(doc)
        f.write(f"""
### {name}

???+ note "method"
    ~~~ocaml
{valsig}
    ~~~

{doc}
""")

    def write_to_md(self, f, module_path):
        if module_path:
            module_path = f"{module_path}.{self.ml_name}"
        else:
            module_path = self.ml_name
        if self.parent_name:
            full_name = f"{self.parent_name}.{self.ml_name}"
        else:
            full_name = self.ml_name
        full_name = re.sub(r'\.', ".\u200b", full_name)
        f.write(f"## module {full_name}\n")
        f.write("```ocaml\n")
        f.write("type t\n")
        f.write("```\n")
        self.constructor.write_to_md(f, module_path)
        for element in self.elements:
            element.write_to_md(f, module_path)
        self._write_fun_md(
            f, 'to_string', 't -> string',
            'Print the object to a human-readable representation.')
        self._write_fun_md(
            f, 'show', 't -> string',
            'Print the object to a human-readable representation.')
        self._write_fun_md(f, 'pp', 'Format.formatter -> t -> unit',
                           'Pretty-print the object to a formatter.')

    def write_examples_to(self, f):
        write_examples(f, self.klass)
        for element in self.elements:
            element.write_examples_to(f)


def remove_none_from_enum(t):
    if not isinstance(t, Enum):
        return t

    def is_none(x):
        return isinstance(x, NoneValue) or (isinstance(x, StringValue)
                                            and x.text == "None")

    none, not_none = partition(t.elements, is_none)
    return simplify_enum(type(t)(not_none))


class Attribute:
    def __init__(self, name, typ):
        self.name = name
        self.ml_name = mlid(self.name)
        self.typ = remove_none_from_enum(typ)
        if self.name.endswith('_'):
            self.ml_name_opt = self.ml_name + 'opt'
        else:
            self.ml_name_opt = self.ml_name + '_opt'

    def write_to_ml(self, f, module_path):
        unwrap = _localize(self.typ.unwrap, module_path)
        # Not sure whether we should raise or return None if the attribute is not found.
        # Maybe conflating attribute not found with attribute is None is a bad idea?
        #
        # Some objects like StandardScaler specify that the attributes
        # can be None is some configurations, but maybe this is more
        # common and not all such cases are documented? In that case
        # testing for "is None" seems like a good idea. On the other hand,
        # it makes using the attributes more complicated.
        #
        # -> providing both raising + option getters
        f.write(f"""
let {self.ml_name_opt} self =
  match Py.Object.get_attr_string self "{self.name}" with
  | None -> failwith "attribute {self.name} not found"
  | Some x -> if Py.is_none x then None else Some ({unwrap} x)

let {self.ml_name} self = match {self.ml_name_opt} self with
  | None -> raise Not_found
  | Some x -> x
""")

    def write_to_mli(self, f, module_path):
        #  XXX TODO extract doc and put it here
        ml_type_ret = _localize(self.typ.ml_type_ret, module_path)
        f.write(f"""
(** Attribute {self.name}: get value or raise Not_found if None.*)
val {self.ml_name} : t -> {ml_type_ret}

(** Attribute {self.name}: get value as an option. *)
val {self.ml_name_opt} : t -> ({ml_type_ret}) option

""")

    def write_to_md(self, f, module_path):
        ml_type_ret = _localize(self.typ.ml_type_ret, module_path)
        f.write(f"""
### {self.ml_name}

???+ note "attribute"
    ~~~ocaml
    val {self.ml_name} : t -> {ml_type_ret}
    val {self.ml_name_opt} : t -> ({ml_type_ret}) option
    ~~~

    This attribute is documented in `create` above. The first version raises Not_found
    if the attribute is None. The _opt version returns an option.
""")

    def write_examples_to(self, f):
        pass

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"Attribute({self.name} : {self.typ})"


def clean_doc(doc):
    if doc is None:
        return doc
    doc = re.sub(r'\*\)', '* )', str(doc))
    doc = re.sub(r'\(\*', '( *', doc)
    doc = re.sub(r'\{\|', '{ |', doc)
    doc = re.sub(r'\|\}', '| }', doc)
    num_quotes = doc.count('"')
    if num_quotes % 2 == 1:
        doc = doc + '"'
    return doc


def format_md_doc(doc):
    doc = re.sub(r'^(.+)\n---+\n', r'\n#### \1\n\n', doc, flags=re.MULTILINE)
    # doc = re.sub(r'^(\s*)(\S+)(\s*:)', r'\1**\2**\3', doc, flags=re.MULTILINE)
    doc = re.sub(r'^(\s*)([\w*]\S*\s*:[^\n]+)',
                 r'\n???+ info "\2"',
                 doc,
                 flags=re.MULTILINE)
    doc = re.sub(r'^.. math::\n(([^\n]|\n[^\n])+)',
                 r'$$\1\n$$\n',
                 doc,
                 flags=re.MULTILINE)
    doc = re.sub(r'$', "\n\n", doc)
    doc = re.sub(r'^(>>>([^\n]|\n[^\n])+)\n\n',
                 r'```python\n\1\n```\n\n',
                 doc,
                 flags=re.MULTILINE)
    return doc


def examples(doc):
    # XXX removed a \n at the end, hoping that would work, check it
    groups = re.findall(r'^(?:>>>(?:[^\n]|\n[^\n])+)\n',
                        doc,
                        flags=re.MULTILINE)
    for group in groups:
        # ex = re.sub('^(>>>|\.\.\.)\s*', '', group, flags=re.MULTILINE)
        ex = group
        yield ex


class Input:
    def __init__(self, env):
        self.text = ''
        self.env = env

    def append(self, line):
        line = re.sub(r'^(>>>|\.\.\.)\s*', '', line)
        line.strip()
        self.text += line

    def write(self, f):
        self.text = re.sub(r'^\s*(from|import).+$',
                           '',
                           self.text,
                           flags=re.MULTILINE)

        constructions = re.findall(r'^\s*([a-w]\w+)\s*=\s*([A-Z]\w+)',
                                   self.text,
                                   flags=re.MULTILINE)
        for construction in constructions:
            var, klass = [x.strip() for x in construction]
            self.env[var] = klass

        def format_ml_args(args):
            ret = re.sub(r'(,\s*)?(\w+)=([^=])', r' ~\2:\3', args)
            ret = re.sub(r'(^|,\s*)(\w+)(\s*,|$)', r'\1~\2\3', ret)
            ret = re.sub(r'\s*,\s*', ' ', ret)
            ret = ret.strip()
            return ret

        def replace_ctor(m):
            obj = m.group(1)
            klass = m.group(2)
            ml_args = format_ml_args(m.group(3))
            return f"{obj} = {klass}.create {ml_args} ()"

        def replace_method(m):
            obj = m.group(1)
            klass = ucfirst(self.env.get(obj, ''))
            method = mlid(m.group(2))
            ml_args = format_ml_args(m.group(3))
            return f"{klass}.{method} {ml_args} {obj}"

        def replace_attribute(m):
            obj = m.group(1)
            klass = ucfirst(self.env.get(obj, ''))
            method = mlid(m.group(2))
            return f"{klass}.{method} {obj}"

        def replace_function(m):
            fun = m.group(1)
            ml_args = format_ml_args(m.group(2))
            return f"{fun} {ml_args} ()"

        def replace_array(m):
            expr = m.group(0)
            if re.search(r'^\s*\[\s*\[', expr):
                wrap = 'matrix'
            else:
                wrap = 'vector'
            expr = re.sub(r'\[', '[|', expr)
            expr = re.sub(r'\]', '|]', expr)
            expr = re.sub(r',', ';', expr)
            if '.' not in expr:
                wrap = f"{wrap}i"
            return f"({wrap} {expr})"

        # Array.
        self.text = re.sub(
            r'\[([^[\],]|\s*\[[^[\]]*\])(\s*,\s*[^[\],]|\s*\[[^[\]]*\])*\]',
            replace_array, self.text)

        self.text = re.sub(r'False', 'false', self.text)
        self.text = re.sub(r'True', 'true', self.text)
        self.text = re.sub(r'\bX\b', 'x', self.text)

        # Constructor.
        self.text = re.sub(
            r'^\s*(\w+)\s*=\s*([A-Z][a-zA-Z_]+)\(([^()]*)\)\s*$',
            replace_ctor,
            self.text,
            flags=re.MULTILINE)

        # Method call.
        self.text = re.sub(r'\b([a-z][a-zA-Z_]+)\.(\w+)\((.*)\)',
                           replace_method, self.text)

        # Attribute.
        self.text = re.sub(r'\b([a-z][a-zA-Z_]+)\.(\w+)$', replace_attribute,
                           self.text)

        # Function call.
        self.text = re.sub(r'\b([a-z][a-zA-Z_]+)\(([^()]*)\)',
                           replace_function, self.text)

        self.text = re.sub(r'^\s*([\w,\s]+)\s*=([^=].*)$',
                           r'let \1 = \2 in',
                           self.text,
                           flags=re.MULTILINE)
        if self.text and not self.text.endswith('in'):
            m = re.search(r'^([A-Z]\w+)\.', self.text)
            if m is not None and '.fit ' in self.text or '.create ' in self.text:
                self.text = f"print {m.group(1)}.pp @@ {self.text}"
            else:
                self.text = f"print_ndarray @@ {self.text}"
            self.text += ';'

        # whitespace
        self.text = re.sub(r'  +', ' ', self.text)
        self.text = re.sub(r'^\s*$', "", self.text, flags=re.MULTILINE)

        if self.text:
            f.write(self.text)
            f.write("\n")


class Output:
    def __init__(self):
        self.lines = []

    def append(self, line):
        self.lines.append(line)

    def _clean_lines(self, lines):
        lines = [line.rstrip() for line in lines]
        while lines and not lines[-1]:
            lines = lines[:-1]
        return lines

    def write(self, f):
        self.lines = self._clean_lines(self.lines)
        f.write("[%expect {|\n")
        ff = Indented(f)
        for line in self.lines:
            ff.write(line)
            ff.write("\n")
        f.write("|}];\n")


class Indented:
    def __init__(self, f, indent=2):
        self.f = f
        self.indent = ' ' * indent

    def write(self, x):
        self.f.write(self.indent)
        self.f.write(x)


class Example:
    def __init__(self, source, name):
        self.name = name
        self.elements = []
        self.env = {}
        for line in source.split("\n"):
            if line.startswith('>>>'):
                element = Input(self.env)
                element.append(line)
                self.elements.append(element)
            elif line.startswith('...'):
                self.elements[-1].append(line)
            else:
                if not self.elements or not isinstance(self.elements[-1],
                                                       Output):
                    self.elements.append(Output())
                self.elements[-1].append(line)

    def write(self, f):
        module = ucfirst(os.path.splitext(os.path.split(f.name)[-1])[0])
        f.write("(* TEST TODO\n")
        f.write(f'let%expect_test "{self.name}" =\n')
        f.write(f"  let open Sklearn.{module} in\n")
        for element in self.elements:
            element.write(Indented(f))
        f.write("\n*)\n\n")


def qualname(obj):
    try:
        return obj.__qualname__
    except AttributeError:
        try:
            return obj.__name__
        except AttributeError:
            return "<no name>"


def write_examples(f, obj):
    doc = inspect.getdoc(obj)
    if not doc:
        return
    name = getattr(obj, '__name__', '<no name>')
    for example in examples(doc):
        f.write(f"(* {name} *)\n")
        f.write("(*\n")
        f.write(example)
        f.write("\n*)\n\n")
        example_ml = Example(example, qualname(obj))
        example_ml.write(f)
        f.write("\n\n")


def make_return_type(type_dict):
    if not type_dict:
        return builtin['object']
    if len(type_dict) == 1:
        if list(type_dict.keys())[0] == 'self':
            return builtin['self']
        return list(type_dict.values())[0]
    return Tuple(list(type_dict.values()))


class NoSignature(Exception):
    pass


def _localize(t, module_path):
    """Attempt to localize a type or expression by removing the necessary
    parts of paths. Working on strings is inherently flawed, the right
    way to do this would be to pass the module path to the things that
    generate t, so that they can generate clean types and
    expressions. Anyway, seems to work in practice.

    """
    path_elts = module_path.split('.')
    ret = t

    for i in range(len(path_elts), 0, -1):
        path = r'\.'.join(path_elts[:i]) + r'\.'
        ret = re.sub(path, '', ret)

    # if "Csr_matrix" in t:
    #     print(f"localize: {t} / {module_path} -> {ret}")
    return ret


class Parameter:
    def __init__(self, python_name, ty, parameter):
        self.python_name = python_name
        self.ml_name = mlid(self.python_name)
        self.ty = ty
        self.parameter = parameter
        self.fixed_value = None
        # If the default value is None, no need to have it as an option in the type.
        if self.parameter.default is None:
            self.ty = remove_none_from_enum(self.ty)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        mark = ''
        if self.is_star():
            mark = '*'
        elif self.is_star_star():
            mark = '**'

        fixed = ''
        if self.fixed_value is not None:
            fixed = f'=>{self.fixed_value}'

        default = ''
        if self.has_default():
            default = f'={self.parameter.default}'

        return f"{mark}{self.python_name}{default}{fixed} : {self.ty}"

    def _has_fixed_value(self):
        return self.fixed_value is not None

    def is_star(self):
        return self.parameter.kind == inspect.Parameter.VAR_POSITIONAL

    def is_star_star(self):
        return self.parameter.kind == inspect.Parameter.VAR_KEYWORD

    def is_named(self):
        return not (self.ml_name in ['self'] or self.is_star())

    def has_default(self):
        return ((self.parameter.default is not inspect.Parameter.empty)
                or self.is_star_star())

    def sig(self, module_path):
        if self._has_fixed_value():
            return None
        ml_type = _localize(self.ty.ml_type, module_path)
        if self.is_named():
            spec = f"{self.ml_name}:{ml_type}"
            if self.has_default():
                spec = f"?{spec}"
            return spec
        else:
            return ml_type

    def decl(self):
        if self._has_fixed_value():
            return None
        spec = f"{self.ml_name}"
        if self.has_default():
            spec = f"?{spec}"
        elif self.is_named():
            spec = f"~{spec}"
        return spec

    def call(self, module_path):
        """Return a tuple:
        - None or (name, value getter)
        - None or *args param name
        - None or **kwargs param name
        """

        ty_wrap = _localize(self.ty.wrap, module_path)

        if ty_wrap == 'Wrap_utils.id':
            pipe_ty_wrap = ''
        else:
            pipe_ty_wrap = f'|> {ty_wrap}'

        if self._has_fixed_value():
            wrap = f'Some({self.fixed_value} {pipe_ty_wrap})'
        elif self.has_default():
            if ty_wrap == 'Wrap_utils.id':
                wrap = self.ml_name
            else:
                wrap = f'Wrap_utils.Option.map {self.ml_name} {ty_wrap}'
        else:
            wrap = f'Some({self.ml_name} {pipe_ty_wrap})'

        kv = (self.python_name, wrap)

        pos_arg = None
        if self.is_star():
            ty_t_wrap = _localize(self.ty.t.wrap, module_path)
            pos_arg = f"(Wrap_utils.pos_arg {ty_t_wrap} {self.ml_name})"
            kv = None

        kw_arg = None
        if self.is_star_star():
            kw_arg = f"(match {self.ml_name} with None -> [] | Some x -> x)"
            kv = None

        return kv, pos_arg, kw_arg


class DummyUnitParameter:
    python_name = ''
    ty = None

    def is_named(self):
        return False

    def has_default(self):
        return False

    def sig(self, module_path):
        return 'unit'

    def decl(self):
        return '()'

    def call(self, module_path):
        return None, None, None

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return '()'


class SelfParameter:
    python_name = 'self'
    ty = Self()

    def is_named(self):
        return False

    def has_default(self):
        return False

    def sig(self, module_path):
        return 't'

    def decl(self):
        return 'self'

    def call(self, module_path):
        return None, None, None

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return str(self.ty)


class Return:
    def __init__(self, ty):
        self.ty = ty

    def sig(self, module_path):
        return _localize(self.ty.ml_type_ret, module_path)

    def call(self, module_path):
        return _localize(self.ty.unwrap, module_path)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return str(self.ty)


class Wrapper:
    def __init__(self, python_name, ml_name, doc, parameters, ret, doc_type,
                 namespace):
        self.parameters = self._fix_parameters(parameters)
        self.ret = ret
        self.python_name = python_name
        self.ml_name = ml_name
        self.doc = clean_doc(doc)
        self.doc_type = doc_type
        self.namespace = namespace

    def _fix_parameters(self, parameters):
        """Put not-named parameters at the end, and add a dummy unit parameter
        at the end if needed.

        """
        named, not_named = partition(parameters, lambda x: x.is_named())
        with_default, no_default = partition(named, lambda x: x.has_default())
        parameters = with_default + no_default + not_named
        if not not_named:
            # we have only named params with a default
            parameters.append(DummyUnitParameter())
        return parameters

    def _join_not_none(self, sep, elements):
        elements = filter(lambda x: x is not None, elements)
        return sep.join(elements)

    def mli(self, module_path, doc=True):
        sig = self._join_not_none(
            ' -> ', [x.sig(module_path)
                     for x in self.parameters] + [self.ret.sig(module_path)])
        if doc:
            return f"val {self.ml_name} : {sig}\n(**\n{self.doc}\n*)\n\n"
        else:
            return f"val {self.ml_name} : {sig}\n"

    def md(self, module_path):
        sig = self._join_not_none(
            " ->\n", [x.sig(module_path)
                      for x in self.parameters] + [self.ret.sig(module_path)])
        sig = indent(sig, 6)
        if self.doc is None:
            doc = ''
        else:
            doc = format_md_doc(self.doc)
        doc = indent(doc)
        return f"""
### {self.ml_name}

???+ note "{self.doc_type}"
    ~~~ocaml
    val {self.ml_name} :
{sig}
    ~~~

{doc}
"""

    def _pos_args(self, module_path):
        pos_args = None
        for param in self.parameters:
            kv, pos, kw = param.call(module_path)

            if pos is not None:
                assert pos_args is None, \
                    f"function has several *args: {pos_args}, {pos}"
                pos_args = pos

        if pos_args is None:
            pos_args = '[||]'

        return pos_args

    def _kw_args(self, module_path):
        pairs = []
        kwargs = None
        for param in self.parameters:
            kv, _pos, kw = param.call(module_path)
            if kv is not None:
                k, v = kv
                pairs.append(f'("{k}", {v})')
            if kw is not None:
                assert kwargs is None, f"wrapper has several **kwargs: {kwargs}, {kw}"
                kwargs = kw
        if pairs:
            ret = f'(Wrap_utils.keyword_args [{"; ".join(pairs)}])'
            if kwargs is not None:
                ret = f'(List.rev_append {ret} {kwargs})'
        else:
            if kwargs is not None:
                ret = kwargs
            else:
                ret = '[]'
        return ret

    def ml(self, module_path):
        arg_decl = self._join_not_none(' ',
                                       [x.decl() for x in self.parameters])
        pos_args = self._pos_args(module_path)
        kw_args = self._kw_args(module_path)

        ret_call = self.ret.call(module_path)
        if ret_call == 'Wrap_utils.id':
            pipe_ret_call = ''
        else:
            pipe_ret_call = f'|> {ret_call}'

        ret = f"""\
                  let {self.ml_name} {arg_decl} =
                     Py.Module.get_function_with_keywords {self.namespace} "{self.python_name}"
                       {pos_args}
                       {kw_args}
                       {pipe_ret_call}
                  """
        return textwrap.dedent(ret)


def parse_type_simple(t, param_name):
    t = remove_ranges(t)
    t = re.sub(r'\s*\(\)\s*', '', t)

    ret = None
    try:
        ret = builtin[t]
    except KeyError:
        pass

    if ret is not None:
        if isinstance(ret, Dict):
            if param_name in ['class_weight']:
                ret = builtin['dict int to float']
        if isinstance(ret, Arr):
            if param_name in ['filenames']:
                ret = builtin['list of string']
        if isinstance(ret, (Arr)):
            if param_name in ['feature_names', 'target_names']:
                ret = builtin['list of string']
            if param_name in ['neigh_ind', 'is_inlier']:
                ret = builtin['ndarray']
        return ret

    elts = parse_enum(t)
    if elts is not None:
        if all([is_string(elt) for elt in elts]):
            return Enum([StringValue(e.strip("\"'")) for e in elts])
        elts = [parse_type_simple(elt, param_name) for elt in elts]
        # print("parsed enum elts:", elts)
        ret = Enum(elts)
        ret = simplify_enum(ret)
        return ret
    elif is_string(t):
        return StringValue(t.strip("'\""))
    elif is_int(t):
        return IntValue(t)
    else:
        print(f"WW failed to parse type: {t}")
        return UnknownType(t)


def parse_types_simple(doc, section='Parameters'):
    if doc is None:
        return {}
    elements = parse_params(doc, section)

    ret = {}
    for element in elements:
        try:
            text = element.text.strip()
            if not text or text.startswith('..'):
                continue
            m = re.match(r'^(\w+)\s*:\s*([^\n]+)', text)
            if m is None:
                mm = re.match(r'^(\w+)\s*\n', text)
                if mm is not None:
                    ret[mm.group(1)] = UnknownType(text)
                    continue
                continue
            param_name = m.group(1)

            value = m.group(2)
            value = remove_default(value)
            value = remove_shape(value)
            value = value.strip(" \n\t,.;")

            if value.startswith('ref:'):
                continue

            ty = parse_type_simple(value, param_name)
            ret[param_name] = ty

        except Exception as e:
            print(f"!!!!!!!!!!!!! error processing element: {element}")
            print(e)
            raise
    return ret


def parse_bunch_fetch(elements):
    ret = {}
    for element in elements:
        for line in element.text.split("\n"):
            m = re.match(
                r"^\s*-?\s*'?(?:data\.|dataset\.|bunch\.)?(\w+)'?\s*[:,]\s*(.*)$",
                line)
            if m is not None:
                name = m.group(1)
                if name in ['bunch', 'dataset']:
                    continue
                value = m.group(2)
                value = remove_shape(value)
                value = value.strip(" \n\t,.;")
                ret[name] = parse_type_simple(value, name)
    ret = Bunch(ret)
    return ret


def parse_bunch_load(elements):
    text = ' '.join(e.text for e in elements)
    attributes = re.findall(r"'([^']+)'", text)
    print(attributes)
    types = dict(data=Arr(),
                 target=Arr(),
                 data_filename=String(),
                 target_filename=String(),
                 DESCR=String(),
                 filename=String(),
                 target_names=Arr(),
                 feature_names=Arr(),
                 images=Arr(),
                 filenames=Arr())
    return Bunch({k: types[k] for k in attributes})


def parse_bunch_simple(doc, section='Returns'):
    elements = parse_params(doc, section)

    if 'attributes are' in elements[0].text or (
            len(elements) > 1 and 'attributes are' in elements[1].text):
        return parse_bunch_load(elements)
    else:
        return parse_bunch_fetch(elements)


class Function:
    def __init__(self,
                 python_name,
                 qualname,
                 function,
                 overrides,
                 namespace='ns'):
        self.overrides = overrides
        self.function = function
        self.qualname = qualname
        doc = inspect.getdoc(function)
        raw_doc = getattr(function, '__doc__', '')
        signature = self._signature()
        fixed_values = overrides.fixed_values(self.qualname)
        parameters = self._build_parameters(signature, raw_doc, fixed_values)
        ret = self._build_ret(signature, raw_doc, fixed_values)
        self.wrapper = Wrapper(python_name, self._ml_name(python_name), doc,
                               parameters, ret, "function", namespace)

    def _ml_name(self, python_name):
        # warning: all overrides are resolved based on the function
        # qualname, not python_name (which may in some rare cases be
        # different)
        ml = self.overrides.ml_name(self.qualname)
        if ml is not None:
            return ml
        return mlid(python_name)

    def _signature(self):
        sig = self.overrides.signature(self.qualname)
        if sig is not None:
            return sig
        try:
            return inspect.signature(self.function)
        except ValueError:
            raise NoSignature(self.function)

    def _build_parameters(self, signature, doc, fixed_values):
        param_types = self.overrides.param_types(self.qualname)
        if param_types is None:
            param_types = parse_types_simple(doc, section='Parameters')
            # Make sure all params are in param_types, so that the
            # overrides trigger (even in case the params were not
            # found parsing the doc).
            for k in signature.parameters.keys():
                if k not in param_types:
                    param_types[k] = UnknownType(
                        f"{k} not found in parsed types {param_types}")
            param_types = self.overrides.types(self.qualname, param_types)

        parameters = []
        for python_name, parameter in signature.parameters.items():
            ty = param_types[python_name]
            param = Parameter(python_name, ty, parameter)
            if python_name in fixed_values:
                param.fixed_value = fixed_values[python_name][0]
            if param.is_star_star():
                if not isinstance(param.ty, UnknownType):
                    print(
                        f"WW overriding {param} to have type (string * Py.Object.t) list"
                    )
                param.ty = StarStar()
            if param.is_star():
                if not isinstance(param.ty, List):
                    if not isinstance(param.ty, UnknownType):
                        print(
                            f"WW overriding {param} to have type List(PyObject())"
                        )
                    param.ty = List(PyObject())

            parameters.append(param)
        return parameters

    def _build_ret(self, signature, doc, fixed_values):
        ret_type = self.overrides.ret_type(self.qualname)
        if ret_type is None:
            if self.overrides.returns_bunch(self.qualname):
                ret_type = parse_bunch_simple(doc, section='Returns')
            else:
                ret_type_elements = parse_types_simple(doc, section='Returns')
                for k, v in fixed_values.items():
                    if v[1] and k in ret_type_elements:
                        del ret_type_elements[k]
                ret_type = make_return_type(ret_type_elements)
        return Return(ret_type)

    def ml(self, module_path):
        return self.wrapper.ml(module_path)

    def mli(self, module_path, doc=True):
        return self.wrapper.mli(module_path, doc=doc)

    def md(self, module_path):
        return self.wrapper.md(module_path)

    def examples(self):
        f = io.StringIO()
        write_examples(f, self.function)
        return f.getvalue()

    def write_to_ml(self, f, module_path):
        f.write(self.ml(module_path))

    def write_to_mli(self, f, module_path):
        f.write(self.mli(module_path))

    def write_to_md(self, f, module_path):
        f.write(self.md(module_path))

    def write_examples_to(self, f):
        write_examples(f, self.function)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        ret = f"{type(self).__name__} {self.wrapper.python_name}{self._signature()}:\n"
        for param in self.wrapper.parameters:
            ret += f"    {param} ->\n"
        ret += f"    {self.wrapper.ret}\n"
        ret += f"  {self.mli('Sklearn', doc=False)}"
        # ret += f"  {self.ml('Sklearn')}"
        return ret


class Method(Function):
    def __init__(self, python_name, qualname, function, overrides):
        super().__init__(python_name,
                         qualname,
                         function,
                         overrides,
                         namespace='self')
        self.wrapper.doc_type = "method"

    def _build_parameters(self, *args, **kwargs):
        super_params = super()._build_parameters(*args, **kwargs)
        if getattr(self.function, '__self__', None) is None and (
                not super_params or super_params[0].python_name != 'self'):
            print(
                f'WW ignoring method, arg 1 != self: {self.qualname}{self._signature()}'
            )
            raise OutsideScope(self.function)
        params = [x for x in super_params if x.python_name != 'self']
        params.append(SelfParameter())
        return params


class Ctor(Function):
    def __init__(self, python_name, qualname, function, overrides):
        super().__init__(python_name,
                         qualname,
                         function,
                         overrides,
                         namespace='ns')
        self.wrapper.doc_type = "constructor and attributes"

    def _build_ret(self, _signature, _doc, _fixed_values):
        return Return(Self())

    def _ml_name(self, _python_name):
        return 'create'


def re_compile(regex):
    if isinstance(regex, type):
        return regex
    try:
        return re.compile(regex)
    except re.error as e:
        print(f"error compiling regex {regex}: {e}")
        raise


def compile_dict_keys(dic):
    return {re_compile(k): v for k, v in dic.items()}


class Overrides:
    def __init__(self, overrides):
        self.overrides = self._compile_regexes(overrides)
        self.triggered = set()

    def _compile_regexes(self, overrides):
        overrides = compile_dict_keys(overrides)
        for k, v in overrides.items():
            if 'types' in v:
                v['types'] = compile_dict_keys(v['types'])
        return overrides

    def _iter_function_matches(self, qualname, name):
        # print(f"overrides: {name}: searching for qualified name {qualname}")
        for re_qn, dic in self.overrides.items():
            if isinstance(re_qn, type):
                continue
            if re.search(re_qn, qualname) is not None:
                # print(f" -> matches re {re_qn}")
                self.triggered.add(re_qn.pattern)
                yield dic

    def proto(self, klass):
        try:
            ret = self.overrides[klass]['proto']
            self.triggered.add(klass)
            return ret
        except KeyError:
            return None

    def returns_bunch(self, qualname):
        for dic in self._iter_function_matches(qualname, 'ret_bunch'):
            if dic.get('ret_bunch', False):
                return True
        return False

    def signature(self, qualname):
        for dic in self._iter_function_matches(qualname, 'signature'):
            try:
                return dic['signature']
            except KeyError:
                pass
        return None

    def param_types(self, qualname):
        ret = []
        for dic in self._iter_function_matches(qualname, 'param_types'):
            if 'param_types' in dic:
                ret.append(dic['param_types'])
        assert len(ret) <= 1, ("several param_types found for function",
                               qualname, ret)
        if ret:
            return ret[0]
        return None

    def ret_type(self, qualname):
        ret = []
        for dic in self._iter_function_matches(qualname, 'ret_type'):
            if 'ret_type' in dic:
                ret.append(dic['ret_type'])
        assert len(ret) <= 1, ("several ret_type found for function", qualname,
                               ret)
        if ret:
            return ret[0]
        return None

    def types(self, qualname, param_types):
        param_types = dict(param_types)
        for dic in self._iter_function_matches(qualname, 'types'):
            # use list() to freeze the list, we will be modifying param_types
            for k in list(param_types.keys()):
                for param_re, ty in dic.get('types', {}).items():
                    if re.search(param_re, k) is not None:
                        param_types[k] = ty
        return param_types

    def fixed_values(self, qualname):
        ret = {}
        for dic in self._iter_function_matches(qualname, 'fixed_values'):
            if 'fixed_values' in dic:
                ret.update(dic['fixed_values'])
        return ret

    def ml_name(self, qualname):
        for dic in self._iter_function_matches(qualname, 'ml_name'):
            if 'ml_name' in dic:
                return dic['ml_name']
        return None

    def has_complete_spec(self, qualname):
        for dic in self._iter_function_matches(
                qualname, 'signature+param_types+ret_type'):
            if 'signature' in dic and 'param_types' in dic and 'ret_type' in dic:
                return True
        return False

    def report_not_triggered(self):
        not_triggered = set()
        for k, _v in self.overrides.items():
            if isinstance(k, re.Pattern):
                k = k.pattern
            if k not in self.triggered:
                not_triggered.add(k)
        if not_triggered:
            print(f"WW overrides: keys not triggered: {not_triggered}")


def dummy_train_test_split(*arrays,
                           test_size=None,
                           train_size=None,
                           random_state=None,
                           shuffle=True,
                           stratify=None):
    pass


train_test_split_signature = inspect.signature(dummy_train_test_split)

def dummy_inverse_transform(self, X=None, y=None):
    pass


sig_inverse_transform = inspect.signature(dummy_inverse_transform)

def dummy_shuffle(*arrays, random_state=None, n_samples=None):
    pass

sig_shuffle = inspect.signature(dummy_shuffle)

overrides = {
    r'Pipeline\.inverse_transform$':
    dict(signature=sig_inverse_transform,
         param_types={
             'self': Self(),
             'X': Arr(),
             'y': Arr()
         },
         ret_type=Arr()),
    r'__getitem__$':
    dict(ml_name='get_item'),
    r'Pipeline.__getitem__$':
    dict(param_types={
        'self': Self(),
        'ind': Enum([Int(), String(), Slice()])
    }),
    r'set_params$':
    dict(ret_type=Self()),
    r'train_test_split$':
    dict(
        signature=train_test_split_signature,
        param_types={
            'arrays': List(Arr()),
            'test_size': Enum([Float(), Int(), NoneValue()]),
            'train_size': Enum([Float(), Int(), NoneValue()]),
            'random_state': Int(),
            'shuffle': Bool(),
            'stratify': Arr()  # was Arr | None
        },
        ret_type=List(Arr())),
    r'\.shuffle$':
    dict(signature=sig_shuffle,
         param_types=dict(
             arrays=List(Arr()),
             random_state=Int(),
             n_samples=Int()
         ),
         ret_type=List(Arr())),
    r'make_regression$':
    dict(fixed_values=dict(coef=('true', False))),
    r'\.radius_neighbors$':
    # dict(ret_type=Tuple([List(Arr()), List(Arr())])),
    dict(ret_type=Tuple([ArrayList(), ArrayList()])),
    r'^NearestCentroid\.fit$':
    dict(param_types=dict(X=Arr(), y=Arr())),
    r'MultiLabelBinarizer\.(fit_transform|fit)':
    dict(types={'^y$': ArrayList()}),
    r'\.(decision_function|predict|predict_proba|fit_predict|transform|fit_transform)$':
    dict(ret_type=Arr(), types={r'^X$': Arr()}),
    r'\.fit$':
    dict(ret_type=Self()),
    r'Pipeline$':
    dict(param_types=dict(steps=transformer_list,
                          memory=Enum([NoneValue(
                          ), String(), JoblibMemory()]),
                          verbose=Bool())),
    r'load_iris$':
    dict(ret_type=Bunch({
        'data': Arr(),
        'target': Arr(),
        'target_names': Arr(),
        'feature_names': Arr(),
        'DESCR': String(),
        'filename': String()
    })),
    r'load_boston$':  # feature_names is missing from docs
    dict(ret_type=Bunch({
        'data': Arr(),
        'target': Arr(),
        'feature_names': Arr(),
        'DESCR': String(),
        'filename': String()
    })),
    r'(fetch_.*|load_(?!iris)(?!svmlight_files).*)$':
    dict(ret_bunch=True, types={
        'r^(data|target|pairs|images)$': Arr(),
    }),
    r'':
    dict(
        fixed_values=dict(return_X_y=('false', True),
                          return_distance=('true', False)),
        types={
            '^shape$': List(Int()),
            '^DESCR$': String(),
            '^target_names$': Arr(),
            '^(intercept_|coef_|classes_)$': Arr(),
            # using (`Int 42) instead of 42 is getting tiring, and
            # RandomState does not seem necessary
            '^random_state$': Int(),
            "^verbose$": Int(),
            '^named_(estimators|steps|transformers)_?$': Dict(),
            # '^y_(true|pred)$': Arr(),
        }),
    r'power_transform$':
    dict(types={
        r'^method$':
        Enum([StringValue("yeo-johnson"),
              StringValue("box-cox")])
    }),
    r'quantile_transform$':
    dict(types={
        r'^X$':
        Arr(),
        r'^output_distribution$':
        Enum([StringValue("uniform"),
              StringValue("normal")])
    },
         ret_type=Arr()),
    r'mquantiles$':
    dict(param_types=dict(a=Arr(),
                          prob=Arr(),
                          alphap=Float(),
                          betap=Float(),
                          axis=Int(),
                          limit=Tuple([Float(), Float()])),
         ret_type=Arr()),
    '\.auc$':
    dict(param_types=dict(x=Arr(), y=Arr()), ret_type=Float()),
    r'\.classification_report$':
    dict(ret_type=Enum([String(), ClassificationReport()])),
    # hamming_loss() is documented as returning int or float, but I
    # don't see how it can ever return an int.
    r'\.hamming_loss$':
    dict(ret_type=Float()),
    r'CV$':
    dict(types={
        r'^param_grid$': Enum([ParamGridDict(),
                               List(ParamGridDict())])
    }),
    r'\.mean_(\w+)_error$':
    dict(
        types={
            r'^multioutput$':
            Enum([
                StringValue("raw_values"),
                StringValue("uniform_average"),
                Arr()
            ])
        })
}


def write_version(build_dir):
    import sklearn.utils
    full_version = sklearn.utils.validation.LooseVersion(
        sklearn.__version__).version
    full_version_ml = '[' + '; '.join([f'"{x}"' for x in full_version]) + ']'
    version_ml = '(' + ', '.join(str(x) for x in full_version[:2]) + ')'
    with open(build_dir / 'version.ml', 'w') as f:
        f.write(f'let full_version = {full_version_ml}\n')
        f.write(f'let version = {version_ml}\n')


def main():
    import sklearn

    # There are FutureWarnings about deprecated items. We want to
    # catch them in order not to wrap them.
    import warnings
    warnings.simplefilter('error', FutureWarning)

    over = Overrides(overrides)

    import pathlib
    build_dir = pathlib.Path('.')
    # build_dir.mkdir(parents=True, exist_ok=True)

    import sys
    if len(sys.argv) <= 1:
        print("skdoc.py: no argument passed, not doing anything")
        return

    mode = sys.argv[1]
    if mode == "build":
        pkg = Package(sklearn, over)
        pkg.write(build_dir)
        # Write this one separately, no sense having it in metrics and
        # it causes cross dep problems.
        # This is scipy.sparse.csr.csr_matrix.
        # The next lines are a horrible hack to have Csr_matrix use Ndarray
        # instead of Arr, therefore avoiding a dependency cycle.
        global Arr
        for name in Arr.names:
            builtin[name] = Ndarray()
        Arr = Ndarray

        Class(sklearn.metrics.pairwise.csr_matrix, "Sklearn",
              over).write(build_dir, "Sklearn", ns="sklearn.metrics.pairwise")
        write_version(build_dir)
    elif mode == "doc":
        pkg = Package(sklearn, over)
        pkg.write_doc(pathlib.Path('./doc'))
    elif mode == "examples":
        pkg = Package(sklearn, over)
        path = pathlib.Path('./_build/examples/auto/')
        print(f"extracting examples to {path}")
        pkg.write_examples(path)

    over.report_not_triggered()


if __name__ == '__main__':
    main()
