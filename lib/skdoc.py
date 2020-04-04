import re
import pkgutil
import importlib
import inspect
import collections

class Section:
    def __init__(self):
        pass


class Parameter:
    def __init__(self):
        pass


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
            text = getattr(self, 'elements', None)
        return f"{self.__class__.__name__}('{text}')"

    ml_type = 'Py.Object.t'
    wrap = 'Wrap_utils.id'

    ml_type_ret = 'Py.Object.t'
    unwrap = 'Wrap_utils.id'

    def tag(self):
        ta = tag(self.__class__.__name__)
        t = f"`{ta} of {self.ml_type}"
        destruct = f"`{ta} x -> {self.wrap} x"
        return t, destruct

class Bunch(Type):
    def __init__(self, elements):
        self.elements = elements
        self.ml_type = ("< " +
                        '; '.join(f"{mlid(name)}: {t.ml_type_ret}" for name, t in self.elements.items()) +
                        " >")
        # for now Bunch is only for returning things
        self.wrap = 'Wrap_utils.id' 

        self.ml_type_ret = self.ml_type
        unwrap_methods = [f'method {mlid(name)} = {t.unwrap} (Py.Object.get_attr_string bunch "{name}" |> Wrap_utils.Option.get)'
                          for name, t in self.elements.items()]
        self.unwrap = f'(fun bunch -> object {" ".join(unwrap_methods)} end)'
        
    def visit(self, f):
        f(self)
        for child in self.elements.values():
            f(child)

        
class UnknownType(Type):
    def __init__(self, text):
        self.text = text

    def tag(self):
        ta = tag(self.text)
        t = f"`{ta} of {self.ml_type}"
        destruct = f"`{ta} x -> {self.wrap} x"
        return t, destruct


class StringValue(Type):
    def __init__(self, text):
        self.text = text.strip("\"'")

    ml_type = 'string'
    wrap = 'Py.String.of_string'

    ml_type_ret = 'string'
    unwrap = 'Py.String.to_string'

    def tag(self):
        ta = tag(self.text)
        t = f"`{ta}"
        destruct = f'`{ta} -> {self.wrap} "{self.text}"'
        return t, destruct


class Enum(Type):
    def __init__(self, elements):
        self.elements = elements
        self.ml_type = "[" + ' | '.join([elt.tag()[0]
                                         for elt in elements]) + "]"
        wrap_cases = [f"| {elt.tag()[1]}\n" for elt in elements]
        self.wrap = f"(function\n{''.join(wrap_cases)})"

    def __str__(self):
        return repr(self)

    def __repr__(self):
        elts = ", ".join(map(str, self.elements))
        return f"{self.__class__.__name__}({elts})"


class RetTuple(Type):
    def __init__(self, elements):
        self.elements = elements
        self.ml_type_ret = ' * '.join(elt.ml_type_ret for elt in elements)
        unwrap_elts = [
            f"({elt.unwrap} (Py.Tuple.get x {i}))"
            for i, elt in enumerate(elements)
        ]
        self.unwrap = f"(fun x -> ({', '.join(unwrap_elts)}))"

    def __str__(self):
        return repr(self)

    def __repr__(self):
        elts = ", ".join(map(str, self.elements))
        return f"{self.__class__.__name__}({elts})"


# class EnumOption(Type):
#     def __init__(self, elements):
#         self.elements = elements

#     def __str__(self):
#         return repr(self)

#     def __repr__(self):
#         elts = ", ".join(map(str, self.elements))
#         return f"{self.__class__.__name__}({elts})"

# class StringEnum(Type):
# def __init__(self, elements):
#     self.elements = elements

# def __str__(self):
#     return repr(self)

# def __repr__(self):
#     elts = ", ".join(self.elements)
#     return f"{self.__class__.__name__}({elts})"


class Builtin(Type):
    def __init__(self):
        pass

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"{self.__class__.__name__}"


class Int(Builtin):
    names = ['int', 'integer', 'integer > 0']
    ml_type = 'int'
    wrap = 'Py.Int.of_int'
    ml_type_ret = 'int'
    unwrap = 'Py.Int.to_int'


class Float(Builtin):
    names = [
        'float', 'floating', 'double', 'positive float',
        'strictly positive float', 'non-negative float'
    ]
    ml_type = 'float'
    wrap = 'Py.Float.of_float'
    ml_type_ret = 'float'
    unwrap = 'Py.Float.to_float'


class Bool(Builtin):
    names = ['bool', 'boolean', 'Boolean', 'Bool']
    ml_type = 'bool'
    wrap = 'Py.Bool.of_bool'
    ml_type_ret = 'bool'
    unwrap = 'Py.Bool.to_bool'


class Ndarray(Builtin):
    names = [
        'ndarray', 'numpy array', 'array of floats', 'nd-array', 'array-like',
        'array', 'array_like', 'indexable', 'float ndarray'
    ]
    ml_type = 'Ndarray.t'
    wrap = 'Ndarray.to_pyobject'  # 'Numpy.of_bigarray'
    ml_type_ret = 'Ndarray.t'
    unwrap = 'Ndarray.of_pyobject'  #'(Numpy.to_bigarray Bigarray.float64 Bigarray.c_layout)'

    
class Ndarrayi(Builtin):
    names = [
        'ndarrayi',
    ]
    ml_type = 'Ndarrayi.t'
    wrap = 'Numpy.of_bigarray'
    ml_type_ret = 'Ndarrayi.t'
    unwrap = '(Numpy.to_bigarray Bigarray.nativeint Bigarray.c_layout)'
    
class Ndarrayi32(Builtin):
    names = [
        'ndarrayi32',
    ]
    ml_type = 'Ndarrayi32.t'
    wrap = 'Numpy.of_bigarray'
    ml_type_ret = 'Ndarrayi32.t'
    unwrap = '(Numpy.to_bigarray Bigarray.int32 Bigarray.c_layout)'

class Ndarrayi64(Builtin):
    names = [
        'ndarrayi64',
    ]
    ml_type = 'Ndarrayi64.t'
    wrap = 'Numpy.of_bigarray'
    ml_type_ret = 'Ndarrayi64.t'
    unwrap = '(Numpy.to_bigarray Bigarray.int64 Bigarray.c_layout)'

class List(Builtin):
    def __init__(self, t):
        self.t = t
        self.names = []
        self.ml_type = f"{t.ml_type} array"
        self.wrap = f'(fun ml -> Py.Array.of_array {t.wrap} (fun _ -> invalid_argument "read-only") ml)'
        self.ml_type_ret = f"{t.ml_type} array"
        self.unwrap = f"(fun py -> let len = Py.Sequence.length py in Array.init len (fun i -> {t.unwrap} (Py.Sequence.get_item py i)))"

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"List[{str(self.t)}]"
        
class FloatList(Builtin):
    names = ['list of floats']
    ml_type = 'float list'
    wrap = '(Py.List.of_list_map Py.Float.of_float)'


class StringList(Builtin):
    names = [
        'list of strings', 'list of string', 'list/tuple of strings',
        'tuple of str', 'list of str', 'strings'
    ]
    ml_type = 'string list'
    wrap = '(Py.List.of_list_map Py.String.of_string)'
    ml_type_ret = 'string list'
    unwrap = '(Py.List.to_list_map Py.String.to_string)'


class String(Builtin):
    names = ['str', 'string', 'unicode']
    ml_type = 'string'
    wrap = 'Py.String.of_string'
    ml_type_ret = 'string'
    unwrap = 'Py.String.to_string'


class ArrayLike(Builtin):
    names = ['list-like', 'list']


class NoneValue(Builtin):
    names = ['None', 'none']


class TrueValue(Builtin):
    names = ['True', 'true']


class FalseValue(Builtin):
    names = ['False', 'false']


class RandomState(Builtin):
    names = ['RandomState instance', 'instance of RandomState', 'RandomState']


class LinearOperator(Builtin):
    names = ['LinearOperator']


class CrossValGenerator(Builtin):
    names = ['cross-validation generator']


class Estimator(Builtin):
    names = [
        'instance BaseEstimator', 'BaseEstimator instance',
        'estimator instance', 'instance estimator', 'estimator object',
        'estimator'
    ]


class JoblibMemory(Builtin):
    names = ['object with the joblib.Memory interface']


class SparseMatrix(Builtin):
    names = ['sparse matrix', 'sparse-matrix', 'CSR matrix',
             'sparse graph in CSR format']
    ml_type = 'Csr_matrix.t'
    wrap = 'Csr_matrix.to_pyobject'
    ml_type_ret = 'Csr_matrix.t'
    unwrap = 'Csr_matrix.of_pyobject'

    
class PyObject(Builtin):
    names = ['object']


class Self(Builtin):
    names = ['self']
    ml_type = 't'
    ml_type_ret = 't'


class DType(Builtin):
    names = ['type']


class TypeList(Builtin):
    names = ['list of type', 'list of types']


class Iterable(Builtin):
    names = ['an iterable', 'iterable']


class Dict(Builtin):
    names = ['dict', 'Dict', 'dictionary']


# emitted as a postprocessing of Dict() based on param name/callable
class DictIntToFloat(Builtin):
    names = ['dict int to float']
    ml_type = '(int * float) list'
    wrap = '(Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float)'


class Callable(Builtin):
    names = ['callable', 'function']


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
    text = re.sub(r'[Dd]efaults\s+to\s+\S+\.?', '', text)
    text = re.sub(r'[Dd]efault\s+is\s+\S+\.?', '', text)
    text = re.sub(r'\(?\s*[Dd]efault\s*[:=]?\s*.+\)?$', '', text)
    text = re.sub(r'\s*optional', '', text)
    text = re.sub(r'\S+\s+by\s+default', '', text)
    return text


def remove_shape(text):
    text = re.sub(
        r'(of\s+)?shape\s*[:=]?\s*[([](\S+,\s*)*\S+?\s*[\])](\s*,?\s*or\s*[([](\S+,\s*)*\S+?\s*[\])])*',
        '', text)
    # two levels of parentheses should be enough for anyone
    text = re.sub(r'(of\s+)?shape\s*[:=]?\s*\([^()]*(\([^()]*\)[^()]*)?\)', '',
                  text)
    text = re.sub(r'(,\s*)?(of\s+)?length \S+', '', text)
    text = re.sub(r'(,\s*)?\[[^[\]()]+,[^[\]()]+\]', '', text)
    text = re.sub(r'if\s+\S+\s*=+\s*\S+\s*', '', text)
    text = re.sub(r'(,\s*)?\s*\d-dimensional\s*', '', text)
    return text


def is_string(x):
    if not x:
        return False
    return (x[0] == x[-1] and x[0] in ("'", '"')) or x in [
        "eigen'"
    ]  # hack for doc bug of RidgeCV


def parse_enum(t):
    """Love.
    """
    if not t:
        return None
    # print(t)
    elts = None
    m = re.match(r'^str(?:ing)?\s*(?:,|in|)\s*(\{.*\}|\[.*\]|\(.*\))$', t)
    if m is not None:
        return parse_enum(m.group(1))
    t = re.sub(r'(?:str\s*in\s*)?(\{[^}]+\})',
               lambda m: re.sub(r'[,|]|\s+or\s+', ' __OR__ ', m.group(1)), t)
    t = re.sub(r'(?:str\s*in\s*)?(\[[^}]+\])',
               lambda m: re.sub(r'[,|]|\s+or\s+', ' __OR__ ', m.group(1)), t)
    t = re.sub(r'(?:str\s*in\s*)?(\([^}]+\))',
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
    # print(elts)
    return elts


builtin_types = [
    Int(),
    Float(),
    Bool(),
    Ndarray(),
    Ndarrayi(),
    Ndarrayi32(),
    Ndarrayi64(),
    FloatList(),
    StringList(),
    SparseMatrix(),
    String(),
    Dict(),
    DictIntToFloat(),
    Callable(),
    Iterable(),
    DType(),
    TypeList(),
    ArrayLike(),
    NoneValue(),
    TrueValue(),
    FalseValue(),
    RandomState(),
    LinearOperator(),
    CrossValGenerator(),
    Estimator(),
    JoblibMemory(),
    PyObject(),
    Self()
]
builtin = {}
for t in builtin_types:
    for name in t.names:
        assert name not in builtin
        builtin[name] = t


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
        enum = type(enum)(not_none + [StringValue('None')])

    # Enum(String, StringValue, StringValue) should just be
    # Enum(StringValue, StringValue) since that is (probably) a
    # misparsing of "str, 'l1' or 'l2'"
    is_string_value, is_not_string_value = partition(
        enum.elements, lambda x: isinstance(x, StringValue))
    if len(is_not_string_value) == 1 and isinstance(is_not_string_value[0],
                                                    String):
        enum = type(enum)(is_string_value)

    enum = type(enum)(remove_duplicates(enum.elements))

    # Enum(NdArray, SparseMatrix) is annoying because it is all over
    # the place, and SparseMatrix is not easy to use from
    # OCaml. Keeping only NdArray for now.
    # OK sparse matrices seem useful in some cases, and are exposed through Sklearn.Csr_matrix.
    # if len(enum.elements) == 2:
    #     a, b = enum.elements
    #     if ((isinstance(a, Ndarray) and isinstance(b, SparseMatrix))
    #             or (isinstance(b, Ndarray) and isinstance(a, SparseMatrix))):
    #         return builtin['ndarray']

    # There is no point having more than one Py.Object tag in an enum.
    is_obj, is_not_obj = partition(
        enum.elements, lambda x: isinstance(x, (PyObject, UnknownType)))
    if len(is_obj) >= 1:
        enum = type(enum)(is_not_obj + [builtin['object']])

    # this is probably the result of a parsing bug
    if len(enum.elements) == 1:
        return enum.elements[0]

    return enum


def remove_ranges(t):
    t = re.sub(r'(float|int)\s*(in\s*(range\s*)?)\s*[()[\]][^\]]+[()[\]]',
               r'\1', t)
    t = re.sub(r'(float|int)\s*,?\s*(greater|smaller)\s+than\s+\d+(\.\d+)?',
               r'\1', t)
    t = re.sub(r'(float|int)\s*,?\s*(<=|<|>=|>)\s*\d+(\.\d+)?', r'\1', t)
    t = re.sub(r'(float|int)\s*,?\s*\(\s*(<=|<|>=|>)\s*\d+(\.\d+)?\s*\)',
               r'\1', t)
    t = t.strip(" \n\t,.;")
    return t


def parse_type(t, function, param_name):
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
        if isinstance(ret, ArrayLike):
            if param_name in ['filenames']:
                ret = builtin['list of string']
        if isinstance(ret, (Ndarray, ArrayLike)):
            if param_name in ['feature_names', 'target_names']:
                ret = builtin['list of string']
            if param_name in ['neigh_ind', 'is_inlier']:
                ret = builtin['ndarrayi']
        return ret

    if param_name in ['DESCR']:
        return builtin['string']

    if param_name in ['target_names']:
        return builtin['list of string']

    if getattr(function, '__name__', '').startswith('fetch_'):
        if param_name in ['data', 'target']:
            return builtin['ndarray']
        if param_name in ['pairs'] and t.startswith('numpy array'):
            return builtin['ndarray']
        if param_name in ['images'] and t.startswith('ndarray'):
            return builtin['ndarray']

    if param_name in ['intercept_', 'coef_']:
        return builtin['ndarray']
        
    elts = parse_enum(t)
    if elts is not None:
        if all([is_string(elt) for elt in elts]):
            return Enum([StringValue(e.strip("\"'")) for e in elts])
        elts = [parse_type(elt, function, param_name) for elt in elts]
        # print("parsed enum elts:", elts)
        ret = Enum(elts)
        ret = simplify_enum(ret)
        return ret
    elif is_string(t):
        return StringValue(t.strip("'\""))
    else:
        return UnknownType(t)

    
def parse_bunch(function, elements):
    print(f"parsing bunch: {elements}")
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
                ret[name] = parse_type(value, function, name)
    ret = Bunch(ret)
    print(f"-> {ret}")
    return ret


def parse_types(function, doc, section='Parameters'):
    function_name = getattr(function, '__name__', '<no function name>')
    qualname = getattr(function, '__qualname__', '<no function name>')

    if section == 'Returns':
        if qualname in ['RadiusNeighborsMixin.radius_neighbors']:
            return {'ret': RetTuple([List(Ndarray()), List(Ndarrayi())])}
        # 'neigh_ind': builtin['ndarrayi']}
        if qualname in ['NearestCentroid.predict', 'NearestCentroid.fit_predict']:
            return {'y': builtin['ndarrayi']}
        if function_name in ['decision_function', 'predict', 'predict_proba', 'fit_predict']:
            # XXX too wide a net?
            return {'y': builtin['array']}
        if function_name in ['__str__']:
            return {'return': builtin['string']}
        if function_name in ['fit']:
            return {'self': builtin['self']}
        
    if section == 'Parameters':
        if function_name in ['__str__']:
            return {'N_CHAR_MAX': builtin['int']}
        if getattr(function, '__qualname__', None) == 'NearestCentroid.fit':
            return { 'X': Enum([builtin['ndarray'], builtin['sparse matrix']]),
                     'y': builtin['ndarrayi']}
        
    # print(doc)
    if doc is None:
        return {}
    elements = parse_params(doc, section)

    if section == 'Returns' and function_name.startswith('fetch_'):
        return {'data': parse_bunch(function, elements) }

    # if section == 'Returns':
    #     print(elements)
    ret = {}
    for element in elements:
        # print(element)
        try:
            text = element.text.strip()
            if not text or text.startswith('..'):
                continue
            m = re.match(r'^(\w+)\s*:\s*([^\n]+)', text)
            if m is None:
                # print(f"not a param: '{text}'")
                mm = re.match(r'^(\w+)\s*\n', text)
                if mm is not None:
                    # print(f"-> found simple param: {mm.group(1)}")
                    ret[mm.group(1)] = UnknownType('')
                    continue
                # print(f"failed to parse param description: {element.text}")
                continue
            param_name = m.group(1)
            value = m.group(2)
            value = remove_default(value)
            # print(f"value without default: {value}")
            value = remove_shape(value)
            value = value.strip(" \n\t,.;")
            if value.startswith('ref:'):
                continue
            # value = re.sub(r'\s*\(\)\s*', '', value)
            ty = parse_type(value, function, param_name)
            ret[param_name] = ty
            # if 'scale' in value:
            #     print(f"parsing type in '{value}': {ty}")

        except Exception as e:
            print(f"!!!!!!!!!!!!! error processing element: {element}")
            print(e)
            raise
        # if 'self' not in ret:
        #     ret['self'] = builtin['self']
    return ret


def classes_in(m):
    return [getattr(m, x) for x in dir(m) if not x.startswith('_')]


def flatten(l):
    return [x for ll in l for x in ll]


def unknown_types(types):
    ret = []

    def visit(t):
        if isinstance(t, UnknownType):
            ret.append(t)

    for t in types:
        t.visit(visit)
    return ret


def modules(pkg):
    if isinstance(pkg, list):
        return pkg

    mods = [
        x.name for x in pkgutil.iter_modules(pkg.__path__)
        if not x.name.startswith('_')
    ]
    prefix = pkg.__name__ + "."
    mods = [prefix + m for m in mods]
    return mods


def report_unknown(pkg):
    all_types = []
    mods = modules(pkg)
    for m in mods:
        print(f"--------- processing module {m}")
        module = importlib.import_module(m)
        classes = classes_in(module)
        types = [
            parse_types(c, c.__doc__) for c in classes if c.__doc__ is not None
        ]
        ts = flatten([list(t.values()) for t in types])
        # unknown_ts = unknown_types(ts)
        # if unknown_ts:
        #     print(f"unknown types: {len(unknown_ts)}, {unknown_ts}")
        all_types += ts
    unk = unknown_types(all_types)
    print(f"overall unknown types: {len(unk)}")
    return unk, all_types


# import sklearn.linear_model
# report_unknown(['sklearn.linear_model'])
# import sklearn
# unk, all_types = report_unknown(sklearn)
# unk


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
    except (NoSignature, OutsideScope, NoDoc) as e:
        # print(f"WW append: caught error building {ctor}: {e}")
        return
    container.append(elt)


class Package:
    def __init__(self, pkg):
        self.pkg = pkg
        self.modules = self._list_modules(pkg)

    def _list_modules(self, pkg):
        ret = []
        for mod in pkgutil.iter_modules(pkg.__path__):
            name = mod.name
            if name.startswith('_'):
                continue
            full_name = f"{pkg.__name__}.{name}"
            module = importlib.import_module(full_name)
            ret.append(Module(
                module,
                parent_name=""))  # not passing Sklearn as parent, too long
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
        print(f"writing package {self.pkg.__name__} to {dire}")
        dire.mkdir(parents=True, exist_ok=True)
        for mod in self.modules:
            mod.write(dire)

    def write_doc(self, path):
        dire = self.output_dir(path)
        dire.mkdir(parents=True, exist_ok=True)
        for mod in self.modules:
            mod.write_doc(dire)

    def write_examples(self, path):
        dire = self.output_dir(path)
        dire.mkdir(parents=True, exist_ok=True)
        for mod in self.modules:
            mod.write_examples(dire)


class OutsideScope(Exception):
    pass


class NoDoc(Exception):
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
    # DESCR -> descr
    if s == s.upper():
        return s.lower()
    s = lcfirst(s)
    if s in ml_keywords:
        return s + '_'
    return s


class Module:
    def __init__(self, module, parent_name):
        self.full_python_name = module.__name__
        if not self.full_python_name.startswith('sklearn'):
            raise OutsideScope
        if '.externals.' in self.full_python_name:
            raise OutsideScope

        self.parent_name = parent_name
        self.python_name = module.__name__.split('.')[-1]
        self.ml_name = ucfirst(self.python_name)
        self.full_ml_name = f"{parent_name}.{self.ml_name}"
        # print(f"building module {self.full_python_name}")
        self.module = module
        self.elements = self._list_elements(module)

    def _list_elements(self, module):
        elts = []
        for name in dir(module):
            if name.startswith('_'):
                continue
            # emit this one separately, not inside sklearn.metrics
            if name == "csr_matrix":
                continue
            item = getattr(module, name)
            parent_name = self.full_ml_name
            if inspect.ismodule(item):
                append(elts, Module, item, parent_name)
            elif inspect.isclass(item):
                append(elts, Class, item, parent_name)
            elif callable(item):
                append(elts, Function, name, item, self.ml_name)
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
        if self.has_callables():
            f.write("let () = Wrap_utils.init ();;\n")
            f.write(f'let ns = Py.import "{self.module.__name__}"\n\n')
        else:
            f.write(
                "(* this module has no callables, skipping init and ns *)\n")

    def write(self, path):
        ml = f"{path / self.python_name}.ml"
        with open(ml, 'w') as f:
            self.write_header(f)
            for element in self.elements:
                element.write_to_ml(f)
        mli = f"{path / self.python_name}.mli"
        with open(mli, 'w') as f:
            for element in self.elements:
                element.write_to_mli(f)

    def write_doc(self, path):
        md = f"{path / self.python_name}.md"
        with open(md, 'w') as f:
            for element in self.elements:
                element.write_to_md(f)

    def write_examples(self, path):
        md = f"{path / self.python_name}.ml"
        with open(md, 'w') as f:
            for element in self.elements:
                element.write_examples_to(f)

    def write_to_ml(self, f):
        f.write(f"module {self.ml_name} = struct\n")
        self.write_header(f)
        for element in self.elements:
            element.write_to_ml(f)
        f.write("\nend\n")

    def write_to_mli(self, f):
        f.write(f"module {self.ml_name} : sig\n")
        for element in self.elements:
            element.write_to_mli(f)
        f.write("\nend\n\n")

    def write_to_md(self, f):
        full_name = f"{self.parent_name}.{self.ml_name}"
        f.write(f"## module {full_name}\n")
        for element in self.elements:
            element.write_to_md(f)

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
    def __init__(self, klass, parent_name):
        self.klass = klass
        self.parent_name = parent_name
        self.constructor = Ctor(self.klass.__name__, self.klass, ucfirst(self.klass.__name__))
        self.elements = self._list_elements()
        
    def _list_elements(self):
        elts = []
        callables = set(
        )  # there may be several times the same function behind the
        # same name Some classes don't have all the methods ready, you
        # need to instantiate an object. For instance,
        # sklearn.neighbors.LocalOutlierFactor has fit_predict as a
        # property (not yet a method), so taking its signature does
        # not work.  So we attempt to instantiate an object and work
        # on that, and fallback on the class if that fails.
        try:
            proto = self.klass()
        except Exception as e:
            print(f"instantiating {self.klass.__name__} did not work: {e}")
            proto = self.klass
        for name in dir(proto):
            if name.startswith('_'):  # and name not in ['__str__']:
                continue
            item = getattr(proto, name, None)
            if item is None:
                continue
            if callable(item):
                if is_hashable(item):
                    if item not in callables:
                        append(elts, Method, name, item, ucfirst(self.klass.__name__))
                        callables.add(item)
                else:
                    append(elts, Method, name, item, ucfirst(self.klass.__name__))
        attributes = parse_types(self.klass,
                                 self.klass.__doc__,
                                 section="Attributes")
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
        # A class needs no import.
        # if self.elements:
        #     f.write(f'let ns = Py.import "{self.klass.__module__}.{self.klass.__name__}"\n\n')

    # XXX TODO there is some copy-paste with Module, share it
    # XXX TODO also there is some ugly copy-paste with write_to_ml/mli
    def write(self, path, module):
        name = self.klass.__name__
        ml = f"{path / name}.ml"
        with open(ml, 'w') as f:
            f.write("let () = Wrap_utils.init ();;\n")
            f.write(f'let ns = Py.import "{module.__name__}"\n\n')
            
            self.write_header(f)
            self.constructor.write_to_ml(f)
            for element in self.elements:
                element.write_to_ml(f)
            f.write("let to_string self = Py.Object.to_string self\n")
            f.write('let show self = to_string self\n')
            f.write(
                'let pp formatter self = Format.fprintf formatter "%s" (show self)\n'
            )
                
        mli = f"{path / name}.mli"
        with open(mli, 'w') as f:
            f.write("type t\n")
            f.write("val of_pyobject : Py.Object.t -> t\n")
            f.write("val to_pyobject : t -> Py.Object.t\n\n")
            self.constructor.write_to_mli(f)
            for element in self.elements:
                element.write_to_mli(f)
            f.write(
                "\n(** Print the object to a human-readable representation. *)\n")
            f.write("val to_string : t -> string\n\n")
            f.write(
                "\n(** Print the object to a human-readable representation. *)\n")
            f.write("val show : t -> string\n\n")
            f.write("(** Pretty-print the object to a formatter. *)\n")
            f.write("val pp : Format.formatter -> t -> unit\n\n")


    def write_doc(self, path):
        name = self.klass.__name__
        md = f"{path / name}.md"
        with open(md, 'w') as f:
            self.constructor.write_to_md(f)
            for element in self.elements:
                element.write_to_md(f)

    def write_to_ml(self, f):
        name = self.klass.__name__
        name = ucfirst(name)
        f.write(f"module {name} = struct\n")
        self.write_header(f)
        self.constructor.write_to_ml(f)
        for element in self.elements:
            element.write_to_ml(f)
        f.write("let to_string self = Py.Object.to_string self\n")
        f.write('let show self = to_string self\n')
        f.write(
            'let pp formatter self = Format.fprintf formatter "%s" (show self)\n'
        )
        f.write("\nend\n")

    def write_to_mli(self, f):
        name = self.klass.__name__
        name = ucfirst(name)
        f.write(f"module {name} : sig\n")
        f.write("type t\n")
        f.write("val of_pyobject : Py.Object.t -> t\n")
        f.write("val to_pyobject : t -> Py.Object.t\n\n")
        self.constructor.write_to_mli(f)
        for element in self.elements:
            element.write_to_mli(f)
        f.write(
            "\n(** Print the object to a human-readable representation. *)\n")
        f.write("val to_string : t -> string\n\n")
        f.write(
            "\n(** Print the object to a human-readable representation. *)\n")
        f.write("val show : t -> string\n\n")
        f.write("(** Pretty-print the object to a formatter. *)\n")
        f.write("val pp : Format.formatter -> t -> unit\n\n")
        f.write("\nend\n\n")

    def _write_fun_md(self, f, name, sig, doc):
        f.write(f"### {name}\n")
        f.write("```ocaml\n")
        f.write(f"val {name} : {sig}\n")
        f.write("```\n")
        f.write(doc)
        f.write("\n")

    def write_to_md(self, f):
        name = self.klass.__name__
        name = ucfirst(name)

        full_name = f"{self.parent_name}.{name}"
        f.write(f"## module {full_name}\n")
        f.write("```ocaml\n")
        f.write("type t\n")
        f.write("```\n")
        self.constructor.write_to_md(f)
        for element in self.elements:
            element.write_to_md(f)
        self._write_fun_md(
            f, 'show', 't -> string',
            'Print the object to a human-readable representation.')
        self._write_fun_md(f, 'pp', 'Format.formatter -> t -> unit',
                           'Pretty-print the object to a formatter.')

    def write_examples_to(self, f):
        write_examples(f, self.klass)
        for element in self.elements:
            element.write_examples_to(f)


class Attribute:
    def __init__(self, name, typ):
        self.name = name
        self.ml_name = mlid(self.name)
        self.typ = typ

    def write_to_ml(self, f):
        f.write(f"let {self.ml_name} self =\n")
        f.write(f'  match Py.Object.get_attr_string self "{self.name}" with\n')
        f.write(
            f'| None -> raise (Wrap_utils.Attribute_not_found "{self.name}")\n'
        )
        f.write(f'| Some x -> {self.typ.unwrap} x\n')

    def write_to_mli(self, f):
        #  XXX TODO extract doc and put it here
        f.write(
            f"\n(** Attribute {self.name}: see constructor for documentation *)\n"
        )
        f.write(f"val {self.ml_name} : t -> {self.typ.ml_type_ret}\n")

    def write_to_md(self, f):
        f.write(f"### {self.ml_name}\n")
        f.write("```ocaml\n")
        f.write(f"val {self.ml_name} : t -> {self.typ.ml_type_ret}\n")
        f.write("```\n\n")

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
                 r'\n???+ note "\2"',
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
    groups = re.findall(r'^(?:>>>(?:[^\n]|\n[^\n])+)\n\n',
                        doc,
                        flags=re.MULTILINE)
    for group in groups:
        # ex = re.sub('^(>>>|\.\.\.)\s*', '', group, flags=re.MULTILINE)
        ex = group
        yield ex


class Input:
    def __init__(self):
        self.text = ''

    def append(self, line):
        line = re.sub(r'^(>>>|\.\.\.)\s*', '', line)
        line.strip()
        self.text += line

    def write(self, f):
        m = re.match(r'^from (\S+)\s+import\s+(\S+)$', self.text)
        if m is not None:
            ns = '.'.join([ucfirst(x) for x in m.group(1).split('.')])
            name = mlid(m.group(2))
            f.write(f"let {name} = {ns}.{name} in\n")
            return

        m = re.match(r'^\s*(\w+(?:\s*,\s*\w+))*\s*=\s*(\w+)\((.*)\)\s*$',
                     self.text)
        if m is not None:
            names = [mlid(x.strip()) for x in m.group(1).split(',')]
            fun = m.group(2)
            args = [mlid(x.strip()) for x in m.group(3).split(',')]
            f.write(f"let {', '.join(names)} = {fun} {' '.join(args)} in\n")
            return

        m = re.match(
            r'^\s*(\w+)\.(\w+)\(\s*([^(),]+(?:\s*,\s*[^(),]+)*)\s*\)\s*$',
            self.text)
        if m is not None:
            obj = m.group(1)
            meth = m.group(2)
            args = ' '.join([mlid(x.strip()) for x in m.group(3).split(',')])
            f.write(f"print @@ {meth} {obj} {args}\n")
            return

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
        f.write("|}]\n")


class Indented:
    def __init__(self, f, indent=4):
        self.f = f
        self.indent = ' ' * indent

    def write(self, x):
        self.f.write(self.indent)
        self.f.write(x)


class Example:
    def __init__(self, source, name):
        self.name = name
        self.elements = []
        for line in source.split("\n"):
            if line.startswith('>>>'):
                element = Input()
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
        f.write("(* TEST TODO\n")
        f.write(f'let%expect_test "{self.name}" =\n')
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
    return RetTuple(list(type_dict.values()))


class NoSignature(Exception):
    pass


class Function:
    def __init__(self, python_name, function, module_name):
        self.function = function
        self.python_name = python_name
        self.module_name = module_name
        try:
            self.signature = inspect.signature(self.function)
        except ValueError:
            raise NoSignature(self.function)
        if python_name is None:
            raise NoSignature(self.function)
        self.doc = clean_doc(inspect.getdoc(self.function))
        if self.doc is None or not self.doc:
            raise NoDoc()
        # does not work atm with getdoc() because of indentation
        # parsing in my code
        doc = str(self.function.__doc__)
        self.types = parse_types(self.function, doc)
        # print(f"return type for {self.function.__name__}: {self.return_type}")
        for k, v in self.signature.parameters.items():
            if k not in self.types:
                self.types[k] = UnknownType('<not found in doc>')
                # self.types[' return'] = builtin['object']

        return_values = parse_types(self.function, doc, section="Returns")
        # print(f"{self.python_name} return values: {return_values}")

        self.return_type = "<not yet determined>"

        function_name = getattr(self.function, '__name__', '<no name>')
        
        # XXX TODO idea for return_: fix everything to true,
        # unless return_X_y: fix to false
        self.fixed_values = {'return_X_y': ('false', True)}
        
        if function_name == 'make_regression':
            self.fixed_values['coef'] = ('true', False)

        # kneighbors()
        self.fixed_values['return_distance'] = ('true', False)
        
        for name, has_default, t, fixed_value in self.arguments():
            if name in self.fixed_values:
                continue
            if 'return' in name:
                print(
                    f"WW {self.function}: return arg {name}, {has_default}, {t}: {self}"
                )
                # m = re.match(r'^return_(.*)$', name)
                # assert m is not None
                # ret_name = m.group(1)
                # if ret_name not in return_values:
                #     print(
                #         f"!! {ret_name} not in return values {return_values}")
                # else:
                #     print(
                #         f"II removing {ret_name} from return values {return_values}"
                #     )
                #     del return_values[ret_name]
        for name in self.fixed_values:
            _, remove = self.fixed_values[name]
            if remove and name in return_values:
                del return_values[name]

        self.return_type = make_return_type(return_values)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        ret = f"{type(self).__name__}({self.python_name})(\n"
        params = ""
        for k, v in self.signature.parameters.items():
            params += k
            if v.default is not inspect.Parameter.empty:
                params += f"={v.default}"
            params += f" : {self.types[k]},\n"
        params += f" -> {self.return_type}"
        ret += indent(params)
        ret += ")"
        return ret

    def arguments(self):
        for k, v in self.signature.parameters.items():
            if v.kind == inspect.Parameter.VAR_KEYWORD: # **kwargs
                continue
            has_default = (v.default is not inspect.Parameter.empty)
            ty = self.types[k]
            fixed_value, _ = self.fixed_values.get(k, (None, False))
            yield k, has_default, ty, fixed_value

    def has_default(self):
        "Whether the function has any default argument."
        for k, has_default, ty, fixed_value in self.arguments():
            if has_default:
                return True
        return False

    def num_params(self):
        ret = len(self.signature.parameters)
        if getattr(self.function, '__self__', None) is not None:
            ret += 1
        return ret
    
    def needs_unit_param(self):
        return self.has_default() or not self.num_params()

    def ml_name(self):
        # if self.python_name == '__str__':
        #     return 'to_string'
        return mlid(self.python_name)

    def ns(self):
        return 'ns'

    def write_to_ml(self, f):
        f.write(f"let {self.ml_name()}")
        for param, has_default, ty, fixed_value in self.arguments():
            if fixed_value is not None:
                continue
            param_ml = mlid(param)
            default_mark = ['~', '?'][has_default]
            if param == 'self':
                default_mark = ''
            f.write(f" {default_mark}{param_ml}")
        if self.needs_unit_param():
            f.write(" ()")
        f.write(" =\n")
        f.write(
            f'Py.Module.get_function_with_keywords {self.ns()} "{self.python_name}" [||] (Wrap_utils.keyword_args [\n'
        )
        for param, has_default, ty, fixed_value in self.arguments():
            # self is not passed as an arg to Python, it is already
            # the arg to get_function_with_keywords()
            if param == "self":
                continue
            ml_type = ty.ml_type
            if ml_type == f"{self.module_name}.t":
                ml_type = 't'
            param_ml = mlid(param)
            ty_wrap = re.sub('^' + f'{self.module_name}' + '\\.', '', ty.wrap)
            if fixed_value is not None:
                f.write(f'"{param}", Some ({ty_wrap} {fixed_value});\n')
                continue

            if has_default:
                f.write(
                    f'"{param}", Wrap_utils.Option.map {param_ml} {ty_wrap};\n'
                )
            else:
                f.write(f'"{param}", Some({param_ml} |> {ty_wrap});\n')

        # XXX TODO factor out this manipulation
        unwrap = re.sub('^' + f'{self.module_name}' + '\\.', '', self.return_type.unwrap)
        f.write(f"]) |> {unwrap}\n")

    def write_to_mli(self, f):
        f.write(f"(** {self.doc}\n**)\n")
        f.write(f"val {self.ml_name()} : ")
        for param, has_default, ty, fixed_value in self.arguments():
            if fixed_value is not None:
                continue
            param_ml = mlid(param)
            default_mark = ['', '?'][has_default]
            ml_type = ty.ml_type
            if ml_type == f"{self.module_name}.t":
                ml_type = 't'
            if param == "self":
                f.write(f" {ml_type} ->")
            else:
                f.write(f" {default_mark}{param_ml} : {ml_type} ->")
        if self.needs_unit_param():
            f.write(" unit ->")
        return_type = self.return_type.ml_type_ret
        if return_type == f"{self.module_name}.t":
            return_type = 't'
        f.write(f" {return_type}\n\n")

    def write_to_md(self, f):
        f.write(f"### {self.ml_name()}\n")
        f.write("```ocaml\n")
        f.write(f"val {self.ml_name()} : ")
        for param, has_default, ty, fixed_value in self.arguments():
            if fixed_value is not None:
                continue
            param_ml = mlid(param)
            default_mark = ['', '?'][has_default]
            f.write(f"\n    {default_mark}{param_ml} : {ty.ml_type} ->")
        if self.needs_unit_param():
            f.write("\n    unit ->")
        return_type = self.return_type.ml_type_ret
        f.write(f"\n    {return_type}\n")
        f.write("```\n")
        f.write(format_md_doc(self.doc))
        f.write("\n")

    def write_examples_to(self, f):
        write_examples(f, self.function)


class Method(Function):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.types['self'] = builtin['self']
        args = list(self.arguments())
        if getattr(self.function, '__self__', None) is None and ((not args) or args[0][0] != 'self'):
            print(
                f"!! skipping {self.function}({args}): first arg is not self")
            raise OutsideScope()

    def arguments(self):
        if getattr(self.function, '__self__', None) is not None:
            yield 'self', False, builtin['self'], None
        for arg in super().arguments():
            yield arg
        
    def ns(self):
        return 'self'


class Ctor(Function):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_type = builtin['self']

    def ml_name(self):
        return 'create'


def main():
    import sklearn
    pkg = Package(sklearn)

    import pathlib
    build_dir = pathlib.Path('.')
    # build_dir.mkdir(parents=True, exist_ok=True)

    import sys
    mode = sys.argv[1]
    if mode == "build":
        pkg.write(build_dir)
        # Write this one separately, no sense having it in metrics and
        # it causes cross dep problems.
        # This is scipy.sparse.csr.csr_matrix.
        Class(sklearn.metrics.pairwise.csr_matrix, "Sklearn").write(build_dir, sklearn.metrics.pairwise)
    elif mode == "doc":
        pkg.write_doc(pathlib.Path('./doc'))
    elif mode == "examples":
        path = pathlib.Path('./examples/auto/')
        print(f"extracting examples to {path}")
        pkg.write_examples(path)


if __name__ == '__main__':
    main()

# import sklearn
# import pathlib
# pkg = Package(sklearn)
# pkg.write_examples(pathlib.Path('./auto-examples'))
